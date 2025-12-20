#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
CI sanity check: export a tiny GPT-2-like model to `.nfpt` and compare one forward step
against PyTorch.

This is meant to be cheap enough for CI while still catching semantic drift (e.g. missing
attention biases, wrong GeLU constant, etc.).

Usage (CI):
  python3 scripts/ci_sanity_forward_step.py --model sshleifer/tiny-gpt2 --seqLen 8
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

try:
    import torch
    from transformers import GPT2Model
except ImportError as e:
    raise SystemExit(
        "Missing deps. Install with e.g.: pip install torch transformers\n"
        f"ImportError: {e}"
    )


def write_header(f, **fields: object) -> None:
    f.write(b"NFP_BINARY_V1\n")
    for key, value in fields.items():
        f.write(f"{key}={value}\n".encode("ascii"))
    f.write(b"BINARY_START\n")


def write_i32(f, data: np.ndarray) -> None:
    arr = np.ascontiguousarray(data, dtype="<i4")
    f.write(arr.tobytes(order="C"))


def write_f64(f, data: np.ndarray) -> None:
    arr = np.ascontiguousarray(data, dtype="<f8")
    f.write(arr.tobytes(order="C"))


def get_bias(param, size: int) -> np.ndarray:
    if param is None:
        return np.zeros(size, dtype=np.float64)
    return param.detach().cpu().numpy()


def export_nfpt(model: GPT2Model, seq_len: int, out_path: Path) -> np.ndarray:
    cfg = model.config
    num_layers = int(cfg.n_layer)
    num_heads = int(cfg.n_head)
    model_dim = int(cfg.n_embd)
    head_dim = model_dim // num_heads
    hidden_dim = int(cfg.n_inner or 4 * model_dim)
    vocab_size = int(cfg.vocab_size)
    layer_norm_eps = float(cfg.layer_norm_epsilon)

    # Deterministic token sequence.
    tokens = (np.arange(seq_len, dtype=np.int64) % min(32, vocab_size)).astype(np.int64)

    # GPT-2 embeddings: wte[token] + wpe[pos].
    wte = model.wte.weight.detach().cpu().numpy()
    wpe = model.wpe.weight.detach().cpu().numpy()
    embeddings = wte[tokens] + wpe[:seq_len]

    with out_path.open("wb") as f:
        write_header(
            f,
            num_layers=num_layers,
            num_heads=num_heads,
            model_dim=model_dim,
            head_dim=head_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            seq_len=seq_len,
            layer_norm_eps=layer_norm_eps,
        )

        write_i32(f, tokens)
        write_f64(f, embeddings)

        for layer_idx in range(num_layers):
            block = model.h[layer_idx]

            c_attn_w = block.attn.c_attn.weight.detach().cpu().numpy()  # (d, 3d)
            c_attn_b = get_bias(block.attn.c_attn.bias, 3 * model_dim)  # (3d,)
            c_proj_w = block.attn.c_proj.weight.detach().cpu().numpy()  # (d, d)
            c_proj_b = get_bias(block.attn.c_proj.bias, model_dim)  # (d,)

            W_Q_all = c_attn_w[:, 0:model_dim]
            W_K_all = c_attn_w[:, model_dim : 2 * model_dim]
            W_V_all = c_attn_w[:, 2 * model_dim : 3 * model_dim]
            b_Q_all = c_attn_b[0:model_dim]
            b_K_all = c_attn_b[model_dim : 2 * model_dim]
            b_V_all = c_attn_b[2 * model_dim : 3 * model_dim]

            for h in range(num_heads):
                start, end = h * head_dim, (h + 1) * head_dim
                write_f64(f, W_Q_all[:, start:end])
                write_f64(f, b_Q_all[start:end])
                write_f64(f, W_K_all[:, start:end])
                write_f64(f, b_K_all[start:end])
                write_f64(f, W_V_all[:, start:end])
                write_f64(f, b_V_all[start:end])
                write_f64(f, c_proj_w[start:end, :])

            write_f64(f, c_proj_b)

            write_f64(f, block.mlp.c_fc.weight.detach().cpu().numpy())
            write_f64(f, get_bias(block.mlp.c_fc.bias, hidden_dim))
            write_f64(f, block.mlp.c_proj.weight.detach().cpu().numpy())
            write_f64(f, get_bias(block.mlp.c_proj.bias, model_dim))

            write_f64(f, block.ln_1.weight.detach().cpu().numpy())
            write_f64(f, get_bias(block.ln_1.bias, model_dim))
            write_f64(f, block.ln_2.weight.detach().cpu().numpy())
            write_f64(f, get_bias(block.ln_2.bias, model_dim))

        write_f64(f, model.ln_f.weight.detach().cpu().numpy())
        write_f64(f, get_bias(model.ln_f.bias, model_dim))
        write_f64(f, wte.T)

    return tokens


def run_lean_dump(model_path: Path, layer: int, pos: int, take: int) -> np.ndarray:
    exe = Path(".lake/build/bin/nfp")
    if not exe.exists():
        raise SystemExit("Missing `.lake/build/bin/nfp`. Run `lake build nfp` first.")
    cmd = [
        str(exe),
        "dump",
        "--kind",
        "afterLayer",
        "--layer",
        str(layer),
        "--pos",
        str(pos),
        "--take",
        str(take),
        str(model_path),
    ]
    out = subprocess.check_output(cmd, text=True)
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    dump_i = None
    for i, ln in enumerate(lines):
        if ln.startswith("DUMP "):
            dump_i = i
    if dump_i is None or dump_i + 2 >= len(lines):
        raise SystemExit(f"Unexpected Lean dump output:\n{out}")
    vals = [float(x) for x in lines[dump_i + 2].split()]
    return np.asarray(vals, dtype=np.float64)


def run_torch(model: GPT2Model, tokens: np.ndarray, layer: int, pos: int, take: int) -> np.ndarray:
    input_ids = torch.tensor(tokens.reshape(1, -1), dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    hs = out.hidden_states
    if hs is None:
        raise SystemExit("Expected output_hidden_states=True to return hidden_states.")
    x = hs[layer + 1][0, pos, :take].detach().cpu().numpy()
    return x.astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sshleifer/tiny-gpt2")
    ap.add_argument("--seqLen", type=int, default=8)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--pos", type=int, default=0)
    ap.add_argument("--take", type=int, default=16)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    m = GPT2Model.from_pretrained(args.model)
    m = m.to(dtype=torch.float64)
    m.eval()

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ci_tiny.nfpt"
        toks = export_nfpt(m, args.seqLen, path)
        lean = run_lean_dump(path, args.layer, args.pos, args.take)
        torch_x = run_torch(m, toks, args.layer, args.pos, args.take)
        diff = float(np.max(np.abs(lean - torch_x)))
        print(f"max_abs_diff={diff:.6g}  tol={args.tol:.6g}")
        if not np.isfinite(diff) or diff > args.tol:
            print("FAIL")
            print("lean :", lean.tolist())
            print("torch:", torch_x.tolist())
            raise SystemExit(1)
        print("OK")


if __name__ == "__main__":
    main()
