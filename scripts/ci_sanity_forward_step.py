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


def write_matrix(f, m: np.ndarray) -> None:
    for row in m:
        f.write(" ".join(f"{float(x):.9g}" for x in row) + "\n")


def write_vector(f, v: np.ndarray) -> None:
    f.write(" ".join(f"{float(x):.9g}" for x in v.reshape(-1)) + "\n")


def write_tokens(f, t: np.ndarray) -> None:
    s = [str(int(x)) for x in t.reshape(-1)]
    for i in range(0, len(s), 64):
        f.write(" ".join(s[i : i + 64]) + "\n")


def export_nfpt(model: GPT2Model, seq_len: int, out_path: Path) -> np.ndarray:
    cfg = model.config
    num_layers = int(cfg.n_layer)
    num_heads = int(cfg.n_head)
    model_dim = int(cfg.n_embd)
    head_dim = model_dim // num_heads
    hidden_dim = int(cfg.n_inner or 4 * model_dim)
    vocab_size = int(cfg.vocab_size)

    # Deterministic token sequence.
    tokens = (np.arange(seq_len, dtype=np.int64) % min(32, vocab_size)).astype(np.int64)

    # GPT-2 embeddings: wte[token] + wpe[pos].
    wte = model.wte.weight.detach().cpu().numpy()
    wpe = model.wpe.weight.detach().cpu().numpy()
    embeddings = wte[tokens] + wpe[:seq_len]

    with out_path.open("w") as f:
        f.write("NFP_TEXT_V2\n")
        f.write(f"num_layers={num_layers}\n")
        f.write(f"num_heads={num_heads}\n")
        f.write(f"model_dim={model_dim}\n")
        f.write(f"head_dim={head_dim}\n")
        f.write(f"hidden_dim={hidden_dim}\n")
        f.write(f"vocab_size={vocab_size}\n")
        f.write(f"seq_len={seq_len}\n\n")

        f.write("TOKENS\n")
        write_tokens(f, tokens)
        f.write("\nEMBEDDINGS\n")
        write_matrix(f, embeddings)

        for layer_idx in range(num_layers):
            block = model.h[layer_idx]
            f.write(f"LAYER {layer_idx}\n")

            c_attn_w = block.attn.c_attn.weight.detach().cpu().numpy()  # (d, 3d)
            c_attn_b = block.attn.c_attn.bias.detach().cpu().numpy()  # (3d,)
            c_proj_w = block.attn.c_proj.weight.detach().cpu().numpy()  # (d, d)
            c_proj_b = block.attn.c_proj.bias.detach().cpu().numpy()  # (d,)

            W_Q_all = c_attn_w[:, 0:model_dim]
            W_K_all = c_attn_w[:, model_dim : 2 * model_dim]
            W_V_all = c_attn_w[:, 2 * model_dim : 3 * model_dim]
            b_Q_all = c_attn_b[0:model_dim]
            b_K_all = c_attn_b[model_dim : 2 * model_dim]
            b_V_all = c_attn_b[2 * model_dim : 3 * model_dim]

            for h in range(num_heads):
                start, end = h * head_dim, (h + 1) * head_dim
                f.write(f"HEAD {h}\n")
                f.write("W_Q\n")
                write_matrix(f, W_Q_all[:, start:end])
                f.write("b_Q\n")
                write_vector(f, b_Q_all[start:end])
                f.write("W_K\n")
                write_matrix(f, W_K_all[:, start:end])
                f.write("b_K\n")
                write_vector(f, b_K_all[start:end])
                f.write("W_V\n")
                write_matrix(f, W_V_all[:, start:end])
                f.write("b_V\n")
                write_vector(f, b_V_all[start:end])
                f.write("W_O\n")
                write_matrix(f, c_proj_w[start:end, :])

            f.write("ATTN_BIAS\n")
            write_vector(f, c_proj_b)

            f.write("MLP\n")
            f.write("W_in\n")
            write_matrix(f, block.mlp.c_fc.weight.detach().cpu().numpy())
            f.write("b_in\n")
            write_vector(f, block.mlp.c_fc.bias.detach().cpu().numpy())
            f.write("W_out\n")
            write_matrix(f, block.mlp.c_proj.weight.detach().cpu().numpy())
            f.write("b_out\n")
            write_vector(f, block.mlp.c_proj.bias.detach().cpu().numpy())

            f.write("LN1_GAMMA\n")
            write_vector(f, block.ln_1.weight.detach().cpu().numpy())
            f.write("LN1_BETA\n")
            write_vector(f, block.ln_1.bias.detach().cpu().numpy())
            f.write("LN2_GAMMA\n")
            write_vector(f, block.ln_2.weight.detach().cpu().numpy())
            f.write("LN2_BETA\n")
            write_vector(f, block.ln_2.bias.detach().cpu().numpy())

        f.write("LN_F_GAMMA\n")
        write_vector(f, model.ln_f.weight.detach().cpu().numpy())
        f.write("LN_F_BETA\n")
        write_vector(f, model.ln_f.bias.detach().cpu().numpy())
        f.write("UNEMBEDDING\n")
        write_matrix(f, wte.T)

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

