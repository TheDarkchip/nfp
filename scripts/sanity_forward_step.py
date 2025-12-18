#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Sanity check: compare one forward step against PyTorch / HuggingFace GPT-2.

This is a correctness guardrail to ensure that the exported `.nfpt` semantics match what
the Lean-side executable actually analyzes (including attention biases).

Usage:
  uv run scripts/sanity_forward_step.py models/gpt2_rigorous.nfpt --layer 0 --pos 0 --take 16
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

try:
    import torch
    from transformers import GPT2Model
except ImportError as e:
    raise SystemExit(
        "Missing deps. Install with e.g.: uv add torch transformers\n"
        f"ImportError: {e}"
    )


def parse_tokens_from_nfpt(path: Path) -> list[int]:
    lines = path.read_text().splitlines()
    try:
        tok_i = lines.index("TOKENS")
    except ValueError:
        raise SystemExit("No TOKENS section found (required for sanity check).")
    try:
        emb_i = lines.index("EMBEDDINGS")
    except ValueError:
        raise SystemExit("No EMBEDDINGS section found.")
    toks: list[int] = []
    for line in lines[tok_i + 1 : emb_i]:
        line = line.strip()
        if not line:
            continue
        toks.extend(int(x) for x in line.split())
    return toks


def has_attn_bias(path: Path) -> bool:
    return "ATTN_BIAS" in path.read_text()


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


def run_torch(model_name: str, tokens: list[int], layer: int, pos: int, take: int) -> np.ndarray:
    m = GPT2Model.from_pretrained(model_name)
    m.eval()
    input_ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        out = m(input_ids=input_ids, output_hidden_states=True)
    hs = out.hidden_states
    if hs is None:
        raise SystemExit("Expected output_hidden_states=True to return hidden_states.")
    # hs[0] = embeddings, hs[layer+1] = after block `layer`.
    x = hs[layer + 1][0, pos, :take].detach().cpu().numpy()
    return x.astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--hf", default="gpt2", help="HuggingFace model name (default: gpt2)")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--pos", type=int, default=0)
    ap.add_argument("--take", type=int, default=16)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Missing file: {args.model}")
    if not has_attn_bias(args.model):
        raise SystemExit(
            "This `.nfpt` appears to be missing ATTENTION biases (ATTN_BIAS).\n"
            "Re-export with the updated exporter."
        )

    tokens = parse_tokens_from_nfpt(args.model)
    if len(tokens) == 0:
        raise SystemExit("No tokens parsed from TOKENS section.")

    lean = run_lean_dump(args.model, args.layer, args.pos, args.take)
    torch_x = run_torch(args.hf, tokens, args.layer, args.pos, args.take)

    diff = np.max(np.abs(lean - torch_x))
    print(f"max_abs_diff={diff:.6g}  tol={args.tol:.6g}")
    if not np.isfinite(diff) or diff > args.tol:
        print("FAIL")
        print("lean :", lean.tolist())
        print("torch:", torch_x.tolist())
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
