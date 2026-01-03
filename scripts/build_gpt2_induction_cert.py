#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Build a softmax-margin certificate for a GPT-2-small induction head.

This script is untrusted and uses floating-point arithmetic to produce a
rational certificate compatible with `nfp induction certify`.

Usage:
  python scripts/build_gpt2_induction_cert.py --output reports/gpt2_induction.cert \
    --layer 5 --head 1 --seq 32 --pattern-length 16 \
    --values-out reports/gpt2_induction.values --value-dim 0
"""

import argparse
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

try:
    import torch
    from transformers import GPT2Model
except ImportError:
    raise SystemExit(
        "Missing dependencies. Install with: uv add transformers torch"
    )


def rat_from_float(x: float, decimals: int) -> Fraction:
    scale = 10 ** decimals
    return Fraction(int(round(x * scale)), scale)


def rat_to_str(q: Fraction) -> str:
    if q.denominator == 1:
        return str(q.numerator)
    return f"{q.numerator}/{q.denominator}"


def build_tokens(seq: int, pattern_len: int, random_pattern: bool, seed: int) -> np.ndarray:
    if random_pattern:
        rng = np.random.default_rng(seed)
        pattern = rng.integers(1000, 30000, size=pattern_len, endpoint=False)
    else:
        pattern = np.arange(pattern_len)
    repeats = (seq // pattern_len) + 1
    return np.tile(pattern, repeats)[:seq]


def build_prev(tokens: np.ndarray) -> np.ndarray:
    prev = np.zeros_like(tokens)
    last_seen = {}
    for idx, tok in enumerate(tokens):
        if idx == 0:
            prev[idx] = 0
        else:
            prev[idx] = last_seen.get(tok, 0)
        last_seen[tok] = idx
    return prev


def compute_scores_weights(model, input_ids, layer: int, head: int, device: str):
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer]
        block = model.h[layer]
        x = block.ln_1(hidden_states)
        qkv = block.attn.c_attn(x)
        n_head = model.config.n_head
        head_dim = qkv.shape[-1] // (3 * n_head)
        q, k, _v = qkv.split(n_head * head_dim, dim=2)
        q = q.view(1, -1, n_head, head_dim).transpose(1, 2)
        k = k.view(1, -1, n_head, head_dim).transpose(1, 2)
        v = _v.view(1, -1, n_head, head_dim).transpose(1, 2)
        qh = q[:, head]
        kh = k[:, head]
        vh = v[:, head]
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(head_dim)
        seq = scores.shape[-1]
        mask = torch.triu(torch.ones(seq, seq, device=device), diagonal=1).bool()
        scores = scores.masked_fill(mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
    return (scores.squeeze(0).cpu().numpy(),
            weights.squeeze(0).cpu().numpy(),
            vh.squeeze(0).cpu().numpy())


def write_scores(path: Path, seq: int, prev: np.ndarray, scores, weights, eps=None, margin=None):
    with path.open("w", encoding="ascii") as f:
        f.write(f"seq {seq}\n")
        if eps is not None:
            f.write(f"eps {rat_to_str(eps)}\n")
        if margin is not None:
            f.write(f"margin {rat_to_str(margin)}\n")
        for q, k in enumerate(prev.tolist()):
            f.write(f"prev {q} {k}\n")
        for q in range(seq):
            for k in range(seq):
                f.write(f"score {q} {k} {rat_to_str(scores[q][k])}\n")
        for q in range(seq):
            for k in range(seq):
                f.write(f"weight {q} {k} {rat_to_str(weights[q][k])}\n")


def write_value_range(path: Path, seq: int, values, decimals: int) -> None:
    vals_rat = [rat_from_float(float(values[k]), decimals) for k in range(seq)]
    lo = min(vals_rat)
    hi = max(vals_rat)
    with path.open("w", encoding="ascii") as f:
        f.write(f"seq {seq}\n")
        f.write(f"lo {rat_to_str(lo)}\n")
        f.write(f"hi {rat_to_str(hi)}\n")
        for k, val in enumerate(vals_rat):
            f.write(f"val {k} {rat_to_str(val)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Path to write certificate")
    parser.add_argument("--scores-out", help="Optional path for raw scores/weights dump")
    parser.add_argument("--layer", type=int, default=0, help="Transformer layer index")
    parser.add_argument("--head", type=int, default=0, help="Attention head index")
    parser.add_argument("--seq", type=int, default=32, help="Sequence length")
    parser.add_argument("--pattern-length", type=int, default=16, help="Pattern length")
    parser.add_argument("--random-pattern", action="store_true", help="Use random token pattern")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for random pattern")
    parser.add_argument("--decimals", type=int, default=6, help="Decimal rounding for rationals")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--values-out", help="Optional path for a value-range certificate")
    parser.add_argument("--value-dim", type=int, default=0,
                        help="Value dimension index for the value-range certificate")
    args = parser.parse_args()

    if args.seq <= 0:
        raise SystemExit("seq must be positive")

    tokens = build_tokens(args.seq, args.pattern_length, args.random_pattern, args.seed)
    prev = build_prev(tokens)

    model = GPT2Model.from_pretrained(args.model)
    model.to(args.device)
    input_ids = torch.tensor(tokens, dtype=torch.long, device=args.device).unsqueeze(0)

    scores, weights, values = compute_scores_weights(model, input_ids, args.layer, args.head,
                                                    args.device)

    scores_rat = [[rat_from_float(float(scores[q, k]), args.decimals) for k in range(args.seq)]
                  for q in range(args.seq)]
    weights_rat = [[rat_from_float(float(weights[q, k]), args.decimals) for k in range(args.seq)]
                   for q in range(args.seq)]

    for q in range(args.seq):
        total = sum(weights_rat[q], Fraction(0))
        if total == 0:
            raise SystemExit(f"zero weight sum at q={q}")
        weights_rat[q] = [w / total for w in weights_rat[q]]

    eps = Fraction(0)
    margin = None
    for q in range(1, args.seq):
        prev_q = prev[q]
        prev_w = weights_rat[q][prev_q]
        max_other = max(weights_rat[q][k] for k in range(args.seq) if k != prev_q)
        deficit = Fraction(1) - prev_w
        eps = max(eps, max(max_other, deficit))

        diffs = [scores_rat[q][prev_q] - scores_rat[q][k]
                 for k in range(args.seq) if k != prev_q]
        if diffs:
            min_diff = min(diffs)
            margin = min_diff if margin is None else min(margin, min_diff)

    if margin is None:
        margin = Fraction(0)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_scores(output_path, args.seq, prev, scores_rat, weights_rat, eps=eps, margin=margin)

    if args.scores_out:
        scores_path = Path(args.scores_out)
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        write_scores(scores_path, args.seq, prev, scores_rat, weights_rat)

    if args.values_out:
        if args.value_dim < 0 or args.value_dim >= values.shape[1]:
            raise SystemExit(f"value-dim must be in [0, {values.shape[1] - 1}]")
        values_path = Path(args.values_out)
        values_path.parent.mkdir(parents=True, exist_ok=True)
        write_value_range(values_path, args.seq, values[:, args.value_dim], args.decimals)

    print(f"Wrote certificate to {output_path}")
    if args.scores_out:
        print(f"Wrote scores dump to {scores_path}")
    if args.values_out:
        print(f"Wrote value-range certificate to {values_path}")


if __name__ == "__main__":
    main()
