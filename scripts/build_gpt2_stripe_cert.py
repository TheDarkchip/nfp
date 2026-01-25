#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Build a stripe-attention certificate for a GPT-2 small head.

This script is untrusted and uses floating-point arithmetic to produce a
rational stripe-attention certificate compatible with `nfp induction verify --stripe-cert`.
Sequence indices in the payload are 0-based.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import numpy as np

try:
    import torch
    from transformers import GPT2Model
except ImportError:
    raise SystemExit("Missing dependencies. Install with: uv add transformers torch")


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


def compute_weights(model, input_ids, layer: int, head: int) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True, use_cache=False)
    if outputs.attentions is None:
        raise SystemExit("model did not return attention weights")
    attn = outputs.attentions[layer]
    return attn[0, head].detach().cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Path to write certificate")
    parser.add_argument("--layer", type=int, required=True, help="Transformer layer index (0-based)")
    parser.add_argument("--head", type=int, required=True, help="Attention head index (0-based)")
    parser.add_argument("--seq", type=int, default=32, help="Sequence length")
    parser.add_argument("--pattern-length", type=int, default=16, help="Pattern length")
    parser.add_argument("--random-pattern", action="store_true", help="Use random token pattern")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for random pattern")
    parser.add_argument("--decimals", type=int, default=6, help="Decimal rounding for rationals")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--stripe-mean-lb", default=None, help="Override stripe-mean lower bound")
    parser.add_argument("--stripe-top1-lb", default=None, help="Override stripe-top1 lower bound")
    args = parser.parse_args()

    if args.seq <= 0:
        raise SystemExit("seq must be positive")
    if args.seq != 2 * args.pattern_length:
        raise SystemExit("seq must equal 2 * pattern-length for repeated sequence diagnostic")

    tokens = build_tokens(args.seq, args.pattern_length, args.random_pattern, args.seed)
    model = GPT2Model.from_pretrained(args.model, attn_implementation=args.attn_implementation)
    model.to(args.device)
    input_ids = torch.tensor(tokens, dtype=torch.long, device=args.device).unsqueeze(0)

    if args.layer < 0 or args.layer >= model.config.n_layer:
        raise SystemExit(f"layer must be in [0, {model.config.n_layer - 1}]")
    if args.head < 0 or args.head >= model.config.n_head:
        raise SystemExit(f"head must be in [0, {model.config.n_head - 1}]")

    weights = compute_weights(model, input_ids, args.layer, args.head)
    if weights is None:
        raise SystemExit("model did not return attention weights")
    weights_rat = [
        [rat_from_float(float(weights[q, k]), args.decimals) for k in range(args.seq)]
        for q in range(args.seq)
    ]

    for q in range(args.seq):
        total = sum(weights_rat[q], Fraction(0))
        if total == 0:
            raise SystemExit(f"zero weight sum at q={q}")
        weights_rat[q] = [w / total for w in weights_rat[q]]

    period = args.pattern_length
    active_q = [q for q in range(args.seq) if q >= period]
    if not active_q:
        raise SystemExit("no active queries for stripe stats")

    stripe_vals = [weights_rat[q][q - period] for q in active_q]
    stripe_mean = sum(stripe_vals, Fraction(0)) / Fraction(len(active_q), 1)
    stripe_top1 = Fraction(
        sum(1 for q in active_q if weights_rat[q][q - period] >= max(weights_rat[q])), 1
    ) / Fraction(len(active_q), 1)

    stripe_mean_lb = stripe_mean
    stripe_top1_lb = stripe_top1
    if args.stripe_mean_lb is not None:
        stripe_mean_lb = Fraction(args.stripe_mean_lb)
    if args.stripe_top1_lb is not None:
        stripe_top1_lb = Fraction(args.stripe_top1_lb)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii") as f:
        f.write(f"seq {args.seq}\n")
        f.write("kind stripe\n")
        f.write(f"period {period}\n")
        f.write(f"stripe-mean-lb {rat_to_str(stripe_mean_lb)}\n")
        f.write(f"stripe-top1-lb {rat_to_str(stripe_top1_lb)}\n")
        for q in range(args.seq):
            for k in range(args.seq):
                f.write(f"weight {q} {k} {rat_to_str(weights_rat[q][k])}\n")

    print(f"Wrote stripe certificate to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
