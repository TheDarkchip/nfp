#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Build an induction-head certificate for a GPT-2-small induction head.

This script is untrusted and uses floating-point arithmetic to produce a
rational induction-head certificate compatible with `nfp induction certify`.
Active induction positions are recorded as `active <q>` lines in the output.
All sequence indices in the certificate are 1-based (literature convention).

Usage:
  python scripts/build_gpt2_induction_cert.py --output reports/gpt2_induction.cert \
    --layer 0 --head 5 --seq 32 --pattern-length 16 \
    --random-pattern --seed 0 \
    --values-out reports/gpt2_induction.values --value-dim 0 \
    --active-eps-max 0.2 --min-margin 0

Optionally, provide a logit-diff direction:
  --direction-target <tok_id> --direction-negative <tok_id>

Note: active positions are filtered by --active-eps-max and --min-margin. If
none qualify, the script exits with an error.
Direction token IDs use the model's raw tokenizer indexing.
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


def build_prev(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prev = np.zeros_like(tokens)
    active = np.zeros_like(tokens, dtype=bool)
    last_seen = {}
    for idx, tok in enumerate(tokens):
        if idx == 0:
            prev[idx] = 0
            active[idx] = False
        else:
            if tok in last_seen:
                prev[idx] = last_seen[tok]
                active[idx] = True
            else:
                prev[idx] = 0
                active[idx] = False
        last_seen[tok] = idx
    return prev, active


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
        scores = scores.masked_fill(mask, -10000.0)
        weights = torch.softmax(scores, dim=-1)
    return (scores.squeeze(0).cpu().numpy(),
            weights.squeeze(0).cpu().numpy(),
            vh.squeeze(0).cpu().numpy())


def write_scores(path: Path, seq: int, prev: np.ndarray, scores, weights, eps=None, margin=None,
                 active=None):
    with path.open("w", encoding="ascii") as f:
        f.write(f"seq {seq}\n")
        if eps is not None:
            f.write(f"eps {rat_to_str(eps)}\n")
        if margin is not None:
            f.write(f"margin {rat_to_str(margin)}\n")
        if active is not None:
            for q in active:
                f.write(f"active {q + 1}\n")
        for q, k in enumerate(prev.tolist()):
            f.write(f"prev {q + 1} {k + 1}\n")
        for q in range(seq):
            for k in range(seq):
                f.write(f"score {q + 1} {k + 1} {rat_to_str(scores[q][k])}\n")
        for q in range(seq):
            for k in range(seq):
                f.write(f"weight {q + 1} {k + 1} {rat_to_str(weights[q][k])}\n")

def write_induction_cert(path: Path, seq: int, prev: np.ndarray, scores, weights,
                         eps, margin, active, eps_at, weight_bound_at,
                         vals, direction_target=None, direction_negative=None) -> None:
    lo = min(vals)
    hi = max(vals)
    with path.open("w", encoding="ascii") as f:
        f.write(f"seq {seq}\n")
        if direction_target is not None and direction_negative is not None:
            f.write(f"direction-target {direction_target}\n")
            f.write(f"direction-negative {direction_negative}\n")
        f.write(f"eps {rat_to_str(eps)}\n")
        f.write(f"margin {rat_to_str(margin)}\n")
        if active is not None:
            for q in active:
                f.write(f"active {q + 1}\n")
        for q, k in enumerate(prev.tolist()):
            f.write(f"prev {q + 1} {k + 1}\n")
        for q in range(seq):
            for k in range(seq):
                f.write(f"score {q + 1} {k + 1} {rat_to_str(scores[q][k])}\n")
        for q in range(seq):
            for k in range(seq):
                f.write(f"weight {q + 1} {k + 1} {rat_to_str(weights[q][k])}\n")
        for q in range(seq):
            f.write(f"eps-at {q + 1} {rat_to_str(eps_at[q])}\n")
        for q in range(seq):
            for k in range(seq):
                f.write(f"weight-bound {q + 1} {k + 1} {rat_to_str(weight_bound_at[q][k])}\n")
        f.write(f"lo {rat_to_str(lo)}\n")
        f.write(f"hi {rat_to_str(hi)}\n")
        for k, val in enumerate(vals):
            val_str = rat_to_str(val)
            f.write(f"val {k + 1} {val_str}\n")
            f.write(f"val-lo {k + 1} {val_str}\n")
            f.write(f"val-hi {k + 1} {val_str}\n")


def write_value_range(path: Path, seq: int, values, decimals: int,
                      direction_target=None, direction_negative=None) -> None:
    vals_rat = [rat_from_float(float(values[k]), decimals) for k in range(seq)]
    lo = min(vals_rat)
    hi = max(vals_rat)
    with path.open("w", encoding="ascii") as f:
        f.write(f"seq {seq}\n")
        if direction_target is not None and direction_negative is not None:
            f.write(f"direction-target {direction_target}\n")
            f.write(f"direction-negative {direction_negative}\n")
        f.write(f"lo {rat_to_str(lo)}\n")
        f.write(f"hi {rat_to_str(hi)}\n")
        for k, val in enumerate(vals_rat):
            f.write(f"val {k + 1} {rat_to_str(val)}\n")


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
    parser.add_argument("--active-eps-max", default="1/2",
                        help="Maximum eps to include an active position (default: 1/2).")
    parser.add_argument("--min-margin", default="0",
                        help="Minimum score gap required for an active position (default: 0).")
    parser.add_argument("--direction-target", type=int,
                        help="Target token id for logit-diff direction (optional)")
    parser.add_argument("--direction-negative", type=int,
                        help="Negative token id for logit-diff direction (optional)")
    args = parser.parse_args()

    if args.seq <= 0:
        raise SystemExit("seq must be positive")

    tokens = build_tokens(args.seq, args.pattern_length, args.random_pattern, args.seed)
    prev, active_mask = build_prev(tokens)
    candidate_positions = [int(i) for i, flag in enumerate(active_mask) if flag]

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
    eps_by_q: dict[int, Fraction] = {}
    margin_by_q: dict[int, Fraction] = {}
    for q in range(args.seq):
        prev_q = prev[q]
        prev_w = weights_rat[q][prev_q]
        if args.seq == 1:
            max_other = Fraction(0)
        else:
            max_other = max(weights_rat[q][k] for k in range(args.seq) if k != prev_q)
        deficit = Fraction(1) - prev_w
        eps_by_q[q] = max(max_other, deficit)

        diffs = [scores_rat[q][prev_q] - scores_rat[q][k]
                 for k in range(args.seq) if k != prev_q]
        margin_by_q[q] = min(diffs) if diffs else Fraction(0)

    eps_threshold = Fraction(args.active_eps_max)
    min_margin = Fraction(args.min_margin)
    active_positions = [
        q for q in candidate_positions
        if eps_by_q[q] <= eps_threshold and margin_by_q[q] >= min_margin
    ]
    if not active_positions:
        raise SystemExit(
            "No active positions satisfy active-eps-max/min-margin. "
            "Try a different head/layer, random-pattern/seed, or relax the thresholds."
        )
    if active_positions:
        eps = max(eps_by_q[q] for q in active_positions)
        margin = min((margin_by_q[q] for q in active_positions), default=Fraction(0))
    else:
        margin = Fraction(0)

    if candidate_positions:
        print(f"Active positions: {len(active_positions)}/{len(candidate_positions)}")

    eps_at = []
    for q in range(args.seq):
        prev_q = prev[q]
        if args.seq == 1:
            max_other = Fraction(0)
        else:
            max_other = max(weights_rat[q][k] for k in range(args.seq) if k != prev_q)
        deficit = Fraction(1) - weights_rat[q][prev_q]
        eps_at.append(max(max_other, deficit))
    weight_bound_at = weights_rat

    direction_target = None
    direction_negative = None
    if (args.direction_target is None) != (args.direction_negative is None):
        raise SystemExit("direction-target and direction-negative must be provided together")
    if args.direction_target is not None:
        wte = model.wte.weight.detach().cpu().numpy()
        if args.direction_target < 0 or args.direction_target >= wte.shape[0]:
            raise SystemExit("direction-target out of vocab range")
        if args.direction_negative < 0 or args.direction_negative >= wte.shape[0]:
            raise SystemExit("direction-negative out of vocab range")
        direction_target = args.direction_target
        direction_negative = args.direction_negative
        direction = wte[direction_target] - wte[direction_negative]
        head_dim = model.config.n_embd // model.config.n_head
        start, end = args.head * head_dim, (args.head + 1) * head_dim
        w_o = model.h[args.layer].attn.c_proj.weight.detach().cpu().numpy()[:, start:end]
        dir_head = w_o.T @ direction
        vals = values @ dir_head
    else:
        if args.value_dim < 0 or args.value_dim >= values.shape[1]:
            raise SystemExit(f"value-dim must be in [0, {values.shape[1] - 1}]")
        vals = values[:, args.value_dim]

    vals_rat = [rat_from_float(float(vals[k]), args.decimals) for k in range(args.seq)]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_induction_cert(
        output_path,
        args.seq,
        prev,
        scores_rat,
        weights_rat,
        eps,
        margin,
        active_positions,
        eps_at,
        weight_bound_at,
        vals_rat,
        direction_target=direction_target,
        direction_negative=direction_negative,
    )

    if args.scores_out:
        scores_path = Path(args.scores_out)
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        write_scores(scores_path, args.seq, prev, scores_rat, weights_rat,
                     active=active_positions)

    if args.values_out:
        values_path = Path(args.values_out)
        values_path.parent.mkdir(parents=True, exist_ok=True)
        write_value_range(values_path, args.seq, vals, args.decimals,
                          direction_target=direction_target,
                          direction_negative=direction_negative)

    print(f"Wrote certificate to {output_path}")
    if args.scores_out:
        print(f"Wrote scores dump to {scores_path}")
    if args.values_out:
        print(f"Wrote value-range certificate to {values_path}")


if __name__ == "__main__":
    main()
