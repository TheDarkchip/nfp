#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Build a residual-bound certificate from a GPT-2 forward pass.

This script is untrusted. It computes per-coordinate absolute bounds by
taking maxima over a fixed input sequence (optionally restricted to active
positions from a softmax-margin certificate). The resulting bounds are
rounded up to rationals for Lean-side checking.

Usage:
  uv run scripts/build_residual_bound_cert.py \
    --output reports/gpt2_residual.bound \
    --seq 32 --pattern-length 16 \
    --scores reports/gpt2_induction.cert

Optional:
  --tokens tokens.txt    # whitespace-separated token ids
  --random-pattern --seed 0
  --decimals 6 --safety 1e-6
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


def parse_tokens(path: Path) -> np.ndarray:
    raw = path.read_text(encoding="ascii")
    tokens = [int(tok) for tok in raw.split() if tok.strip()]
    if not tokens:
        raise SystemExit(f"no tokens found in {path}")
    return np.array(tokens, dtype=np.int64)


def parse_active_positions(path: Path) -> tuple[int | None, list[int]]:
    seq = None
    active: list[int] = []
    for line in path.read_text(encoding="ascii").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "seq" and len(parts) >= 2:
            seq = int(parts[1])
        elif parts[0] == "active" and len(parts) >= 2:
            active.append(int(parts[1]))
    return seq, active


def ceil_rat(x: float, decimals: int, safety: float) -> Fraction:
    scale = 10 ** decimals
    scaled = abs(x) * (1.0 + safety) * scale
    return Fraction(int(math.ceil(scaled)), scale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Path to write certificate")
    parser.add_argument("--seq", type=int, default=32, help="Sequence length")
    parser.add_argument("--pattern-length", type=int, default=16, help="Pattern length")
    parser.add_argument("--random-pattern", action="store_true",
                        help="Use random token pattern")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for random pattern")
    parser.add_argument("--tokens", help="Optional path to whitespace-separated tokens")
    parser.add_argument("--scores", help="Optional softmax-margin certificate for active queries")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--decimals", type=int, default=6,
                        help="Decimal rounding for rationals (ceil)")
    parser.add_argument("--safety", type=float, default=1e-6,
                        help="Relative safety slack added before rounding")
    args = parser.parse_args()

    if args.seq <= 0:
        raise SystemExit("seq must be positive")
    if args.decimals < 0:
        raise SystemExit("decimals must be nonnegative")
    if args.safety < 0:
        raise SystemExit("safety must be nonnegative")

    if args.tokens:
        tokens = parse_tokens(Path(args.tokens))
        seq = len(tokens)
    else:
        seq = args.seq
        tokens = build_tokens(seq, args.pattern_length, args.random_pattern, args.seed)

    positions = list(range(seq))
    if args.scores:
        cert_seq, active = parse_active_positions(Path(args.scores))
        if cert_seq is not None and cert_seq != seq:
            raise SystemExit(f"seq mismatch: scores={cert_seq} tokens={seq}")
        if active:
            positions = active

    model = GPT2Model.from_pretrained(args.model)
    model.to(args.device)
    model.eval()
    input_ids = torch.tensor(tokens, dtype=torch.long, device=args.device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids)
        hidden = outputs.last_hidden_state.squeeze(0).cpu().numpy()

    if hidden.shape[0] != seq:
        raise SystemExit(f"hidden state seq mismatch: {hidden.shape[0]} vs {seq}")

    chosen = hidden[positions]
    max_abs = np.max(np.abs(chosen), axis=0)
    bounds = [ceil_rat(float(val), args.decimals, args.safety) for val in max_abs]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii") as f:
        f.write(f"dim {len(bounds)}\n")
        for i, bound in enumerate(bounds):
            f.write(f"bound {i} {rat_to_str(bound)}\n")

    print(f"Wrote residual-bound certificate to {output_path}")


if __name__ == "__main__":
    main()
