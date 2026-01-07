#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Build a residual-interval certificate from a GPT-2 forward pass.

This script is untrusted. It computes per-coordinate min/max bounds by
taking extrema over a fixed input sequence (optionally restricted to active
positions from a softmax-margin certificate). The resulting intervals are
expanded slightly and rounded outwards to rationals for checking by
`nfp induction certify_end_to_end_model`.

Usage:
  uv run scripts/build_residual_interval_cert.py \
    --output reports/gpt2_residual.interval \
    --seq 32 --pattern-length 16 \
    --scores reports/gpt2_induction.cert

Optional:
  --tokens tokens.txt    # whitespace-separated token ids
  --nfpt model.nfpt      # read tokens from binary model
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


def parse_tokens_from_nfpt(path: Path) -> np.ndarray:
    header: dict[str, str] = {}
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise SystemExit("unexpected EOF while reading header")
            text = line.decode("ascii").strip()
            if text == "BINARY_START":
                break
            if "=" in text:
                key, value = text.split("=", 1)
                header[key.strip()] = value.strip()
        seq_len_raw = header.get("seq_len")
        if seq_len_raw is None:
            raise SystemExit("header missing seq_len")
        seq_len = int(seq_len_raw)
        token_bytes = f.read(seq_len * 4)
        if len(token_bytes) != seq_len * 4:
            raise SystemExit("unexpected EOF while reading tokens")
        tokens = np.frombuffer(token_bytes, dtype="<i4").astype(np.int64, copy=False)
    if tokens.size == 0:
        raise SystemExit(f"no tokens found in {path}")
    return tokens


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


def expand_lo(val: float, safety: float) -> float:
    slack = safety * max(1.0, abs(val))
    return val - slack


def expand_hi(val: float, safety: float) -> float:
    slack = safety * max(1.0, abs(val))
    return val + slack


def floor_rat(val: float, decimals: int) -> Fraction:
    scale = 10 ** decimals
    return Fraction(int(math.floor(val * scale)), scale)


def ceil_rat(val: float, decimals: int) -> Fraction:
    scale = 10 ** decimals
    return Fraction(int(math.ceil(val * scale)), scale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Path to write certificate")
    parser.add_argument("--seq", type=int, default=32, help="Sequence length")
    parser.add_argument("--pattern-length", type=int, default=16, help="Pattern length")
    parser.add_argument("--random-pattern", action="store_true",
                        help="Use random token pattern")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for random pattern")
    parser.add_argument("--tokens", help="Optional path to whitespace-separated tokens")
    parser.add_argument("--nfpt", help="Optional .nfpt file to read tokens from")
    parser.add_argument("--scores", help="Optional softmax-margin certificate for active queries")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--decimals", type=int, default=6,
                        help="Decimal rounding for rationals (outward)")
    parser.add_argument("--safety", type=float, default=1e-6,
                        help="Relative safety slack added before rounding")
    args = parser.parse_args()

    if args.seq <= 0:
        raise SystemExit("seq must be positive")
    if args.decimals < 0:
        raise SystemExit("decimals must be nonnegative")
    if args.safety < 0:
        raise SystemExit("safety must be nonnegative")

    if args.nfpt:
        tokens = parse_tokens_from_nfpt(Path(args.nfpt))
        seq = len(tokens)
    elif args.tokens:
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
    mins = np.min(chosen, axis=0)
    maxs = np.max(chosen, axis=0)

    lo_bounds = []
    hi_bounds = []
    for lo_val, hi_val in zip(mins.tolist(), maxs.tolist(), strict=True):
        lo_adj = expand_lo(float(lo_val), args.safety)
        hi_adj = expand_hi(float(hi_val), args.safety)
        lo_rat = floor_rat(lo_adj, args.decimals)
        hi_rat = ceil_rat(hi_adj, args.decimals)
        if lo_rat > hi_rat:
            lo_rat, hi_rat = hi_rat, lo_rat
        lo_bounds.append(lo_rat)
        hi_bounds.append(hi_rat)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii") as f:
        f.write(f"dim {len(lo_bounds)}\n")
        for i, (lo, hi) in enumerate(zip(lo_bounds, hi_bounds, strict=True)):
            f.write(f"lo {i} {rat_to_str(lo)}\n")
            f.write(f"hi {i} {rat_to_str(hi)}\n")

    print(f"Wrote residual-interval certificate to {output_path}")


if __name__ == "__main__":
    main()
