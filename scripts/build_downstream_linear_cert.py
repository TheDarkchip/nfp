#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Build a downstream linear certificate from externally computed bounds.

This script is untrusted: it only formats rational inputs into the certificate
format expected by `nfp induction certify_end_to_end`.

Usage:
  python scripts/build_downstream_linear_cert.py \
    --output reports/gpt2_downstream.cert \
    --gain 3/2 \
    --input-bound 5/4

Optional:
  --error 15/8  # override gain * input-bound
"""

import argparse
from fractions import Fraction
from pathlib import Path


def parse_rat(raw: str) -> Fraction:
    if "/" in raw:
        num, den = raw.split("/", 1)
        return Fraction(int(num.strip()), int(den.strip()))
    return Fraction(int(raw.strip()), 1)


def rat_to_str(q: Fraction) -> str:
    if q.denominator == 1:
        return str(q.numerator)
    return f"{q.numerator}/{q.denominator}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Path to write certificate")
    parser.add_argument("--gain", required=True, help="Nonnegative gain bound (Rat)")
    parser.add_argument("--input-bound", required=True,
                        help="Nonnegative input bound (Rat)")
    parser.add_argument("--error",
                        help="Optional error override (Rat). Defaults to gain * input-bound.")
    args = parser.parse_args()

    gain = parse_rat(args.gain)
    input_bound = parse_rat(args.input_bound)
    if gain < 0 or input_bound < 0:
        raise SystemExit("gain and input-bound must be nonnegative")
    error = parse_rat(args.error) if args.error else gain * input_bound
    if error < 0:
        raise SystemExit("error must be nonnegative")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii") as f:
        f.write(f"error {rat_to_str(error)}\n")
        f.write(f"gain {rat_to_str(gain)}\n")
        f.write(f"input-bound {rat_to_str(input_bound)}\n")
    print(f"Wrote downstream certificate to {output_path}")


if __name__ == "__main__":
    main()
