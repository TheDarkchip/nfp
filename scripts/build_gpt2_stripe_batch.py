#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Build a batch of stripe-attention certificates for GPT-2 small.

This script shells out to build_gpt2_stripe_cert.py for each seed and then
writes a batch file listing the resulting certs.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-out", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="Max seeds to try (default: count * 5).")
    parser.add_argument("--layer", type=int, required=True, help="0-based layer index")
    parser.add_argument("--head", type=int, required=True, help="0-based head index")
    parser.add_argument("--seq", type=int, default=32)
    parser.add_argument("--pattern-length", type=int, default=16)
    parser.add_argument("--random-pattern", action="store_true")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-stripe-mean", default=None)
    parser.add_argument("--min-stripe-top1", default=None)
    parser.add_argument("--min-avg-stripe-mean", default=None)
    parser.add_argument("--min-avg-stripe-top1", default=None)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_out = args.batch_out or (out_dir / "gpt2_stripe.batch")

    items: list[Path] = []
    script = Path(__file__).with_name("build_gpt2_stripe_cert.py")

    max_attempts = args.max_attempts
    if max_attempts is None:
        max_attempts = args.count * 5

    successes = 0
    attempt = 0
    seed = args.seed_start
    while successes < args.count and attempt < max_attempts:
        cert_path = out_dir / f"cert_{successes:04d}.cert"
        cmd = [
            "python",
            str(script),
            "--output",
            str(cert_path),
            "--layer",
            str(args.layer),
            "--head",
            str(args.head),
            "--seq",
            str(args.seq),
            "--pattern-length",
            str(args.pattern_length),
            "--seed",
            str(seed),
            "--model",
            str(args.model),
            "--device",
            str(args.device),
        ]
        if args.random_pattern:
            cmd.append("--random-pattern")
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            items.append(cert_path)
            successes += 1
        else:
            print(f"warning: stripe certificate generation failed for seed {seed}")
        attempt += 1
        seed += 1

    if successes < args.count:
        raise SystemExit(
            f"only generated {successes} certs after {attempt} attempts"
        )

    lines = []
    if args.min_stripe_mean is not None:
        lines.append(f"min-stripe-mean {args.min_stripe_mean}")
    if args.min_stripe_top1 is not None:
        lines.append(f"min-stripe-top1 {args.min_stripe_top1}")
    if args.min_avg_stripe_mean is not None:
        lines.append(f"min-avg-stripe-mean {args.min_avg_stripe_mean}")
    if args.min_avg_stripe_top1 is not None:
        lines.append(f"min-avg-stripe-top1 {args.min_avg_stripe_top1}")
    for cert_path in items:
        lines.append(f"item {cert_path}")

    batch_out.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"Wrote stripe batch file to {batch_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
