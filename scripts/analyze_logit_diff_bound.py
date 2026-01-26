#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Analyze logit-diff lower bounds from an induction-head certificate.

This mirrors the three variants used by the verifier:
- base: global lo/hi with epsAt
- tight: per-query loAt with epsAt
- weighted: per-key weight bounds with valsLo
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path
from typing import Iterable


def parse_rat(text: str) -> Fraction:
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise SystemExit(f"invalid rational literal: {text}") from exc


def load_cert(path: Path) -> dict:
    data: dict[str, list[tuple[int, int, Fraction]] | list[tuple[int, Fraction]] | Fraction | int | list[int]] = {}
    seq = None
    active: list[int] = []
    prev: dict[int, int] = {}
    eps_at: dict[int, Fraction] = {}
    weight_bound: dict[tuple[int, int], Fraction] = {}
    vals: dict[int, Fraction] = {}
    vals_lo: dict[int, Fraction] = {}
    vals_hi: dict[int, Fraction] = {}
    lo = None
    hi = None

    for raw in path.read_text(encoding="ascii").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tag = parts[0]
        if tag == "seq":
            seq = int(parts[1])
        elif tag == "active":
            active.append(int(parts[1]))
        elif tag == "prev":
            prev[int(parts[1])] = int(parts[2])
        elif tag == "eps-at":
            eps_at[int(parts[1])] = parse_rat(parts[2])
        elif tag == "weight-bound":
            weight_bound[(int(parts[1]), int(parts[2]))] = parse_rat(parts[3])
        elif tag == "val":
            vals[int(parts[1])] = parse_rat(parts[2])
        elif tag == "val-lo":
            vals_lo[int(parts[1])] = parse_rat(parts[2])
        elif tag == "val-hi":
            vals_hi[int(parts[1])] = parse_rat(parts[2])
        elif tag == "lo":
            lo = parse_rat(parts[1])
        elif tag == "hi":
            hi = parse_rat(parts[1])

    if seq is None:
        raise SystemExit("missing seq in cert")
    if not prev or not eps_at or not weight_bound:
        raise SystemExit("missing prev/eps-at/weight-bound in cert")
    if not vals:
        raise SystemExit("missing val entries in cert")

    if not vals_lo:
        vals_lo = vals.copy()
    if not vals_hi:
        vals_hi = vals.copy()
    if lo is None:
        lo = min(vals_lo.values())
    if hi is None:
        hi = max(vals_hi.values())

    if not active:
        active = list(range(seq))

    return {
        "seq": seq,
        "active": active,
        "prev": prev,
        "eps_at": eps_at,
        "weight_bound": weight_bound,
        "vals": vals,
        "vals_lo": vals_lo,
        "vals_hi": vals_hi,
        "lo": lo,
        "hi": hi,
    }


def min_with_arg(items: Iterable[tuple[int, Fraction]]) -> tuple[int, Fraction] | None:
    best = None
    for q, val in items:
        if best is None or val < best[1]:
            best = (q, val)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cert", type=Path, required=True)
    args = parser.parse_args()

    cert = load_cert(args.cert)
    seq = cert["seq"]
    active = cert["active"]
    prev = cert["prev"]
    eps_at = cert["eps_at"]
    weight_bound = cert["weight_bound"]
    vals = cert["vals"]
    vals_lo = cert["vals_lo"]
    lo = cert["lo"]
    hi = cert["hi"]

    gap = hi - lo
    base_terms = []
    tight_terms = []
    weighted_terms = []

    for q in active:
        pq = prev[q]
        val_prev = vals[pq]
        eps = eps_at[q]
        base = val_prev - eps * gap
        base_terms.append((q, base))

        others = [vals_lo[k] for k in range(seq) if k != pq]
        lo_at = min(others) if others else lo
        delta = vals_lo[pq] - lo_at
        tight = vals_lo[pq] - eps * max(Fraction(0), delta)
        tight_terms.append((q, tight))

        weighted_gap = Fraction(0)
        for k in range(seq):
            diff = vals_lo[pq] - vals_lo[k]
            if diff > 0:
                weighted_gap += weight_bound[(q, k)] * diff
        weighted = vals_lo[pq] - weighted_gap
        weighted_terms.append((q, weighted))

    base_min = min_with_arg(base_terms)
    tight_min = min_with_arg(tight_terms)
    weighted_min = min_with_arg(weighted_terms)
    candidates = [t for t in [base_min, tight_min, weighted_min] if t is not None]
    overall = max((val for _, val in candidates), default=None)

    print(f"cert={args.cert}")
    print(f"seq={seq} active={len(active)}")
    if base_min is not None:
        q, val = base_min
        print(f"base_lb={val} (q={q}, prev={prev[q]}, epsAt={eps_at[q]}, gap={gap})")
    if tight_min is not None:
        q, val = tight_min
        pq = prev[q]
        others = [vals_lo[k] for k in range(seq) if k != pq]
        lo_at = min(others) if others else lo
        print(
            f"tight_lb={val} (q={q}, prev={pq}, epsAt={eps_at[q]}, loAt={lo_at})"
        )
    if weighted_min is not None:
        q, val = weighted_min
        print(f"weighted_lb={val} (q={q}, prev={prev[q]})")
    if overall is not None:
        print(f"overall_lb={overall}")
    else:
        print("overall_lb=None (empty active set)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
