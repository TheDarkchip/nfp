#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

from fractions import Fraction


def population_variance(xs: list[Fraction]) -> Fraction:
    n = len(xs)
    if n == 0:
        return Fraction(0)
    mean = sum(xs, Fraction(0)) / n
    return sum((x - mean) ** 2 for x in xs) / n


def variance_bound_range(delta: Fraction, n: int) -> Fraction:
    if n <= 0 or delta <= 0:
        return Fraction(0)
    return (delta * delta) / (2 * n)


def main() -> None:
    xs = [Fraction(0), Fraction(1, 2), Fraction(1)]
    n = len(xs)
    delta = max(xs) - min(xs)
    var = population_variance(xs)
    old_bound = Fraction(n - 1, n * n) * (delta**2)
    new_bound = variance_bound_range(delta, n)
    print(f"x = {xs}")
    print(f"n = {n}, delta = {delta}")
    print(f"variance = {var} = {float(var):.6f}")
    print(f"old_bound = {old_bound} = {float(old_bound):.6f}")
    print(f"new_bound = {new_bound} = {float(new_bound):.6f}")
    print(f"old_bound <= variance? {old_bound <= var}")
    print(f"new_bound <= variance? {new_bound <= var}")

    # Quick brute-force sanity check on a small grid.
    grid = [Fraction(0), Fraction(1, 2), Fraction(1)]
    for a in grid:
        for b in grid:
            for c in grid:
                xs = [a, b, c]
                delta = max(xs) - min(xs)
                var = population_variance(xs)
                bound = variance_bound_range(delta, len(xs))
                if bound > var:
                    raise SystemExit(
                        f"violation: xs={xs} delta={delta} var={var} bound={bound}"
                    )
    print("grid_check=OK")


if __name__ == "__main__":
    main()
