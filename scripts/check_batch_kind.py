#!/usr/bin/env python3
"""Check batch constraints vs cert kind.

Fails if a batch file specifies onehot-only constraints (min-margin/max-eps)
while any referenced cert is induction-aligned.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parse_batch(batch_path: Path) -> tuple[dict[str, str], list[tuple[Path, Path]]]:
    constraints: dict[str, str] = {}
    items: list[tuple[Path, Path]] = []
    for raw in batch_path.read_text(encoding="ascii").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "item":
            if len(parts) < 3:
                raise SystemExit(f"Malformed item line: {line}")
            items.append((Path(parts[1]), Path(parts[2])))
        else:
            if len(parts) < 2:
                raise SystemExit(f"Malformed constraint line: {line}")
            constraints[parts[0]] = " ".join(parts[1:])
    return constraints, items


def read_cert_kind(cert_path: Path) -> str:
    for raw in cert_path.read_text(encoding="ascii").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("kind "):
            return line.split(" ", 1)[1].strip()
    raise SystemExit(f"Missing kind line in cert: {cert_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args()

    constraints, items = parse_batch(args.batch)
    has_onehot_constraints = any(k in constraints for k in ("min-margin", "max-eps"))
    if not has_onehot_constraints:
        return 0

    bad = []
    for cert_path, _ in items:
        kind = read_cert_kind(cert_path)
        if kind == "induction-aligned":
            bad.append(cert_path)

    if bad:
        bad_list = "\n".join(f"  - {p}" for p in bad)
        print(
            "Batch file has onehot-only constraints (min-margin/max-eps) but contains "
            f"induction-aligned certs:\n{bad_list}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
