#!/usr/bin/env python3
"""Ensure a `.nfpt` header has `gelu_kind`, optionally writing a patched copy."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def has_gelu_kind(path: Path) -> bool:
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError("unexpected EOF while reading header")
            stripped = line.strip()
            if stripped.startswith(b"gelu_kind=") or stripped.startswith(b"gelu_deriv="):
                return True
            if stripped == b"BINARY_START":
                return False


def patch_header(path: Path, output: Path, default_kind: str) -> None:
    with path.open("rb") as f:
        header_lines: list[bytes] = []
        gelu_present = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError("unexpected EOF while reading header")
            stripped = line.strip()
            if stripped.startswith(b"gelu_kind=") or stripped.startswith(b"gelu_deriv="):
                gelu_present = True
            header_lines.append(line)
            if stripped == b"BINARY_START":
                break
        payload = f.read()

    output.parent.mkdir(parents=True, exist_ok=True)
    if gelu_present:
        shutil.copyfile(path, output)
        print(f"gelu_kind already present; copied to {output}")
        return

    patched: list[bytes] = []
    inserted = False
    for line in header_lines:
        if line.strip() == b"BINARY_START" and not inserted:
            patched.append(f"gelu_kind={default_kind}\n".encode("ascii"))
            inserted = True
        patched.append(line)

    if not inserted:
        raise ValueError("missing BINARY_START while patching header")

    with output.open("wb") as f:
        f.writelines(patched)
        f.write(payload)
    print(f"added gelu_kind={default_kind}; wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--default", default="tanh")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        return 0 if has_gelu_kind(args.input) else 1

    if args.output is None:
        raise SystemExit("missing --output (use --check to test only)")

    if args.output.resolve() == args.input.resolve():
        raise SystemExit("refusing in-place patch; pass a distinct --output path")

    patch_header(args.input, args.output, args.default)
    return 0


if __name__ == "__main__":
    sys.exit(main())
