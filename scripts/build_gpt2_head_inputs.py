#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Build an induction head input file from an NFP_BINARY_V1 model.

This is an untrusted helper that extracts a single head slice plus the
prompt embeddings from an `.nfpt` file and writes the text format consumed by
`nfp induction certify_head`.
"""

from __future__ import annotations

import argparse
import math
import struct
from fractions import Fraction
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def rat_from_float_exact(x: float) -> Fraction:
    if not math.isfinite(x):
        raise SystemExit(f"non-finite float encountered: {x}")
    num, den = x.as_integer_ratio()
    return Fraction(num, den)


def rat_to_str(q: Fraction) -> str:
    if q.denominator == 1:
        return str(q.numerator)
    return f"{q.numerator}/{q.denominator}"


def parse_header(f) -> Dict[str, str]:
    header: Dict[str, str] = {}
    magic = f.readline().decode("ascii").strip()
    if magic != "NFP_BINARY_V1":
        raise SystemExit(f"Unsupported magic header: {magic}")
    while True:
        line = f.readline()
        if line == b"":
            raise SystemExit("Unexpected EOF while reading header.")
        text = line.decode("ascii").strip()
        if text == "BINARY_START":
            break
        if "=" in text:
            key, value = text.split("=", 1)
            header[key.strip()] = value.strip()
    return header


def read_i32(f, count: int) -> np.ndarray:
    raw = f.read(count * 4)
    if len(raw) != count * 4:
        raise SystemExit("Unexpected EOF while reading int32 payload.")
    return np.frombuffer(raw, dtype="<i4").astype(np.int64, copy=False)


def read_f64(f, count: int) -> np.ndarray:
    raw = f.read(count * 8)
    if len(raw) != count * 8:
        raise SystemExit("Unexpected EOF while reading float64 payload.")
    return np.frombuffer(raw, dtype="<f8").astype(np.float64, copy=False)


def skip_f64(f, count: int) -> None:
    offset = count * 8
    f.seek(offset, 1)


def build_prev(tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    prev = np.zeros_like(tokens)
    active = np.zeros_like(tokens, dtype=bool)
    last_seen: Dict[int, int] = {}
    for idx, tok in enumerate(tokens.tolist()):
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


def read_head_weights(
    f,
    num_layers: int,
    num_heads: int,
    model_dim: int,
    head_dim: int,
    hidden_dim: int,
    layer: int,
    head: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = (layer, head)
    wq = wk = wv = wo = None
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            wq_block = read_f64(f, model_dim * head_dim).reshape(model_dim, head_dim)
            _ = read_f64(f, head_dim)  # b_Q
            wk_block = read_f64(f, model_dim * head_dim).reshape(model_dim, head_dim)
            _ = read_f64(f, head_dim)  # b_K
            wv_block = read_f64(f, model_dim * head_dim).reshape(model_dim, head_dim)
            _ = read_f64(f, head_dim)  # b_V
            wo_block = read_f64(f, head_dim * model_dim).reshape(head_dim, model_dim)
            if (layer_idx, head_idx) == target:
                wq = wq_block
                wk = wk_block
                wv = wv_block
                wo = wo_block
        # Skip per-layer non-head data.
        skip_f64(f, model_dim)  # attn_bias
        skip_f64(f, model_dim * hidden_dim)  # w_in
        skip_f64(f, hidden_dim)  # b_in
        skip_f64(f, hidden_dim * model_dim)  # w_out
        skip_f64(f, model_dim)  # b_out
        skip_f64(f, model_dim)  # ln1_gamma
        skip_f64(f, model_dim)  # ln1_beta
        skip_f64(f, model_dim)  # ln2_gamma
        skip_f64(f, model_dim)  # ln2_beta
    if wq is None or wk is None or wv is None or wo is None:
        raise SystemExit("Failed to locate head weights.")
    return wq, wk, wv, wo


def read_unembed_columns(
    f,
    start: int,
    model_dim: int,
    vocab_size: int,
    target: int,
    negative: int,
) -> Tuple[np.ndarray, np.ndarray]:
    row_bytes = vocab_size * 8
    col_t = np.zeros(model_dim, dtype=np.float64)
    col_n = np.zeros(model_dim, dtype=np.float64)
    for row in range(model_dim):
        base = start + row * row_bytes
        f.seek(base + target * 8)
        col_t[row] = struct.unpack("<d", f.read(8))[0]
        f.seek(base + negative * 8)
        col_n[row] = struct.unpack("<d", f.read(8))[0]
    return col_t, col_n


def write_head_inputs(
    path: Path,
    scale: Fraction,
    tokens: np.ndarray,
    embeddings: np.ndarray,
    prev: np.ndarray,
    active: np.ndarray,
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
    wo: np.ndarray,
    direction_target: int,
    direction_negative: int,
    direction: np.ndarray,
) -> None:
    seq, model_dim = embeddings.shape
    _, head_dim = wq.shape
    with path.open("w", encoding="ascii") as f:
        f.write(f"seq {seq}\n")
        f.write(f"d_model {model_dim}\n")
        f.write(f"d_head {head_dim}\n")
        f.write(f"scale {rat_to_str(scale)}\n")
        for q, flag in enumerate(active.tolist()):
            if flag:
                f.write(f"active {q}\n")
        for q, k in enumerate(prev.tolist()):
            f.write(f"prev {q} {k}\n")
        for q in range(seq):
            for d in range(model_dim):
                f.write(f"embed {q} {d} {rat_to_str(rat_from_float_exact(float(embeddings[q, d])))}\n")
        for i in range(model_dim):
            for j in range(head_dim):
                f.write(f"wq {i} {j} {rat_to_str(rat_from_float_exact(float(wq[i, j])))}\n")
        for i in range(model_dim):
            for j in range(head_dim):
                f.write(f"wk {i} {j} {rat_to_str(rat_from_float_exact(float(wk[i, j])))}\n")
        for i in range(model_dim):
            for j in range(head_dim):
                f.write(f"wv {i} {j} {rat_to_str(rat_from_float_exact(float(wv[i, j])))}\n")
        for i in range(model_dim):
            for j in range(head_dim):
                f.write(f"wo {i} {j} {rat_to_str(rat_from_float_exact(float(wo[i, j])))}\n")
        f.write(f"direction-target {direction_target}\n")
        f.write(f"direction-negative {direction_negative}\n")
        for d in range(model_dim):
            f.write(f"direction {d} {rat_to_str(rat_from_float_exact(float(direction[d])))}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True, help="Path to NFP_BINARY_V1 model")
    ap.add_argument("--layer", type=int, required=True, help="Layer index")
    ap.add_argument("--head", type=int, required=True, help="Head index")
    ap.add_argument("--output", type=Path, required=True, help="Path for the head input file")
    ap.add_argument("--direction-target", type=int, required=True, help="Target token id")
    ap.add_argument("--direction-negative", type=int, required=True, help="Negative token id")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Missing model file: {args.model}")

    with args.model.open("rb") as f:
        header = parse_header(f)
        num_layers = int(header["num_layers"])
        num_heads = int(header["num_heads"])
        model_dim = int(header["model_dim"])
        head_dim = int(header["head_dim"])
        vocab_size = int(header["vocab_size"])
        seq_len = int(header["seq_len"])
        hidden_dim = int(header["hidden_dim"])

        if args.layer < 0 or args.layer >= num_layers:
            raise SystemExit("layer index out of range")
        if args.head < 0 or args.head >= num_heads:
            raise SystemExit("head index out of range")
        if not (0 <= args.direction_target < vocab_size):
            raise SystemExit("direction-target out of vocab range")
        if not (0 <= args.direction_negative < vocab_size):
            raise SystemExit("direction-negative out of vocab range")

        tokens = read_i32(f, seq_len)
        embeddings = read_f64(f, seq_len * model_dim).reshape(seq_len, model_dim)

        wq, wk, wv, wo_raw = read_head_weights(
            f,
            num_layers,
            num_heads,
            model_dim,
            head_dim,
            hidden_dim,
            args.layer,
            args.head,
        )

        # Skip final layer norm parameters.
        skip_f64(f, model_dim)  # ln_f_gamma
        skip_f64(f, model_dim)  # ln_f_beta

        unembed_start = f.tell()
        col_target, col_negative = read_unembed_columns(
            f,
            unembed_start,
            model_dim,
            vocab_size,
            args.direction_target,
            args.direction_negative,
        )

    prev, active = build_prev(tokens)
    direction = col_target - col_negative
    scale_denom = int(math.isqrt(head_dim))
    if scale_denom * scale_denom != head_dim:
        scale = rat_from_float_exact(1.0 / math.sqrt(head_dim))
    else:
        scale = Fraction(1, scale_denom)

    # Stored W_O is (head_dim, model_dim); transpose to model_dim × head_dim.
    wo = wo_raw.T

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_head_inputs(
        args.output,
        scale,
        tokens,
        embeddings,
        prev,
        active,
        wq,
        wk,
        wv,
        wo,
        args.direction_target,
        args.direction_negative,
        direction,
    )

    print(f"Wrote head inputs to {args.output}")
    print(f"seq={seq_len} d_model={model_dim} d_head={head_dim}")


if __name__ == "__main__":
    main()
