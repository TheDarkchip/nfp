#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Callable, Dict, List, Tuple


def parse_header(lines: List[str], path: Path) -> Tuple[Dict[str, str], int]:
    if not lines:
        raise ValueError(f"{path}: empty file")
    magic = lines[0].strip()
    if not magic.startswith("NFP_TEXT"):
        raise ValueError(f"{path}: unexpected header '{magic}'")
    header: Dict[str, str] = {}
    i = 1
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if line == "":
            break
        if "=" in line:
            key, value = line.split("=", 1)
            header[key.strip()] = value.strip()
    return header, i


def next_nonempty(lines: List[str], i: int, path: Path) -> Tuple[int, str]:
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    if i >= len(lines):
        raise ValueError(f"{path}: unexpected EOF")
    return i, lines[i].strip()


def expect_line(lines: List[str], i: int, expected: str, path: Path) -> int:
    i, line = next_nonempty(lines, i, path)
    if line != expected:
        raise ValueError(f"{path}: expected '{expected}', got '{line}'")
    return i + 1


def expect_prefix(lines: List[str], i: int, prefix: str, path: Path) -> int:
    i, line = next_nonempty(lines, i, path)
    if not line.startswith(prefix):
        raise ValueError(f"{path}: expected '{prefix} ...', got '{line}'")
    return i + 1


def read_numbers(
    lines: List[str],
    i: int,
    count: int,
    path: Path,
    cast: Callable[[str], float],
) -> Tuple[List[float], int]:
    out: List[float] = []
    while len(out) < count:
        if i >= len(lines):
            raise ValueError(f"{path}: unexpected EOF while reading numbers")
        line = lines[i].strip()
        i += 1
        if line == "":
            continue
        for tok in line.split():
            try:
                out.append(cast(tok))
            except ValueError as exc:
                raise ValueError(f"{path}: invalid number '{tok}'") from exc
            if len(out) == count:
                break
    return out, i


def read_input_embeddings(path: Path) -> Tuple[Dict[str, str], List[int], List[float]]:
    lines = path.read_text().splitlines()
    header, i = parse_header(lines, path)
    seq_len = int(header["seq_len"])
    model_dim = int(header["model_dim"])
    i = expect_line(lines, i, "TOKENS", path)
    tokens_f, i = read_numbers(lines, i, seq_len, path, int)
    tokens = [int(t) for t in tokens_f]
    i = expect_line(lines, i, "EMBEDDINGS", path)
    embeddings_f, _ = read_numbers(lines, i, seq_len * model_dim, path, float)
    return header, tokens, embeddings_f


def read_model_weights(path: Path) -> Tuple[Dict[str, str], Dict[str, object]]:
    lines = path.read_text().splitlines()
    header, i = parse_header(lines, path)
    num_layers = int(header["num_layers"])
    num_heads = int(header["num_heads"])
    model_dim = int(header["model_dim"])
    head_dim = int(header["head_dim"])
    hidden_dim = int(header["hidden_dim"])

    layers: List[Dict[str, object]] = []
    for _ in range(num_layers):
        i = expect_prefix(lines, i, "LAYER", path)
        heads: List[Dict[str, List[float]]] = []
        for _ in range(num_heads):
            i = expect_prefix(lines, i, "HEAD", path)
            i = expect_line(lines, i, "W_Q", path)
            w_q, i = read_numbers(lines, i, model_dim * head_dim, path, float)
            i = expect_line(lines, i, "b_Q", path)
            b_q, i = read_numbers(lines, i, head_dim, path, float)
            i = expect_line(lines, i, "W_K", path)
            w_k, i = read_numbers(lines, i, model_dim * head_dim, path, float)
            i = expect_line(lines, i, "b_K", path)
            b_k, i = read_numbers(lines, i, head_dim, path, float)
            i = expect_line(lines, i, "W_V", path)
            w_v, i = read_numbers(lines, i, model_dim * head_dim, path, float)
            i = expect_line(lines, i, "b_V", path)
            b_v, i = read_numbers(lines, i, head_dim, path, float)
            i = expect_line(lines, i, "W_O", path)
            w_o, i = read_numbers(lines, i, head_dim * model_dim, path, float)
            heads.append(
                {
                    "W_Q": w_q,
                    "b_Q": b_q,
                    "W_K": w_k,
                    "b_K": b_k,
                    "W_V": w_v,
                    "b_V": b_v,
                    "W_O": w_o,
                }
            )
        i = expect_line(lines, i, "ATTN_BIAS", path)
        attn_bias, i = read_numbers(lines, i, model_dim, path, float)
        i = expect_line(lines, i, "MLP", path)
        i = expect_line(lines, i, "W_in", path)
        w_in, i = read_numbers(lines, i, model_dim * hidden_dim, path, float)
        i = expect_line(lines, i, "b_in", path)
        b_in, i = read_numbers(lines, i, hidden_dim, path, float)
        i = expect_line(lines, i, "W_out", path)
        w_out, i = read_numbers(lines, i, hidden_dim * model_dim, path, float)
        i = expect_line(lines, i, "b_out", path)
        b_out, i = read_numbers(lines, i, model_dim, path, float)
        i = expect_line(lines, i, "LN1_GAMMA", path)
        ln1_gamma, i = read_numbers(lines, i, model_dim, path, float)
        i = expect_line(lines, i, "LN1_BETA", path)
        ln1_beta, i = read_numbers(lines, i, model_dim, path, float)
        i = expect_line(lines, i, "LN2_GAMMA", path)
        ln2_gamma, i = read_numbers(lines, i, model_dim, path, float)
        i = expect_line(lines, i, "LN2_BETA", path)
        ln2_beta, i = read_numbers(lines, i, model_dim, path, float)
        layers.append(
            {
                "heads": heads,
                "attn_bias": attn_bias,
                "w_in": w_in,
                "b_in": b_in,
                "w_out": w_out,
                "b_out": b_out,
                "ln1_gamma": ln1_gamma,
                "ln1_beta": ln1_beta,
                "ln2_gamma": ln2_gamma,
                "ln2_beta": ln2_beta,
            }
        )

    return header, {"layers": layers}


def write_i32(f, data: List[int]) -> None:
    if not data:
        return
    f.write(struct.pack("<" + "i" * len(data), *data))


def write_f64(f, data: List[float]) -> None:
    if not data:
        return
    f.write(struct.pack("<" + "d" * len(data), *data))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a tiny NFP_TEXT fixture into NFP_BINARY_V1."
    )
    parser.add_argument(
        "--model",
        default="tests/fixtures/tiny_sound_model.nfpt",
        help="Text model fixture path (NFP_TEXT_V2).",
    )
    parser.add_argument(
        "--input",
        default="tests/fixtures/tiny_sound_input.nfpt",
        help="Text input fixture path (NFP_TEXT_V2).",
    )
    parser.add_argument(
        "--output",
        default="tests/fixtures/tiny_sound_binary.nfpt",
        help="Output binary fixture path.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)

    input_header, tokens, embeddings = read_input_embeddings(input_path)
    model_header, model_payload = read_model_weights(model_path)

    if int(input_header["model_dim"]) != int(model_header["model_dim"]):
        raise ValueError("input/model model_dim mismatch")
    if int(input_header["seq_len"]) != int(model_header["seq_len"]):
        raise ValueError("input/model seq_len mismatch")
    input_eps = input_header.get("layer_norm_eps") or input_header.get("eps")
    model_eps = model_header.get("layer_norm_eps") or model_header.get("eps")
    if model_eps is None:
        raise ValueError("model header missing layer_norm_eps")
    if input_eps is not None and input_eps != model_eps:
        raise ValueError("input/model layer_norm_eps mismatch")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(b"NFP_BINARY_V1\n")
        for key in [
            "num_layers",
            "num_heads",
            "model_dim",
            "head_dim",
            "hidden_dim",
            "vocab_size",
            "seq_len",
        ]:
            f.write(f"{key}={model_header[key]}\n".encode("ascii"))
        f.write(f"layer_norm_eps={model_eps}\n".encode("ascii"))
        f.write(b"BINARY_START\n")

        write_i32(f, tokens)
        write_f64(f, embeddings)

        layers = model_payload["layers"]
        for layer in layers:
            for head in layer["heads"]:
                write_f64(f, head["W_Q"])
                write_f64(f, head["b_Q"])
                write_f64(f, head["W_K"])
                write_f64(f, head["b_K"])
                write_f64(f, head["W_V"])
                write_f64(f, head["b_V"])
                write_f64(f, head["W_O"])
            write_f64(f, layer["attn_bias"])
            write_f64(f, layer["w_in"])
            write_f64(f, layer["b_in"])
            write_f64(f, layer["w_out"])
            write_f64(f, layer["b_out"])
            write_f64(f, layer["ln1_gamma"])
            write_f64(f, layer["ln1_beta"])
            write_f64(f, layer["ln2_gamma"])
            write_f64(f, layer["ln2_beta"])

        model_dim = int(model_header["model_dim"])
        vocab_size = int(model_header["vocab_size"])
        write_f64(f, [1.0] * model_dim)
        write_f64(f, [0.0] * model_dim)
        write_f64(f, [0.0] * (model_dim * vocab_size))

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
