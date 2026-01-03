#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Generate a "Rigorous Induction" dataset for NFP verification.
Constructs a sequence of random single-token words repeated in a fixed pattern.

Dataset intent (heuristic, model-agnostic):
- The next token is deterministically defined by a previous occurrence in the sequence.
- Randomized content reduces semantic cues but does not guarantee model probabilities.

This aims to isolate induction-style copying from semantic completion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from transformers import GPT2Model, GPT2Tokenizer
except ImportError:
    print("Error: transformers library not installed.")
    print("Install with: uv add transformers torch")
    sys.exit(1)


def select_vocab_candidates(
    tokenizer,
    vocab_min: int,
    vocab_max: int,
    min_word_length: int,
    require_leading_space: bool,
) -> list[int]:
    candidates = []
    for tid in range(vocab_min, vocab_max):
        word = tokenizer.decode([tid])
        if len(word.strip()) <= min_word_length:
            continue
        if require_leading_space and not word.startswith(" "):
            continue
        candidates.append(tid)
    return candidates


def export_rigorous_induction(
    output_path: str = "models/gpt2_rigorous.nfpt",
    seq_len: int = 256,
    pattern_len: int = 20,
    seed: int = 1337,
    vocab_min: int = 1000,
    vocab_max: int = 5000,
    min_word_length: int = 4,
    require_leading_space: bool = True,
    model_name: str = "gpt2",
) -> None:
    print(f"Loading {model_name}...")
    model = GPT2Model.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    config = model.config
    layer_norm_eps = float(config.layer_norm_epsilon)

    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if pattern_len <= 0 or pattern_len > seq_len:
        raise ValueError("pattern_len must be between 1 and seq_len")
    if vocab_min < 0 or vocab_max <= vocab_min:
        raise ValueError("invalid vocab range")

    vocab_candidates = select_vocab_candidates(
        tokenizer,
        vocab_min=vocab_min,
        vocab_max=vocab_max,
        min_word_length=min_word_length,
        require_leading_space=require_leading_space,
    )
    if len(vocab_candidates) < pattern_len:
        raise ValueError(
            f"Need at least {pattern_len} vocab candidates; only found {len(vocab_candidates)}"
        )

    np.random.seed(seed)
    unique_pattern = np.random.choice(vocab_candidates, size=pattern_len, replace=False)

    repeats = (seq_len // pattern_len) + 1
    full_sequence = np.tile(unique_pattern, repeats)[:seq_len]

    last_token = full_sequence[-1]
    prev_idx = seq_len - 1 - pattern_len
    target_token = full_sequence[prev_idx + 1]

    print("\nSequence Structure:")
    print(f"  Pattern Length: {pattern_len}")
    print(f"  Total Length:   {seq_len}")
    print(f"  Seed:           {seed}")
    print(f"  Vocab Range:    [{vocab_min}, {vocab_max})")
    print(
        f"  Token Filter:   min_len>{min_word_length}, "
        f"leading_space={require_leading_space}"
    )
    print(f"  Last Token:     '{tokenizer.decode([last_token])}' (ID: {last_token})")
    print(f"  Previous Occur: Index {prev_idx}")
    print(
        f"  True Target:    '{tokenizer.decode([target_token])}' (ID: {target_token})"
    )

    wte = model.wte.weight.detach().numpy()
    wpe = model.wpe.weight.detach().numpy()
    sample_embeddings = wte[full_sequence] + wpe[:seq_len]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nExporting to {output_path}...")
    with output_file.open("wb") as f:
        write_header(
            f,
            num_layers=config.n_layer,
            num_heads=config.n_head,
            model_dim=768,
            head_dim=64,
            hidden_dim=config.n_inner or 4 * 768,
            vocab_size=config.vocab_size,
            seq_len=seq_len,
            layer_norm_eps=layer_norm_eps,
            gelu_kind="tanh",
        )

        write_i32(f, full_sequence)
        write_f64(f, sample_embeddings)

        for layer_idx in range(config.n_layer):
            block = model.h[layer_idx]

            c_attn = block.attn.c_attn.weight.detach().numpy()
            c_attn_bias = get_bias(block.attn.c_attn.bias, 3 * 768)
            c_proj = block.attn.c_proj.weight.detach().numpy()
            c_proj_bias = get_bias(block.attn.c_proj.bias, 768)

            w_q_all = c_attn[:, 0:768]
            w_k_all = c_attn[:, 768 : 2 * 768]
            w_v_all = c_attn[:, 2 * 768 : 3 * 768]
            b_q_all = c_attn_bias[0:768]
            b_k_all = c_attn_bias[768 : 2 * 768]
            b_v_all = c_attn_bias[2 * 768 : 3 * 768]

            for h in range(12):
                start, end = h * 64, (h + 1) * 64
                write_f64(f, w_q_all[:, start:end])
                write_f64(f, b_q_all[start:end])
                write_f64(f, w_k_all[:, start:end])
                write_f64(f, b_k_all[start:end])
                write_f64(f, w_v_all[:, start:end])
                write_f64(f, b_v_all[start:end])
                write_f64(f, c_proj[start:end, :])

            write_f64(f, c_proj_bias)

            write_f64(f, block.mlp.c_fc.weight.detach().numpy())
            write_f64(f, get_bias(block.mlp.c_fc.bias, config.n_inner or 4 * 768))
            write_f64(f, block.mlp.c_proj.weight.detach().numpy())
            write_f64(f, get_bias(block.mlp.c_proj.bias, 768))

            write_f64(f, block.ln_1.weight.detach().numpy())
            write_f64(f, get_bias(block.ln_1.bias, 768))
            write_f64(f, block.ln_2.weight.detach().numpy())
            write_f64(f, get_bias(block.ln_2.bias, 768))

        write_f64(f, model.ln_f.weight.detach().numpy())
        write_f64(f, get_bias(model.ln_f.bias, 768))
        write_f64(f, wte.T)

    print("Done.")


def write_header(f, **fields: object) -> None:
    f.write(b"NFP_BINARY_V1\n")
    for key, value in fields.items():
        f.write(f"{key}={value}\n".encode("ascii"))
    f.write(b"BINARY_START\n")


def write_i32(f, data: np.ndarray) -> None:
    arr = np.ascontiguousarray(data, dtype="<i4")
    f.write(arr.tobytes(order="C"))


def write_f64(f, data: np.ndarray) -> None:
    arr = np.ascontiguousarray(data, dtype="<f8")
    f.write(arr.tobytes(order="C"))


def get_bias(param, size: int) -> np.ndarray:
    if param is None:
        return np.zeros(size, dtype=np.float64)
    return param.detach().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="models/gpt2_rigorous.nfpt")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--pattern-len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--vocab-min", type=int, default=1000)
    parser.add_argument("--vocab-max", type=int, default=5000)
    parser.add_argument("--min-word-length", type=int, default=4)
    parser.add_argument("--require-leading-space", action="store_true", default=True)
    parser.add_argument(
        "--allow-no-leading-space",
        action="store_true",
        help="Permit tokens without a leading space",
    )
    parser.add_argument("--model", default="gpt2")
    args = parser.parse_args()

    require_leading_space = args.require_leading_space and not args.allow_no_leading_space
    export_rigorous_induction(
        output_path=args.output,
        seq_len=args.seq_len,
        pattern_len=args.pattern_len,
        seed=args.seed,
        vocab_min=args.vocab_min,
        vocab_max=args.vocab_max,
        min_word_length=args.min_word_length,
        require_leading_space=require_leading_space,
        model_name=args.model,
    )
