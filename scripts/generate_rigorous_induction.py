#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Generate a "Rigorous Induction" dataset for NFP verification.
Constructs a sequence of random English words repeated K times.

Mathematical Guarantee:
- Probability of semantic prediction P(next|prev) ≈ 0 (random sequence)
- Probability of induction prediction P(next|prev_occurrence) = 1.0 (deterministic repetition)

This isolates the induction head mechanism from semantic completion.
"""

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


def export_rigorous_induction(output_path: str = "models/gpt2_rigorous.nfpt"):
    print("Loading GPT-2 Small...")
    model = GPT2Model.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    config = model.config

    # 1. Define a vocabulary of common, distinct English nouns
    # These token IDs are single-token words in GPT-2 (e.g., " apple", " logic")
    # This prevents fragmentation issues.
    # IDs: 1000-5000 range usually contains common words.
    # We filter for length > 4 to ensure they are substantive.
    vocab_candidates = []
    for tid in range(1000, 5000):
        word = tokenizer.decode([tid])
        if len(word.strip()) > 4 and word.startswith(" "):
            vocab_candidates.append(tid)

    # 2. Construct the Random Repeated Sequence
    # Pattern length L=30, Repeats K=10 -> Total 300 tokens
    # We truncate to 256 for the model.
    seq_len = 256
    pattern_len = 20

    np.random.seed(1337)  # Fixed seed for reproducibility
    unique_pattern = np.random.choice(vocab_candidates, size=pattern_len, replace=False)

    # [A, B, C, ...] -> [A, B, C, ..., A, B, C, ...]
    repeats = (seq_len // pattern_len) + 1
    full_sequence = np.tile(unique_pattern, repeats)[:seq_len]

    # 3. Verify the "Induction Target" property
    # The last token T[N] must have appeared at T[N - pattern_len].
    # The target is T[N - pattern_len + 1].
    last_token = full_sequence[-1]
    prev_idx = seq_len - 1 - pattern_len
    target_token = full_sequence[prev_idx + 1]

    print(f"\nSequence Structure:")
    print(f"  Pattern Length: {pattern_len}")
    print(f"  Total Length:   {seq_len}")
    print(f"  Last Token:     '{tokenizer.decode([last_token])}' (ID: {last_token})")
    print(f"  Previous Occur: Index {prev_idx}")
    print(
        f"  True Target:    '{tokenizer.decode([target_token])}' (ID: {target_token})"
    )

    # 4. Compute Embeddings & Weights
    wte = model.wte.weight.detach().numpy()
    wpe = model.wpe.weight.detach().numpy()
    sample_embeddings = wte[full_sequence] + wpe[:seq_len]

    # 5. Export
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nExporting to {output_path}...")
    with open(output_path, "wb") as f:
        write_header(
            f,
            num_layers=config.n_layer,
            num_heads=config.n_head,
            model_dim=768,
            head_dim=64,
            hidden_dim=config.n_inner or 4 * 768,
            vocab_size=config.vocab_size,
            seq_len=seq_len,
        )

        write_i32(f, full_sequence)
        write_f64(f, sample_embeddings)

        # Export Layers (Standard Loop)
        for layer_idx in range(config.n_layer):
            block = model.h[layer_idx]

            c_attn = block.attn.c_attn.weight.detach().numpy()
            c_attn_bias = get_bias(block.attn.c_attn.bias, 3 * 768)
            c_proj = block.attn.c_proj.weight.detach().numpy()
            c_proj_bias = get_bias(block.attn.c_proj.bias, 768)

            W_Q_all = c_attn[:, 0:768]
            W_K_all = c_attn[:, 768 : 2 * 768]
            W_V_all = c_attn[:, 2 * 768 : 3 * 768]
            b_Q_all = c_attn_bias[0:768]
            b_K_all = c_attn_bias[768 : 2 * 768]
            b_V_all = c_attn_bias[2 * 768 : 3 * 768]

            for h in range(12):
                start, end = h * 64, (h + 1) * 64
                write_f64(f, W_Q_all[:, start:end])
                write_f64(f, b_Q_all[start:end])
                write_f64(f, W_K_all[:, start:end])
                write_f64(f, b_K_all[start:end])
                write_f64(f, W_V_all[:, start:end])
                write_f64(f, b_V_all[start:end])
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


def write_header(f, **fields: int) -> None:
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
    export_rigorous_induction()
