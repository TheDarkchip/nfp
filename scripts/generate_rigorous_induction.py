#!/usr/bin/env python3
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
    with open(output_path, "w") as f:
        # Header
        f.write("NFP_TEXT_V2\n")
        f.write(f"num_layers={config.n_layer}\n")
        f.write(f"num_heads={config.n_head}\n")
        f.write(f"model_dim={768}\n")
        f.write(f"head_dim={64}\n")
        f.write(f"hidden_dim={config.n_inner or 4 * 768}\n")
        f.write(f"vocab_size={config.vocab_size}\n")
        f.write(f"seq_len={seq_len}\n\n")

        f.write("TOKENS\n")
        write_tokens(f, full_sequence)
        f.write("\nEMBEDDINGS\n")
        write_matrix(f, sample_embeddings)

        # Export Layers (Standard Loop)
        for layer_idx in range(config.n_layer):
            block = model.h[layer_idx]
            f.write(f"LAYER {layer_idx}\n")

            c_attn = block.attn.c_attn.weight.detach().numpy()
            c_proj = block.attn.c_proj.weight.detach().numpy()

            W_Q_all = c_attn[:, 0:768]
            W_K_all = c_attn[:, 768 : 2 * 768]
            W_V_all = c_attn[:, 2 * 768 : 3 * 768]

            for h in range(12):
                start, end = h * 64, (h + 1) * 64
                f.write(f"HEAD {h}\n")
                f.write("W_Q\n")
                write_matrix(f, W_Q_all[:, start:end])
                f.write("W_K\n")
                write_matrix(f, W_K_all[:, start:end])
                f.write("W_V\n")
                write_matrix(f, W_V_all[:, start:end])
                f.write("W_O\n")
                write_matrix(f, c_proj[start:end, :])

            f.write("MLP\n")
            f.write("W_in\n")
            write_matrix(f, block.mlp.c_fc.weight.detach().numpy())
            f.write("b_in\n")
            write_vector(f, block.mlp.c_fc.bias.detach().numpy())
            f.write("W_out\n")
            write_matrix(f, block.mlp.c_proj.weight.detach().numpy())
            f.write("b_out\n")
            write_vector(f, block.mlp.c_proj.bias.detach().numpy())

            f.write("LN1_GAMMA\n")
            write_vector(f, block.ln_1.weight.detach().numpy())
            f.write("LN1_BETA\n")
            write_vector(f, block.ln_1.bias.detach().numpy())
            f.write("LN2_GAMMA\n")
            write_vector(f, block.ln_2.weight.detach().numpy())
            f.write("LN2_BETA\n")
            write_vector(f, block.ln_2.bias.detach().numpy())

        f.write("LN_F_GAMMA\n")
        write_vector(f, model.ln_f.weight.detach().numpy())
        f.write("LN_F_BETA\n")
        write_vector(f, model.ln_f.bias.detach().numpy())
        f.write("UNEMBEDDING\n")
        write_matrix(f, wte.T)

    print("Done.")


def write_matrix(f, m):
    for row in m:
        f.write(" ".join(f"{x:.9g}" for x in row) + "\n")


def write_vector(f, v):
    f.write(" ".join(f"{x:.9g}" for x in v) + "\n")


def write_tokens(f, t):
    s = [str(x) for x in t]
    for i in range(0, len(s), 64):
        f.write(" ".join(s[i : i + 64]) + "\n")


if __name__ == "__main__":
    export_rigorous_induction()
