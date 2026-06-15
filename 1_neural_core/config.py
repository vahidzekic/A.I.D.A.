"""
=============================================================================
A.I.D.A. — Artificially Intelligent Digital Assistant
=============================================================================
FILE: config.py
PHASE: 1 (Foundation)
PURPOSE: Central hyperparameter registry for our "Baby LLM".

All model dimensions live here in a single Python dataclass.
To scale the model up or down, you ONLY need to edit this file.

DESIGN NOTES:
    - We use @dataclass for clean, immutable-style config objects.
    - Every parameter has a comment explaining what it controls.
    - The "Baby" config is intentionally tiny (~15-30M params) so you
      can train and run it on a single laptop GPU or even CPU.
=============================================================================
"""

from dataclasses import dataclass


@dataclass
class BabyLLMConfig:
    """
    Hyperparameters for our Neuro-Symbolic Baby LLM.

    Architecture Overview:
    ┌─────────────────────────────────────────────────────┐
    │  Token Embedding  (vocab_size → d_model)            │
    │  Position Embedding (max_seq_len → d_model)         │
    │  ┌───────────────────────────────────────────────┐  │
    │  │  Transformer Block × n_layers                 │  │
    │  │  ┌─────────────────────────────────────────┐  │  │
    │  │  │  LayerNorm → CausalSelfAttention        │  │  │
    │  │  │  LayerNorm → FeedForward (SwiGLU)       │  │  │
    │  │  └─────────────────────────────────────────┘  │  │
    │  └───────────────────────────────────────────────┘  │
    │  Final LayerNorm → Linear Head (d_model → vocab)    │
    └─────────────────────────────────────────────────────┘
    """

    # ── Vocabulary & Sequence ──────────────────────────────────────────
    # vocab_size: How many unique tokens the model can understand.
    #   - We'll use a character-level or BPE tokenizer later.
    #   - For now, 8192 is a reasonable small-vocab BPE size.
    vocab_size: int = 8192

    # max_seq_len: Maximum number of tokens in a single input sequence.
    #   - This defines the "context window" of the model.
    #   - 512 tokens ≈ ~1-2 paragraphs of text.
    max_seq_len: int = 512

    # ── Model Dimensions ───────────────────────────────────────────────
    # d_model: The dimensionality of token embeddings and all hidden states.
    #   - Every token is represented as a vector of this size.
    #   - Bigger = more capacity, but more compute and memory.
    #   - GPT-2 Small uses 768. We use 256 for our "Baby".
    d_model: int = 256

    # n_heads: Number of attention heads in Multi-Head Attention.
    #   - Each head independently learns different "types" of relationships.
    #   - d_model MUST be divisible by n_heads.
    #   - head_dim = d_model // n_heads = 256 // 4 = 64
    n_heads: int = 4

    # n_layers: Number of stacked Transformer blocks.
    #   - More layers = deeper reasoning, but slower and more memory.
    #   - GPT-2 Small uses 12. We use 4 for our "Baby".
    n_layers: int = 4

    # d_ff: Dimensionality of the FeedForward network's hidden layer.
    #   - Standard practice: d_ff = 4 * d_model = 1024.
    #   - With SwiGLU, we use (4 * d_model * 2/3) rounded to nearest 64
    #     for efficiency, but we keep it simple here.
    d_ff: int = 1024

    # ── Regularization ─────────────────────────────────────────────────
    # dropout: Probability of randomly zeroing activations during training.
    #   - Prevents overfitting by forcing the model to be redundant.
    #   - 0.1 = 10% of neurons are randomly turned off each forward pass.
    dropout: float = 0.1

    # ── Training ───────────────────────────────────────────────────────
    # learning_rate: Step size for the optimizer (AdamW).
    learning_rate: float = 3e-4

    # batch_size: Number of sequences processed in parallel per step.
    batch_size: int = 32

    # max_epochs: Maximum number of full passes through the dataset.
    max_epochs: int = 10

    def __post_init__(self):
        """Validate critical constraints at instantiation time."""
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by "
            f"n_heads ({self.n_heads}). "
            f"Got remainder: {self.d_model % self.n_heads}"
        )
        self.head_dim = self.d_model // self.n_heads


# ═══════════════════════════════════════════════════════════════════════════
# QUICK TEST — Run this file directly to verify the config instantiates.
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cfg = BabyLLMConfig()
    print("=" * 60)
    print("  A.I.D.A. Baby LLM — Configuration")
    print("=" * 60)
    print(f"  Vocabulary size  : {cfg.vocab_size:,}")
    print(f"  Max sequence len : {cfg.max_seq_len}")
    print(f"  Model dimension  : {cfg.d_model}")
    print(f"  Attention heads  : {cfg.n_heads}")
    print(f"  Head dimension   : {cfg.head_dim}")
    print(f"  Transformer layers: {cfg.n_layers}")
    print(f"  FF hidden dim    : {cfg.d_ff}")
    print(f"  Dropout          : {cfg.dropout}")
    print(f"  Learning rate    : {cfg.learning_rate}")
    print(f"  Batch size       : {cfg.batch_size}")
    print("=" * 60)

    # Quick parameter count estimate (rough)
    # Embedding: vocab_size * d_model
    # Pos Embed: max_seq_len * d_model
    # Per layer: ~4 * d_model^2 (attention) + ~2 * d_model * d_ff (FF)
    embed_params = cfg.vocab_size * cfg.d_model + cfg.max_seq_len * cfg.d_model
    attn_params_per_layer = 4 * cfg.d_model ** 2  # Q, K, V, Out projections
    ff_params_per_layer = 2 * cfg.d_model * cfg.d_ff  # Up + Down projections
    total_est = embed_params + cfg.n_layers * (attn_params_per_layer + ff_params_per_layer)
    print(f"  Estimated params : ~{total_est / 1e6:.1f}M")
    print("=" * 60)
    print("  ✅ Config validated successfully!")
