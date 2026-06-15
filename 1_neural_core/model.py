"""
=============================================================================
A.I.D.A. — Artificially Intelligent Digital Assistant
=============================================================================
FILE: model.py
PHASE: 1 & 2 (Custom LLM Architecture)
PURPOSE: The complete Transformer-based "Baby LLM" — built from pure PyTorch.

This file contains four core building blocks, bottom-up:

    1. CausalSelfAttention  — The "eyes" of the model. Learns WHICH tokens
                               to pay attention to via Q·Kᵀ dot products,
                               masked so the model can't cheat by looking
                               at future tokens.

    2. FeedForward (SwiGLU)  — The "thinking" layer. A non-linear MLP that
                               processes each token independently after
                               attention has mixed information across tokens.

    3. TransformerBlock      — One "layer" of the model: Attention + FFN,
                               each wrapped with LayerNorm and residual
                               connections (Pre-Norm architecture).

    4. NeuroSymbolicBabyLLM  — The full model: Embeddings → N×Blocks → Head.
                               Maps token IDs → logits over vocabulary.

MATHEMATICAL REFERENCES:
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - "GLU Variants Improve Transformer" (Shazeer, 2020) — SwiGLU
    - "On Layer Normalization in the Transformer Architecture" (Xiong, 2020)

TENSOR SHAPE CONVENTIONS (used throughout):
    B   = Batch size           (number of sequences processed in parallel)
    T   = Sequence length      (number of tokens per sequence)
    C   = d_model              (embedding / hidden dimension, e.g. 256)
    H   = n_heads              (number of attention heads, e.g. 4)
    D   = head_dim             (C // H = per-head dimension, e.g. 64)
    V   = vocab_size           (size of the token vocabulary, e.g. 8192)
    F   = d_ff                 (feedforward hidden dimension, e.g. 1024)
=============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BabyLLMConfig


# ═══════════════════════════════════════════════════════════════════════════
# 1. CAUSAL SELF-ATTENTION
# ═══════════════════════════════════════════════════════════════════════════
class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal (Masked) Self-Attention.

    ┌──────────────────────────────────────────────────────────────────┐
    │  THE CORE IDEA:                                                  │
    │                                                                  │
    │  For every token in the sequence, we want to ask:                │
    │    "Which OTHER tokens (before me) should I pay attention to?"   │
    │                                                                  │
    │  We do this by computing three vectors for each token:           │
    │    Q (Query)  = "What am I looking for?"                         │
    │    K (Key)    = "What do I contain?"                             │
    │    V (Value)  = "What information do I provide if selected?"     │
    │                                                                  │
    │  The attention score = softmax(Q · Kᵀ / √d_k) · V               │
    │                                                                  │
    │  The "causal mask" ensures token at position i can only attend   │
    │  to tokens at positions 0, 1, ..., i (not future tokens).        │
    │  This is what makes it AUTOREGRESSIVE — it generates left→right. │
    └──────────────────────────────────────────────────────────────────┘

    Multi-Head Mechanism:
        Instead of one big attention operation, we split the embedding
        into H independent "heads", each with dimension D = C/H.
        Each head learns a DIFFERENT type of relationship:
            - Head 1 might learn syntactic relationships (subject-verb)
            - Head 2 might learn positional proximity
            - Head 3 might learn semantic similarity
            - Head 4 might learn coreference (pronouns → nouns)
        The outputs of all heads are concatenated and projected back to C.
    """

    def __init__(self, config: BabyLLMConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim  # D = C // H
        self.d_model = config.d_model    # C

        # ── Linear Projections ─────────────────────────────────────────
        # These are the learned weight matrices Wq, Wk, Wv, Wo.
        #
        # Instead of creating H separate small matrices of shape [C, D],
        # we use ONE big matrix of shape [C, C] that computes all heads
        # at once — this is much more efficient on the GPU.
        #
        # Shape of each weight matrix: [C, C] = [256, 256]
        #
        # Wq transforms input → queries:  x @ Wq = Q   [B, T, C] → [B, T, C]
        # Wk transforms input → keys:     x @ Wk = K   [B, T, C] → [B, T, C]
        # Wv transforms input → values:   x @ Wv = V   [B, T, C] → [B, T, C]
        # Wo projects concatenated heads:  attn @ Wo    [B, T, C] → [B, T, C]
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)

        # Dropout applied to attention weights (randomly zeroes some
        # attention connections during training for regularization).
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # ── Causal Mask ────────────────────────────────────────────────
        # This is a LOWER-TRIANGULAR matrix of ones:
        #
        #   [[1, 0, 0, 0],    Token 0 can see: [0]
        #    [1, 1, 0, 0],    Token 1 can see: [0, 1]
        #    [1, 1, 1, 0],    Token 2 can see: [0, 1, 2]
        #    [1, 1, 1, 1]]    Token 3 can see: [0, 1, 2, 3]
        #
        # We register it as a buffer (not a parameter — no gradients).
        # Shape: [1, 1, max_seq_len, max_seq_len] — the two leading 1s
        # allow broadcasting across Batch and Head dimensions.
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Multi-Head Causal Self-Attention.

        Args:
            x: Input tensor of shape [B, T, C]
               B = batch size, T = sequence length, C = d_model

        Returns:
            Output tensor of shape [B, T, C]
        """
        B, T, C = x.shape  # e.g. [32, 128, 256]

        # ── Step 1: Compute Q, K, V ────────────────────────────────────
        # Each linear layer: [B, T, C] @ [C, C] → [B, T, C]
        # All heads are computed together in one matrix multiply.
        q = self.W_q(x)  # [B, T, C] → [B, T, C]  e.g. [32, 128, 256]
        k = self.W_k(x)  # [B, T, C] → [B, T, C]
        v = self.W_v(x)  # [B, T, C] → [B, T, C]

        # ── Step 2: Reshape into multiple heads ────────────────────────
        # We need to split the C dimension into H heads of D dimensions.
        #   [B, T, C] → [B, T, H, D] → [B, H, T, D]
        #
        # The transpose(1, 2) swaps the T and H dimensions so that
        # each head's tokens are contiguous in memory, enabling
        # efficient batched matrix multiplication.
        #
        # Example with our config:
        #   [32, 128, 256] → [32, 128, 4, 64] → [32, 4, 128, 64]
        #    B    T    C      B    T   H   D      B   H   T    D
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # q, k, v are now all [B, H, T, D] = [32, 4, 128, 64]

        # ── Step 3: Scaled Dot-Product Attention ───────────────────────
        #
        #   Attention(Q, K, V) = softmax( Q · Kᵀ / √d_k ) · V
        #
        # WHY the dot product?
        #   Q · Kᵀ measures the SIMILARITY between every pair of tokens.
        #   High dot product = "these two tokens are relevant to each other."
        #
        # WHY scale by √d_k?
        #   Without scaling, large d_k values cause dot products to grow
        #   very large, pushing softmax into regions with tiny gradients
        #   (the "vanishing gradient" problem). Dividing by √d_k keeps
        #   the variance of the dot products at ~1.0 regardless of d_k.
        #
        # Matrix multiply: [B, H, T, D] @ [B, H, D, T] → [B, H, T, T]
        #   For each head: every token's Q attends to every token's K.
        #   Result is a T×T "attention map" per head per batch.
        scale = math.sqrt(self.head_dim)  # √64 = 8.0
        attn_scores = (q @ k.transpose(-2, -1)) / scale
        # attn_scores shape: [B, H, T, T] = [32, 4, 128, 128]

        # ── Step 4: Apply Causal Mask ──────────────────────────────────
        # We set future positions to -infinity BEFORE softmax.
        # After softmax, e^(-inf) → 0, so future tokens get ZERO attention.
        #
        # The mask is [1, 1, max_seq_len, max_seq_len] but we only
        # slice [:T, :T] because the actual sequence might be shorter
        # than max_seq_len.
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0,
            float('-inf')
        )
        # attn_scores shape: still [B, H, T, T], but upper triangle = -inf

        # ── Step 5: Softmax → Attention Weights ───────────────────────
        # Softmax along the last dimension (the "key" dimension).
        # Converts raw scores into a probability distribution:
        #   - Each row sums to 1.0
        #   - Higher scores → higher probability → more attention
        attn_weights = torch.softmax(attn_scores, dim=-1)
        # attn_weights shape: [B, H, T, T] = [32, 4, 128, 128]

        # Apply dropout to attention weights (randomly zero some connections)
        attn_weights = self.attn_dropout(attn_weights)

        # ── Step 6: Weighted Sum of Values ─────────────────────────────
        # Multiply attention weights by values:
        #   [B, H, T, T] @ [B, H, T, D] → [B, H, T, D]
        #
        # Each token's output is a weighted average of ALL value vectors,
        # where the weights come from the attention scores.
        # Token i's output = Σⱼ (attention_weight[i,j] * V[j])
        attn_output = attn_weights @ v
        # attn_output shape: [B, H, T, D] = [32, 4, 128, 64]

        # ── Step 7: Concatenate Heads & Project ────────────────────────
        # Reverse the reshape: [B, H, T, D] → [B, T, H, D] → [B, T, C]
        # transpose(1, 2): swap H and T back
        # contiguous(): ensure memory layout is contiguous after transpose
        # view(): flatten last two dims (H * D = n_heads * head_dim = C)
        attn_output = (
            attn_output
            .transpose(1, 2)       # [B, H, T, D] → [B, T, H, D]
            .contiguous()          # ensure contiguous memory layout
            .view(B, T, C)         # [B, T, H, D] → [B, T, C]  (H*D = C)
        )
        # attn_output shape: [B, T, C] = [32, 128, 256]

        # Final output projection: mix information across heads.
        # [B, T, C] @ [C, C] → [B, T, C]
        output = self.resid_dropout(self.W_o(attn_output))
        # output shape: [B, T, C] = [32, 128, 256]

        return output


# ═══════════════════════════════════════════════════════════════════════════
# 2. FEED-FORWARD NETWORK (SwiGLU Variant)
# ═══════════════════════════════════════════════════════════════════════════
class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network with SwiGLU activation.

    ┌──────────────────────────────────────────────────────────────────┐
    │  THE CORE IDEA:                                                  │
    │                                                                  │
    │  After attention has mixed information ACROSS tokens (horizontal │
    │  mixing), the FFN processes each token INDEPENDENTLY (vertical   │
    │  processing). It's a two-layer MLP:                              │
    │                                                                  │
    │  Standard:  FFN(x) = W₂ · ReLU(W₁ · x)                         │
    │  SwiGLU:    FFN(x) = W₂ · (Swish(W_gate · x) ⊙ (W_up · x))    │
    │                                                                  │
    │  SwiGLU (Shazeer 2020) replaces ReLU with a GATED activation:   │
    │    1. W_gate projects x to hidden dim and applies Swish          │
    │    2. W_up projects x to hidden dim (no activation)              │
    │    3. Element-wise multiply (⊙) = the "gate" controls flow      │
    │    4. W_down projects back to model dimension                    │
    │                                                                  │
    │  WHY SwiGLU? It consistently outperforms plain ReLU/GELU in     │
    │  modern LLMs (used by LLaMA, PaLM, Mistral, Gemma, etc.)       │
    └──────────────────────────────────────────────────────────────────┘

    Tensor Shapes:
        Input:  [B, T, C]    = [32, 128, 256]
        After W_gate & W_up: [B, T, F] = [32, 128, 1024]
        After gating (⊙):   [B, T, F] = [32, 128, 1024]
        After W_down:        [B, T, C] = [32, 128, 256]
    """

    def __init__(self, config: BabyLLMConfig):
        super().__init__()

        # W_gate: Projects to hidden dim for the Swish gate
        # Shape: [C, F] = [256, 1024]
        self.W_gate = nn.Linear(config.d_model, config.d_ff, bias=False)

        # W_up: Projects to hidden dim (ungated path)
        # Shape: [C, F] = [256, 1024]
        self.W_up = nn.Linear(config.d_model, config.d_ff, bias=False)

        # W_down: Projects back from hidden dim to model dim
        # Shape: [F, C] = [1024, 256]
        self.W_down = nn.Linear(config.d_ff, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C] — input from attention + residual

        Returns:
            [B, T, C] — processed output, same shape as input
        """
        # Gate path: Swish activation (also called SiLU = x * sigmoid(x))
        # Swish is smooth, non-monotonic, and allows small negative values
        # through (unlike ReLU which hard-clips at 0).
        #
        # [B, T, C] @ [C, F] → [B, T, F], then apply Swish element-wise
        gate = F.silu(self.W_gate(x))  # [B, T, F] = [32, 128, 1024]

        # Up path: no activation, just linear projection
        # [B, T, C] @ [C, F] → [B, T, F]
        up = self.W_up(x)              # [B, T, F] = [32, 128, 1024]

        # Element-wise gating: the gate controls how much of "up" passes
        # through. This is the key insight of GLU (Gated Linear Units).
        # [B, T, F] ⊙ [B, T, F] → [B, T, F]
        hidden = gate * up             # [B, T, F] = [32, 128, 1024]

        # Project back down to model dimension
        # [B, T, F] @ [F, C] → [B, T, C]
        output = self.dropout(self.W_down(hidden))  # [B, T, C] = [32, 128, 256]

        return output


# ═══════════════════════════════════════════════════════════════════════════
# 3. TRANSFORMER BLOCK (Pre-Norm Architecture)
# ═══════════════════════════════════════════════════════════════════════════
class TransformerBlock(nn.Module):
    """
    A single Transformer layer: Attention + FeedForward, both with
    Pre-LayerNorm and residual connections.

    ┌──────────────────────────────────────────────────────────────────┐
    │  WHY PRE-NORM (vs. Post-Norm)?                                   │
    │                                                                  │
    │  Original Transformer (Post-Norm):                               │
    │    x = LayerNorm(x + Attention(x))                               │
    │    x = LayerNorm(x + FFN(x))                                     │
    │                                                                  │
    │  Modern / Pre-Norm (what we use):                                │
    │    x = x + Attention(LayerNorm(x))                               │
    │    x = x + FFN(LayerNorm(x))                                     │
    │                                                                  │
    │  Pre-Norm is more stable during training because:                │
    │    1. Gradients flow through residual connections unimpeded       │
    │    2. Each sub-layer receives a normalized input                  │
    │    3. This allows training deeper models without warmup tricks    │
    │                                                                  │
    │  Used by: GPT-2, GPT-3, LLaMA, Mistral, Gemma, etc.            │
    └──────────────────────────────────────────────────────────────────┘

    Residual Connection:
        The "+x" (skip connection) is CRITICAL. Without it, gradients
        would need to flow through every layer sequentially, dying
        exponentially. With it, gradients have a "highway" directly
        from output back to input.

        Mathematically:  output = input + f(input)
        Gradient:        ∂output/∂input = 1 + ∂f/∂input
        The "1" ensures the gradient is ALWAYS at least 1.
    """

    def __init__(self, config: BabyLLMConfig):
        super().__init__()

        # LayerNorm normalizes across the C (feature) dimension.
        # For each token vector of size C, it computes:
        #   LN(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
        # This stabilizes activations and speeds up training.
        self.ln_1 = nn.LayerNorm(config.d_model)  # Before attention
        self.ln_2 = nn.LayerNorm(config.d_model)  # Before FFN

        self.attention = CausalSelfAttention(config)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C] — input from previous block (or embeddings)

        Returns:
            [B, T, C] — output to next block
        """
        # Sub-layer 1: Multi-Head Causal Self-Attention with residual
        # Normalize → Attend → Add back to original (residual)
        x = x + self.attention(self.ln_1(x))  # [B, T, C] + [B, T, C] → [B, T, C]

        # Sub-layer 2: Feed-Forward Network with residual
        # Normalize → FFN → Add back to original (residual)
        x = x + self.ffn(self.ln_2(x))        # [B, T, C] + [B, T, C] → [B, T, C]

        return x


# ═══════════════════════════════════════════════════════════════════════════
# 4. THE FULL MODEL — NeuroSymbolicBabyLLM
# ═══════════════════════════════════════════════════════════════════════════
class NeuroSymbolicBabyLLM(nn.Module):
    """
    Complete autoregressive language model.

    ┌──────────────────────────────────────────────────────────────────┐
    │  FULL FORWARD PASS:                                              │
    │                                                                  │
    │  token_ids [B, T]  (integers, e.g. [32, 128])                    │
    │       │                                                          │
    │       ▼                                                          │
    │  Token Embedding [B, T, C]    (lookup table: ID → vector)        │
    │       +                                                          │
    │  Position Embedding [B, T, C] (learned position encoding)        │
    │       │                                                          │
    │       ▼                                                          │
    │  Dropout                                                         │
    │       │                                                          │
    │       ▼                                                          │
    │  ┌─── TransformerBlock 0 ───┐                                    │
    │  │  LN → Attention → +     │                                     │
    │  │  LN → FFN → +           │                                     │
    │  └──────────────────────────┘                                    │
    │       │                                                          │
    │       ▼                                                          │
    │  ┌─── TransformerBlock 1 ───┐                                    │
    │  │  ...                     │                                     │
    │  └──────────────────────────┘                                    │
    │       │        × n_layers                                        │
    │       ▼                                                          │
    │  Final LayerNorm [B, T, C]                                       │
    │       │                                                          │
    │       ▼                                                          │
    │  Linear Head [B, T, V]   (project C → vocab_size for logits)     │
    │       │                                                          │
    │       ▼                                                          │
    │  logits [B, T, V]  (raw scores over vocabulary for each token)   │
    └──────────────────────────────────────────────────────────────────┘

    WHY "NeuroSymbolic"?
        The "Neuro" part is this model — it handles fuzzy, probabilistic
        language understanding via neural attention.
        The "Symbolic" part (Phase 4) will be deterministic Python tools
        that handle exact computation, database lookups, etc.
        The Agentic Loop (Phase 5) will bridge them: the LLM decides
        WHEN and WHICH tool to call via structured JSON output.
    """

    def __init__(self, config: BabyLLMConfig):
        super().__init__()
        self.config = config

        # ── Embedding Layers ───────────────────────────────────────────
        # Token Embedding: maps each token ID to a dense vector.
        # This is a learnable lookup table of shape [V, C] = [8192, 256].
        # Input:  token_ids [B, T] (integers in range [0, vocab_size))
        # Output: [B, T, C] — each token becomes a 256-dim vector.
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Position Embedding: encodes WHERE each token is in the sequence.
        # Without this, the model has NO concept of token order (attention
        # is permutation-invariant). This is a learned lookup table of
        # shape [max_seq_len, C] = [512, 256].
        # Input:  position indices [T] = [0, 1, 2, ..., T-1]
        # Output: [T, C] → broadcast to [B, T, C]
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Dropout on embeddings (standard practice)
        self.embed_dropout = nn.Dropout(config.dropout)

        # ── Transformer Blocks ─────────────────────────────────────────
        # Stack n_layers identical blocks. nn.ModuleList ensures PyTorch
        # properly registers them (tracks parameters, moves to GPU, etc.)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # ── Output Head ────────────────────────────────────────────────
        # Final LayerNorm (Pre-Norm architecture requires one at the end)
        self.ln_final = nn.LayerNorm(config.d_model)

        # Linear projection from hidden dim to vocabulary size.
        # This produces "logits" — raw (unnormalized) scores for each
        # token in the vocabulary. During training, we apply cross-entropy
        # loss directly to logits. During generation, we apply softmax
        # to get probabilities.
        # Shape: [C, V] = [256, 8192]
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # ── Weight Tying ───────────────────────────────────────────────
        # CRITICAL TRICK: Share weights between token_embedding and lm_head.
        #
        # Why? The embedding maps token_id → vector (what does this word mean?).
        # The lm_head maps vector → token_id (what word does this vector represent?).
        # These are conceptually INVERSE operations, so sharing weights:
        #   1. Reduces parameter count significantly (saves V*C = ~2M params)
        #   2. Creates a consistent latent space (embed and un-embed agree)
        #   3. Empirically improves performance
        #
        # Used by GPT-2, GPT-3, LLaMA, T5, BERT, and most modern LLMs.
        self.lm_head.weight = self.token_embedding.weight

        # ── Initialize Weights ─────────────────────────────────────────
        # Proper initialization is CRUCIAL for training stability.
        self.apply(self._init_weights)

        # Count and report total parameters
        n_params = sum(p.numel() for p in self.parameters())
        # Subtract double-counted tied weights
        n_params -= self.token_embedding.weight.numel()
        print(f"  NeuroSymbolicBabyLLM initialized: {n_params / 1e6:.2f}M parameters")

    def _init_weights(self, module: nn.Module):
        """
        Custom weight initialization following GPT-2's scheme.

        - Linear layers: Normal distribution with std=0.02
        - Embeddings: Normal distribution with std=0.02
        - LayerNorm: gamma=1, beta=0 (the default, but explicit is better)

        WHY 0.02?
            The Xavier/He init suggests std ∝ 1/√fan_in. For d_model=256,
            1/√256 = 0.0625. GPT-2 uses 0.02 which is slightly smaller,
            providing a more conservative start that works well in practice.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Full forward pass of the Baby LLM.

        Args:
            token_ids: [B, T] — integer token IDs
            targets:   [B, T] — target token IDs for loss computation
                       (optional — None during inference/generation)

        Returns:
            logits: [B, T, V] — raw predictions over vocabulary
            loss:   scalar tensor or None — cross-entropy loss if targets given
        """
        B, T = token_ids.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        )

        # ── Step 1: Create position indices ────────────────────────────
        # Generate [0, 1, 2, ..., T-1] on the same device as token_ids.
        # Shape: [T] — will broadcast across batch dimension.
        positions = torch.arange(0, T, dtype=torch.long, device=token_ids.device)
        # positions shape: [T] = [128]

        # ── Step 2: Embed tokens + positions ───────────────────────────
        # Token embedding:    [B, T] → lookup → [B, T, C]
        # Position embedding: [T]    → lookup → [T, C] → broadcast → [B, T, C]
        # Sum them element-wise: token identity + positional information
        tok_emb = self.token_embedding(token_ids)  # [B, T, C] = [32, 128, 256]
        pos_emb = self.position_embedding(positions)  # [T, C] → broadcast [B, T, C]

        x = self.embed_dropout(tok_emb + pos_emb)  # [B, T, C] = [32, 128, 256]

        # ── Step 3: Pass through all Transformer blocks ────────────────
        # Each block: [B, T, C] → [B, T, C]
        # Information flows deeper with each layer, building increasingly
        # abstract representations of the input.
        for block in self.blocks:
            x = block(x)
        # x shape: [B, T, C] = [32, 128, 256]

        # ── Step 4: Final LayerNorm ────────────────────────────────────
        x = self.ln_final(x)  # [B, T, C] = [32, 128, 256]

        # ── Step 5: Project to vocabulary logits ───────────────────────
        # [B, T, C] @ [C, V] → [B, T, V]
        # For each token position, we get a score for EVERY word in the vocab.
        # The highest score = the model's best guess for the next token.
        logits = self.lm_head(x)  # [B, T, V] = [32, 128, 8192]

        # ── Step 6: Compute loss (if targets are provided) ─────────────
        loss = None
        if targets is not None:
            # Cross-entropy expects:
            #   input:  [N, C] where N = total tokens, C = vocab_size
            #   target: [N]    where each value is the correct class index
            #
            # We reshape:
            #   logits:  [B, T, V] → [B*T, V]   = [4096, 8192]
            #   targets: [B, T]    → [B*T]       = [4096]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # [B*T, V]
                targets.view(-1),                   # [B*T]
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.

        The model generates ONE token at a time:
          1. Feed the current sequence through the model → get logits
          2. Take logits for the LAST position only (next-token prediction)
          3. Apply temperature scaling and optional top-k filtering
          4. Sample from the distribution (or take argmax)
          5. Append the new token and repeat

        Args:
            token_ids:      [B, T] — initial prompt token IDs
            max_new_tokens: how many tokens to generate
            temperature:    >1.0 = more random, <1.0 = more deterministic
            top_k:          if set, only sample from top-k highest prob tokens

        Returns:
            [B, T + max_new_tokens] — the full sequence with generated tokens
        """
        self.eval()  # Disable dropout for generation

        for _ in range(max_new_tokens):
            # Crop to max_seq_len if the sequence has grown too long.
            # We keep the MOST RECENT tokens (sliding window).
            context = token_ids if token_ids.size(1) <= self.config.max_seq_len \
                else token_ids[:, -self.config.max_seq_len:]

            # Forward pass — only need logits, not loss
            logits, _ = self(context)

            # Take logits for the LAST token position only
            # [B, T, V] → [B, V]
            logits = logits[:, -1, :]  # [B, V]

            # Temperature scaling: divide logits by temperature before softmax
            # Higher temp → flatter distribution → more randomness
            # Lower temp → sharper distribution → more deterministic
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering: zero out all logits except the top-k
            if top_k is not None:
                # Get the k-th largest value as threshold
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold = top_k_values[:, -1].unsqueeze(-1)  # [B, 1]
                logits[logits < threshold] = float('-inf')

            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)  # [B, V]

            # Sample one token from the distribution
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

            # Append to the sequence
            token_ids = torch.cat([token_ids, next_token], dim=1)  # [B, T+1]

        return token_ids


# ═══════════════════════════════════════════════════════════════════════════
# 5. VERIFICATION — Compile & Shape Check
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  A.I.D.A. — NeuroSymbolicBabyLLM Architecture Verification")
    print("=" * 70)

    # 1. Create config
    config = BabyLLMConfig()
    print(f"\n  Config: d_model={config.d_model}, n_heads={config.n_heads}, "
          f"n_layers={config.n_layers}, vocab={config.vocab_size}")

    # 2. Initialize model
    print("\n  Initializing model...")
    model = NeuroSymbolicBabyLLM(config)

    # 3. Create a dummy input — random token IDs
    #    Shape: [B, T] = [2, 64] (small batch for testing)
    batch_size = 2
    seq_len = 64
    dummy_input = torch.randint(
        low=0, high=config.vocab_size,
        size=(batch_size, seq_len)
    )
    print(f"\n  Input shape:  {list(dummy_input.shape)}  "
          f"(B={batch_size}, T={seq_len})")

    # 4. Forward pass — without targets (inference mode)
    print("  Running forward pass (inference)...")
    logits, loss = model(dummy_input)
    print(f"  Output logits shape: {list(logits.shape)}  "
          f"(B={batch_size}, T={seq_len}, V={config.vocab_size})")
    print(f"  Loss: {loss}  (None expected — no targets provided)")

    # 5. Forward pass — with targets (training mode)
    print("\n  Running forward pass (training with targets)...")
    dummy_targets = torch.randint(
        low=0, high=config.vocab_size,
        size=(batch_size, seq_len)
    )
    logits, loss = model(dummy_input, targets=dummy_targets)
    print(f"  Output logits shape: {list(logits.shape)}")
    print(f"  Loss: {loss.item():.4f}  (random init ≈ ln(vocab_size) = "
          f"{math.log(config.vocab_size):.4f})")

    # 6. Test generation
    print("\n  Testing autoregressive generation...")
    prompt = torch.randint(0, config.vocab_size, (1, 5))  # [1, 5] — single prompt
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"  Prompt shape:    {list(prompt.shape)}")
    print(f"  Generated shape: {list(generated.shape)}  "
          f"(5 prompt + 20 generated = 25)")

    # 7. Detailed parameter breakdown
    print("\n" + "─" * 70)
    print("  Parameter Breakdown:")
    print("─" * 70)
    total = 0
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        print(f"  {name:45s}  {str(list(param.shape)):20s}  {count:>10,}")
    print("─" * 70)
    print(f"  {'TOTAL':45s}  {'':20s}  {total:>10,}")
    print("─" * 70)

    print("\n  ✅ All shape checks passed! Architecture compiles successfully.")
    print("=" * 70)
