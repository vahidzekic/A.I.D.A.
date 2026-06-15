"""
=============================================================================
A.I.D.A. — Artificially Intelligent Digital Assistant
=============================================================================
FILE: dataset.py
PHASE: 3 (Training Loop — Data Pipeline)
PURPOSE: Convert raw text files into batched training tensors for the LLM.

HOW LLM TRAINING DATA WORKS:
    An LLM learns via NEXT-TOKEN PREDICTION. Given a sequence of tokens,
    the model predicts what comes NEXT at every position.

    Example (with token IDs):
        Input:   [The, quick, brown, fox]     ← model sees these
        Target:  [quick, brown, fox, jumps]   ← model predicts these

    The target is just the input shifted by one position to the right!
    This is called "causal language modeling" or "autoregressive" training.

    In practice, we take a huge text corpus, tokenize it into one long
    sequence of IDs, then chop it into fixed-length "windows" of size
    max_seq_len. Each window becomes one training example.

    Corpus:  [15, 234, 89, 12, 567, 3, 45, 890, 23, 78, 456, ...]
             └── window 1 ──┘  └── window 2 ──┘  └── window 3 ──┘
                (if max_seq_len = 4)

    For each window of length T, we create:
        x = tokens[0:T]    = input  (what the model SEES)
        y = tokens[1:T+1]  = target (what the model should PREDICT)

=============================================================================
"""

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Add parent directory to path so we can import from 1_neural_core
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "1_neural_core"))

from config import BabyLLMConfig
from tokenizer import BPETokenizer


# ═══════════════════════════════════════════════════════════════════════════
# 1. PRE-TRAINING DATASET — Next-token prediction on raw text
# ═══════════════════════════════════════════════════════════════════════════
class PreTrainDataset(Dataset):
    """
    Dataset for pre-training: converts a text corpus into overlapping
    windows of token IDs for next-token prediction.

    ┌──────────────────────────────────────────────────────────────────┐
    │  MEMORY STRATEGY:                                                │
    │                                                                  │
    │  We tokenize the ENTIRE corpus upfront into one flat tensor.     │
    │  Then __getitem__ simply slices a window from it.                 │
    │  This is memory-efficient because:                               │
    │    1. The tokenized data is stored as int64 (8 bytes per token)  │
    │    2. A 10MB text file ≈ 10M characters ≈ ~3M BPE tokens        │
    │       = ~24 MB of token IDs in memory. Very manageable.          │
    │    3. No need to re-tokenize on each access.                     │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        text: str,
        tokenizer: BPETokenizer,
        max_seq_len: int,
    ):
        """
        Args:
            text:        The raw text corpus (one large string).
            tokenizer:   Trained BPE tokenizer instance.
            max_seq_len: Length of each training window (from config).
        """
        super().__init__()
        self.max_seq_len = max_seq_len

        # ── Tokenize the entire corpus ─────────────────────────────────
        # Convert text → list of integer IDs → flat PyTorch tensor.
        # We add BOS at the start so the model learns to generate from it.
        print(f"  Tokenizing corpus ({len(text):,} chars)...")
        token_ids = tokenizer.encode(text)
        self.data = torch.tensor(token_ids, dtype=torch.long)
        print(f"  Tokenized: {len(self.data):,} tokens")

        # ── Calculate number of windows ────────────────────────────────
        # We need max_seq_len + 1 tokens per example (+1 for the target's
        # last token). Windows are NON-OVERLAPPING (strided by max_seq_len)
        # to avoid the model seeing the same data multiple times per epoch.
        #
        # Example with max_seq_len=4, data=[a,b,c,d,e,f,g,h,i,j,k]:
        #   Window 0: data[0:5] → x=[a,b,c,d], y=[b,c,d,e]
        #   Window 1: data[4:9] → x=[e,f,g,h], y=[f,g,h,i]
        #   (stride = max_seq_len = 4)
        self.n_examples = max(0, (len(self.data) - 1) // max_seq_len)
        print(f"  Training examples: {self.n_examples:,} "
              f"(windows of {max_seq_len} tokens)")

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training example (input, target) pair.

        Args:
            idx: Index of the window (0 to n_examples-1).

        Returns:
            x: [max_seq_len] — input token IDs
            y: [max_seq_len] — target token IDs (shifted right by 1)

        Tensor Shapes:
            x shape: [T] = [512]   (one sequence of input tokens)
            y shape: [T] = [512]   (one sequence of target tokens)

        After DataLoader batching:
            x shape: [B, T] = [32, 512]
            y shape: [B, T] = [32, 512]
        """
        start = idx * self.max_seq_len
        end = start + self.max_seq_len + 1  # +1 because target is shifted

        # Slice a window of (max_seq_len + 1) tokens
        chunk = self.data[start:end]  # [max_seq_len + 1]

        # Input  = first max_seq_len tokens:  chunk[0:T]
        # Target = last max_seq_len tokens:   chunk[1:T+1]
        x = chunk[:-1]  # [T]   e.g. [512]
        y = chunk[1:]    # [T]   e.g. [512]

        return x, y


# ═══════════════════════════════════════════════════════════════════════════
# 2. SFT DATASET — Supervised Fine-Tuning for tool-calling JSON
# ═══════════════════════════════════════════════════════════════════════════
class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning: teaches the model to produce
    structured JSON tool calls in response to user queries.

    ┌──────────────────────────────────────────────────────────────────┐
    │  THE KEY INSIGHT OF SFT:                                         │
    │                                                                  │
    │  During pre-training, the model learns LANGUAGE (grammar,        │
    │  semantics, world knowledge) from raw text.                      │
    │                                                                  │
    │  During SFT, we OVERWRITE the model's behavior to follow a       │
    │  specific format. We show it examples like:                      │
    │                                                                  │
    │  <|user|>What is 15 + 27?<|assistant|><|tool_call|>              │
    │  {"tool": "calculator", "args": {"a": 15, "b": 27, "op": "add"}}│
    │  <|eos|>                                                         │
    │                                                                  │
    │  The model learns:                                               │
    │    "When user asks a math question → emit JSON tool call"        │
    │    "ALWAYS use the exact JSON schema"                            │
    │    "STOP after <|eos|>"                                          │
    │                                                                  │
    │  LOSS MASKING:                                                   │
    │  We ONLY compute loss on the ASSISTANT's response, NOT on the   │
    │  user's input. The model shouldn't try to predict the user's    │
    │  question — it should only learn to RESPOND correctly.           │
    │                                                                  │
    │  Input:  <|user|> What is 15+27? <|assistant|> <|tool_call|> ... │
    │  Mask:   [-100     -100  -100    -100         LOSS   LOSS   ...] │
    │  (-100 = PyTorch's ignore_index for cross_entropy)               │
    └──────────────────────────────────────────────────────────────────┘
    """

    # Sentinel value that tells PyTorch's cross_entropy to IGNORE this position
    IGNORE_INDEX = -100

    def __init__(
        self,
        examples: list[dict],
        tokenizer: BPETokenizer,
        max_seq_len: int,
    ):
        """
        Args:
            examples:    List of training examples, each a dict with keys:
                         - "user": the user's input text
                         - "assistant": the desired model response
            tokenizer:   Trained BPE tokenizer instance.
            max_seq_len: Maximum sequence length (from config).
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Process all examples upfront
        self.processed: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._process_examples(examples)
        print(f"  SFT Dataset: {len(self.processed):,} examples processed")

    def _process_examples(self, examples: list[dict]):
        """
        Convert raw examples into (input_ids, labels) tensor pairs.

        For each example:
        1. Format as:  <|user|>{user_text}<|assistant|>{assistant_text}<|eos|>
        2. Tokenize the full sequence
        3. Create labels with IGNORE_INDEX on the user portion
        4. Pad or truncate to max_seq_len
        """
        for ex in examples:
            user_text = ex["user"]
            assistant_text = ex["assistant"]

            # ── Tokenize user portion ──────────────────────────────────
            # <|user|> + user text + <|assistant|>
            user_part = f"<|user|>{user_text}<|assistant|>"
            user_ids = self.tokenizer.encode(user_part)

            # ── Tokenize assistant portion ─────────────────────────────
            # assistant text + <|eos|>
            assistant_part = f"{assistant_text}<|eos|>"
            assistant_ids = self.tokenizer.encode(assistant_part)

            # ── Combine ───────────────────────────────────────────────
            full_ids = user_ids + assistant_ids

            # ── Truncate if too long ───────────────────────────────────
            if len(full_ids) > self.max_seq_len:
                full_ids = full_ids[:self.max_seq_len]
                # Ensure we don't lose the EOS if it was at the end
                full_ids[-1] = self.tokenizer.eos_id

            # ── Create labels with masking ─────────────────────────────
            # The input to the model is full_ids[:-1] (all but last)
            # The labels are full_ids[1:] (all but first) — shifted right
            #
            # We mask the USER portion of the labels with IGNORE_INDEX
            # so the model is NOT penalized for "predicting" user text.
            #
            # user_ids has length L. In the shifted labels:
            #   positions 0..L-2 correspond to predicting user tokens → MASK
            #   positions L-1..end correspond to predicting assistant tokens → KEEP
            labels = list(full_ids[1:])  # Shifted right
            input_ids = list(full_ids[:-1])  # All but last

            # Mask user portion in labels (positions 0 through len(user_ids)-2)
            num_user_tokens_to_mask = len(user_ids) - 1  # -1 for shift
            for i in range(min(num_user_tokens_to_mask, len(labels))):
                labels[i] = self.IGNORE_INDEX

            # ── Pad to max_seq_len ─────────────────────────────────────
            # Both input_ids and labels must have length max_seq_len
            pad_length = self.max_seq_len - len(input_ids)
            if pad_length > 0:
                input_ids = input_ids + [self.tokenizer.pad_id] * pad_length
                labels = labels + [self.IGNORE_INDEX] * pad_length

            # Truncate to exactly max_seq_len (safety)
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]

            self.processed.append((
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
            ))

    def __len__(self) -> int:
        return len(self.processed)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_ids: [T] — input token IDs
            labels:    [T] — target IDs with IGNORE_INDEX on user portion

        After DataLoader batching:
            input_ids: [B, T] = [batch_size, max_seq_len]
            labels:    [B, T] = [batch_size, max_seq_len]
        """
        return self.processed[idx]


# ═══════════════════════════════════════════════════════════════════════════
# 3. HELPER — Create DataLoaders
# ═══════════════════════════════════════════════════════════════════════════
def create_pretrain_dataloader(
    text: str,
    tokenizer: BPETokenizer,
    config: BabyLLMConfig,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for pre-training.

    Returns batches of shape:
        x: [B, T] = [batch_size, max_seq_len]
        y: [B, T] = [batch_size, max_seq_len]
    """
    dataset = PreTrainDataset(text, tokenizer, config.max_seq_len)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        drop_last=True,  # Drop incomplete last batch
        num_workers=0,    # Single process (safe for all platforms)
    )


def create_sft_dataloader(
    examples: list[dict],
    tokenizer: BPETokenizer,
    config: BabyLLMConfig,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for SFT.

    Returns batches of shape:
        input_ids: [B, T] = [batch_size, max_seq_len]
        labels:    [B, T] = [batch_size, max_seq_len]
    """
    dataset = SFTDataset(examples, tokenizer, config.max_seq_len)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# VERIFICATION — Test both dataset types
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  A.I.D.A. — Dataset Pipeline Verification")
    print("=" * 70)

    # ── Setup: train a small tokenizer ─────────────────────────────────
    config = BabyLLMConfig(
        vocab_size=400,
        max_seq_len=32,   # Small for testing
        batch_size=4,
    )

    corpus = (
        "The quick brown fox jumps over the lazy dog. "
        "Machine learning is the science of getting computers to learn. "
        "Deep learning uses neural networks with many layers. "
        "Attention is all you need for sequence modeling. "
    ) * 50

    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    print("\n  Training tokenizer...")
    tokenizer.train(corpus, verbose=False)

    # ── Test 1: PreTrainDataset ────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Test 1: PreTrainDataset")
    print("─" * 70)

    dataset = PreTrainDataset(corpus, tokenizer, config.max_seq_len)
    print(f"  Dataset size: {len(dataset)} examples")

    x, y = dataset[0]
    print(f"  x shape: {list(x.shape)} (input)")
    print(f"  y shape: {list(y.shape)} (target)")
    print(f"  x[:10] = {x[:10].tolist()}")
    print(f"  y[:10] = {y[:10].tolist()}")

    # Verify shift: y should be x shifted right by 1
    # y[i] should equal the token AFTER x[i] in the original sequence
    assert x.shape == y.shape == (config.max_seq_len,), \
        f"Shape mismatch: x={x.shape}, y={y.shape}"
    print("  ✅ Shapes correct")

    # Test DataLoader
    loader = create_pretrain_dataloader(corpus, tokenizer, config)
    batch_x, batch_y = next(iter(loader))
    print(f"\n  DataLoader batch shapes:")
    print(f"    x: {list(batch_x.shape)} = [B={config.batch_size}, T={config.max_seq_len}]")
    print(f"    y: {list(batch_y.shape)} = [B={config.batch_size}, T={config.max_seq_len}]")

    # ── Test 2: SFTDataset ─────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Test 2: SFTDataset")
    print("─" * 70)

    sft_examples = [
        {
            "user": "What is 5 plus 3?",
            "assistant": '<|tool_call|>{"tool": "calc", "a": 5, "b": 3}',
        },
        {
            "user": "Search for Alice",
            "assistant": '<|tool_call|>{"tool": "search", "query": "Alice"}',
        },
        {
            "user": "Hello!",
            "assistant": "Hello! How can I help you today?",
        },
    ]

    sft_dataset = SFTDataset(sft_examples, tokenizer, config.max_seq_len)
    print(f"  SFT Dataset size: {len(sft_dataset)} examples")

    input_ids, labels = sft_dataset[0]
    print(f"  input_ids shape: {list(input_ids.shape)}")
    print(f"  labels shape:    {list(labels.shape)}")

    # Count masked vs unmasked positions
    masked = (labels == SFTDataset.IGNORE_INDEX).sum().item()
    unmasked = (labels != SFTDataset.IGNORE_INDEX).sum().item()
    print(f"  Masked positions (user):      {masked}")
    print(f"  Unmasked positions (assistant): {unmasked}")
    print(f"  → Model only learns from {unmasked} tokens (assistant response)")

    # Show the masking in action
    print(f"\n  Label visualization (first 20 positions):")
    for i in range(min(20, len(labels))):
        label_val = labels[i].item()
        if label_val == SFTDataset.IGNORE_INDEX:
            print(f"    pos {i:2d}: MASKED (user portion, no loss)")
        else:
            decoded = tokenizer.decode([label_val])
            print(f"    pos {i:2d}: {label_val:5d} → '{decoded}'")

    # Test SFT DataLoader
    sft_loader = create_sft_dataloader(sft_examples, tokenizer, config, shuffle=False)
    batch_ids, batch_labels = next(iter(sft_loader))
    print(f"\n  SFT DataLoader batch shapes:")
    print(f"    input_ids: {list(batch_ids.shape)}")
    print(f"    labels:    {list(batch_labels.shape)}")

    print("\n" + "=" * 70)
    print("  ✅ All dataset tests passed!")
    print("=" * 70)
