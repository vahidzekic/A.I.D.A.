"""
=============================================================================
A.I.D.A. — Artificially Intelligent Digital Assistant
=============================================================================
FILE: tokenizer.py
PHASE: 1 (Foundation — prerequisite for Training Loop)
PURPOSE: Byte Pair Encoding (BPE) tokenizer, built entirely from scratch.

WHAT IS A TOKENIZER?
    A tokenizer converts raw text ←→ integer IDs that the model understands.
    
    "Hello world" → [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
                     (character-level, as a starting point)
    
    After BPE training, frequent pairs get merged into single tokens:
    "Hello world" → [15234, 1937]    (much more efficient!)

WHAT IS BPE (Byte Pair Encoding)?
    BPE is the tokenization algorithm used by GPT-2/3/4, LLaMA, Mistral, etc.
    
    The algorithm:
    1. Start with individual bytes/characters as the base vocabulary.
    2. Count every adjacent pair of tokens in the corpus.
    3. Find the MOST FREQUENT pair → merge them into a new token.
    4. Repeat steps 2-3 until we reach the desired vocabulary size.
    
    Example training on "aaabdaaabac":
        Base vocab: {a, b, c, d}
        Step 1: Most frequent pair = (a, a) → merge → new token "aa"
                Corpus: "aa|a|b|d|aa|a|b|a|c"
        Step 2: Most frequent pair = (aa, a) → merge → new token "aaa"
                Corpus: "aaa|b|d|aaa|b|a|c"  
        Step 3: Most frequent pair = (aaa, b) → merge → new token "aaab"
                Corpus: "aaab|d|aaab|a|c"
        ...and so on.
    
    The beauty of BPE:
    - Common words become single tokens (efficient)
    - Rare/unknown words decompose into subword pieces (no OOV!)
    - Byte-level BPE can handle ANY text, ANY language, even binary

DESIGN DECISIONS:
    - We use BYTE-LEVEL BPE: start from raw UTF-8 bytes (0-255).
      This means our base vocabulary is always 256, and we can
      encode literally any text without "unknown token" issues.
    - Special tokens are reserved at the start of the vocabulary
      for agentic control flow (tool calls, EOS, padding, etc.)
    - The tokenizer is fully self-contained — no external libraries.

=============================================================================
"""

import json
import os
import re
from pathlib import Path
from collections import Counter


# ═══════════════════════════════════════════════════════════════════════════
# SPECIAL TOKENS — Reserved for agentic control flow
# ═══════════════════════════════════════════════════════════════════════════
# These tokens have SPECIAL MEANING in our agentic framework.
# They are inserted into the vocabulary at fixed positions so the model
# can learn to emit them during SFT (Supervised Fine-Tuning).
SPECIAL_TOKENS = {
    "<|pad|>":          0,   # Padding token (fills unused positions)
    "<|unk|>":          1,   # Unknown token (safety fallback)
    "<|bos|>":          2,   # Beginning of sequence
    "<|eos|>":          3,   # End of sequence — model should STOP here
    "<|tool_call|>":    4,   # Signals: "I want to call a tool"
    "<|tool_result|>":  5,   # Signals: "Here is the tool's output"
    "<|user|>":         6,   # Start of user message
    "<|assistant|>":    7,   # Start of assistant response
    "<|system|>":       8,   # Start of system prompt
}

# Number of reserved special token slots (we reserve extra for future use)
NUM_SPECIAL_TOKENS = 16  # IDs 0-15 are reserved


class BPETokenizer:
    """
    Byte-level Byte Pair Encoding tokenizer — built from scratch.
    
    Vocabulary Layout:
    ┌────────────────────────────────────────────────────────────┐
    │  IDs 0-15:     Special tokens (<|pad|>, <|eos|>, etc.)    │
    │  IDs 16-271:   Raw byte tokens (0x00 through 0xFF)        │
    │  IDs 272+:     BPE merge tokens (learned from corpus)     │
    │  ...up to vocab_size                                       │
    └────────────────────────────────────────────────────────────┘
    
    The total vocab_size = NUM_SPECIAL_TOKENS + 256 + num_merges.
    For our Baby LLM config (vocab_size=8192):
        num_merges = 8192 - 16 - 256 = 7920 merge operations.
    """
    
    def __init__(self, vocab_size: int = 8192):
        """
        Initialize the tokenizer.
        
        Args:
            vocab_size: Total vocabulary size including specials + bytes + merges.
                        Must be >= NUM_SPECIAL_TOKENS + 256 = 272.
        """
        assert vocab_size >= NUM_SPECIAL_TOKENS + 256, (
            f"vocab_size ({vocab_size}) must be >= {NUM_SPECIAL_TOKENS + 256} "
            f"(specials + byte tokens)"
        )
        
        self.vocab_size = vocab_size
        
        # Number of BPE merges to learn = remaining slots after specials + bytes
        self.num_merges = vocab_size - NUM_SPECIAL_TOKENS - 256
        
        # ── Special Tokens ─────────────────────────────────────────────
        # Maps: special string → integer ID
        self.special_tokens = dict(SPECIAL_TOKENS)
        # Inverse: integer ID → special string
        self.inverse_special = {v: k for k, v in self.special_tokens.items()}
        
        # ── Merge Rules ────────────────────────────────────────────────
        # Learned during training. Each merge is: (token_a, token_b) → new_id
        # Stored as an ordered list — ORDER MATTERS during encoding!
        # Earlier merges are applied first (they represent more frequent pairs).
        self.merges: dict[tuple[int, int], int] = {}  # (a, b) → merged_id
        
        # ── Vocabulary ─────────────────────────────────────────────────
        # Maps: integer ID → byte sequence (for decoding)
        # Initialized with raw byte tokens at positions 16-271.
        self.vocab: dict[int, bytes] = {}
        self._build_base_vocab()
    
    def _build_base_vocab(self):
        """
        Build the base vocabulary: 256 raw byte tokens at IDs 16-271.
        
        Each byte value (0-255) maps to a single-byte bytes object.
        For example:
            ID 16  → b'\\x00'  (null byte)
            ID 113 → b'a'      (ASCII 'a' = byte 97, id = 97 + 16 = 113)
            ID 271 → b'\\xff'  (byte 255)
        """
        for byte_val in range(256):
            token_id = byte_val + NUM_SPECIAL_TOKENS
            self.vocab[token_id] = bytes([byte_val])
    
    def _byte_to_id(self, byte_val: int) -> int:
        """Convert a raw byte value (0-255) to its token ID."""
        return byte_val + NUM_SPECIAL_TOKENS
    
    def _text_to_byte_ids(self, text: str) -> list[int]:
        """
        Convert text to a list of byte-level token IDs.
        
        "Hi" → UTF-8 bytes [72, 105] → token IDs [88, 121]
        
        This is the STARTING POINT before any BPE merges are applied.
        """
        raw_bytes = text.encode("utf-8")  # str → bytes
        return [self._byte_to_id(b) for b in raw_bytes]
    
    # ═══════════════════════════════════════════════════════════════════
    # TRAINING — Learn BPE merge rules from a text corpus
    # ═══════════════════════════════════════════════════════════════════
    
    def _count_pairs(self, token_ids: list[int]) -> Counter:
        """
        Count all adjacent pairs of token IDs.
        
        Example:
            [1, 2, 3, 2, 3] → Counter({(2, 3): 2, (1, 2): 1, (3, 2): 1})
        
        The most frequent pair will be the next merge candidate.
        """
        pairs = Counter()
        for i in range(len(token_ids) - 1):
            pair = (token_ids[i], token_ids[i + 1])
            pairs[pair] += 1
        return pairs
    
    def _merge_pair(
        self, 
        token_ids: list[int], 
        pair: tuple[int, int], 
        new_id: int
    ) -> list[int]:
        """
        Replace all occurrences of `pair` in `token_ids` with `new_id`.
        
        Example:
            token_ids = [1, 2, 3, 1, 2, 4]
            pair = (1, 2)
            new_id = 99
            result = [99, 3, 99, 4]
        
        This is the core merge operation of BPE.
        We scan left-to-right, greedily merging whenever we find the pair.
        """
        merged = []
        i = 0
        while i < len(token_ids):
            # Check if current position matches the pair
            if (i < len(token_ids) - 1 
                and token_ids[i] == pair[0] 
                and token_ids[i + 1] == pair[1]):
                merged.append(new_id)
                i += 2  # Skip both tokens (they've been merged)
            else:
                merged.append(token_ids[i])
                i += 1
        return merged
    
    def train(self, text: str, verbose: bool = True):
        """
        Train the BPE tokenizer on a text corpus.
        
        THE BPE TRAINING ALGORITHM:
        ┌──────────────────────────────────────────────────────────┐
        │  1. Convert entire corpus to byte-level token IDs        │
        │  2. Repeat `num_merges` times:                           │
        │     a. Count all adjacent token pairs                    │
        │     b. Find the most frequent pair                       │
        │     c. Create a new token ID for this pair               │
        │     d. Replace ALL occurrences of the pair in the corpus │
        │     e. Store the merge rule: (a, b) → new_id             │
        │  3. Save the ordered merge list for use during encoding  │
        └──────────────────────────────────────────────────────────┘
        
        Args:
            text:    The training corpus (a single large string).
            verbose: If True, print progress every 500 merges.
        """
        if verbose:
            print(f"  Training BPE tokenizer...")
            print(f"  Corpus size: {len(text):,} characters")
            print(f"  Target merges: {self.num_merges:,}")
        
        # Step 1: Convert text → byte-level IDs
        token_ids = self._text_to_byte_ids(text)
        initial_len = len(token_ids)
        
        if verbose:
            print(f"  Initial token count: {initial_len:,} (byte-level)")
        
        # Step 2: Iteratively merge the most frequent pairs
        next_id = NUM_SPECIAL_TOKENS + 256  # First merge ID = 272
        
        for merge_idx in range(self.num_merges):
            # 2a. Count all adjacent pairs
            pair_counts = self._count_pairs(token_ids)
            
            if not pair_counts:
                # No more pairs to merge (corpus is fully compressed)
                if verbose:
                    print(f"  ⚠ Stopped early at merge {merge_idx}: no pairs left")
                break
            
            # 2b. Find the most frequent pair
            best_pair = pair_counts.most_common(1)[0][0]
            best_count = pair_counts[best_pair]
            
            if best_count < 2:
                # Don't merge pairs that only appear once — not worth it
                if verbose:
                    print(f"  ⚠ Stopped early at merge {merge_idx}: "
                          f"no pair appears more than once")
                break
            
            # 2c. Assign a new token ID
            new_id = next_id
            next_id += 1
            
            # 2d. Merge all occurrences in the corpus
            token_ids = self._merge_pair(token_ids, best_pair, new_id)
            
            # 2e. Store the merge rule
            self.merges[best_pair] = new_id
            
            # Also store the concatenated bytes in vocab for decoding
            # The new token's bytes = bytes_of(a) + bytes_of(b)
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            # Progress logging
            if verbose and (merge_idx + 1) % 500 == 0:
                compression = initial_len / len(token_ids)
                print(f"  Merge {merge_idx + 1:>5d}/{self.num_merges}: "
                      f"pair={best_pair} count={best_count:>6,} → "
                      f"id={new_id}  "
                      f"tokens: {len(token_ids):,} "
                      f"(compression: {compression:.2f}x)")
        
        if verbose:
            final_len = len(token_ids)
            compression = initial_len / final_len if final_len > 0 else float('inf')
            print(f"\n  ✅ Training complete!")
            print(f"  Final token count: {final_len:,}")
            print(f"  Compression ratio: {compression:.2f}x")
            print(f"  Learned merges: {len(self.merges):,}")
            print(f"  Effective vocab: {NUM_SPECIAL_TOKENS + 256 + len(self.merges):,}")
    
    # ═══════════════════════════════════════════════════════════════════
    # ENCODING — Text → Token IDs
    # ═══════════════════════════════════════════════════════════════════
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:
        """
        Encode a text string into a list of BPE token IDs.
        
        ENCODING ALGORITHM:
        ┌──────────────────────────────────────────────────────────┐
        │  1. Convert text → byte-level token IDs                  │
        │  2. Apply merge rules IN ORDER (earliest merges first):  │
        │     For each merge rule (a, b) → c:                      │
        │       Scan the sequence and replace all (a, b) with c    │
        │  3. Optionally prepend <|bos|> and append <|eos|>        │
        └──────────────────────────────────────────────────────────┘
        
        WHY IN ORDER?
            BPE merges are hierarchical. Merge #1 creates tokens used by
            merge #2, which creates tokens used by merge #3, etc.
            If we applied them out of order, we'd get wrong tokenizations.
        
        Args:
            text: The input string to encode.
            add_special_tokens: If True, wrap with <|bos|> ... <|eos|>.
        
        Returns:
            List of integer token IDs.
        """
        # Handle special token strings embedded in the text
        # Split text around special tokens, encode normal parts, insert specials
        parts = self._split_on_special_tokens(text)
        
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_tokens[part_text])
            else:
                # Encode normal text through BPE
                ids = self._encode_chunk(part_text)
                all_ids.extend(ids)
        
        # Optionally wrap with BOS/EOS
        if add_special_tokens:
            all_ids = [self.special_tokens["<|bos|>"]] + all_ids + \
                      [self.special_tokens["<|eos|>"]]
        
        return all_ids
    
    def _encode_chunk(self, text: str) -> list[int]:
        """Encode a single chunk of normal text (no special tokens)."""
        if not text:
            return []
        
        # Start with byte-level IDs
        token_ids = self._text_to_byte_ids(text)
        
        # Apply merges in order (earliest = most frequent = applied first)
        for pair, new_id in self.merges.items():
            token_ids = self._merge_pair(token_ids, pair, new_id)
            if len(token_ids) < 2:
                break  # Nothing left to merge
        
        return token_ids
    
    def _split_on_special_tokens(self, text: str) -> list[tuple[str, bool]]:
        """
        Split text into segments, separating special token strings.
        
        Example:
            "<|user|>Hello<|eos|>"
            → [("<|user|>", True), ("Hello", False), ("<|eos|>", True)]
        
        Returns:
            List of (text, is_special) tuples.
        """
        if not self.special_tokens:
            return [(text, False)]
        
        # Build regex pattern that matches any special token
        # Escape special regex characters in token strings
        pattern = "(" + "|".join(
            re.escape(token) for token in sorted(
                self.special_tokens.keys(), key=len, reverse=True
            )
        ) + ")"
        
        parts = re.split(pattern, text)
        result = []
        for part in parts:
            if not part:
                continue
            is_special = part in self.special_tokens
            result.append((part, is_special))
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════
    # DECODING — Token IDs → Text
    # ═══════════════════════════════════════════════════════════════════
    
    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a list of token IDs back to a text string.
        
        DECODING ALGORITHM:
        ┌──────────────────────────────────────────────────────────┐
        │  For each token ID:                                      │
        │    - If it's a special token → look up its string        │
        │    - If it's a byte/merge token → look up its bytes      │
        │  Concatenate all bytes and decode from UTF-8.            │
        └──────────────────────────────────────────────────────────┘
        
        Args:
            token_ids: List of integer token IDs.
        
        Returns:
            Decoded text string.
        """
        raw_bytes = bytearray()
        
        for token_id in token_ids:
            if token_id in self.inverse_special:
                # Special token — insert as UTF-8 encoded string
                special_str = self.inverse_special[token_id]
                raw_bytes.extend(special_str.encode("utf-8"))
            elif token_id in self.vocab:
                # Normal token — append its raw bytes
                raw_bytes.extend(self.vocab[token_id])
            else:
                # Unknown ID — shouldn't happen, but safety fallback
                raw_bytes.extend(b"<?>")
        
        # Decode bytes → string, replacing invalid UTF-8 sequences
        return raw_bytes.decode("utf-8", errors="replace")
    
    # ═══════════════════════════════════════════════════════════════════
    # SAVE / LOAD — Persist tokenizer to disk
    # ═══════════════════════════════════════════════════════════════════
    
    def save(self, path: str):
        """
        Save the trained tokenizer to a JSON file.
        
        We store:
            - vocab_size
            - merges as a list of [pair_a, pair_b, new_id] triples
            - special_tokens mapping
        """
        save_dir = os.path.dirname(path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Convert merge dict to serializable list
        # Order is preserved (Python 3.7+ dicts are ordered)
        merge_list = [
            [pair[0], pair[1], new_id]
            for pair, new_id in self.merges.items()
        ]
        
        data = {
            "vocab_size": self.vocab_size,
            "num_special_tokens": NUM_SPECIAL_TOKENS,
            "special_tokens": self.special_tokens,
            "merges": merge_list,
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"  💾 Tokenizer saved to: {path}")
        print(f"     ({len(self.merges):,} merges, "
              f"vocab={NUM_SPECIAL_TOKENS + 256 + len(self.merges):,})")
    
    def load(self, path: str):
        """
        Load a trained tokenizer from a JSON file.
        
        This restores the merge rules so encode/decode work identically
        to when the tokenizer was trained.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.vocab_size = data["vocab_size"]
        self.special_tokens = data["special_tokens"]
        self.inverse_special = {v: k for k, v in self.special_tokens.items()}
        
        # Rebuild base vocab
        self.vocab = {}
        self._build_base_vocab()
        
        # Rebuild merges and vocab from the saved merge list
        self.merges = {}
        for pair_a, pair_b, new_id in data["merges"]:
            pair = (pair_a, pair_b)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair_a] + self.vocab[pair_b]
        
        print(f"  📂 Tokenizer loaded from: {path}")
        print(f"     ({len(self.merges):,} merges, "
              f"vocab={NUM_SPECIAL_TOKENS + 256 + len(self.merges):,})")
    
    # ═══════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════════
    
    @property
    def eos_id(self) -> int:
        return self.special_tokens["<|eos|>"]
    
    @property
    def bos_id(self) -> int:
        return self.special_tokens["<|bos|>"]
    
    @property
    def pad_id(self) -> int:
        return self.special_tokens["<|pad|>"]
    
    @property
    def effective_vocab_size(self) -> int:
        """Actual number of tokens (may be less than vocab_size if training stopped early)."""
        return NUM_SPECIAL_TOKENS + 256 + len(self.merges)


# ═══════════════════════════════════════════════════════════════════════════
# VERIFICATION — Train on a small corpus and test encode/decode
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  A.I.D.A. — BPE Tokenizer Verification")
    print("=" * 70)
    
    # A small but representative training corpus
    corpus = """
    The quick brown fox jumps over the lazy dog. The quick brown fox is very quick.
    A.I.D.A. is an Artificially Intelligent Digital Assistant built from scratch.
    The model uses Multi-Head Causal Self-Attention to understand context.
    Each Transformer block contains attention and feed-forward sub-layers.
    The tokenizer converts text to token IDs using Byte Pair Encoding.
    Machine learning is the science of getting computers to learn from data.
    Deep learning uses neural networks with many layers to find patterns.
    The attention mechanism allows the model to focus on relevant parts.
    Training involves minimizing the cross-entropy loss function.
    The optimizer adjusts weights using gradients computed via backpropagation.
    """ * 20  # Repeat to make pairs more frequent
    
    # 1. Initialize with a small vocab for testing
    test_vocab_size = 400  # 16 special + 256 bytes + 128 merges
    tokenizer = BPETokenizer(vocab_size=test_vocab_size)
    
    # 2. Train
    print()
    tokenizer.train(corpus, verbose=True)
    
    # 3. Test encoding
    print("\n" + "─" * 70)
    print("  Encoding Tests:")
    print("─" * 70)
    
    test_texts = [
        "Hello world!",
        "The quick brown fox",
        "attention mechanism",
        "<|user|>What is 2+2?<|eos|>",
    ]
    
    for text in test_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        print(f"\n  Original:  '{text}'")
        print(f"  Token IDs: {ids}")
        print(f"  Decoded:   '{decoded}'")
        print(f"  Tokens:    {len(ids)}")
        assert decoded.replace("<|user|>", "").replace("<|eos|>", "") == \
               text.replace("<|user|>", "").replace("<|eos|>", "") or \
               decoded == text, \
               f"Round-trip FAILED: '{text}' → '{decoded}'"
    
    # 4. Test encode with special tokens
    print("\n" + "─" * 70)
    print("  Special Token Test:")
    print("─" * 70)
    ids_with_special = tokenizer.encode("Hello", add_special_tokens=True)
    decoded_with_special = tokenizer.decode(ids_with_special)
    print(f"  encode('Hello', add_special=True) = {ids_with_special}")
    print(f"  decode → '{decoded_with_special}'")
    assert ids_with_special[0] == tokenizer.bos_id
    assert ids_with_special[-1] == tokenizer.eos_id
    
    # 5. Test save/load round-trip
    print("\n" + "─" * 70)
    print("  Save/Load Test:")
    print("─" * 70)
    test_path = "/tmp/aida_test_tokenizer.json"
    tokenizer.save(test_path)
    
    tokenizer2 = BPETokenizer(vocab_size=test_vocab_size)
    tokenizer2.load(test_path)
    
    # Verify that loaded tokenizer produces identical results
    for text in test_texts:
        ids1 = tokenizer.encode(text)
        ids2 = tokenizer2.encode(text)
        assert ids1 == ids2, f"Save/load mismatch for '{text}'"
    print("  ✅ Save/load round-trip: PASSED")
    
    # 6. Compression statistics
    print("\n" + "─" * 70)
    print("  Compression Analysis:")
    print("─" * 70)
    sample = "The attention mechanism allows the model to focus on relevant parts."
    byte_ids = tokenizer._text_to_byte_ids(sample)
    bpe_ids = tokenizer.encode(sample)
    print(f"  Text: '{sample}'")
    print(f"  Characters:  {len(sample)}")
    print(f"  Byte tokens: {len(byte_ids)}")
    print(f"  BPE tokens:  {len(bpe_ids)}")
    print(f"  Compression: {len(byte_ids) / len(bpe_ids):.2f}x")
    
    # Cleanup
    os.remove(test_path)
    
    print("\n" + "=" * 70)
    print("  ✅ All tokenizer tests passed!")
    print("=" * 70)
