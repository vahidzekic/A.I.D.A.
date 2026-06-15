"""
=============================================================================
A.I.D.A. — Artificially Intelligent Digital Assistant
=============================================================================
FILE: sft_trainer.py
PHASE: 3 (Training Loop — Supervised Fine-Tuning)
PURPOSE: Fine-tune the pre-trained LLM to follow instructions and
         emit strict JSON tool calls.

WHAT IS SFT (Supervised Fine-Tuning)?
    After pre-training, the model is a "next-token prediction machine."
    It can complete text, but it doesn't know HOW to be an assistant.
    
    SFT takes the pre-trained model and trains it on STRUCTURED examples:
    
    Before SFT:  "What is 5+3?"  → "Well, let me think about math..."
    After SFT:   "What is 5+3?"  → '{"tool": "calculator", "args": ...}'
    
    The model learns:
    1. To RECOGNIZE when a tool is needed (vs. a direct text response)
    2. To produce VALID JSON in the exact schema we define
    3. To STOP after <|eos|> (crucial for the agentic loop)

HOW IS SFT DIFFERENT FROM PRE-TRAINING?
    ┌───────────────────┬────────────────────┬───────────────────────┐
    │                   │   PRE-TRAINING      │   SFT                 │
    ├───────────────────┼────────────────────┼───────────────────────┤
    │ Data              │ Raw text (books,   │ Curated (user, asst)  │
    │                   │ web, articles)      │ pairs with JSON tools │
    │ Loss              │ All token positions │ ONLY assistant tokens │
    │                   │                    │ (user portion masked) │
    │ Objective         │ Learn language     │ Learn to follow       │
    │                   │                    │ instructions & format │
    │ Learning Rate     │ Higher (3e-4)      │ Lower (1e-5 to 5e-5) │
    │ Duration          │ Long (many epochs) │ Short (1-3 epochs)    │
    │ Data Size         │ Huge (GB-TB)       │ Small (MB)            │
    └───────────────────┴────────────────────┴───────────────────────┘
    
    KEY INSIGHT: SFT uses a MUCH lower learning rate because we don't
    want to "forget" the language knowledge from pre-training. We're
    just gently nudging the model's behavior toward the desired format.
    This is called "catastrophic forgetting" prevention.

=============================================================================
"""

import os
import sys
import math
import time
import json
import torch
import torch.nn as nn
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "1_neural_core"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "2_training_loop"))

from config import BabyLLMConfig
from tokenizer import BPETokenizer
from model import NeuroSymbolicBabyLLM
from dataset import SFTDataset, create_sft_dataloader
from pretrain import get_lr, load_checkpoint


# ═══════════════════════════════════════════════════════════════════════════
# 1. SFT TRAINING DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════
def generate_sft_examples() -> list[dict]:
    """
    Generate synthetic SFT training examples.
    
    In a real project, you'd curate these by hand or use a larger model
    to generate them. For our Baby LLM, we create a diverse set covering:
    
    1. Calculator tool calls (math questions → JSON)
    2. Database search tool calls (lookup questions → JSON)
    3. Direct text responses (greetings, simple questions)
    
    Each example is a dict with:
        "user":      The user's input text
        "assistant": The desired model response (JSON or plain text)
    
    THE CRITICAL PATTERN:
        For tool calls, the assistant response MUST follow this format:
        <|tool_call|>{"tool": "tool_name", "args": {...}}
        
        This teaches the model to emit the <|tool_call|> special token
        BEFORE the JSON, which the agentic loop (Phase 5) will detect
        to know "I should parse JSON and execute a tool."
    """
    examples = []
    
    # ── Calculator Tool Calls ──────────────────────────────────────────
    # Teach the model: math question → calculator JSON
    calc_templates = [
        # Addition
        ("What is {a} plus {b}?", "add"),
        ("Calculate {a} + {b}", "add"),
        ("Add {a} and {b}", "add"),
        ("How much is {a} plus {b}?", "add"),
        ("Sum of {a} and {b}?", "add"),
        # Subtraction
        ("What is {a} minus {b}?", "subtract"),
        ("Calculate {a} - {b}", "subtract"),
        ("Subtract {b} from {a}", "subtract"),
        ("{a} minus {b} equals what?", "subtract"),
        # Multiplication
        ("What is {a} times {b}?", "multiply"),
        ("Calculate {a} * {b}", "multiply"),
        ("Multiply {a} by {b}", "multiply"),
        ("{a} multiplied by {b}?", "multiply"),
    ]
    
    import random
    random.seed(42)  # Reproducibility
    
    for template, op in calc_templates:
        # Generate multiple examples with different numbers
        for _ in range(8):
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            user_text = template.format(a=a, b=b)
            tool_json = json.dumps({
                "tool": "calculator",
                "args": {"a": a, "b": b, "op": op}
            })
            examples.append({
                "user": user_text,
                "assistant": f"<|tool_call|>{tool_json}",
            })
    
    # ── Database Search Tool Calls ─────────────────────────────────────
    # Teach the model: lookup question → search JSON
    search_templates = [
        "Find information about {name}",
        "Search for {name}",
        "Look up {name} in the database",
        "Who is {name}?",
        "Tell me about {name}",
        "What do we know about {name}?",
    ]
    
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank",
             "Grace", "Hank", "Ivy", "Jack"]
    
    for template in search_templates:
        for name in names:
            user_text = template.format(name=name)
            tool_json = json.dumps({
                "tool": "search_db",
                "args": {"query": name}
            })
            examples.append({
                "user": user_text,
                "assistant": f"<|tool_call|>{tool_json}",
            })
    
    # ── Direct Text Responses ──────────────────────────────────────────
    # Teach the model: NOT everything needs a tool call!
    # Simple greetings and conversational responses should be plain text.
    direct_responses = [
        ("Hello!", "Hello! How can I help you today?"),
        ("Hi there!", "Hi! What can I do for you?"),
        ("Good morning!", "Good morning! How may I assist you?"),
        ("Thank you!", "You're welcome! Is there anything else?"),
        ("Goodbye!", "Goodbye! Have a great day!"),
        ("What can you do?", 
         "I can help with math calculations and database searches."),
        ("Who are you?", 
         "I am A.I.D.A., an Artificially Intelligent Digital Assistant."),
        ("How are you?", 
         "I'm doing well, thank you for asking! How can I help?"),
    ]
    
    for user_text, assistant_text in direct_responses:
        # Repeat each direct response a few times to balance with tool calls
        for _ in range(10):
            examples.append({
                "user": user_text,
                "assistant": assistant_text,
            })
    
    random.shuffle(examples)
    return examples


# ═══════════════════════════════════════════════════════════════════════════
# 2. SFT TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════
def sft_train(
    model: NeuroSymbolicBabyLLM,
    examples: list[dict],
    tokenizer: BPETokenizer,
    config: BabyLLMConfig,
    output_dir: str = "checkpoints",
    sft_lr: float = 5e-5,
    sft_epochs: int = 3,
    log_interval: int = 10,
    max_grad_norm: float = 1.0,
    warmup_fraction: float = 0.05,
) -> dict:
    """
    Supervised Fine-Tuning loop.
    
    This is structurally similar to pretrain(), but with key differences:
    
    1. LOWER learning rate (sft_lr << pretrain_lr)
       → Preserve pre-trained knowledge
    
    2. MASKED loss (only on assistant tokens)
       → Model learns to RESPOND, not to predict user input
    
    3. FEWER epochs
       → SFT data is small; too many epochs → overfitting
    
    4. ignore_index=-100 in cross_entropy
       → PyTorch skips positions where label == -100

    Args:
        model:           Pre-trained model to fine-tune.
        examples:        List of {"user": ..., "assistant": ...} dicts.
        tokenizer:       Trained BPE tokenizer.
        config:          Hyperparameters (max_seq_len, batch_size used).
        output_dir:      Directory to save SFT checkpoints.
        sft_lr:          Peak learning rate for SFT (much lower than pretrain).
        sft_epochs:      Number of SFT epochs.
        log_interval:    Print loss every N steps.
        max_grad_norm:   Max gradient norm for clipping.
        warmup_fraction: Fraction of steps for LR warmup.
    
    Returns:
        Dictionary with training metrics.
    """
    # ── Setup ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")
    print(f"\n  Device: {device}")
    model = model.to(device)
    model.train()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ── Create SFT DataLoader ──────────────────────────────────────────
    dataloader = create_sft_dataloader(examples, tokenizer, config)
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * sft_epochs
    warmup_steps = int(total_steps * warmup_fraction)
    
    print(f"  SFT Examples:     {len(examples)}")
    print(f"  Steps per epoch:  {steps_per_epoch}")
    print(f"  Total steps:      {total_steps}")
    print(f"  SFT Learning rate: {sft_lr:.2e}")
    
    # ── Optimizer ──────────────────────────────────────────────────────
    # Use the same AdamW but with LOWER learning rate for SFT.
    # This prevents "catastrophic forgetting" of pre-trained knowledge.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=sft_lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,  # Less weight decay for SFT
        eps=1e-8,
    )
    
    # ── Training Metrics ───────────────────────────────────────────────
    metrics = {
        "sft_losses": [],
        "learning_rates": [],
        "best_loss": float("inf"),
    }
    
    # ── Training Loop ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  🎯 Starting Supervised Fine-Tuning...")
    print("=" * 70)
    
    global_step = 0
    start_time = time.time()
    
    for epoch in range(sft_epochs):
        epoch_loss = 0.0
        epoch_tokens = 0  # Count of NON-MASKED tokens (actual training signal)
        epoch_steps = 0
        
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            # ── Move to device ─────────────────────────────────────────
            # input_ids: [B, T] — input token IDs
            # labels:    [B, T] — target IDs with -100 on user positions
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # ── Update learning rate ───────────────────────────────────
            lr = get_lr(
                step=global_step,
                max_lr=sft_lr,
                min_lr=sft_lr * 0.1,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            
            # ── Forward Pass ───────────────────────────────────────────
            # The model returns logits [B, T, V] and we compute loss
            # manually here because we need ignore_index=-100 support.
            logits, _ = model(input_ids)
            # logits: [B, T, V] — raw vocabulary predictions
            
            # ── Compute Masked Loss ────────────────────────────────────
            # Cross-entropy with ignore_index=-100:
            #   - Positions where label == -100 are EXCLUDED from loss
            #   - Only assistant tokens contribute to the gradient
            #
            # Reshape for cross_entropy:
            #   logits: [B, T, V] → [B*T, V]
            #   labels: [B, T]    → [B*T]
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),  # [B*T, V]
                labels.view(-1),                    # [B*T]
                ignore_index=SFTDataset.IGNORE_INDEX,  # -100
            )
            
            # Count how many tokens actually contributed to loss
            active_tokens = (labels != SFTDataset.IGNORE_INDEX).sum().item()
            
            # ── Backward + Optimize ────────────────────────────────────
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # ── Track Metrics ──────────────────────────────────────────
            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_tokens += active_tokens
            epoch_steps += 1
            global_step += 1
            
            metrics["sft_losses"].append(loss_val)
            metrics["learning_rates"].append(lr)
            
            if loss_val < metrics["best_loss"]:
                metrics["best_loss"] = loss_val
            
            # ── Logging ────────────────────────────────────────────────
            if global_step % log_interval == 0:
                print(
                    f"  Step {global_step:>4d}/{total_steps} │ "
                    f"Loss: {loss_val:.4f} │ "
                    f"LR: {lr:.2e} │ "
                    f"Grad: {grad_norm:.2f} │ "
                    f"Active tokens: {active_tokens}"
                )
        
        # ── Epoch Summary ──────────────────────────────────────────────
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - start_time
        print(f"\n  ═══ SFT Epoch {epoch + 1}/{sft_epochs} ═══ "
              f"Avg Loss: {avg_epoch_loss:.4f} │ "
              f"Active Tokens: {epoch_tokens:,} │ "
              f"Elapsed: {elapsed:.1f}s")
    
    # ── Save Final SFT Checkpoint ──────────────────────────────────────
    sft_path = os.path.join(output_dir, "final_sft.pt")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "sft_metrics": metrics,
    }
    torch.save(checkpoint, sft_path)
    print(f"\n  💾 SFT checkpoint saved: {sft_path}")
    
    total_time = time.time() - start_time
    print(f"\n  ✅ SFT complete!")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Best loss:  {metrics['best_loss']:.4f}")
    print(f"  Final loss: {metrics['sft_losses'][-1]:.4f}")
    
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# VERIFICATION — Run a quick SFT test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  A.I.D.A. — SFT Trainer Verification")
    print("=" * 70)
    
    # ── Small config for fast testing ──────────────────────────────────
    config = BabyLLMConfig(
        vocab_size=400,
        max_seq_len=128,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        batch_size=8,
        learning_rate=1e-3,
    )
    
    # ── Train tokenizer on diverse text (needs tool-call vocab) ────────
    print("\n  Step 1: Training tokenizer...")
    tokenizer_corpus = (
        "Hello! How can I help you today? "
        "What is 15 plus 27? Calculate 5 + 3. Add 10 and 20. "
        "Search for Alice. Find information about Bob. "
        '{"tool": "calculator", "args": {"a": 15, "b": 27, "op": "add"}} '
        '{"tool": "search_db", "args": {"query": "Alice"}} '
        "The quick brown fox jumps over the lazy dog. "
        "I am A.I.D.A., an Artificially Intelligent Digital Assistant. "
        "Good morning! Goodbye! Thank you! "
        '<|tool_call|><|tool_result|><|user|><|assistant|><|eos|> '
    ) * 100
    
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    tokenizer.train(tokenizer_corpus, verbose=False)
    print(f"  Tokenizer ready: {tokenizer.effective_vocab_size} tokens")
    
    # ── Initialize model (simulating "pre-trained") ────────────────────
    print("\n  Step 2: Initializing model (simulating pre-trained)...")
    model = NeuroSymbolicBabyLLM(config)
    
    # ── Generate SFT examples ──────────────────────────────────────────
    print("\n  Step 3: Generating SFT training data...")
    sft_examples = generate_sft_examples()
    print(f"  Generated {len(sft_examples)} SFT examples")
    
    # Show a few examples
    print("\n  Sample SFT examples:")
    for ex in sft_examples[:3]:
        print(f"    User:      {ex['user']}")
        print(f"    Assistant: {ex['assistant'][:80]}...")
        print()
    
    # ── Run SFT ────────────────────────────────────────────────────────
    print("  Step 4: Running SFT...")
    metrics = sft_train(
        model=model,
        examples=sft_examples,
        tokenizer=tokenizer,
        config=config,
        output_dir="/tmp/aida_test_sft",
        sft_lr=5e-4,  # Higher for testing (small model)
        sft_epochs=2,
        log_interval=5,
    )
    
    # ── Verify loss decreased ──────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  SFT Loss Analysis:")
    print("─" * 70)
    losses = metrics["sft_losses"]
    first_5 = losses[:5]
    last_5 = losses[-5:]
    avg_first = sum(first_5) / len(first_5)
    avg_last = sum(last_5) / len(last_5)
    print(f"  First 5 steps avg loss: {avg_first:.4f}")
    print(f"  Last 5 steps avg loss:  {avg_last:.4f}")
    print(f"  Loss decreased: {'✅ YES' if avg_last < avg_first else '❌ NO'}")
    
    # ── Test inference after SFT ───────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Post-SFT Generation Test:")
    print("─" * 70)
    # Move model to CPU for generation test (avoids MPS device mismatch
    # when prompt tensors are created on CPU).
    model = model.cpu()
    model.eval()
    
    test_prompts = [
        "<|user|>What is 10 plus 5?<|assistant|>",
        "<|user|>Hello!<|assistant|>",
        "<|user|>Search for Alice<|assistant|>",
    ]
    
    for prompt in test_prompts:
        prompt_ids = tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)
        
        with torch.no_grad():
            generated = model.generate(
                prompt_tensor,
                max_new_tokens=40,
                temperature=0.7,
                top_k=50,
            )
        
        full_text = tokenizer.decode(generated[0].tolist())
        # Extract only the generated part (after the prompt)
        response = full_text[len(prompt):]
        print(f"  Prompt:   {prompt}")
        print(f"  Response: {response[:100]}")
        print()
    
    # Cleanup
    import shutil
    shutil.rmtree("/tmp/aida_test_sft", ignore_errors=True)
    
    print("  (Note: with tiny model & synthetic data, outputs will be noisy.")
    print("   A properly pre-trained model with real data produces much better results.)")
    
    print("\n" + "=" * 70)
    print("  ✅ SFT trainer verified successfully!")
    print("=" * 70)
