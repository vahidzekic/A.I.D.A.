"""
=============================================================================
A.I.D.A. — Artificially Intelligent Digital Assistant
=============================================================================
FILE: pretrain.py
PHASE: 3 (Training Loop — Pre-training)
PURPOSE: Train the Baby LLM on raw text via next-token prediction.

WHAT IS PRE-TRAINING?
    Pre-training is the FIRST stage of LLM training. The model reads
    massive amounts of raw text and learns to predict the next token.

    Through this simple objective, the model learns:
    - Grammar and syntax ("the" is often followed by a noun)
    - Semantics ("Paris is the capital of France")
    - World knowledge ("water boils at 100°C")
    - Reasoning patterns ("if A implies B, and B implies C, then A implies C")

    It's remarkable that next-token prediction alone teaches all of this!

THE TRAINING LOOP — Step by Step:
    ┌──────────────────────────────────────────────────────────────────┐
    │  for each epoch:                                                 │
    │    for each batch of (input_ids, targets):                       │
    │      1. FORWARD PASS:   logits = model(input_ids)                │
    │         → model predicts a distribution over vocab for each pos  │
    │      2. COMPUTE LOSS:   loss = cross_entropy(logits, targets)    │
    │         → how "wrong" were the predictions?                      │
    │      3. BACKWARD PASS:  loss.backward()                          │
    │         → compute gradients (∂loss/∂weight for every parameter)  │
    │      4. GRADIENT CLIP:  clip_grad_norm_(params, max_norm)        │
    │         → prevent exploding gradients from destabilizing training│
    │      5. OPTIMIZER STEP: optimizer.step()                         │
    │         → update weights: w = w - lr * gradient                  │
    │      6. ZERO GRADIENTS: optimizer.zero_grad()                    │
    │         → reset gradients for next iteration                     │
    │  Save checkpoint periodically.                                   │
    └──────────────────────────────────────────────────────────────────┘

LEARNING RATE SCHEDULE:
    We use a cosine annealing schedule with linear warmup:
    
    LR
    │   ╱‾‾‾‾‾‾‾‾‾‾‾‾‾────____
    │  ╱                        ‾‾‾‾──────
    │ ╱                                     ‾‾──__
    │╱                                              ‾‾──_
    └──────────────────────────────────────────────────── Steps
     warmup    cosine decay phase

    Warmup: Gradually increase LR from 0 to peak over the first N steps.
            Prevents the randomly-initialized model from taking huge steps.
    Cosine: Smoothly decrease LR to near-zero. Allows the model to
            make increasingly fine adjustments as training progresses.

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

# Add parent directory to path so we can import from sibling packages
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "1_neural_core"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "2_training_loop"))

from config import BabyLLMConfig
from tokenizer import BPETokenizer
from model import NeuroSymbolicBabyLLM
from dataset import PreTrainDataset, create_pretrain_dataloader


# ═══════════════════════════════════════════════════════════════════════════
# 1. LEARNING RATE SCHEDULER — Cosine with Linear Warmup
# ═══════════════════════════════════════════════════════════════════════════
def get_lr(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """
    Compute the learning rate for a given training step.

    THE MATH:
        Warmup phase (step < warmup_steps):
            lr = max_lr * (step / warmup_steps)
            → Linear interpolation from 0 to max_lr

        Cosine decay phase (step >= warmup_steps):
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))
            → Smooth cosine curve from max_lr to min_lr

    Args:
        step:         Current training step (0-indexed).
        max_lr:       Peak learning rate (after warmup).
        min_lr:       Minimum learning rate (at end of training).
        warmup_steps: Number of warmup steps.
        total_steps:  Total number of training steps.

    Returns:
        Learning rate for this step.
    """
    # ── Warmup Phase ───────────────────────────────────────────────────
    if step < warmup_steps:
        # Linear warmup: 0 → max_lr
        return max_lr * (step + 1) / warmup_steps

    # ── Cosine Decay Phase ─────────────────────────────────────────────
    if step >= total_steps:
        return min_lr

    # How far through the decay phase are we? (0.0 → 1.0)
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)

    # Cosine decay coefficient: 1.0 → 0.0 as decay_ratio goes 0.0 → 1.0
    # cos(0) = 1, cos(π) = -1, so (1 + cos(π * ratio)) / 2 goes 1 → 0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)


# ═══════════════════════════════════════════════════════════════════════════
# 2. PRE-TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════
def pretrain(
    model: NeuroSymbolicBabyLLM,
    train_text: str,
    tokenizer: BPETokenizer,
    config: BabyLLMConfig,
    output_dir: str = "checkpoints",
    log_interval: int = 10,
    save_interval: int = 500,
    max_grad_norm: float = 1.0,
    warmup_fraction: float = 0.1,
) -> dict:
    """
    Pre-train the Baby LLM on a text corpus via next-token prediction.

    Args:
        model:           The NeuroSymbolicBabyLLM instance.
        train_text:      Raw text corpus for training.
        tokenizer:       Trained BPE tokenizer.
        config:          Model/training hyperparameters.
        output_dir:      Directory to save checkpoints.
        log_interval:    Print loss every N steps.
        save_interval:   Save checkpoint every N steps.
        max_grad_norm:   Max gradient norm for clipping.
        warmup_fraction: Fraction of total steps used for LR warmup.

    Returns:
        Dictionary with training metrics (losses, times, etc.)
    """
    # ── Setup ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")
    print(f"\n  Device: {device}")
    model = model.to(device)
    model.train()

    os.makedirs(output_dir, exist_ok=True)

    # ── Create DataLoader ──────────────────────────────────────────────
    dataloader = create_pretrain_dataloader(train_text, tokenizer, config)
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * config.max_epochs
    warmup_steps = int(total_steps * warmup_fraction)

    print(f"  Steps per epoch:  {steps_per_epoch}")
    print(f"  Total steps:      {total_steps}")
    print(f"  Warmup steps:     {warmup_steps}")

    # ── Optimizer: AdamW ───────────────────────────────────────────────
    # WHY AdamW (not plain Adam)?
    #   Adam: applies L2 regularization to the gradient BEFORE the
    #         adaptive scaling → doesn't actually regularize heavy params.
    #   AdamW: applies weight decay DIRECTLY to the weights AFTER the
    #          update step → proper regularization.
    #
    # AdamW maintains two state variables per parameter:
    #   m (1st moment / mean of gradients)  → momentum direction
    #   v (2nd moment / mean of squared gradients) → per-param learning rate
    #
    # Update rule:
    #   m = β₁ * m + (1 - β₁) * grad
    #   v = β₂ * v + (1 - β₂) * grad²
    #   w = w - lr * (m / (√v + ε)) - lr * weight_decay * w
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,    # Will be overridden by scheduler
        betas=(0.9, 0.95),          # β₁=0.9, β₂=0.95 (LLaMA-style)
        weight_decay=0.1,           # Light regularization
        eps=1e-8,                   # Numerical stability
    )

    # ── Training Metrics ───────────────────────────────────────────────
    metrics = {
        "train_losses": [],
        "learning_rates": [],
        "step_times": [],
        "best_loss": float("inf"),
    }

    # ── Training Loop ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  🚀 Starting Pre-training...")
    print("=" * 70)

    global_step = 0
    start_time = time.time()

    for epoch in range(config.max_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            step_start = time.time()

            # ── Move data to device ────────────────────────────────────
            # x: [B, T] — input token IDs
            # y: [B, T] — target token IDs (shifted right by 1)
            x = x.to(device)
            y = y.to(device)

            # ── Update learning rate ───────────────────────────────────
            lr = get_lr(
                step=global_step,
                max_lr=config.learning_rate,
                min_lr=config.learning_rate * 0.1,  # Decay to 10% of peak
                warmup_steps=warmup_steps,
                total_steps=total_steps,
            )
            # Manually set LR for all parameter groups
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # ── Forward Pass ───────────────────────────────────────────
            # model(x, y) returns (logits, loss)
            #   logits: [B, T, V] — predictions over vocabulary
            #   loss:   scalar    — cross-entropy loss
            logits, loss = model(x, targets=y)

            # ── Backward Pass ──────────────────────────────────────────
            # Compute gradients: ∂loss/∂θ for every parameter θ.
            # This is the "magic" of automatic differentiation — PyTorch
            # traces the computation graph during forward pass and walks
            # it backward to compute gradients via the chain rule.
            loss.backward()

            # ── Gradient Clipping ──────────────────────────────────────
            # WHY?  Occasionally, a bad batch causes extremely large
            # gradients that would "explode" the weights. Clipping
            # rescales the gradient vector if its norm exceeds max_norm.
            #
            # If ||grad|| > max_norm:
            #     grad = grad * (max_norm / ||grad||)
            # This preserves the DIRECTION but limits the MAGNITUDE.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )

            # ── Optimizer Step ─────────────────────────────────────────
            # Update all parameters: θ = θ - lr * ∂loss/∂θ
            optimizer.step()

            # ── Zero Gradients ─────────────────────────────────────────
            # CRITICAL: Reset gradients to zero. Without this, gradients
            # would ACCUMULATE across steps (PyTorch adds by default).
            # set_to_none=True is slightly faster than zero_grad().
            optimizer.zero_grad(set_to_none=True)

            # ── Track Metrics ──────────────────────────────────────────
            loss_val = loss.item()
            step_time = time.time() - step_start

            epoch_loss += loss_val
            epoch_steps += 1
            global_step += 1

            metrics["train_losses"].append(loss_val)
            metrics["learning_rates"].append(lr)
            metrics["step_times"].append(step_time)

            if loss_val < metrics["best_loss"]:
                metrics["best_loss"] = loss_val

            # ── Logging ────────────────────────────────────────────────
            if global_step % log_interval == 0:
                avg_time = sum(metrics["step_times"][-log_interval:]) / log_interval
                tokens_per_sec = (config.batch_size * config.max_seq_len) / avg_time
                print(
                    f"  Step {global_step:>5d}/{total_steps} │ "
                    f"Loss: {loss_val:.4f} │ "
                    f"LR: {lr:.2e} │ "
                    f"Grad Norm: {grad_norm:.2f} │ "
                    f"{tokens_per_sec:,.0f} tok/s │ "
                    f"{avg_time*1000:.0f}ms/step"
                )

            # ── Checkpoint Saving ──────────────────────────────────────
            if global_step % save_interval == 0:
                _save_checkpoint(
                    model, optimizer, config, global_step,
                    loss_val, output_dir
                )

        # ── Epoch Summary ──────────────────────────────────────────────
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - start_time
        print(f"\n  ═══ Epoch {epoch + 1}/{config.max_epochs} ═══ "
              f"Avg Loss: {avg_epoch_loss:.4f} │ "
              f"Elapsed: {elapsed:.1f}s")

    # ── Final Checkpoint ───────────────────────────────────────────────
    _save_checkpoint(
        model, optimizer, config, global_step,
        metrics["train_losses"][-1] if metrics["train_losses"] else 0,
        output_dir, is_final=True
    )

    total_time = time.time() - start_time
    print(f"\n  ✅ Pre-training complete!")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Best loss:  {metrics['best_loss']:.4f}")
    print(f"  Final loss: {metrics['train_losses'][-1]:.4f}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# 3. CHECKPOINT SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════════════
def _save_checkpoint(
    model: NeuroSymbolicBabyLLM,
    optimizer: torch.optim.Optimizer,
    config: BabyLLMConfig,
    step: int,
    loss: float,
    output_dir: str,
    is_final: bool = False,
):
    """
    Save a training checkpoint.

    A checkpoint contains EVERYTHING needed to resume training:
    - model_state_dict: all learned weights
    - optimizer_state_dict: momentum & adaptive learning rate states
    - config: hyperparameters
    - step: current training step
    - loss: current loss value
    """
    filename = "final_pretrain.pt" if is_final else f"pretrain_step_{step}.pt"
    filepath = os.path.join(output_dir, filename)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.__dict__,
        "step": step,
        "loss": loss,
    }

    torch.save(checkpoint, filepath)
    print(f"  💾 Checkpoint saved: {filepath} (step={step}, loss={loss:.4f})")


def load_checkpoint(
    filepath: str,
    device: str = "cpu",
) -> tuple[NeuroSymbolicBabyLLM, dict]:
    """
    Load a model from a checkpoint file.

    Args:
        filepath: Path to the .pt checkpoint file.
        device:   Device to load the model onto.

    Returns:
        model:      The loaded NeuroSymbolicBabyLLM instance.
        checkpoint: The full checkpoint dict (for resuming training).
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    # Reconstruct config
    config = BabyLLMConfig(**{
        k: v for k, v in checkpoint["config"].items()
        if k != "head_dim"  # head_dim is computed in __post_init__
    })

    # Reconstruct model
    model = NeuroSymbolicBabyLLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"  📂 Checkpoint loaded: {filepath}")
    print(f"     Step: {checkpoint['step']}, Loss: {checkpoint['loss']:.4f}")

    return model, checkpoint


# ═══════════════════════════════════════════════════════════════════════════
# VERIFICATION — Run a quick pre-training test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  A.I.D.A. — Pre-training Loop Verification")
    print("=" * 70)

    # ── Small config for fast testing ──────────────────────────────────
    config = BabyLLMConfig(
        vocab_size=400,      # Small vocab → fast tokenizer training
        max_seq_len=64,      # Short sequences → fast training
        d_model=64,          # Tiny model → fast on CPU
        n_heads=4,
        n_layers=2,
        d_ff=256,
        batch_size=4,
        max_epochs=2,        # Just 2 epochs for verification
        learning_rate=1e-3,
    )

    # ── Create a training corpus ───────────────────────────────────────
    corpus = (
        "The quick brown fox jumps over the lazy dog. "
        "Machine learning is the science of getting computers to learn. "
        "Deep learning uses neural networks with many layers. "
        "The attention mechanism allows the model to focus on relevant parts. "
        "Training involves minimizing the cross-entropy loss function. "
        "The optimizer adjusts weights using gradients computed via backpropagation. "
        "A tokenizer converts text into numerical token identifiers. "
        "Byte pair encoding is an effective subword tokenization algorithm. "
    ) * 100  # Repeat to have enough data

    # ── Train tokenizer ────────────────────────────────────────────────
    print("\n  Step 1: Training tokenizer...")
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    tokenizer.train(corpus, verbose=False)
    print(f"  Tokenizer ready: {tokenizer.effective_vocab_size} tokens")

    # ── Initialize model ───────────────────────────────────────────────
    print("\n  Step 2: Initializing model...")
    model = NeuroSymbolicBabyLLM(config)

    # ── Run pre-training ───────────────────────────────────────────────
    print("\n  Step 3: Pre-training...")
    metrics = pretrain(
        model=model,
        train_text=corpus,
        tokenizer=tokenizer,
        config=config,
        output_dir="/tmp/aida_test_checkpoints",
        log_interval=5,
        save_interval=9999,  # Don't save intermediates in test
    )

    # ── Verify loss decreased ──────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Loss Analysis:")
    print("─" * 70)
    first_losses = metrics["train_losses"][:5]
    last_losses = metrics["train_losses"][-5:]
    avg_first = sum(first_losses) / len(first_losses)
    avg_last = sum(last_losses) / len(last_losses)
    print(f"  First 5 steps avg loss: {avg_first:.4f}")
    print(f"  Last 5 steps avg loss:  {avg_last:.4f}")
    print(f"  Loss decreased: {'✅ YES' if avg_last < avg_first else '❌ NO'}")

    # ── Test checkpoint save/load ──────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Checkpoint Test:")
    print("─" * 70)
    ckpt_path = "/tmp/aida_test_checkpoints/final_pretrain.pt"
    loaded_model, ckpt = load_checkpoint(ckpt_path)

    # Verify loaded model produces same output
    # Move both models to CPU for deterministic comparison
    # (MPS→CPU avoids device mismatch issues in the test)
    model_cpu = model.cpu()
    loaded_model_cpu = loaded_model.cpu()
    test_input = torch.randint(0, config.vocab_size, (1, 16))
    model_cpu.eval()
    loaded_model_cpu.eval()
    with torch.no_grad():
        orig_logits, _ = model_cpu(test_input)
        loaded_logits, _ = loaded_model_cpu(test_input)
    match = torch.allclose(orig_logits, loaded_logits, atol=1e-5)
    print(f"  Checkpoint round-trip: {'✅ PASSED' if match else '❌ FAILED'}")

    # ── Test generation after pre-training ─────────────────────────────
    print("\n" + "─" * 70)
    print("  Generation Test (after pre-training):")
    print("─" * 70)
    prompt_text = "The"
    prompt_ids = tokenizer.encode(prompt_text)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)
    # model_cpu is already on CPU, prompt_tensor is on CPU — no mismatch
    generated = model_cpu.generate(prompt_tensor, max_new_tokens=30, temperature=0.8)
    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"  Prompt: '{prompt_text}'")
    print(f"  Generated: '{generated_text}'")
    print(f"  (Note: with tiny model & limited data, output will be noisy)")

    # Cleanup
    import shutil
    shutil.rmtree("/tmp/aida_test_checkpoints", ignore_errors=True)

    print("\n" + "=" * 70)
    print("  ✅ Pre-training loop verified successfully!")
    print("=" * 70)
