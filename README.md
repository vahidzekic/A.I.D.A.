<div align="center">

# 🧠 A.I.D.A.

### **Artificially Intelligent Deterministic Agent**

*A complete Neuro-Symbolic Agentic AI framework bridging probabilistic Deep Learning*
*with deterministic Symbolic Logic — built entirely from scratch.*

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Pure-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Dependencies](https://img.shields.io/badge/External_LLM_Deps-Zero-gold?style=for-the-badge)](#)
[![Built](https://img.shields.io/badge/Built-From_Scratch-blueviolet?style=for-the-badge)](#)

---

**No HuggingFace. No LangChain. No Ollama. No LlamaIndex.**
**Every tensor shape annotated. Every algorithm explained. Every line intentional.**

[Architecture](#-architecture-deep-dive) •
[Innovations](#-key-technical-innovations) •
[Showcase](#-showcase--trace-logs) •
[Get Started](#-getting-started) •
[Roadmap](#-future-roadmap--the-brain-transplant)

</div>

---

## 📖 What is A.I.D.A.?

A.I.D.A. is a **Neuro-Symbolic Agentic AI framework** that combines two fundamentally different paradigms of intelligence into one unified system:

| | 🧠 Neural (The Brain) | ⚙️ Symbolic (The Hands) |
|---|---|---|
| **Nature** | Probabilistic, fuzzy, learned | Deterministic, exact, hard-coded |
| **Strength** | Language understanding, reasoning | Math, data lookup, code execution |
| **Weakness** | Hallucination, imprecise math | Zero language understanding |
| **In A.I.D.A.** | Custom Transformer LLM | Tool Registry + Safe Executor |

The **Agent** is the bridge — it lets the neural network's language understanding *drive* the symbolic engine's precise computation. Best of both worlds.

```
  "What is Alice's balance       ┌─────────────┐     ┌──────────────┐
   with 5% interest?"  ────────▶ │  🧠 Neural   │────▶│  ⚙️ Symbolic  │
                                 │  LLM Brain   │◀────│  Tool Engine │
                        ◀─────── │  (Decides)    │     │  (Executes)  │
  "Alice has $12,750.50.         └─────────────┘     └──────────────┘
   With 5% interest:
   $13,388.03."
```

---

## 🤔 Why Build From Scratch?

> *"What I cannot create, I do not understand."* — **Richard Feynman**

Frameworks like LangChain and HuggingFace Transformers are extraordinary tools — for *production*. But they are **black boxes** when it comes to *understanding*.

Building A.I.D.A. from scratch provides:

| Benefit | What It Means |
|---|---|
| 🔬 **Deep Understanding** | You understand *why* `Q·Kᵀ/√d_k` works, not just that `model.generate()` exists |
| 🛡️ **Security by Design** | No transitive dependency vulnerabilities. No supply-chain attacks. Every line is auditable |
| 🎛️ **Absolute Control** | Modify attention patterns, loss masking, tool schemas — no fighting framework abstractions |
| 📐 **Mathematical Rigor** | Tensor shapes annotated at every step: `[B, T, C]` → `[B, n_heads, T, head_dim]` |
| 🧩 **Modular Architecture** | Swap the Baby LLM for a 120B-parameter model without touching the symbolic engine |

This project proves that the **entire LLM-to-Agent pipeline** — from raw bytes to tool-calling AI — can be understood, built, and controlled by a single engineer.

---

## 🏗️ Architecture Deep-Dive

### The 5 Phases

```
A.I.D.A./
│
├── 📁 1_neural_core/               ← THE BRAIN (Phase 1 & 2)
│   ├── config.py                    BabyLLMConfig — Central hyperparameter registry
│   ├── tokenizer.py                 Byte-level BPE tokenizer (from scratch)
│   └── model.py                     Full Transformer: Attention → SwiGLU → Pre-Norm
│
├── 📁 2_training_loop/             ← THE EDUCATION (Phase 3)
│   ├── dataset.py                   PreTrainDataset + SFTDataset (with loss masking)
│   ├── pretrain.py                  Next-token prediction (cosine LR + AdamW)
│   └── sft_trainer.py              JSON tool-call fine-tuning
│
├── 📁 3_symbolic_engine/           ← THE HANDS & SHIELD (Phase 4)
│   ├── tools.py                     Tool ABC + Registry + Calculator (AST-safe!) + MockDB
│   └── parser.py                    Bulletproof 6-layer JSON parser with auto-repair
│
├── 📁 4_agent_orchestration/       ← THE SOUL (Phase 5)
│   └── react_loop.py               ReAct Agent: Thought → Action → Observation loop
│
├── .gitignore
└── README.md                        ← You are here
```

### How the Pieces Connect

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         A.I.D.A. SYSTEM ARCHITECTURE                    │
│                                                                         │
│  ┌───────────────────────────── NEURAL SIDE ─────────────────────────┐  │
│  │                                                                    │  │
│  │  ┌─────────────┐    ┌───────────────┐    ┌──────────────────┐     │  │
│  │  │  Tokenizer   │───▶│  Transformer   │───▶│  Training Loop   │     │  │
│  │  │  (BPE)       │    │  (Attention +  │    │  (PreTrain + SFT)│     │  │
│  │  │  Phase 1     │    │   SwiGLU)      │    │  Phase 3         │     │  │
│  │  │              │    │  Phase 2       │    │                  │     │  │
│  │  └─────────────┘    └───────┬───────┘    └──────────────────┘     │  │
│  │                             │                                      │  │
│  └─────────────────────────────┼──────────────────────────────────────┘  │
│                                │ generates text                          │
│                                ▼                                         │
│  ┌──────────────────── SYMBOLIC SIDE ────────────────────────────────┐  │
│  │                                                                    │  │
│  │  ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐    │  │
│  │  │  JSON Parser  │───▶│  Tool Registry │───▶│  Tool Executor   │    │  │
│  │  │  (6-layer     │    │  (Schema +     │    │  (Calculator,    │    │  │
│  │  │   auto-repair)│    │   Dispatch)    │    │   Database, ...) │    │  │
│  │  │  Phase 4      │    │  Phase 4       │    │  Phase 4         │    │  │
│  │  └──────────────┘    └───────────────┘    └──────────────────┘    │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌────────────────────── ORCHESTRATION ──────────────────────────────┐  │
│  │                                                                    │  │
│  │                    ⚡ ReAct Agent (Phase 5)                        │  │
│  │         Thought → Action → Observation → Response                  │  │
│  │         Multi-tool chaining · Self-correction · Graceful fallback │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase-by-Phase Breakdown

#### Phase 1 — Neural Core: Config & Tokenizer
- **`BabyLLMConfig`** — A `@dataclass` registry for all hyperparameters (vocab size, model dimension, attention heads, sequence length, etc.)
- **`BPETokenizer`** — Byte Pair Encoding built from first principles. Starts with 256 raw bytes, iteratively merges the most frequent pairs. Includes special tokens: `<|tool_call|>`, `<|tool_result|>`, `<|user|>`, `<|assistant|>`, `<|eos|>`

#### Phase 2 — Neural Core: Transformer Architecture
- **`CausalSelfAttention`** — Multi-head masked self-attention with the classic `Q·Kᵀ/√d_k` formula
- **`FeedForward`** — SwiGLU activation (used in LLaMA, Gemma, Mistral): `SwiGLU(x) = (xW₁ ⊙ Swish(xW_gate)) · W₂`
- **`TransformerBlock`** — Pre-LayerNorm architecture with residual connections
- **`NeuroSymbolicBabyLLM`** — Full GPT-style autoregressive model with weight-tied embeddings and `generate()` with top-k sampling

#### Phase 3 — Training Loop
- **Pre-training** — Next-token prediction on raw text. Cosine LR schedule with linear warmup. AdamW with LLaMA-style β₂=0.95
- **SFT (Supervised Fine-Tuning)** — Trains the model to emit `<|tool_call|>{"tool": "...", "args": {...}}` JSON. Uses **masked cross-entropy loss** — only assistant tokens contribute to the gradient; user tokens are masked with `ignore_index=-100`

#### Phase 4 — Symbolic Engine
- **`ToolRegistry`** — Register, discover, and dispatch tools. Generates JSON schemas dynamically for the system prompt
- **`CalculatorTool`** — Safe math evaluation via AST parsing (see [Security](#-ast-safe-execution) below)
- **`AccountBalanceTool`** — Mock database with deterministic lookups
- **`parse_llm_output()`** — The bulletproof parser (see [Parser](#-bulletproof-6-layer-output-parser) below)

#### Phase 5 — Agentic Orchestration
- **`ReActAgent`** — The grand orchestrator implementing the Reason+Act loop with conversation history, dynamic system prompt injection, and a 3-branch control flow

---

## 🔑 Key Technical Innovations

### 🛡️ AST-Safe Execution

The calculator tool **never uses `eval()`**. Instead, it parses mathematical expressions into an Abstract Syntax Tree and walks it node-by-node, allowing **only** arithmetic operations on literal numbers.

```python
# ❌ DANGEROUS — eval() executes arbitrary code
eval("__import__('os').system('rm -rf /')")  # 💀 Catastrophic!

# ✅ A.I.D.A. — AST-safe execution
#    Parses "2 + 3" into a tree, only allows: +, -, *, /, **, //, %
#    __import__, function calls, attribute access → REJECTED at AST level
```

```
A.I.D.A. Security Test Results:
  ✅ '2 + 3'                              → 5
  ✅ '10 * (3 + 2)'                       → 50
  ✅ '10 / 0'                             → Error: Division by zero
  ✅ '__import__("os").system("ls")'       → BLOCKED: "Unsafe expression element"
  ✅  Missing arguments                    → Error: "Missing required arguments"
```

### 🔧 Bulletproof 6-Layer Output Parser

LLMs — especially small ones — produce messy output. A.I.D.A.'s parser handles it all:

```
Layer 1: Detect <|tool_call|> special token
Layer 2: Extract JSON from markdown ```json ... ``` code blocks
Layer 3: Balanced-brace matching to find { ... } in garbage text
Layer 4: json.loads() on the extracted string
Layer 5: Auto-repair (trailing commas, single→double quotes, unclosed braces)
Layer 6: Structured error return for ReAct self-correction (NEVER crashes)
```

```
Parser Test Results:
  ✅ Perfect JSON                          → Parsed
  ✅ JSON buried in conversational text    → Extracted & parsed
  ✅ {'single': 'quotes', 'trailing': ,}  → REPAIRED & parsed
  ✅ Truncated JSON (unclosed braces)      → REPAIRED & parsed
  ✅ Markdown ```json ... ``` blocks       → Extracted & parsed
  ✅ Completely garbled nonsense           → Structured error (no crash!)
  🛡️ Script completed without crashing on ANY input
```

### 🔄 Agentic Self-Correction

When the parser detects malformed output, it doesn't crash — it feeds the error message **back into the LLM's conversation history** as a new observation. This gives the model a chance to correct itself:

```
Step 1: 🧠 LLM outputs:  {tool: calculator, args: {expression: 5+5}}   ← broken!
        ⚠️ Parser:        "Malformed JSON. Please use double quotes."
        🔄 Error injected back into history

Step 2: 🧠 LLM retries:  {"tool": "calculator", "args": {"expression": "5+5"}}  ← fixed!
        ✅ Parser:        Valid tool call detected
        ⚙️ Executor:      calculator("5+5") → 10

Step 3: 🧠 LLM responds:  "The answer is 10."
        💬 Final answer returned to user
```

---

## 🎬 Showcase — Trace Logs

### Multi-Tool Chaining: Balance Lookup + Interest Calculation

```
══════════════════════════════════════════════════════════════════════
  👤 USER: What is Alice's balance, and what would it be with 5% interest?
══════════════════════════════════════════════════════════════════════

  ┌── ReAct Step 1/5 ─────────────────────────────────────────────
  │ 🧠 THOUGHT: I need to look up Alice's balance first.
  │ ⚙️  ACTION:  Calling tool 'get_balance' with args: {"username": "alice"}
  │ 📊 OBSERVE:  ✅ {"status": "success", "result": {"username": "alice",
  │              "balance": 12750.50, "currency": "USD", "account_type": "savings"}}
  └── 🔄 Continuing loop (tool result received)

  ┌── ReAct Step 2/5 ─────────────────────────────────────────────
  │ 🧠 THOUGHT: Now I need to calculate 5% interest on $12,750.50.
  │ ⚙️  ACTION:  Calling tool 'calculator' with args: {"expression": "12750.50 * 1.05"}
  │ 📊 OBSERVE:  ✅ {"status": "success", "result": 13388.025}
  └── 🔄 Continuing loop (tool result received)

  ┌── ReAct Step 3/5 ─────────────────────────────────────────────
  │ 🧠 THOUGHT: I have both results. Let me formulate the answer.
  │ 💬 RESPOND:  Alice's current balance is $12,750.50. With 5% interest,
  │              it would grow to $13,388.03.
  └── ✅ Loop complete (text response)
```

### Graceful Degradation: Real Baby LLM Chaos Test

```
══════════════════════════════════════════════════════════════════════
  👤 USER: What is the balance for john?
══════════════════════════════════════════════════════════════════════

  ┌── ReAct Step 1/3 ─────────────────────────────────────────────
  │ 🧠 THOUGHT: [garbled bytes from untrained 0.14M parameter model]
  │ ⚠️  PARSE ERROR: Malformed JSON detected
  └── 🔄 Continuing loop (self-correction attempt)

  ┌── ReAct Step 2/3 ─────────────────────────────────────────────
  │ 🧠 THOUGHT: [more noise — model accidentally generates valid-ish JSON]
  │ ⚙️  ACTION:  calculator("25 * 4") → 100
  └── 🔄 Continuing loop (tool result received)

  ┌── ReAct Step 3/3 ─────────────────────────────────────────────
  │ 💬 RESPOND:  [noisy but graceful response]
  └── ✅ Loop complete — NO CRASH, max_steps handled safely
```

> The orchestrator **never crashes**, even when connected to a chaotic, untrained neural network. This is the power of the symbolic shield.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.12+**
- **PyTorch 2.0+** (CPU, CUDA, or MPS)
- No other dependencies required

### Installation

```bash
# Clone the repository
git clone https://github.com/vahidzekic/A.I.D.A..git
cd A.I.D.A.

# Create virtual environment
python3 -m venv aida-venv
source aida-venv/bin/activate  # On Windows: aida-venv\Scripts\activate

# Install PyTorch (CPU version — ~200MB)
pip install torch

# That's it. No other dependencies.
```

### Quick Start — Run the Agent

```bash
# Run the full ReAct loop verification (mocked + real LLM tests)
python 4_agent_orchestration/react_loop.py
```

### Train the Baby LLM Yourself

```bash
# Phase 1: Verify the tokenizer
python 1_neural_core/tokenizer.py

# Phase 2: Verify the model architecture
python 1_neural_core/model.py

# Phase 3: Run pre-training (next-token prediction)
python 2_training_loop/pretrain.py

# Phase 3: Run SFT (tool-call fine-tuning)
python 2_training_loop/sft_trainer.py

# Phase 4: Verify tools & parser independently
python 3_symbolic_engine/tools.py
python 3_symbolic_engine/parser.py
```

Every file is self-contained with an `if __name__ == '__main__'` verification block. Run any file independently to see its component in action.

---

## 🗺️ Future Roadmap — The "Brain Transplant"

The A.I.D.A. architecture is **intentionally decoupled**. The symbolic engine (tools, parser, registry) knows *nothing* about the neural core. The ReAct agent communicates with the LLM through a single, clean interface: **text in → text out**.

This means the 0.14M parameter "Baby LLM" can be **hot-swapped** for any model — without changing a single line in the symbolic engine:

```python
# Current: Custom Baby LLM (built from scratch, 0.14M params)
agent = ReActAgent(model=baby_llm, tokenizer=bpe_tokenizer, registry=registry)

# Future: Swap in a production-grade local model
# agent = ReActAgent(model=Llama3_70B, tokenizer=llama_tokenizer, registry=registry)
# agent = ReActAgent(model=Nemotron_Super, tokenizer=nemo_tokenizer, registry=registry)
# agent = ReActAgent(model=Qwen3_72B, tokenizer=qwen_tokenizer, registry=registry)
```

**The vision:** Enterprise-grade intelligence powered by massive local models (fully on-premise, zero API calls), channeled through A.I.D.A.'s battle-tested symbolic engine for reliable, auditable, and secure tool execution.

| Milestone | Status |
|---|---|
| Custom Transformer Architecture | ✅ Complete |
| Byte-level BPE Tokenizer | ✅ Complete |
| Pre-training Loop (Next-Token Prediction) | ✅ Complete |
| SFT Loop (JSON Tool-Call Fine-Tuning) | ✅ Complete |
| AST-Safe Tool Execution Engine | ✅ Complete |
| 6-Layer Bulletproof JSON Parser | ✅ Complete |
| ReAct Agentic Loop with Self-Correction | ✅ Complete |
| Multi-Tool Chaining | ✅ Complete |
| Integration with Local LLMs (Ollama/vLLM) | 🔜 Planned |
| RAG (Retrieval-Augmented Generation) | 🔜 Planned |
| Persistent Memory (Vector Store) | 🔜 Planned |
| Web UI (Interactive Chat Interface) | 🔜 Planned |
| CUDA-Optimized Training Pipeline | 🔜 Planned |

---

## 🧮 By the Numbers

```
Lines of Code:        ~3,500 (hand-written, zero boilerplate)
External LLM Deps:    0
Tensor Shape Comments: Every single operation annotated
Training Verified:     Loss 5.98 → 2.62 (pre-train), 5.94 → 4.66 (SFT)
Parser Tests:          12/12 passed (including malformed, truncated, garbled)
Tool Tests:            All passed (including code injection blocked)
ReAct Tests:           4/4 passed (single-tool, multi-tool, error recovery, chaos)
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with 🧠 and ⚡ by [Vahid Žekić](https://github.com/vahidzekic)**

*"The best way to understand a system is to build it from nothing."*

</div>