# 🗺️ ROADMAP — SLM from Scratch

A phase-by-phase **learn-by-yourself curriculum** for building a GPT-2-style Small Language Model (title → scientific abstract) from scratch in pure PyTorch, on a single **RTX 3060 (12 GB)**.

Each phase has two parts:
- **📚 Study** — the theory and the "why," with curated resources, taught *before* writing code.
- **🔨 Implement** — building the component, then validating it works.

> **Two iterations of the whole project:**
> 1. **Make it work** — learn each phase and get the SLM training and generating.
> 2. **Make it clean** — refine code and documentation so other programmers can learn from it.

**Anchor resource:** Andrej Karpathy — [*Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html), especially *"Let's build GPT from scratch"* and *"Let's build the GPT Tokenizer."*

---

## Phase 0 — Foundations & Setup
**📚 Study:** the language-model mental model (next-token prediction); PyTorch tensors & autograd (`.backward()`); what CUDA/VRAM actually do.
**🔨 Implement:** create the venv, install PyTorch with CUDA, verify the GPU is visible, write `requirements.txt`.
**✅ Done when:** `torch.cuda.is_available()` is `True` and the environment is reproducible.

## Phase 1 — Tokenization
**📚 Study:** why text must become numbers; characters vs. words vs. subwords; Byte-Pair Encoding (BPE) theory.
**🔨 Implement:** build a BPE tokenizer from scratch (train merges, encode, decode).
**✅ Done when:** the tokenizer round-trips text (`decode(encode(x)) == x`) and reports a sensible vocab.

## Phase 2 — Data Pipeline
**📚 Study:** datasets vs. dataloaders; train/validation split; batching and why we shuffle.
**🔨 Implement:** fetch + clean arXiv title↔abstract pairs (narrow field); tokenize; build `Dataset` / `DataLoader`.
**✅ Done when:** a batch of (title, abstract) tensors loads with correct shapes.

## Phase 3 — Transformer Architecture
**📚 Study:** token & positional embeddings; **self-attention** (Q/K/V); multi-head attention; residual connections; layer norm; the feed-forward block.
**🔨 Implement:** assemble the GPT decoder in PyTorch, module by module.
**✅ Done when:** a forward pass on a batch returns logits of the expected shape.

## Phase 4 — Training Loop
**📚 Study:** cross-entropy loss for next-token prediction; the AdamW optimizer; learning-rate scheduling; the backprop step.
**🔨 Implement:** write the training loop; overfit a tiny batch first as a sanity check.
**✅ Done when:** loss goes down on a small sample (the model can memorize).

## Phase 5 — VRAM Engineering
**📚 Study:** what consumes VRAM (params, activations, optimizer states); **mixed precision (AMP)**; **gradient accumulation**; gradient checkpointing.
**🔨 Implement:** apply AMP + gradient accumulation; tune batch size and sequence length to fit 12 GB.
**✅ Done when:** a real training run is stable with no CUDA OOM.

## Phase 6 — Evaluation
**📚 Study:** **perplexity** (fluency); **ROUGE** (content overlap); sampling strategies — temperature, top-k, top-p.
**🔨 Implement:** compute validation perplexity; generate sample abstracts; inspect quality.
**✅ Done when:** we can measure the model and read coherent generated abstracts.

## Phase 7 — Inference & Showcase
**📚 Study:** packaging a model for use; designing a clear demo.
**🔨 Implement:** a `generate.py` script (title in → abstract out); polished README with examples.
**✅ Done when:** anyone can run a title through the model and get an abstract.

---

## Progress Tracker

| Phase | Iter 1 (work) | Iter 2 (clean) |
|-------|:---:|:---:|
| 0 — Foundations & Setup | ✅ | ⬜ |
| 1 — Tokenization | ✅ | 🔄 |
| 2 — Data Pipeline | ⬜ | ⬜ |
| 3 — Transformer Architecture | ⬜ | ⬜ |
| 4 — Training Loop | ⬜ | ⬜ |
| 5 — VRAM Engineering | ⬜ | ⬜ |
| 6 — Evaluation | ⬜ | ⬜ |
| 7 — Inference & Showcase | ⬜ | ⬜ |

⬜ not started · 🔄 in progress · ✅ done

> **📍 You are here:** Phase 1 complete — byte-level BPE tokenizer built, reviewed, and refactored into a `Tokenizer` class (`src/tokenizer.py`), plus an interactive study Codex (`docs/index.html`). **Next: Phase 2 — Data Pipeline** (start with the *Study* column, then arXiv title↔abstract pairs → tokenize → `Dataset`/`DataLoader`).
