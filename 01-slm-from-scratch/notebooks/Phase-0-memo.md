# 🧠 Phase 0 Memo — Foundations & Setup

> Quick-reference study sheet for Pillar 1 (SLM from Scratch).
> Covers the **theory** of Phase 0. Read top-to-bottom in ~5 minutes to reload the mental model.

---

## 🎥 Re-watch video (the foundation, from scratch)

**Andrej Karpathy — "The spelled-out intro to neural networks and backpropagation: building micrograd"**
🔗 https://www.youtube.com/watch?v=VMj-3S1tku0

Why this one: it builds **gradients, the chain rule, backprop, and autograd from scratch** for tiny networks — which is exactly Pillars 1 & 2 below. After watching, `loss.backward()` will feel obvious, not magical.

---

## Pillar 1 — The mental model of a language model

- An LM has **one job**: given the text so far, **predict the next token**. It's a *next-token prediction machine*.
- The model's output is **not** a single word — it's a **probability distribution over the whole vocabulary**:
  - a vector of shape `(vocab_size,)`,
  - **position** = *which* word (fixed index→word mapping),
  - **value** = *how likely* that word is next,
  - the values **sum to 1**.
- Raw model outputs are called **logits** (they don't sum to 1). **Softmax** turns logits → a clean probability distribution. *(Meet it for real in Phase 3.)*
- **Generation = autoregressive loop:** predict next token → **pick** one → **append** it to the text → feed the longer text back in → repeat.
  - **Picking** is a real decision: always take the most likely word, or sample a less-likely one? → **sampling** (temperature, top-k, top-p), *Phase 6*.
  - **Stopping:** the model learns a special **end-of-sequence token**; generation halts when it predicts that. *(Designed in Phase 1.)*

## How a model learns (the core loop)

A fresh model is **random** → produces garbage. Training improves it:

1. **Forward** — run the model → prediction → compute the **loss**.
2. **Loss** — a single number measuring *how wrong* the prediction was vs. the correct answer. High = bad, low = good. (Next-token version = **cross-entropy loss**, *Phase 4*.) Training = **drive the loss down**.
3. **Backward** — compute **gradients** (see Pillar 2).
4. **Update** — nudge each weight a small step **downhill**.
5. Repeat.

This 4-step loop is **the whole game**; everything else is detail bolted on.

---

## Pillar 2 — Tensors, gradients & autograd

- **The model = a big bag of numbers.** Parameters = **weights** (+ **biases**) = `float32` numbers organized into **matrices/vectors**. "124M parameters" = 124M numbers. Training = *find better values for them*.
- **Tensor** = a NumPy array with **two superpowers**: (1) it can live on the **GPU**, (2) it can **track its own gradient**.
- **Gradient** = for one weight, *"how much does the loss change per tiny nudge of this weight?"* → a **slope (derivative)**.
  - **Direction**: move the weight the way that **lowers** the loss.
  - **Magnitude**: how strongly this weight affects the loss → take a step **proportional** to it (big influence → big step). Near the minimum the slope flattens → steps shrink → settle gently.
  - **Learning rate** = a global "caution dial" that scales **every** step down so we don't overshoot. *(Phase 4.)*
- **The full gradient** = one number per weight, same shape as the weights → "nudge me this much, this way."

### Why `loss.backward()` is cheap (the chain rule)

- **Brute force** (nudge each weight, re-measure loss) would cost ~**124M+1 model runs per step** → infeasible.
- Every basic op has a **known local derivative** (multiply `q=a*b` → rate `b`; square `L=q²` → rate `2q`; add → rate `1`).
- **Chain rule:** multiply local derivatives **along the path** from a weight to the loss → its gradient. **One backward sweep** fills in *all* gradients (cost ∝ number of **operations**, not operations × weights), because the upstream gradient is computed once and **reused** at every branch.
- **Autograd:** PyTorch records every op into a **computation graph** during the forward pass; **`loss.backward()`** walks it in reverse and fills in every gradient. ⚠️ It's `.backward()` — **no "s".**

**Worked example (proves math = experiment):**
```
x = 2 (input, fixed), w = 3 (weight)
q = w * x = 6
L = q²    = 36          (pretend loss)

dL/dq = 2q = 12
dq/dw = x  = 2
dL/dw = 12 * 2 = 24     ← chain rule

Check by nudging w → 3.001:
q = 6.002, L = 36.024004, slope ≈ 0.024004 / 0.001 ≈ 24  ✓
```

### What "from scratch" means here
We **build by hand**: the BPE tokenizer, the transformer architecture (embeddings, attention, layer norm, FFN), the training loop, sampling, VRAM techniques.
We **stand on PyTorch** for: tensor math, **autograd (`.backward()`)**, and CUDA kernels. We do *not* reimplement autodiff. *(Curious? Watch micrograd above.)*

---

## Pillar 3 — CUDA & VRAM (the hardware constraint)

- **CPU** = few smart cores, optimized for **latency** (finish one complex task fast).
- **GPU (RTX 3060)** = thousands of simple cores, optimized for **throughput** (huge volume of simple identical tasks).
- Training is **matrix multiplication** = millions of **independent** multiply-and-adds = **"embarrassingly parallel"** → fits the GPU perfectly.
- For the cores to work, data must physically sit in the GPU's own memory — **VRAM (12 GB)**. Doesn't fit → **CUDA Out-Of-Memory (OOM)**.

### What occupies VRAM during a training step

| Consumer | Size (relative to model) | Notes |
|---|---|---|
| **Weights** | 1× | the 124M numbers |
| **Gradients** | 1× | one per weight, same shape |
| **Optimizer states (AdamW)** | 2× | momentum + variance, one each per weight |
| **Activations** | depends on `batch_size × seq_length` | forward values cached for backward; often the **biggest** consumer |
| Current **batch** of data | tiny | full dataset stays on disk/RAM, streamed batch-by-batch |

➡️ **Static cost ≈ 4× the model** (weights + gradients + AdamW) **before any compute.** "Fits in memory" ≠ "trains in memory."

### First OOM rescue knob
**Lower the batch size** → shrinks **activations** (the consumer you directly control).
- Trade-off: less parallel work per step → **slower**, and noisier gradient estimates → less stable.
- **Gradient accumulation** (*Phase 5*) fixes the trade-off: process several small batches, **sum their gradients**, then update once → big-batch quality at small-batch memory.
- Related Phase 5 levers: **mixed precision (AMP)**, **gradient checkpointing**.

---

## ✅ Phase 0 "Done when"
- `torch.cuda.is_available()` returns **`True`**.
- The environment is **reproducible** (isolated venv + pinned `requirements.txt`).
- *(Implement step — install PyTorch matching your CUDA version, verify the GPU is visible.)*
