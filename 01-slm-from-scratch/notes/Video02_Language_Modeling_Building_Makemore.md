# 🎥 Video 02 — Language Modeling: building makemore (Part 1, the bigram model)

- **Presenter:** Andrej Karpathy — *Neural Networks: Zero to Hero* (#2)
- **Link:** https://www.youtube.com/watch?v=PaCmpygFfXo
- **In one line:** builds a **character-level bigram language model** on a names dataset — first by **counting**, then by **gradient descent on a tiny neural net** — and shows the two are equivalent. This is the conceptual seed of every later model (and our SLM).

> Study notes for active recall. Pair with [Phase-0-memo.md](../notebooks/Phase-0-memo.md). Callbacks to Phase 0 are marked 🔁.

---

## 1. What is makemore?
- A model that "makes more" of its input. Fed ~32k names, it learns to **generate new name-like strings**.
- **Character-level:** it predicts the next *character*, not word. 🔁 Same next-token idea from Phase 0, with characters as tokens.

## 2. The data & the vocabulary
- Load names; build the character vocabulary: 26 letters + **one special token `.`** used for **both start and end** of a word (the boundary marker). 🔁 This is the "end-of-sequence" token idea.
- Build two lookups: **`stoi`** (string→index) and **`itos`** (index→string). 27 tokens total → indices `0..26` (`.` = 0).

## 3. The bigram model (the idea)
- **Bigram = a pair of consecutive characters.** The model predicts the next char using **only the single previous char**. Very weak, but the perfect first step.
- Training data = every consecutive pair in every name, e.g. `emma` → `.e`, `em`, `mm`, `ma`, `a.` (with boundary `.`). Get pairs with `zip(chars, chars[1:])`.

## 4. Approach A — counting
- Build a **count matrix `N`** of shape **(27, 27)**: `N[i, j]` = how many times char `j` follows char `i`. Just tally every bigram.
- Visualize it (Karpathy plots the grid) — you can literally see which letters follow which.

## 5. Counts → probabilities (vectorized + broadcasting) ⚠️
- Normalize **each row** so it sums to 1 (each row = a probability distribution over the next char):
  ```python
  P = N / N.sum(1, keepdim=True)
  ```
- **`N.sum(1, keepdim=True)`** → shape **(27, 1)** = each row's total. **Broadcasting** stretches that column across all 27 columns → every element divided by its **row** sum. ✅
- 🐛 **The `keepdim` gotcha:** drop `keepdim=True` → shape `(27,)` → treated as `(1, 27)` → broadcasts across **rows** → divides by the wrong totals, **silently**. Broadcasting rule: align shapes **from the right**; dims must be **equal or one is 1**.
- 💡 Vectorize (not Python loops) for speed — 🔁 the throughput lesson.

## 6. Sampling — generating names
- 🔁 The **autoregressive loop** from Phase 0, in code:
  1. Start at index `0` (`.`). 2. Take that row's probabilities `p`. 3. **Sample** the next index. 4. Append. 5. Repeat until you sample `0` (`.`) again → name done.
- Tool: **`torch.multinomial(p, num_samples=1, replacement=True, generator=g)`** — a weighted dice roll that returns index `i` with probability `p[i]`.
- `g = torch.Generator().manual_seed(...)` → reproducible "randomness."
- Output names are bad (a 1-char-context model is dumb) but clearly better than uniform-random → the model learned *something*.

## 7. Model smoothing
- Add fake counts (e.g. `N + 1`) → **smoothing**. Prevents any probability being exactly **0** (which would make `log(0) = −∞` → infinite loss). More smoothing → more uniform distribution.

## 8. The loss — negative log likelihood (NLL)
- **Likelihood** of the data = **product** of the probabilities the model gives to each *actual* bigram. We want it high.
- 🐛 Product of many tiny numbers **underflows** → take **`log`**: `log(a·b)=log a+log b`, so product → **sum**.
- **Log likelihood** = `Σ log(pᵢ)` (always ≤ 0). **Negate** → **NLL** = `−Σ log(pᵢ)` so lower = better (a proper loss). **Average:** `mean NLL = −(1/n) Σ log(pᵢ)`.
- Intuition: perfect model → `p=1` → `log 1 = 0` → loss 0. Tiny `p` on a real bigram → large loss.
- 🔁 This is the **loss you invented in Phase 0**, and its formal name is **cross-entropy** (Phase 4). **GOAL = minimize average NLL.**

## 9. Approach B — the same model as a neural network
Same bigram task, now learned by **gradient descent** (the bridge to everything later):
- **Training set:** `xs` = input char indices, `ys` = target (next) char indices.
- **One-hot encode** inputs: `F.one_hot(xs, num_classes=27).float()` → each char becomes a 27-dim vector (a single `1`, rest `0`).
- **One linear layer:** weights `W` shape **(27, 27)**; `logits = xenc @ W` (matrix multiply). 🔁 `W` is the bag of weights we tune.
- **Softmax** turns logits into probabilities: `counts = logits.exp()` → `probs = counts / counts.sum(1, keepdim=True)`. So **logits = log-counts**; `exp` makes them positive "counts"; normalize → distribution. (`W` ≈ the log of the count matrix `N`.)
- **Loss:** negative log likelihood of the correct next chars, read off `probs`.
- **Train:** `loss.backward()` (🔁 autograd from micrograd!) → `W.data -= lr * W.grad` → repeat. `W` converges to (almost) the same loss as counting.
- **Regularization:** add `+ λ * (W**2).mean()` to the loss (**L2 reg**) → the gradient-based equivalent of **smoothing** (pushes `W` toward 0 → more uniform probs).

## 10. Why this matters (the punchline)
- Counting and the neural net reach the **same answer** — but the neural-net framework **scales**: you can feed it **more context** (more previous chars), which pure counting can't (the table explodes). 🔁 This is the path that leads to MLPs → transformers → our SLM.

---

## 🧠 Key takeaways
- A **bigram model** predicts the next char from the previous one; train it by **counting** *or* by **gradient descent** — same result.
- **`softmax`** (exp + normalize) turns raw `logits` into a probability distribution; **`logits` = log-counts**.
- **Sampling** = `torch.multinomial` in an **autoregressive loop**; **`keepdim=True`** matters for broadcasting.
- **Loss = mean negative log likelihood = cross-entropy**; minimize it. **Regularization ≈ smoothing.**
- The neural-net framing is the **scalable** one — it's why we don't just count.

## ❓ My open questions (fill in as they come up)
-
