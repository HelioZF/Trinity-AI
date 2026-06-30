# 🎥 Video 01 — Neural Networks & Backpropagation: building micrograd

- **Presenter:** Andrej Karpathy — *Neural Networks: Zero to Hero* (#1)
- **Link:** https://www.youtube.com/watch?v=VMj-3S1tku0
- **In one line:** builds **micrograd** (a tiny scalar autograd engine) from scratch, then trains a small neural net — demystifying **backpropagation** and what `loss.backward()` really does.

> These are study notes for active recall. Pair them with [Phase-0-memo.md](../notebooks/Phase-0-memo.md).

---

## 1. What is a derivative? (the foundation)
- **Derivative = sensitivity:** how much the output changes when you nudge an input by a tiny amount. It's the **slope** of the function at a point.
- **Numerical estimate (finite differences):** `(f(x+h) − f(x)) / h` for a tiny `h`.
- **Sign** = direction the output moves; **magnitude** = how strongly this input affects the output.

## 2. The `Value` object — the heart of micrograd
Each `Value` wraps **one scalar** and carries the bookkeeping that makes backprop possible:
- `.data` — the actual number.
- `.grad` — **dL/d(this value)**: how the *final* output (loss) responds to this node. Starts at `0`.
- `._prev` (children) — the Values that produced this one.
- `._op` — which operation created it (`+`, `*`, `tanh`, …).

➡️ Chaining operations on `Value`s automatically **builds a computation graph (a DAG)**.

## 3. Forward pass
- Evaluating the expression computes each node's `.data` **and** records the graph (children + op) as it goes.

## 4. Backpropagation — the main event
- **Goal:** fill every node's `.grad` with `dL/d(node)`.
- **Chain rule:** `dL/dx = dL/d(parent) × d(parent)/dx` = **(upstream gradient) × (local derivative)**.
- Every op knows its **local derivative**, so backprop is just multiply-and-accumulate backward:
  | Op | Gradient sent to inputs |
  |---|---|
  | `c = a + b` | `a.grad += out.grad`, `b.grad += out.grad` (addition **distributes** the gradient) |
  | `c = a * b` | `a.grad += b.data * out.grad`, `b.grad += a.data * out.grad` |
  | `c = tanh(a)` | `a.grad += (1 − tanh(a)²) * out.grad` |
- **Seed:** set `L.grad = 1` (because `dL/dL = 1`), then walk **backward**.
- **Order matters:** process nodes in **reverse topological order** — a node's gradient is only complete once *everything that depends on it* has already passed its gradient back.
- `.backward()` = topological sort → seed `self.grad = 1` → call each node's local `_backward()` in reverse.

## 5. ⚠️ The `+=` (accumulation) subtlety
- If a node feeds into **multiple** places, its gradients must **add up** (multivariable chain rule). That's why we use `grad += ...`, never `grad = ...`. Forgetting this silently corrupts gradients.

## 6. From scalars to a neural network
- **Neuron:** weights `w` (one per input) + bias `b` → output = `tanh(Σ wᵢ·xᵢ + b)`.
- **Layer:** a list of neurons. **MLP:** a list of layers.
- A forward pass through the MLP produces a prediction — itself a `Value` graph.

## 7. Loss & the training loop
- **Loss** = how wrong the prediction is. Here: **Mean Squared Error** = `Σ (pred − target)²`.
- **One training iteration (memorize this order):**
  1. **Forward** — compute predictions → loss.
  2. **Zero the gradients** — set every `param.grad = 0`. ← **must happen before backward**
  3. **Backward** — `loss.backward()` fills every gradient.
  4. **Update** — `param.data -= learning_rate * param.grad` (step **downhill** = gradient descent).
  5. Repeat.
- **Learning rate** = step size. Too big → overshoot / diverge; too small → painfully slow.

## 8. 🐛 The famous bug: forgetting `zero_grad`
- Because gradients **accumulate** (`+=`), if you don't reset them to `0` each iteration, gradients from previous steps **pile up** → wrong, often exploding, updates. Karpathy demonstrates this exact bug. **Fix:** zero the grads at the start of every step.

## 9. Bridge to PyTorch
- micrograd's `Value` ≈ a PyTorch **tensor** — but a tensor holds **arrays** (not one scalar) and can run on the **GPU**.
- PyTorch's `loss.backward()` does **exactly this graph-walk**, just industrial-strength and vectorized.

---

## 🧠 Key takeaways
- A neural net is **just a math expression** (a big graph of simple operations).
- **Training = forward → loss → backward (chain rule) → nudge weights downhill → repeat.**
- **Backprop = recursive chain rule** over the graph, computed in **one efficient backward pass**.
- You now know what's *inside* `.backward()` — it's not magic.

## ❓ My open questions (fill in as they come up)
-
