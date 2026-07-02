# 🧠 Phase 1 Memo — Tokenization (BPE from Scratch)

> Quick-reference study sheet for Pillar 1 (SLM from Scratch).
> Covers the **theory + build** of Phase 1. Read top-to-bottom in ~5 minutes to reload the mental model.

---

## 🎥 Re-watch video

**Andrej Karpathy — "Let's build the GPT Tokenizer"**
🔗 https://www.youtube.com/watch?v=zduSFxRajkE

Why re-watch: the first half (Unicode → BPE training) clicks fast, but the **second half is the tricky part** — the `encode` merge-order logic, the regex pre-split, and special tokens. Re-watch from the `encode`/`decode` section onward; that's where the subtlety lives.
*(Next up for Phase 3: "Let's build GPT from scratch.")*

---

## Pillar 1 — What tokenization is, and the byte pipeline

- A neural net is a **math machine — it only eats numbers.** The tokenizer is the **bidirectional translator** between text and integer **token IDs** (via a **vocabulary**): `encode` (text → ids) and `decode` (ids → text).
- **The pipeline:** `text → code points (Unicode) → bytes (UTF-8)`.
  - **Code point** = the integer Unicode assigns a character (`ord('A')=65`, `😀`=128512). ~150k defined.
  - **Byte** = 8 bits → **256 values, range 0–255** (not 256!).
  - **UTF-8** encodes each code point into **1–4 bytes**, *determined by the code point's size* (ASCII → 1 byte; `😀` → 4 bytes). Chosen because it's **ASCII-compatible**, **compact**, and yields a clean **byte stream (0–255)**.

## Pillar 2 — Why subwords, and the vocab-size hyperparameter

- **Granularity tradeoff** (vocab size vs. sequence length are **inverse**):

  | Unit | Vocab | Seq length | Problem |
  |---|---|---|---|
  | Raw bytes | 256 (fixed) | **longest** | sequences too long |
  | Characters | small | long | little meaning/token |
  | Words | **huge** | shortest | **OOV** (unseen words) |
  | **Subwords (BPE)** | moderate | moderate | 🥇 the Goldilocks sweet spot, **no OOV** |

- **Vocab size is a hyperparameter.** Two axes that pull opposite ways:
  - **Embedding + output params** grow **linearly** with vocab (one row per token; table = `vocab × d_model`).
  - **Attention** grows **quadratically** with **sequence length** (`n×n` matrix) — *not* with vocab (vocab affects it only indirectly, via seq length).
  - → For a **small model on 12 GB**, lean **moderate/smaller**: a huge vocab lets the embedding/output tables dominate the parameter budget (and OOM risk).
- `d_model` (embedding width) is a **separate** hyperparameter, independent of vocab; per-layer params grow ~`d_model²`.

## Pillar 3 — BPE: the algorithm we built

**BPE = Byte-Pair Encoding.** Start from 256 byte tokens; repeatedly **merge the most frequent adjacent pair** into a new token.

- `vocab_size = 256 + num_merges` (+ special tokens). GPT-2's `50,257 = 256 + 50,000 + 1 (<|endoftext|>)`.
- **Training loop:** `get_stats` (count adjacent **pairs**) → pick the **winner** (max count) → `merge` (mint new id, replace all occurrences) → **record the merge in order** → repeat to target vocab.
- **Merges stack** (cumulative): a later token is built from earlier ones. Real example our tokenizer learned:
  ```
  ' token' (263) = ' to'(260) + 'ken'(261)
                    ' t'(256)+'o'    'k'+'en'(257)
  ```

## Pillar 4 — encode / decode / round-trip (the subtle core)

- 🔑 **Train by frequency · Encode by order · Decode by lookup.**
- **`encode(text)`:** bytes → repeatedly apply the **earliest-learned applicable merge** — `min(stats, key=lambda p: merges.get(p, inf))`. Order is **mandatory** because later merges depend on earlier ones *and* it must reproduce the exact tokenization the model trained on.
  - Two stops: `while len(ids) >= 2` (structural — ≥2 tokens to have a pair) **and** `if pair not in merges: break` (semantic — nothing left to merge).
- **`decode(ids)`:** look up each id's **bytes** in `vocab`, concatenate (`b"".join(...)`), then `bytes.decode("utf-8", errors="replace")`. No ordering needed — it's a direct lookup. `errors="replace"` guards against **invalid UTF-8** byte sequences (byte-level BPE can emit them) → substitutes `�` instead of crashing.
- 🎯 **Correctness property:** **`decode(encode(x)) == x`** (round-trip).
- **Two dicts, two jobs:** `merges` (pair→id, in order) powers **encode**; `vocab` (id→bytes) powers **decode**.

## Pillar 5 — Real tokenizers & why LLMs act weird

- **Regex pre-split:** GPT splits text (letters / numbers / punctuation / whitespace) **before** BPE so merges **never cross category boundaries** (e.g. `"dog"`+`"."` won't fuse; digits won't glue to words).
- **Special tokens** (e.g. `<|endoftext|>`): added **outside** BPE; adding one means resizing the model's embedding/output layers.
- **tiktoken vs. SentencePiece:** tiktoken = **bytes-first, inference-only** (can't train); SentencePiece = **code-points-first** (+ `byte_fallback`), **can train**.
- **LLM "weirdness" = tokenizer artifacts:** bad spelling/reversing (chars hidden inside tokens), weak arithmetic (inconsistent number splits), worse non-English (more tokens/word), `SolidGoldMagikarp` (token in tokenizer's data but ~never in model's data → untrained embedding → unstable).

---

## 🔨 What we built — `src/tokenizer.py`

Six functions, pure standard library (no dependencies):

| Function | Role |
|---|---|
| `get_stats(ids)` | count adjacent pairs → `{pair: count}` (🔁 same as makemore bigram counting) |
| `merge(ids, pair, idx)` | replace every occurrence of `pair` with `idx` (returns a new list; `while` + manual index, skip-by-2; bounds check `i < len(ids)-1`) |
| `train(text, vocab_size)` | learn merges to the target vocab; returns `(merges, vocab)` |
| `encode(text, merges)` | text → ids, replaying merges by lowest index |
| `decode(ids, vocab)` | ids → text, byte lookup + `errors="replace"` |

### ✅ Validation
- **All round-trips pass:** training text, unseen (OOV) words, empty string, single char, accents + emoji, whitespace runs, digits.
- **Sensible vocab:** with 20 merges, `341 bytes → 236 tokens` (**1.44×** compression). More merges → more compression (GPT-2 ≈ 4× on English).

### ⚠️ Gotchas learned the hard way
- **Fresh `counts = {}` inside `get_stats`** (not module-level) — else counts pile up across calls.
- **`merge` bounds check** must be `i < len(ids) - 1` (guarding `ids[i+1]`), *not* `i < len(ids)`.
- **`vocab[idx] = vocab[pair[0]] + vocab[pair[1]]`** — bracket placement matters (concatenate two byte lookups, don't index by their sum).
- **Windows console = cp1252** → printing emoji crashes with `UnicodeEncodeError`. Fix: `sys.stdout.reconfigure(encoding="utf-8")` (or `PYTHONIOENCODING=utf-8`). **Remember for the Phase 2 data pipeline: always pass `encoding="utf-8"` when reading/writing files.**

---

## ✅ Phase 1 "Done when"
- `decode(encode(x)) == x` — the round-trip holds (incl. edge cases). ✅
- The tokenizer reports a **sensible vocab** and compresses text. ✅

**Next → Phase 2: Data Pipeline** (fetch + clean arXiv title↔abstract pairs, tokenize with this tokenizer, build `Dataset`/`DataLoader`).
