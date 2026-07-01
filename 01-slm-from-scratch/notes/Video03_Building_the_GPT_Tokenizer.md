# 🎥 Video 03 — Let's build the GPT Tokenizer

- **Presenter:** Andrej Karpathy — *Neural Networks: Zero to Hero* (tokenizer)
- **Link:** https://www.youtube.com/watch?v=zduSFxRajkE
- **In one line:** builds a **byte-level BPE tokenizer** from scratch (train → `encode` → `decode`), then explains how the real GPT-2/GPT-4 tokenizers work — and why tokenization is the hidden cause of many LLM "weirdnesses."

> Study notes for active recall. This is the **Phase 1** reference. Callbacks to earlier phases marked 🔁.

---

## 1. What tokenization is (and why it matters)
- 🔁 The model is a **math machine — it eats numbers**. **Tokenization = the translator** between text and integer **token IDs** (via a **vocabulary**).
- 🔁 In makemore you used the simplest version: **character-level** (`a→1, b→2…`). A real tokenizer uses **subwords**.
- **Tokenization silently causes many LLM failures:** bad spelling / character tasks, weak arithmetic, worse performance in non-English, old Python-whitespace issues, trailing-whitespace bugs, and "trigger" tokens (see §11).

## 2. Unicode, code points, UTF-8
- A **code point** = the unique integer Unicode assigns to a character. `ord('A')=65`, `ord('😀')=128512`; `chr(65)='A'`. (~150k defined.)
- ⚠️ **Code point ≠ byte.** **UTF-8** *encodes* each code point into **1–4 bytes** (ASCII → 1 byte; `😀` → 4 bytes `[240,159,152,128]`).
- **Pipeline:** text → code points (Unicode) → bytes (UTF-8). UTF-8 chosen over UTF-16/32: ASCII-compatible, compact, a clean **byte stream**.

## 3. Why subwords? (the tradeoff)
- **Raw bytes** (vocab 256): 🔁 **no OOV ever**, but sequences too **long** → attention cost is **quadratic** in length.
- **Characters vs words vs subwords:** char = tiny vocab / long seqs; word = huge vocab / OOV; **subword = Goldilocks**.
- **Vocab-size tradeoff:** embedding + output layers grow **linearly** with vocab; attention grows **quadratically** with sequence length. → a **sweet spot**. For a *small* model, an oversized vocab's embedding/output can **dominate** params → lean **moderate/smaller**.

## 4. BPE — training (learning the merges)
- **Start** at the **256 byte** tokens.
- **Loop:** `get_stats` (count adjacent **pairs**) → pick the **most frequent** pair → **merge** it into a **new token** (id `256, 257, …`) → replace all occurrences → repeat until target vocab.
- Merges are recorded **in order**. **`vocab_size = 256 + num_merges`** (+ special tokens). GPT-2 = **50,257** = 256 + 50,000 + 1.
- **Tiny example:** `aaabdaaabac` → merge `aa`→Z → `ZabdZabac` → merge `ab`→Y → `ZYdZYac` → merge `ZY`→X → `XdXac`.

## 5. `encode` and `decode`
- **`decode(ids)`:** map each id → its bytes, concatenate, then `bytes.decode("utf-8", errors="replace")` (the `errors="replace"` handles invalid byte sequences → `�`).
- **`encode(text)`:** text → UTF-8 bytes → **repeatedly apply the earliest-learned applicable merge**:
  - `pair = min(stats, key=lambda p: merges.get(p, inf))` → picks the pair with the **lowest merge index** present.
  - 🔑 **Why lowest index, not most frequent?** Merges must be replayed **in the order they were trained**, because later merges are built on earlier ones. Frequency was for *training*; **order** is for *encoding*.
  - Stop when no present pair is in `merges`. 🐛 Guard `if len(tokens) < 2: break` (else `min` over empty → error).
- 🎯 **Done-when property:** **`decode(encode(x)) == x`** (round-trip).

## 6. The regex pre-split (real GPT tokenizers)
- Before BPE, GPT-2 **splits text with a regex** into chunks (letters, numbers, punctuation, whitespace) so **merges never cross category boundaries** — e.g. `"dog"` and `"."` won't fuse into one token, and numbers don't glue to words.
- **GPT-4** uses a **different pattern** (case-insensitive contractions, whitespace handling, numbers capped at ~3 digits).

## 7. tiktoken, GPT-2 vs GPT-4
- **`tiktoken`** = OpenAI's official BPE library (**inference only** — it can't *train* new tokenizers).
- GPT-4 differs from GPT-2 mainly in the **regex pattern** and **whitespace merging**, plus a larger vocab (~100k).

## 8. Special tokens
- Tokens like **`<|endoftext|>`** delimit documents; chat/FIM models add more (e.g. `<|im_start|>`).
- They're **not** produced by BPE — they're inserted **outside** the merge process and handled specially in `encode`. Adding one means **extending the vocab + resizing** the model's embedding/output layers.

## 9. SentencePiece (the other major library)
- Used by **Llama, T5**. Runs **BPE or Unigram** directly on **code points** (not bytes-first), with **`byte_fallback`** for rare code points, and bakes **whitespace** in as `▁`.
- 🔁 Contrast: **tiktoken = bytes-first**; **SentencePiece = code-points-first, bytes as a safety net**. SentencePiece has many (historically messy) config options but *can train* tokenizers.

## 10. Choosing / extending vocab size
- Vocab size is a **hyperparameter**. Growing it later means **adding rows** to the embedding table and **outputs** to the final layer (and training those new params).

## 11. Why tokenization causes LLM "weirdness" 🐛
- **Spelling / reversing strings:** characters are **hidden inside** multi-char tokens → the model can't easily see them.
- **Arithmetic:** numbers get split **inconsistently** (e.g. `677` vs `6`+`77`) → hard to compute.
- **Non-English worse:** those languages need **more tokens per word** (less efficient, less training) → weaker performance.
- **Python (old GPT-2):** every **space** was its own token → wasted context on indentation.
- **Trailing whitespace & "SolidGoldMagikarp":** rare/never-trained tokens behave unstably → weird outputs.

---

## 🧠 Key takeaways
- Tokenization = **text ↔ integer tokens**, via a vocabulary; it's the foundation the whole model is built on (**Phase 1** for a reason).
- **Byte-level BPE:** start at **256** bytes, merge the **most frequent pair** repeatedly; `vocab = 256 + merges`; **no OOV**.
- **Train by frequency, encode by merge-order**; verify with the **round-trip** `decode(encode(x)) == x`.
- Real tokenizers add a **regex pre-split** + **special tokens**; **tiktoken** (bytes) vs **SentencePiece** (code points).
- Many LLM quirks are **tokenizer artifacts**, not model stupidity.

## ❓ My open questions (fill in as they come up)
-
