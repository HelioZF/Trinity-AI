# 📝 Phase 2 Memo — Data Pipeline

**Status:** ✅ complete (Iteration 1). Done-when met: a shuffled batch of (title, abstract) tensors loads with correct shapes — `(32, 577)`, `torch.int64`, pad fraction ~0.40.

## What was built
- `src/fetch_arxiv.py` — arXiv API pipeline: `parse_entries` + `clean` (Atom XML → cleaned tuples), `build_dataset` (**idempotent**: seeds `seen` from the manifest, `target` = total), `fetch_by_ids` + `restore_from_manifest` (rebuild the *exact* dataset anywhere from the committed id manifest).
- **Corpus:** 2,000 cs.LG papers (2.28 MB of abstracts; avg abstract 1,142 chars, title 73). Data gitignored; `data/paper_ids.txt` committed.
- **Vocab experiment** (`vocab 2048` chosen): trained once at 4,096 on a 400-abstract sample, evaluated merge-list *prefixes* (nested merges) on 30 held-out abstracts. 2,048 → 3.83 bytes/token, 289 toks/abstract, 79% vocab alive, 1.57 M emb+out params @384. Sequences 22% shorter than vocab-1,024 ⇒ attention ~38% cheaper ((1−0.22)² ≈ 0.61).
- `src/train_tokenizer.py` — trained on titles+abstracts of all 2,000 papers (365 s); artifacts **committed**: `models/arxiv_2048.model` (canonical) + `.vocab` (human-readable lineage) via .gitignore exceptions.
- **Special tokens** (appointed, not learned): `SEP_ID=2048`, `EOT_ID=2049`, `PAD_ID=2050` → model vocab **2,051**. Registered decode-side after every vocab rebuild (`train` *and* `load`). Unforgeable from text (unreachable through the merge loop).
- `src/prepare_data.py` — recipe `encode(title) + [SEP] + encode(abstract) + [EOT]` for all pairs (144 s) → `data/tokenized_cs_LG.jsonl`. Lengths: **avg 317 · max 769 · min 51**.
- `src/dataset.py` — `TitleAbstractDataset` (`__init__`/`__len__`/`__getitem__`) + `DataLoader(shuffle, batch_size=32, collate_fn)` padding ragged sequences to the batch max with `PAD_ID`.

## Key concepts (study list)
Dataset vs DataLoader (bookshelf vs librarian) · train/val split & overfitting · batching tradeoff (VRAM/throughput/gradient noise) · shuffling for unbiased batches · commit-the-recipe-not-the-data (manifest) · idempotency · nested BPE merges · the two costs (params ∝ vocab vs attention ∝ n²) · padding + collate · calculate → measure → judge.

## Decisions & why
- **vocab 2048** — attention discount beats liveness cost on 12 GB; reversible (truncation).
- **Minimal specials (2+pad)** — every special is an embedding row; position 0 needs no marker.
- **Pad = dedicated id** (never reuse a meaningful token); loss will `ignore_index` it in Phase 4.
- **Context window: NOT yet decided** — max 769 vs avg 317 long tail; decide at Phase 3 with percentiles (truncate vs pay n²).

## Gotchas that bit us (remember these)
- `.gitignore` has **no trailing comments** (pattern becomes literal text).
- Windows cp1252: always `encoding="utf-8"`; console can't print emoji.
- Debugger slows tight Python loops badly — plain runs for training.
- `str + list` TypeError: nothing crosses into the integer world except through `encode()`.

## Next: Phase 3 — Transformer architecture
📚 **Watch first:** Karpathy, *Let's build GPT: from scratch* (the phase's anchor video) → quiz → handwritten summary → build (see the Codex Phase 3 field guide: config d_model 384 / 6 heads / 6 layers, build order, pitfalls).
