# train_tokenizer.py — trains the project tokenizer on the arXiv corpus
import json, time
from tokenizer import Tokenizer

# corpus = titles + abstracts (the tokenizer must see the text it will meet)
recs = [json.loads(line) for line in open("data/arxiv_cs_LG.jsonl", encoding="utf-8")]
text = "\n".join(r["title"] + "\n" + r["abstract"] for r in recs)
print(f"corpus: {len(text):,} chars from {len(recs)} papers", flush=True)

t0 = time.time()
tok = Tokenizer()
tok.train(text, 2048)                      # the decision you defended
print(f"trained to vocab 2048 (1,792 merges) in {time.time()-t0:.0f}s")

tok.save("models/arxiv_2048.model")        # exact, loadable
tok.save_vocab("models/arxiv_2048.vocab")  # human-readable lineage

# sanity: round-trip on real data
t, a = recs[0]["title"], recs[0]["abstract"]
print("round-trip title   :", tok.decode(tok.encode(t)) == t)
print("round-trip abstract:", tok.decode(tok.encode(a)) == a)
ids = tok.encode(a)
print(f"abstract: {len(a.encode('utf-8'))} bytes -> {len(ids)} tokens "
    f"({len(a.encode('utf-8'))/len(ids):.2f}x compression)")
