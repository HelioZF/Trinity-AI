# prepare_data.py — turn (title, abstract) pairs into training sequences
import json, time
from tokenizer import Tokenizer

tok = Tokenizer()
tok.load("models/arxiv_2048.model")        # ← get the trained tokenizer (do NOT train!)

recs = [json.loads(line) for line in open("data/arxiv_cs_LG.jsonl", encoding="utf-8")]

t0 = time.time()
lengths = []
with open("data/tokenized_cs_LG.jsonl", "w", encoding="utf-8") as out:
    for r in recs:
        ids = tok.encode(r["title"]) + [tok.SEP_ID] + tok.encode(r["abstract"]) + [tok.EOT_ID]
        lengths.append(len(ids))           #    use the class constants, not bare 2048/2049!)
        out.write(json.dumps({"id": r["id"], "ids": ids}) + "\n")

print(f"tokenized {len(lengths)} pairs in {time.time()-t0:.0f}s")
print(f"lengths — avg: {sum(lengths)/len(lengths):.0f} | max: {max(lengths)} | min: {min(lengths)}")

# sanity: the first sequence, decoded — expect: title<|sep|>abstract<|endoftext|>
first = json.loads(open("data/tokenized_cs_LG.jsonl", encoding="utf-8").readline())
print(tok.decode(first["ids"])[:200])
