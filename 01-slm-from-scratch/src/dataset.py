# dataset.py — serve tokenized (title,abstract) sequences as padded batches
import json, torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer

class TitleAbstractDataset(Dataset):
    def __init__(self, path: str):
        self.seqs = [json.loads(line)["ids"] for line in open(path, encoding="utf-8")]

    def __len__(self):
        return len(self.seqs)                    

    def __getitem__(self, i):
        return torch.tensor(self.seqs[i], dtype=torch.long)   # ← "hand over example i" (from self.seqs)

def collate(batch):
    """Glue ragged 1-D tensors into one [B, T_max] rectangle, filling with PAD."""
    T = max(len(s) for s in batch)                              # longest in THIS batch
    out = torch.full((len(batch), T), Tokenizer.PAD_ID, dtype=torch.long)   # rectangle of pads
    for row, seq in enumerate(batch):
        out[row, :len(seq)] = seq                               # copy real ids over the pads
    return out

if __name__ == "__main__":
    ds = TitleAbstractDataset("data/tokenized_cs_LG.jsonl")
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate)
    batch = next(iter(loader))
    print("dataset size :", len(ds))
    print("batch shape  :", tuple(batch.shape), "| dtype:", batch.dtype)
    print("pad fraction :", (batch == Tokenizer.PAD_ID).float().mean().item())
    tok = Tokenizer(); tok.load("models/arxiv_2048.model")
    print("row 0 decoded:", tok.decode(batch[0].tolist())[:160])
