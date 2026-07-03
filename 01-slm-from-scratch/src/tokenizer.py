# tokenizer

class Tokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        
    def save(self, path: str) -> None:
        """
        Save tokenizer merges and vocab into disk
        """
        with open(path, 'w', encoding='utf-8') as file:
            for (p0, p1) in self.merges:
                file.write(f"{p0} {p1}\n")
    
    @staticmethod
    def _render(token_bytes: bytes) -> str:
        s = token_bytes.decode('utf-8', errors='replace')          # bytes -> str (invalid -> �)
        return "".join(ch if ch.isprintable() else f"\\u{ord(ch):04x}" for ch in s)

    def save_vocab(self, path: str) -> None:
        """
        Save a human view friendly version of the vocabulary into disk
        """
        inverted = {idx: pair for pair, idx in self.merges.items()}
        with open(path, 'w', encoding='utf-8') as file:
            for idx, token_bytes in self.vocab.items():
                s = self._render(token_bytes)
                if idx in inverted:
                    p0, p1 = inverted[idx]
                    s0 = self._render(self.vocab[p0])
                    s1 = self._render(self.vocab[p1])
                    file.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    file.write(f"[{s}] {idx}\n")
    
    
    def load(self, path: str) -> None:
        """
        Load tokenizer merges and vocab from disk
        """
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        idx = 256
        
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                p0, p1 = map(int, line.split())
                merges[(p0, p1)] = idx
                vocab[idx] = vocab[p0] + vocab[p1]
                idx += 1
                
        self.merges = merges
        self.vocab = vocab
    
    @staticmethod
    def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
        """
        Get the counts of each pair of consecutive ids in the list.
        
        ids: list[int]
            A list of integer ids.
            
        Returns:
            dict[tuple[int, int], int]
                A dictionary of pairs of consecutive ids and their counts.
        """
        counts = {}
        pairs = zip(ids, ids[1:]) # pairs each element with the next one
        for p1, p2 in pairs:
            counts[(p1, p2)] = counts.get((p1, p2), 0) + 1
        return counts
    
    @staticmethod
    def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
        """
        Merge the pair of consecutive ids in the list.
        
        ids: list[int]
            A list of integer ids
            
        pair: tuple[int, int]
            the target pair to be merged
            
        idx: int
            the new id to be attributed to the pair
            
        Returns:
            list[int]
                A list of integer ids with the pair merged.
        """
        
        newids = []
        i = 0
        while i < len(ids):
            if (i < len(ids) - 1) and (ids[i], ids[i+1]) == pair:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, text: str, vocab_size: int) -> None:
        """
        Train a tokenizer on the given text.
        
        text: str
            The text to be tokenized.
            
        vocab_size: int
            The size of the vocabulary.
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        
        ids = list(text.encode('utf-8'))
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)} 
        
        for i in range(num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self.merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            
        self.merges = merges
        self.vocab = vocab

    def decode(self, ids: list[int]) -> str:
        """
        Decode the given list of ids using the given vocabulary.
        
        ids: list[int]
            A list of integer ids.
            
        Returns:
            str
                The decoded string.
        """
        
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode('utf-8', errors='replace')
        return text

    def encode(self, text: str) -> list[int]:
        """
        Encode the given text using the given merges.
        
        text: str
            The text to be encoded.
            
        Returns:
            list[int]
                A list of integer ids.
        """
        
        ids = list(text.encode('utf-8'))
        
        while len(ids) >= 2:
            stats = self.get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = self.merge(ids, pair, idx)
            
        return ids