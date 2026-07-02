# tokenizer
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

def train(text: str, vocab_size: int) -> tuple[dict, dict]:
    """
    Train a tokenizer on the given text.
    
    text: str
        The text to be tokenized.
        
    vocab_size: int
        The size of the vocabulary.
        
    Returns:
        tuple[dict, dict]
            A tuple of two dictionaries.
    """
    assert vocab_size >= 256
    num_merges = vocab_size - 256
    
    ids = list(text.encode('utf-8'))
    merges = {}
    vocab = {i: bytes([i]) for i in range(256)} 
    
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        ids = merge(ids, pair, idx)
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        
    return merges, vocab

def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    """
    Decode the given list of ids using the given vocabulary.
    
    ids: list[int]
        A list of integer ids.
        
    vocab: dict[int, bytes]
        A dictionary of integer ids and their corresponding bytes.
        
    Returns:
        str
            The decoded string.
    """
    
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode('utf-8', errors='replace')
    return text

def encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    """
    Encode the given text using the given merges.
    
    text: str
        The text to be encoded.
        
    merges: dict[tuple[int, int], int]
        A dictionary of pairs of consecutive ids and their corresponding new id.
        
    Returns:
        list[int]
            A list of integer ids.
    """
    
    ids = list(text.encode('utf-8'))
    
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float('inf')))
        if pair not in merges:
            break
        idx = merges[pair]
        ids = merge(ids, pair, idx)
        
    return ids