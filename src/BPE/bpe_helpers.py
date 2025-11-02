from typing import List


def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, idx, pair):
    new_ids = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

class BPEHelpers:

    def __init__(self):
        self.merges = {}
        self.special_tokens = {}
        self.vocab = self._build_vocab()
    
    def train(self, text:str, vocab_size:int, verbose=False):
        raise NotImplementedError

    def encode(self, text:str) -> List:
        raise NotImplementedError
    
    def decode(self, ids:List) -> str:
        raise NotImplementedError
    
    def load(self, file_name: str):
        raise NotImplementedError
    
    def save(self, file_name: str):
        raise NotImplementedError
    
    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab