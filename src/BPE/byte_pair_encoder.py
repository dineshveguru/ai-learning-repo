"""
A byte pair encoder, a part of the transformer that encodes the given string
input to a format that can be understood by the LLM.
We will implement byte-level BPE that can transform the given string to encoded format
by using a trained encoder.
This code contains both the training part and the inference part.

Process
------------------
* Training
bag of strings (training data [corpus]) -> base vocabulary (individual characters in that bag of strings) -> Merging Process (training) -> New words added to vocabulary -> Assign indexes to new symbols (words)

* Inferencing
Given string tokenize based on the encoder training process -> get index of that particular token -> Return the array
"""

from bpe_helpers import BPEHelpers, get_stats, merge
import numpy as np
class BasicTokenizer(BPEHelpers):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        text_bytes = text.encode("utf-8")
        tokens = list(map(int, text_bytes))
        n_before = len(tokens)
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(tokens)
            max_pair = max(stats, key=stats.get)
            idx = 256 + i
            tokens = merge(tokens, idx, max_pair)
            merges[max_pair] = idx
            vocab[idx] = vocab[max_pair[0]] + vocab[max_pair[1]]
            if verbose:
                print(f"merge {i + 1}/{num_merges}: {max_pair} -> {idx} ({vocab[idx]}) had {stats[max_pair]} occurrences")
        self.merges = merges
        self.vocab = vocab
        n_after = len(tokens)
        print(f"Compression Ratio: {n_after / n_before:.2f}X")
        self.save("trained_corpus")

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def encode(self, text):
        text_bytes = text.encode("utf-8")
        tokens = list(text_bytes)
        new_tokens = []
        i = 0
        n = len(tokens)
        while i < n:
            if i < n - 1:
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    new_tokens.append(self.merges[pair])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def save(self, file_name):
        with open(f"./{file_name}.model", "w") as f:
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        with open(f"./{file_name}.vocab", "w") as f:
            for idx, val in self.vocab.items():
                f.write(f"{idx} {val}\n")

    def load(self, file_name):
        assert file_name.endswith(".model")
        merges = {}
        idx = 256
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.vocab = self._build_vocab()

    def visualize_tokens(self, text):
        template = """
<html>
<body>
"""
        encoded_text = self.encode(text)
        for i in encoded_text:
            random_colour = np.random.choice(range(128, 256), 3) # lighter colours
            template += f'<span style="background-color: rgb{int(random_colour[0]), int(random_colour[1]), int(random_colour[2])}">{self.decode([i])}</span>'
        template += """
</body></html>
"""
        with open("token_viz.html", "w") as f:
            f.write(template)

