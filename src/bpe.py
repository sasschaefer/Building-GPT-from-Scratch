import os
from collections import Counter
from typing import List, Tuple, Iterable
import pandas as pd
import data_utils as du

def word_to_symbols(word: str) -> Tuple[str, ...]:
    return tuple(list(word) + [du.WORD_END])

def corpus_to_symbol_sequences(words: Iterable[str]) -> List[Tuple[str, ...]]:
    return [word_to_symbols(w) for w in words]

def get_pair_counts(seqs: List[Tuple[str, ...]]) -> Counter:
    counts = Counter()
    for s in seqs:
        for i in range(len(s) - 1):
            counts[(s[i], s[i+1])] += 1
    return counts

def merge_pair_in_sequence(seq: Tuple[str, ...], a: str, b: str, ab: str) -> Tuple[str, ...]:
    out = []
    i, L = 0, len(seq)
    while i < L:
        if i < L-1 and seq[i] == a and seq[i+1] == b:
            out.append(ab); i += 2
        else:
            out.append(seq[i]); i += 1
    return tuple(out)

def apply_merge(seqs: List[Tuple[str, ...]], a: str, b: str) -> List[Tuple[str, ...]]:
    ab = a + b
    return [merge_pair_in_sequence(seq, a, b, ab) for seq in seqs]



class BPETokenizer:
    """Word-internal BPE with </w>. Train on TRAIN for k merges."""
    def __init__(self):
        self.merges: List[Tuple[str, str]] = []
        self.vocab:  List[str] = []
        self.normalization: str = "standard"

    def fit(self, corpus_path: str, k: int, normalization: str = "standard") -> None:
        self.normalization = normalization
        words = du.words_from_file_norm(corpus_path, normalization=normalization)
        seqs  = corpus_to_symbol_sequences(words)
        for _ in range(k):
            pair_counts = get_pair_counts(seqs)
            if not pair_counts: break
            (a, b), cnt = pair_counts.most_common(1)[0]
            if cnt < 2: break
            seqs = apply_merge(seqs, a, b)
            self.merges.append((a, b))
        vocab = set()
        for s in seqs:
            vocab.update(s)
        self.vocab = sorted(vocab, key=lambda t: (len(t), t))

def save_tokens_csv(tokens: List[str], csv_path: str):
    """Save token list to CSV with columns: rank, token, length, is_special."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    specials = {du.WORD_END}
    df = pd.DataFrame({
        "rank":   list(range(len(tokens))),
        "token":  tokens,
        "length": [len(t) for t in tokens],
        "is_special": [t in specials for t in tokens],
    })
    df.to_csv(csv_path, index=False, encoding="utf-8")







def train_and_eval_clean_task1(k: int, train_path: str, test_path: str, 
                               corpus, other_set,
                               normalization: str = "standard"):
    """
    Train BPE on TRAIN, write artifacts, evaluate on TRAIN/TEST/OTHER_SET,
    and return evaluation rows (incl. sanity).
    """
    # 1) Train
    bpe = BPETokenizer()
    bpe.fit(train_path, k=k, normalization=normalization)

    # 2) Save artifacts
    # 2a) always save normalization-suffixed copies
    toks_txt_norm = os.path.join(corpus, f"{normalization}_bpe_tokens_k{k}.txt")
    mrg_txt_norm  = os.path.join(corpus, f"{normalization}_bpe_merges_k{k}.txt")
    with open(toks_txt_norm, "w", encoding="utf-8") as f:
        for t in bpe.vocab: f.write(t + "\n")
    with open(mrg_txt_norm, "w", encoding="utf-8") as f:
        for a,b in bpe.merges: f.write(f"{a} {b}\n")
    save_tokens_csv(bpe.vocab, os.path.join(corpus, f"{normalization}_bpe_tokens_k{k}.csv"))

    # 2b) write CANONICAL filenames (exact names required by your assignment)
    if normalization == "standard":
        with open(os.path.join(corpus, f"bpe_tokens with k = {k}.txt"), "w", encoding="utf-8") as f:
            for t in bpe.vocab: f.write(t + "\n")
        with open(os.path.join(corpus, f"bpe_merges with k = {k}.txt"), "w", encoding="utf-8") as f:
            for a,b in bpe.merges: f.write(f"{a} {b}\n")
        save_tokens_csv(bpe.vocab, os.path.join(corpus, f"bpe_tokens with k = {k}.csv"))

    # 3) Sanity (expected_min_vocab ≈ |char_vocab| + k + 1 for </w>)
    train_words_for_char = du.words_from_file_norm(train_path, normalization=normalization)
    char_vocab = set("".join(train_words_for_char))
    expected_min_vocab = len(char_vocab) + k + 1
    sanity_row = {
        "normalization": normalization, "k": k, "set": "sanity",
        "merges_count": len(bpe.merges), "vocab_size": len(bpe.vocab),
        "expected_min_vocab": expected_min_vocab
    }

    # 4) Evaluate on TRAIN / TEST / OTHER_SET
    results = []
    for set_name, path in [("train", train_path), ("test", test_path), ("webtext", other_set)]:
        if not path or not os.path.exists(path): continue
        text = du.read_text(path)
        toks = du.tokenize_text_with_merges(text, bpe.merges, normalization=normalization)
        stats = du.compute_token_stats(toks)
        results.append({"normalization": normalization, "k": k, "set": set_name, **stats})

    results.append(sanity_row)
    return results
