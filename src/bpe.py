import os, json
from collections import Counter
from typing import List, Tuple, Iterable, Dict, Optional
import pandas as pd
import src.data_utils as du

# -----------------------
# Helpers
# -----------------------

def word_to_symbols(word: str) -> Tuple[str, ...]:
    """Split word into characters + end-of-word marker."""
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
    out, i, L = [], 0, len(seq)
    while i < L:
        if i < L-1 and seq[i] == a and seq[i+1] == b:
            out.append(ab); i += 2
        else:
            out.append(seq[i]); i += 1
    return tuple(out)

def apply_merge(seqs: List[Tuple[str, ...]], a: str, b: str) -> List[Tuple[str, ...]]:
    ab = a + b
    return [merge_pair_in_sequence(seq, a, b, ab) for seq in seqs]

# -----------------------
# Unified Tokenizer
# -----------------------

class BPETokenizer:
    """
    Unified BPE Tokenizer.
    - Train from corpus (discover merges + vocab).
    - Or load existing merges for encode/decode.
    """
    def __init__(self, merges: Optional[List[Tuple[str, str]]] = None,
                 extra_tokens: Optional[List[str]] = None):
        self.merges: List[Tuple[str, str]] = merges or []
        self.vocab: List[str] = []
        self.normalization: str = "standard"
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []
        self.extra_tokens = extra_tokens or []

    # -----------------
    # Training
    # -----------------
    def fit(self, corpus_path: str, k: int, normalization: str = "standard") -> None:
        """Learn BPE merges from a training corpus."""
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
        vocab.update(self.extra_tokens)
        self.vocab = sorted(vocab, key=lambda t: (len(t), t))
        self._build_index()

    # -----------------
    # Vocab management
    # -----------------
    def _build_index(self):
        """Build token<->id mappings."""
        self.id_to_token = list(self.vocab)
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}

    # -----------------
    # Encoding / Decoding
    # -----------------
    def encode_word(self, word: str) -> List[str]:
        """Apply merges to a single word."""
        return du.apply_merges_to_word(word.lower(), self.merges)

    def encode_words(self, words: Iterable[str]) -> List[str]:
        toks: List[str] = []
        for w in words:
            toks.extend(self.encode_word(w))
        return toks

    def encode_text(self, text: str) -> List[int]:
        """Tokenize text into ids."""
        lines_tok = du.tokenize_lines_with_merges(text, self.merges)
        ids = []
        for line in lines_tok:
            ids.extend([self.token_to_id[t] for t in line if t in self.token_to_id])
        return ids
    

    def decode_tokens(self, token_stream: List[str]) -> List[str]:
        """Convert BPE tokens back into words (joins on WORD_END)."""
        words, buf = [], []
        for t in token_stream:
            buf.append(t)
            if t.endswith(du.WORD_END):
                chars = []
                for sub in buf:
                    if sub.endswith(du.WORD_END):
                        chars.extend(list(sub[:-len(du.WORD_END)]))
                    else:
                        chars.extend(list(sub))
                words.append("".join(chars))
                buf = []
        # flush
        if buf:
            chars = []
            for sub in buf:
                if sub.endswith(du.WORD_END):
                    chars.extend(list(sub[:-len(du.WORD_END)]))
                else:
                    chars.extend(list(sub))
            if chars:
                words.append("".join(chars))
        return words

    # -----------------
    # Save / Load
    # -----------------
    def save_json(self, path: str):
        """Save merges + vocab as JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "merges": self.merges,
                "id_to_token": self.id_to_token,
                "token_to_id": self.token_to_id,
                "extra_tokens": self.extra_tokens,
                "normalization": self.normalization,
            }, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_json(path: str) -> "BPETokenizer":
        """Load tokenizer from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        tok = BPETokenizer(merges=obj["merges"], extra_tokens=obj.get("extra_tokens", []))
        tok.id_to_token = obj["id_to_token"]
        tok.token_to_id = {k: int(v) for k, v in obj["token_to_id"].items()}
        tok.normalization = obj.get("normalization", "standard")
        return tok

# -----------------------
# Utility: Save tokens CSV (assignment requirement)
# -----------------------

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
