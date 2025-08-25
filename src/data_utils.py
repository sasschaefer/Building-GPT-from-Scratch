import os, re, glob, shutil
from collections import Counter
from typing import List, Tuple, Dict

# BPE end-of-word symbol (used later)
WORD_END = "</w>"

def ensure_in_corpus(filename: str, corpus):
    """Search filename in workspace and place it under Corpus/."""
    dst = os.path.join(corpus, filename)
    if os.path.exists(dst):
        return
    matches = glob.glob(f"**/{filename}", recursive=True)
    if not matches and os.path.exists(filename):
        matches = [filename]
    if matches:
        src = matches[0]
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)

# Simple IO
def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# ---- RAW snapshot helpers
def _raw_compose_report(path: str, label: str) -> Dict[str, object]:
    if not os.path.exists(path):
        return {"set": label, "exists": False}
    txt = read_text(path)
    toks_ws = txt.split()
    n_ws = len(toks_ws)
    return {
        "set": label,
        "exists": True,
        "chars_raw": len(txt),
        "lines": (txt.count("\n") + 1) if txt else 0,
        "whitespace_tokens": n_ws,
        "alphabetic_words_[A-Za-z]+": len(re.findall(r"[A-Za-z]+", txt)),
        "digit_tokens": sum(1 for t in toks_ws if re.search(r"\d", t)),
        "punct_tokens": sum(1 for t in toks_ws if re.search(r"[^\w]", t)),
        "avg_token_len": (sum(len(t) for t in toks_ws) / n_ws) if n_ws else 0.0,
    }

##################################
#         NORMALIZATION          #
##################################

_wsre = re.compile(r"\s+")

def words_from_text_norm(text: str, normalization: str = "standard") -> List[str]:
    """
    normalization:
      - 'standard': lowercase + ONLY letters → regex [a-z]+  (class expectation)
      - 'aggressive_clean': lowercase + keep only [a-z0-9] + whitespace
    """
    text = text.lower()
    if normalization == "aggressive_clean":
        text = re.sub(r"[^a-z0-9\s]+", " ", text)
        return [w for w in _wsre.split(text.strip()) if w]
    return re.findall(r"[a-z]+", text)

def words_from_file_norm(path: str, normalization: str = "standard") -> List[str]:
    return words_from_text_norm(read_text(path), normalization=normalization)

##################################
#         DATA  OVERVIEW         #
##################################

def corpus_basic_stats_from_words(words: List[str]) -> Dict[str, float]:
    n_words = len(words)
    uniq = len(set(words))
    avg_len = sum(len(w) for w in words) / max(1, n_words)
    return {"words": n_words, "unique_words": uniq, "avg_word_length": avg_len}

def top_n_words(words: List[str], n: int = 20) -> List[Tuple[str, int]]:
    return Counter(words).most_common(n)

def summarize_split(path: str, label: str, train_vocab: set = None,
                    normalization: str = "standard"):
    if not path or not os.path.exists(path):
        return {"set": label, "exists": False}, []
    text = read_text(path)
    chars_raw = len(text)
    words = words_from_text_norm(text, normalization=normalization)
    stats = corpus_basic_stats_from_words(words)
    row = {"set": label, "exists": True, "chars_raw": chars_raw, **stats}
    if train_vocab is not None and len(words) > 0:
        oov = sum(1 for w in words if w not in train_vocab)
        row["oov_rate_vs_train"] = oov / len(words)
    else:
        row["oov_rate_vs_train"] = None
    return row, words

##################################
#          TOKEN HELPER          #
##################################

def apply_merges_to_word(word: str, merges: List[Tuple[str, str]]) -> List[str]:
    symbols = tuple(list(word) + [WORD_END])
    for a, b in merges:
        out = []
        i, L = 0, len(symbols)
        while i < L:
            if i < L-1 and symbols[i] == a and symbols[i+1] == b:
                out.append(a + b); i += 2
            else:
                out.append(symbols[i]); i += 1
        symbols = tuple(out)
    return list(symbols)

def tokenize_text_with_merges(text: str, merges: List[Tuple[str, str]],
                              normalization: str = "standard") -> List[List[str]]:
    words = words_from_text_norm(text, normalization=normalization)
    return [apply_merges_to_word(w, merges) for w in words]

def compute_token_stats(tokenized_words: List[List[str]]) -> Dict[str, float]:
    total_words  = len(tokenized_words)
    total_tokens = sum(len(toks) for toks in tokenized_words)
    avg_tokens_per_word = total_tokens / max(1, total_words)
    word_as_token = sum(1 for toks in tokenized_words if len(toks) == 1)
    word_as_token_rate = word_as_token / max(1, total_words)
    multi_char_tokens = sum(1 for toks in tokenized_words for t in toks if len(t) > 1)
    merge_use_rate = multi_char_tokens / max(1, total_tokens)
    unique_tokens = len({t for toks in tokenized_words for t in toks})
    unique_words  = len({tuple(toks) for toks in tokenized_words})
    type_compression = unique_tokens / max(1, unique_words)
    return {
        "total_words": float(total_words),
        "total_tokens": float(total_tokens),
        "avg_tokens_per_word": float(avg_tokens_per_word),
        "word_as_token_rate": float(word_as_token_rate),
        "merge_use_rate": float(merge_use_rate),
        "type_compression": float(type_compression),
    }


############# TASK 2 ######################

# Task 2 specific tokens
EOS = "<eos>"
BOS = "<bos>"


def words_from_text(text: str, lowercase: bool = True) -> List[str]:
    if lowercase:
        text = text.lower()
    return [w for w in _wsre.split(text.strip()) if w]

def tokenize_lines_with_merges(text: str, merges: List[Tuple[str, str]]) -> List[List[str]]:
    """Convert text to line-based token sequences for n-gram training."""
    token_lines: List[List[str]] = []
    for line in text.strip().splitlines():
        words = words_from_text(line)
        if not words:
            continue
        toks: List[str] = []
        for w in words:
            toks.extend(apply_merges_to_word(w, merges))
        toks.append(EOS)
        token_lines.append(toks)
    return token_lines

def load_merges(merges_path: str) -> List[Tuple[str, str]]:
    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                merges.append((parts[0], parts[1]))
    return merges

# DATA LOADING FUNCTIONS

def find_merges_file(k: int, corpus, verbose: bool = True) -> str:
    """Find BPE merges file with flexible naming conventions."""
    candidates = [
        os.path.join(corpus, f"bpe_merges with k = {k}.txt"),
        os.path.join(corpus, f"standard_bpe_merges_k{k}.txt"),
        os.path.join(corpus, f"aggressive_clean_bpe_merges_k{k}.txt"),
        os.path.join(corpus, f"bpe_merges_k{k}.txt"),
        os.path.join(corpus, f"bpe_merges_k{k}_webtext_clean.txt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            if verbose:
                print(f"[Found] Using merges file: {path}")
            return path
    raise FileNotFoundError(f"No merges file found for k={k}. Tried: {candidates}")

def load_token_lines_for_k(k: int, corpus, train, valid, test):
    """Load and tokenize train/validation/test data for given k."""
    merges_path = find_merges_file(k, corpus, verbose=True)
    merges = load_merges(merges_path)

    tr_text = read_text(train)
    va_text = read_text(valid)
    te_text = read_text(test)

    tr_tok = tokenize_lines_with_merges(tr_text, merges)
    va_tok = tokenize_lines_with_merges(va_text, merges)
    te_tok = tokenize_lines_with_merges(te_text, merges)
    return merges, tr_tok, va_tok, te_tok