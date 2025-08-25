import random
import os, re, glob, shutil
from collections import Counter
from typing import Dict, List, Tuple, Optional, Iterable
import pandas as pd
from IPython.display import display
import matplotlib
import matplotlib.pyplot as plt
import time, math
import data_utils as du
import bpe as bpe
import ngram as ngram


def flatten_for_eval(token_lines: List[List[str]]) -> List[str]:
    """Flatten token lines for evaluation (BOS handling done in perplexity)."""
    flat: List[str] = []
    for line in token_lines:
        flat.extend(line)
    return flat

def perplexity(model: ngram.NGramLM, token_lines: List[List[str]], mode: str = "laplace",
               lambdas: Optional[List[float]] = None) -> float:
    """Calculate perplexity following slide methodology exactly."""
    log_prob_sum = 0.0
    count = 0

    for line in token_lines:
        # Initialize context with BOS tokens at start of each sentence
        context = [bpe.BOS] * (model.n - 1) if model.n > 1 else []

        # Process each token in the line (including EOS)
        for token in line:
            # Create context tuple for probability calculation
            ctx = tuple(context[-(model.n-1):]) if model.n > 1 else tuple()

            # Calculate probability based on smoothing method
            if mode == "ml":
                p = model.p_ml(ctx, token)
            elif mode == "laplace":
                p = model.p_laplace(ctx, token)
            elif mode == "interp":
                p = model.p_interpolated(ctx, token, lambdas=lambdas, use_laplace=True)
            elif mode == "backoff":
                p = model.p_backoff(ctx, token)
            elif mode == "katz":
                p = model.p_backoff_katz(ctx, token)
            else:
                raise ValueError("mode must be one of: ml, laplace, interp, backoff, katz")

            # Add log probability
            if p > 0:
                log_prob_sum += math.log(p)
            else:
                log_prob_sum += float('-inf')
            count += 1

            # Update context window
            context = (context + [token])[-(model.n - 1):] if model.n > 1 else context

            # Reset context at sentence boundary
            if token == bpe.EOS:
                context = [bpe.BOS] * (model.n - 1) if model.n > 1 else []

    # Calculate final perplexity
    avg_log_prob = log_prob_sum / count if count > 0 else float('-inf')
    return math.exp(-avg_log_prob)



def grid_simplex_lambdas(n: int, step: float = 0.2) -> List[List[float]]:
    """Generate lambda weight combinations that sum to 1.0."""
    if n == 1:
        return [[1.0]]

    grids = []
    def rec(prefix, remaining, slots):
        if slots == 1:
            grids.append(prefix + [round(remaining, 10)])
            return
        t = 0.0
        while t <= remaining + 1e-9:
            rec(prefix + [round(t,10)], round(remaining - t,10), slots-1)
            t = round(t + step, 10)

    rec([], 1.0, n)
    return [g for g in grids if abs(sum(g) - 1.0) < 1e-6]

def train_and_eval_for_k(corpus, train, valid, test, k: int, n_max: int = 4, tune_interp: bool = True) -> pd.DataFrame:
    """Train and evaluate n-gram models for given BPE vocabulary size k."""
    print(f"\n=== Processing k={k} ===")
    try:
        _, tr_tok, va_tok, te_tok = du.load_token_lines_for_k(k, corpus, train, valid, test)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return pd.DataFrame()

    results = []
    for n in range(1, n_max+1):
        print(f"Training {n}-gram model...")
        lm = ngram.NGramLM(n, tr_tok)

        # Basic evaluations (ML, Laplace)
        pp_valid_ml = perplexity(lm, va_tok, mode="ml")
        pp_valid_laplace = perplexity(lm, va_tok, mode="laplace")
        pp_test_ml = perplexity(lm, te_tok, mode="ml")
        pp_test_laplace = perplexity(lm, te_tok, mode="laplace")

        results.extend([
            {"k":k, "n":n, "mode":"ml", "lambdas":"N/A", "split":"valid", "perplexity":pp_valid_ml},
            {"k":k, "n":n, "mode":"laplace", "lambdas":"N/A", "split":"valid", "perplexity":pp_valid_laplace},
            {"k":k, "n":n, "mode":"ml", "lambdas":"N/A", "split":"test", "perplexity":pp_test_ml},
            {"k":k, "n":n, "mode":"laplace", "lambdas":"N/A", "split":"test", "perplexity":pp_test_laplace},
        ])

        # Interpolation with lambda tuning on validation set
        if tune_interp and n > 1:
            print(f"Tuning interpolation for {n}-gram...")
            best_pp, best_lmb = float("inf"), None
            for lambdas in grid_simplex_lambdas(n=n, step=0.2):
                pp = perplexity(lm, va_tok, mode="interp", lambdas=lambdas)
                if pp < best_pp:
                    best_pp, best_lmb = pp, lambdas

            if best_lmb is not None:
                pp_test_interp = perplexity(lm, te_tok, mode="interp", lambdas=best_lmb)
                results.extend([
                    {"k":k, "n":n, "mode":"interp", "lambdas":str(best_lmb), "split":"valid", "perplexity":best_pp},
                    {"k":k, "n":n, "mode":"interp", "lambdas":str(best_lmb), "split":"test", "perplexity":pp_test_interp},
                ])

        # Backoff evaluation (Stupid Backoff implementation)
        pp_valid_backoff = perplexity(lm, va_tok, mode="backoff")
        pp_test_backoff = perplexity(lm, te_tok, mode="backoff")
        results.extend([
            {"k":k, "n":n, "mode":"backoff", "lambdas":"N/A", "split":"valid", "perplexity":pp_valid_backoff},
            {"k":k, "n":n, "mode":"backoff", "lambdas":"N/A", "split":"test", "perplexity":pp_test_backoff},
        ])

    return pd.DataFrame(results)

def bigram_vs_k(k_list: List[int], corpus, train, valid, test) -> pd.DataFrame:
    """Analyze bigram performance across different k values."""
    rows = []
    for k in k_list:
        merges_path = os.path.join(corpus, f"bpe_merges with k = {k}.txt")
        if not os.path.exists(merges_path):
            print(f"[Skip] No merges for k={k} at {merges_path}")
            continue
        dfk = train_and_eval_for_k(corpus, train, valid, test, k=k, n_max=2, tune_interp=True)  # n=2 only
        rows.append(dfk[dfk["n"] == 2])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()




# TEXT GENERATION (Extrinsic Evaluation)

def bpe_encode_words(words: Iterable[str], merges: List[Tuple[str, str]]) -> List[str]:
    """Lowercase and convert words to BPE subwords (including </w>)."""
    toks: List[str] = []
    for w in (w.lower() for w in words):
        toks.extend(du.apply_merges_to_word(w, merges))
    return toks

def bpe_decode_to_words(token_stream: List[str]) -> List[str]:
    """Convert BPE subword list back to words (split at </w>)."""
    words: List[str] = []
    buf: List[str] = []
    for t in token_stream:
        if t == bpe.EOS:
            break
        buf.append(t)
        if t.endswith(du.WORD_END):  # word boundary
            chars: List[str] = []
            for sub in buf:
                if sub.endswith(du.WORD_END):
                    chars.extend(list(sub[:-len(du.WORD_END)]))
                else:
                    chars.extend(list(sub))
            words.append("".join(chars))
            buf = []
    # Handle any leftover fragments
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

def _get_unigram_model(model: ngram.NGramLM) -> ngram.NGramLM:
    """Return the base unigram model (n=1) from an n-gram chain."""
    m = model
    while m.lower is not None:
        m = m.lower
    return m

def _unigram_fallback(model: ngram.NGramLM, strategy: str = "most") -> str:
    """
    Fallback token choice when distribution is empty:
      - 'most': most frequent unigram (excluding BOS/EOS)
      - 'avg' : token whose Laplace probability is closest to average unigram probability
    """
    um = _get_unigram_model(model)
    total = sum(um.ng_counts.values())
    V = um.V if hasattr(um, "V") else len(um.vocab)

    if strategy == "most":
        best_tok, best_count = None, -1
        for (tok_tuple, cnt) in um.ng_counts.items():
            t = tok_tuple[0]
            if t in (bpe.BOS, bpe.EOS):
                continue
            if cnt > best_count:
                best_count = cnt
                best_tok = t
        return best_tok if best_tok is not None else bpe.EOS

    else:  # 'avg' strategy
        avg_p = 1.0 / V  # Laplace average probability
        best_tok, best_gap = None, float("inf")
        for (tok_tuple, cnt) in um.ng_counts.items():
            t = tok_tuple[0]
            if t in (bpe.BOS, bpe.EOS):
                continue
            p = (cnt + 1) / (total + V)
            gap = abs(p - avg_p)
            if gap < best_gap:
                best_gap, best_tok = gap, t
        return best_tok if best_tok is not None else bpe.EOS

def _next_token_argmax_or_sample(dist: Dict[str, float],
                                 temperature: float = 1.0,
                                 sample: bool = False) -> str:
    """Choose next token from a probability distribution (argmax or temperature sampling)."""
    if not dist:
        return None
    if not sample:
        return max(dist.items(), key=lambda kv: kv[1])[0]
    tokens, probs = zip(*dist.items())
    probs = list(probs)
    if temperature <= 0:
        return tokens[int(max(range(len(probs)), key=lambda i: probs[i]))]
    if temperature != 1.0:
        probs = [p ** (1.0/temperature) for p in probs]
        Z = sum(probs) or 1.0
        probs = [p / Z for p in probs]
    return random.choices(tokens, weights=probs, k=1)[0]

def generate_sentence(
    corpus,
    train,
    valid,
    test,
    k: int,
    n: int,
    prompt_words: List[str],
    mode: str = "interp",
    lambdas: Optional[List[float]] = None,
    max_new_words: int = 30,
    temperature: float = 1.0,
    sample: bool = False,
    fallback_strategy: str = "most",   # 'most' or 'avg'
) -> str:
    """
    Extrinsic evaluation: generate continuation from a prompt using an n-gram model.
    - mode: 'ml' | 'laplace' | 'interp' | 'backoff' | 'katz'
    - sample=False → argmax; sample=True → temperature sampling
    - fallback: unigram choice ('most' or 'avg') if no distribution is found
    - Stops when EOS appears or max_new_words is reached
    """
    merges, tr_tok, _, _ = du.load_token_lines_for_k(corpus, train, valid, test, k)
    lm = ngram.NGramLM(n, tr_tok)

    # 1) Encode prompt to BPE tokens
    prompt_toks = bpe_encode_words(prompt_words, merges)

    # 2) Initialize context (BOS*(n-1) + prompt)
    context: List[str] = ([bpe.BOS] * (n - 1)) + prompt_toks if n > 1 else prompt_toks[:]

    out_tokens: List[str] = []
    words_generated = 0

    def _dist(ctx_tokens: List[str]) -> Dict[str, float]:
        ctx = tuple(ctx_tokens[-(lm.n - 1):]) if lm.n > 1 else tuple()
        probs: Dict[str, float] = {}
        for tok in lm.vocab:
            if tok == bpe.BOS:
                continue
            if mode == "ml":
                p = lm.p_ml(ctx, tok)
            elif mode == "laplace":
                p = lm.p_laplace(ctx, tok)
            elif mode == "interp":
                p = lm.p_interpolated(ctx, tok, lambdas=lambdas, use_laplace=True)
            elif mode == "backoff":
                p = lm.p_backoff(ctx, tok)
            elif mode == "katz":
                p = lm.p_backoff_katz(ctx, tok)
            else:
                raise ValueError("Invalid mode.")
            if p > 0:
                probs[tok] = p
        Z = sum(probs.values())
        if Z > 0:
            for t in probs:
                probs[t] /= Z
        return probs

    while words_generated < max_new_words:
        dist = _dist(context)

        if not dist:
            next_tok = _unigram_fallback(lm, strategy=fallback_strategy)
        else:
            next_tok = _next_token_argmax_or_sample(dist, temperature=temperature, sample=sample)

        out_tokens.append(next_tok)
        context.append(next_tok)

        if next_tok == bpe.EOS:
            break
        if next_tok.endswith(du.WORD_END):
            words_generated += 1

    return " ".join(bpe_decode_to_words(out_tokens))
