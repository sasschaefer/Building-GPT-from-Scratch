from collections import Counter
from typing import List, Tuple, Optional
import bpe as bpe

# N-GRAM BUILDING FUNCTIONS

def add_bos_context(line_tokens: List[str], n: int) -> List[str]:
    """Add beginning-of-sentence tokens for n-gram context."""
    if n <= 1:
        return line_tokens
    return [bpe.BOS] * (n-1) + line_tokens

def build_ngrams(token_lines: List[List[str]], n: int):
    """Build n-gram counts and vocabulary from tokenized lines."""
    vocab = set()
    ngram_counts = Counter()
    context_counts = Counter()

    for line in token_lines:
        line = add_bos_context(line, n)
        vocab.update(line)
        for i in range(n-1, len(line)):
            context = tuple(line[i-n+1:i])
            token   = line[i]
            ngram_counts[context + (token,)] += 1
            context_counts[context]          += 1

    return ngram_counts, context_counts, sorted(vocab)

 # N-GRAM LANGUAGE MODEL CLASS

class NGramLM:
    """N-gram Language Model with multiple smoothing techniques."""

    def __init__(self, n: int, token_lines: List[List[str]]):
        assert n >= 1
        self.n = n
        self.ng_counts, self.ctx_counts, self.vocab = build_ngrams(token_lines, n)
        self.V = len(self.vocab)
        self.lower = NGramLM(n-1, token_lines) if n > 1 else None

    def p_ml(self, context: Tuple[str, ...], token: str) -> float:
        """Maximum Likelihood probability estimation."""
        if self.n == 1:
            total = sum(self.ng_counts.values())
            return self.ng_counts.get((token,), 0) / max(1, total)
        c = self.ctx_counts.get(context, 0)
        if c == 0:
            return 0.0
        return self.ng_counts.get(context + (token,), 0) / c

    def p_laplace(self, context: Tuple[str, ...], token: str) -> float:
        """Laplace (add-one) smoothing probability estimation."""
        if self.n == 1:
            num = self.ng_counts.get((token,), 0) + 1
            den = sum(self.ng_counts.values()) + self.V
            return num / den
        c   = self.ctx_counts.get(context, 0)
        num = self.ng_counts.get(context + (token,), 0) + 1
        den = c + self.V
        return num / max(1, den)

    def p_interpolated(self, context: Tuple[str, ...], token: str,
                       lambdas: Optional[List[float]] = None, use_laplace: bool = True) -> float:
        """Linear interpolation of different n-gram orders."""
        if lambdas is None:
            lambdas = [1.0/self.n] * self.n
        assert len(lambdas) == self.n

        prob = 0.0
        current_model = self

        for order in range(self.n, 0, -1):
            if order == 1:
                p = current_model.p_laplace((), token) if use_laplace else current_model.p_ml((), token)
            else:
                need = order - 1
                if len(context) >= need:
                    ctx = context[-need:]
                else:
                    padding_needed = need - len(context)
                    ctx = tuple([bpe.BOS] * padding_needed) + context

                p = current_model.p_laplace(ctx, token) if use_laplace else current_model.p_ml(ctx, token)

            prob += lambdas[order-1] * p

            if current_model.lower is not None:
                current_model = current_model.lower

        return prob

    def p_backoff_katz(self, context: Tuple[str, ...], token: str) -> float:
        """Simplified Katz backoff (without Good-Turing discounting)."""
        if self.n == 1:
            return self.p_laplace((), token)

        need = self.n - 1
        if len(context) >= need:
            ctx = context[-need:]
        else:
            padding_needed = need - len(context)
            ctx = tuple([bpe.BOS] * padding_needed) + context

        c_ctx = self.ctx_counts.get(ctx, 0)
        c_ng = self.ng_counts.get(ctx + (token,), 0)

        if c_ng > 0:
            # Discounted ML estimate (simplified)
            discount = 0.75  # Simple absolute discounting
            prob_discounted = max(c_ng - discount, 0) / c_ctx
            return prob_discounted
        else:
            # Backoff with alpha weight
            alpha = 0.4  # Simplified backoff weight
            return alpha * self.lower.p_backoff_katz(context, token)

    def p_backoff(self, context: Tuple[str, ...], token: str) -> float:
        """Stupid Backoff (not a true probability distribution)."""
        if self.n == 1:
            return self.p_laplace((), token)

        need = self.n - 1
        if len(context) >= need:
            ctx = context[-need:]
        else:
            padding_needed = need - len(context)
            ctx = tuple([bpe.BOS] * padding_needed) + context

        c_ctx = self.ctx_counts.get(ctx, 0)
        c_ng = self.ng_counts.get(ctx + (token,), 0)

        if c_ctx > 0 and c_ng > 0:
            return c_ng / c_ctx  # ML estimate if seen
        else:
            # Backoff with penalty
            return 0.4 * self.lower.p_backoff(context, token)