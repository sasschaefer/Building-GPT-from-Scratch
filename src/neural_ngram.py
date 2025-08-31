"""
ngram_hard_embedding.py

Pure-NumPy neural n-gram model with trainable embeddings, optimizer (SGD/Adam),
early stopping, checkpointing (top-k), perplexity measurement, and conditional generation.

Usage: import functions or run the example in __main__.
"""
import os
import json
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

# --------------------------
# Utilities
# --------------------------
def softmax_stable(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    logits: (B, V)
    targets: (B,) int
    returns: (loss float, probs (B, V))
    """
    probs = softmax_stable(logits)
    eps = 1e-12
    nll = -np.log(probs[np.arange(len(targets)), targets] + eps)
    return float(nll.mean()), probs

# --------------------------
# Model
# --------------------------
class NeuralNGramHard:
    """
    Neural n-gram: uses embedding matrix E and an MLP (one hidden layer) to predict next token.
    Context length = n-1. Input to MLP is flattened embeddings of the n-1 context tokens.
    Pure NumPy implementation.
    """
    def __init__(self, vocab_size:int, n:int, embd:int, hidden:int, seed:int=1337):
        assert n >= 2
        self.vocab_size = int(vocab_size)
        self.n = int(n)
        self.embd = int(embd)
        self.hidden = int(hidden)
        rng = np.random.RandomState(seed)

        # Parameters
        self.E = rng.normal(0, 0.02, size=(self.vocab_size, self.embd)).astype(np.float32)   # (V, embd)
        self.W1 = rng.normal(0, 0.02, size=((self.n-1)*self.embd, self.hidden)).astype(np.float32)  # ((n-1)*embd, hidden)
        self.b1 = np.zeros((self.hidden,), dtype=np.float32)
        self.W2 = rng.normal(0, 0.02, size=(self.hidden, self.vocab_size)).astype(np.float32) # (hidden, V)
        self.b2 = np.zeros((self.vocab_size,), dtype=np.float32)

        # grads (same shapes)
        self.grad_E = np.zeros_like(self.E)
        self.grad_W1 = np.zeros_like(self.W1)
        self.grad_b1 = np.zeros_like(self.b1)
        self.grad_W2 = np.zeros_like(self.W2)
        self.grad_b2 = np.zeros_like(self.b2)

    def forward(self, X_ids: np.ndarray):
        """
        X_ids: (B, n-1) int
        returns: logits (B, V), cache dict
        """
        X = np.asarray(X_ids, dtype=np.int64)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        B = X.shape[0]
        h0 = self.E[X].reshape(B, -1)                 # (B, (n-1)*embd)
        z1 = h0 @ self.W1 + self.b1                   # (B, hidden)
        a1 = np.maximum(0, z1)                        # ReLU
        logits = a1 @ self.W2 + self.b2               # (B, V)
        cache = {"X": X, "h0": h0, "z1": z1, "a1": a1}
        return logits, cache

    def backward(self, cache: dict, d_logits: np.ndarray):
        """
        d_logits: (B, V) gradient of loss w.r.t logits
        Fills gradient containers grad_*
        """
        X = cache["X"]          # (B, n-1)
        h0 = cache["h0"]        # (B, (n-1)*embd)
        z1 = cache["z1"]
        a1 = cache["a1"]

        # W2, b2
        self.grad_W2 = a1.T @ d_logits                  # (hidden, V)
        self.grad_b2 = d_logits.sum(axis=0).astype(np.float32)  # (V,)

        # backprop to hidden
        dh1 = d_logits @ self.W2.T                       # (B, hidden)
        dz1 = dh1 * (z1 > 0).astype(np.float32)         # ReLU grad

        # W1, b1
        self.grad_W1 = h0.T @ dz1                       # ((n-1)*embd, hidden)
        self.grad_b1 = dz1.sum(axis=0).astype(np.float32)   # (hidden,)

        # to embeddings
        dh0 = dz1 @ self.W1.T                           # (B, (n-1)*embd)
        dh0 = dh0.reshape(X.shape[0], self.n-1, self.embd)  # (B, n-1, embd)

        self.grad_E.fill(0.0)
        # accumulate per-index (handles repeated indices)
        for i in range(self.n - 1):
            # np.add.at supports broadcasting for dh0[:,i,:]
            np.add.at(self.grad_E, X[:, i], dh0[:, i, :])

    # ---------------------------
    # Parameter update helpers (SGD, Adam)
    # ---------------------------
    def _sgd_step(self, lr: float):
        self.E  -= lr * self.grad_E
        self.W1 -= lr * self.grad_W1
        self.b1 -= lr * self.grad_b1
        self.W2 -= lr * self.grad_W2
        self.b2 -= lr * self.grad_b2

    def _init_adam_buffers(self):
        # Lazy init for Adam moments
        if not hasattr(self, "_adam_m"):
            self._adam_m = {n: np.zeros_like(getattr(self, n)) for n in ["E","W1","b1","W2","b2"]}
            self._adam_v = {n: np.zeros_like(getattr(self, n)) for n in ["E","W1","b1","W2","b2"]}
            self._adam_t = 0

    def _adam_step(self, lr: float, beta1=0.9, beta2=0.999, eps=1e-8):
        self._init_adam_buffers()
        self._adam_t += 1
        t = self._adam_t
        for name in ["E","W1","b1","W2","b2"]:
            g = getattr(self, "grad_" + name)
            m = self._adam_m[name]; v = self._adam_v[name]
            m[:] = beta1 * m + (1 - beta1) * g
            v[:] = beta2 * v + (1 - beta2) * (g * g)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            getattr(self, name)[:] -= update

    # Generic step
    def step(self, optimizer: str, lr: float, **opt_kwargs):
        if optimizer.lower() == "sgd":
            self._sgd_step(lr)
        elif optimizer.lower() == "adam":
            self._adam_step(lr, **opt_kwargs)
        else:
            raise ValueError("Unknown optimizer: " + str(optimizer))

    # ---------------------------
    # Train / eval helpers
    # ---------------------------
    def train_batch(self, X_batch: np.ndarray, Y_batch: np.ndarray, lr=0.1, optimizer="sgd"):
        logits, cache = self.forward(X_batch)
        loss, probs = cross_entropy_loss(logits, Y_batch)

        # gradient wrt logits (average over batch)
        d_logits = probs.copy()
        d_logits[np.arange(len(Y_batch)), Y_batch] -= 1.0
        d_logits /= float(len(Y_batch))

        self.backward(cache, d_logits)
        self.step(optimizer, lr)
        return loss

    def eval_loss(self, ids: np.ndarray, n:int, batch_size:int=256):
        """
        ids: 1D array of token ids; evaluate loss over sliding windows (n-1 -> predict nth)
        """
        ids = np.asarray(ids, dtype=np.int64)
        N = len(ids) - (n-1)
        if N <= 0:
            raise ValueError("Sequence too short for evaluation")
        total_loss = 0.0
        num_batches = 0
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_idx = np.arange(start, end)
            X_ids = np.stack([ids[i:i+n-1] for i in batch_idx])
            Y_ids = np.array([ids[i+n-1] for i in batch_idx], dtype=np.int64)
            logits, _ = self.forward(X_ids)
            loss, _ = cross_entropy_loss(logits, Y_ids)
            total_loss += loss
            num_batches += 1
        return total_loss / max(1, num_batches)

    def forward_probs(self, X_batch: np.ndarray):
        logits, _ = self.forward(X_batch)
        return softmax_stable(logits)

    def save(self, path: str):
        np.savez(path, E=self.E, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path: str):
        d = np.load(path)
        self.E = d["E"].astype(np.float32)
        self.W1 = d["W1"].astype(np.float32)
        self.b1 = d["b1"].astype(np.float32)
        self.W2 = d["W2"].astype(np.float32)
        self.b2 = d["b2"].astype(np.float32)

    # ---------------------------
    # Conditional generation
    # ---------------------------
    def generate_next(self, context_ids: List[int], top_k: Optional[int]=None, temperature: float=1.0):
        """
        Given a context list of length n-1, return a sampled next id.
        - Greedy if top_k==1
        - If top_k is None: sample from full distribution
        """
        assert len(context_ids) == self.n - 1
        X = np.asarray(context_ids, dtype=np.int64)[None, :]
        probs = self.forward_probs(X)[0]  # (V,)
        if temperature != 1.0:
            # adjust temperature
            logits = np.log(np.maximum(probs, 1e-12)) / temperature
            probs = np.exp(logits - logits.max()); probs /= probs.sum()
        if top_k is not None:
            if top_k <= 1:
                return int(np.argmax(probs))
            else:
                top_idx = np.argpartition(-probs, top_k-1)[:top_k]
                top_probs = probs[top_idx]
                top_probs = top_probs / top_probs.sum()
                choice = np.random.choice(len(top_idx), p=top_probs)
                return int(top_idx[choice])
        else:
            return int(np.random.choice(len(probs), p=probs))

# --------------------------
# Training loop with early stopping & checkpointing
# --------------------------
def train_with_early_stopping(model: NeuralNGramHard, train_ids: np.ndarray, valid_ids: np.ndarray,
                              n:int, batch_size:int=128, lr:float=0.1, epochs:int=10,
                              optimizer:str="sgd", patience:int=3, top_k_checkpoints:int=3,
                              eval_every_steps:int=200, seed:int=0):
    """
    Train model on train_ids (1D array of token ids), evaluate on valid_ids.
    Saves top_k_checkpoints best models by validation loss in ./checkpoints/ as .npz
    Returns training history (list of dicts).
    """
    np.random.seed(seed)
    os.makedirs("checkpoints", exist_ok=True)
    N_train = len(train_ids) - (n-1)
    steps_per_epoch = max(1, N_train // batch_size)

    # helper to iterate minibatches (randomized)
    def batch_generator(ids_arr):
        N = len(ids_arr) - (n-1)
        idxs = np.arange(0, N)
        np.random.shuffle(idxs)
        for i in range(0, len(idxs), batch_size):
            batch_idx = idxs[i:i+batch_size]
            X = np.stack([ids_arr[j:j+n-1] for j in batch_idx])
            Y = np.array([ids_arr[j+n-1] for j in batch_idx], dtype=np.int64)
            yield X, Y

    history = []
    best_val = float("inf")
    best_list: List[Tuple[float, str]] = []   # (val_loss, path)
    patience_counter = 0
    step = 0

    for epoch in range(1, epochs+1):
        # training sweep
        for X_batch, Y_batch in batch_generator(train_ids):
            step += 1
            loss = model.train_batch(X_batch, Y_batch, lr=lr, optimizer=optimizer)

            if step % eval_every_steps == 0:
                val_loss = model.eval_loss(valid_ids, n=n, batch_size=batch_size)
                train_loss = model.eval_loss(train_ids, n=n, batch_size=batch_size//2 if batch_size>1 else 1)
                train_ppl = float(np.exp(train_loss))
                val_ppl = float(np.exp(val_loss))
                print(f"[step {step}] train_loss={train_loss:.4f} train_ppl={train_ppl:.2f}  val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")

                history.append({
                    "step": step, "epoch": epoch, "train_loss": train_loss, "train_ppl": train_ppl,
                    "val_loss": val_loss, "val_ppl": val_ppl
                })

                # checkpoint top-K by val_loss
                ckpt_name = f"checkpoints/model_step{step}_val{val_loss:.6f}.npz"
                model.save(ckpt_name)
                best_list.append((val_loss, ckpt_name))
                best_list = sorted(best_list, key=lambda x: x[0])[:top_k_checkpoints]
                # remove older ones not in best_list
                valid_paths = set(p for _, p in best_list)
                for _, p in list(best_list):
                    pass
                # prune other files in checkpoints directory (safe: delete those not in valid_paths)
                for fname in os.listdir("checkpoints"):
                    fpath = os.path.join("checkpoints", fname)
                    if fpath not in valid_paths and fname.endswith(".npz"):
                        try:
                            os.remove(fpath)
                        except Exception:
                            pass

                # early stopping logic
                if val_loss < best_val - 1e-12:
                    best_val = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"  patience {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    return history

    return history

# --------------------------
# Notes on RAM and batch sizing
# --------------------------
RAM_NOTE = """
Notes:
- The model keeps embeddings of size V x embd in RAM. For vocab 50k and embd 256,
  that's ~50_000 * 256 * 4 bytes ≈ 50 MB (float32) — embeddings often dominate memory.
- Batch size increases temporary arrays in forward/backward; keep batch sizes moderate
  (32 or less) if RAM is limited. For pure NumPy training on CPU, 16 is a safe starting point.
"""

# --------------------------
# Example usage (toy)
# --------------------------
if __name__ == "__main__":
    # toy vocab & random ids for a quick run
    V = 2000
    n = 4
    embd = 64
    hidden = 256

    rng = np.random.RandomState(42)
    # create random token stream
    tokens = rng.randint(0, V, size=20000, dtype=np.int64)
    # split into train/valid
    split = int(0.9 * len(tokens))
    train_ids = tokens[:split]
    valid_ids = tokens[split:]

    model = NeuralNGramHard(vocab_size=V, n=n, embd=embd, hidden=hidden, seed=0)

    history = train_with_early_stopping(model, train_ids, valid_ids, n=n,
                                        batch_size=64, lr=0.05, epochs=5,
                                        optimizer="adam", patience=3,
                                        top_k_checkpoints=2, eval_every_steps=200, seed=0)

    print("Done. History entries:", len(history))
    print(RAM_NOTE)
