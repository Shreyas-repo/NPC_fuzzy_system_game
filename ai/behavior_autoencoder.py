"""
Unsupervised Behavior Autoencoder — Neural feedback loop beneath the Fuzzy System.

Architecture:
  Encoder: Input(6) → Dense(12, ReLU) → Latent(4, ReLU)
  Decoder: Latent(4) → Dense(12, ReLU) → Output(6, Sigmoid)

Training signal (fully unsupervised):
  1. MSE reconstruction loss  — learns the manifold of normal NPC behavior
  2. Centroid-alignment loss  — latent vectors of same K-Means cluster pulled together

Outputs per NPC:
  novelty_score  (0..1)  — normalized reconstruction error; high = anomalous / novel behavior
  latent_vec     (4-dim) — compressed embedding; passed to ClusteringEngine for
                           latent-space K-Means (replaces raw 6-dim behavior vectors)

Feedback loop:
  Game Engine ──► ClusteringEngine ──► BehaviorAutoencoderManager.train_async()
                                                │
                              novelty_score ◄───┘
                                   │
                              FuzzySocialController.recommend(novelty_score=...)
                                   │ (explore_bias injected for anomalous NPCs)
                                   ▼
                              NPC Action Output
"""

import math
import os
import threading
import time

import numpy as np

# ── Config (with safe fallbacks) ──────────────────────────────────────────────
try:
    from config import AUTOENCODER_ENABLED
except ImportError:
    AUTOENCODER_ENABLED = True

try:
    from config import AUTOENCODER_LATENT_DIM
except ImportError:
    AUTOENCODER_LATENT_DIM = 4

try:
    from config import AUTOENCODER_HIDDEN_DIM
except ImportError:
    AUTOENCODER_HIDDEN_DIM = 12

try:
    from config import AUTOENCODER_LR
except ImportError:
    AUTOENCODER_LR = 0.003

try:
    from config import AUTOENCODER_EPOCHS
except ImportError:
    AUTOENCODER_EPOCHS = 5

try:
    from config import AUTOENCODER_BATCH_SIZE
except ImportError:
    AUTOENCODER_BATCH_SIZE = 16

try:
    from config import AUTOENCODER_CENTROID_WEIGHT
except ImportError:
    AUTOENCODER_CENTROID_WEIGHT = 0.3

try:
    from config import AUTOENCODER_NOVELTY_THRESHOLD
except ImportError:
    AUTOENCODER_NOVELTY_THRESHOLD = 0.55

try:
    from config import AUTOENCODER_WEIGHT_PATH
except ImportError:
    AUTOENCODER_WEIGHT_PATH = "behavior_autoencoder_weights.npz"

INPUT_DIM = 6   # [mood, energy, hunger, social_need, wealth, trust]
LATENT_DIM = int(AUTOENCODER_LATENT_DIM)
HIDDEN_DIM = int(AUTOENCODER_HIDDEN_DIM)


# ═══════════════════════════════════════════════════════════════════════════════
#  Pure-numpy activations
# ═══════════════════════════════════════════════════════════════════════════════

def _relu(x):
    return np.maximum(0.0, x)

def _relu_grad(x):
    return (x > 0).astype(np.float64)

def _sigmoid(x):
    # Numerically stable sigmoid
    pos = x >= 0
    result = np.where(pos,
                      1.0 / (1.0 + np.exp(-np.where(pos, x, 0))),
                      np.exp(np.where(~pos, x, 0)) / (1.0 + np.exp(np.where(~pos, x, 0))))
    return result

def _sigmoid_grad(x):
    s = _sigmoid(x)
    return s * (1.0 - s)

def _he_init(fan_in, fan_out):
    std = math.sqrt(2.0 / max(1, fan_in))
    return np.random.randn(fan_in, fan_out).astype(np.float64) * std

def _xavier_init(fan_in, fan_out):
    std = math.sqrt(2.0 / max(1, fan_in + fan_out))
    return np.random.randn(fan_in, fan_out).astype(np.float64) * std


# ═══════════════════════════════════════════════════════════════════════════════
#  BehaviorAutoencoder — pure-numpy encoder + decoder with Adam optimizer
# ═══════════════════════════════════════════════════════════════════════════════

class BehaviorAutoencoder:
    """
    Autoencoder over 6-dim NPC behavior vectors.

    Encoder: [6] → [HIDDEN, ReLU] → [LATENT, ReLU]
    Decoder: [LATENT] → [HIDDEN, ReLU] → [6, Sigmoid]
    """

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                 latent_dim=LATENT_DIM, lr=AUTOENCODER_LR):
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr         = float(lr)

        # ── Encoder weights ───────────────────────────────────────
        self.We1 = _he_init(input_dim, hidden_dim)
        self.be1 = np.zeros(hidden_dim, dtype=np.float64)
        self.We2 = _he_init(hidden_dim, latent_dim)
        self.be2 = np.zeros(latent_dim, dtype=np.float64)

        # ── Decoder weights ───────────────────────────────────────
        self.Wd1 = _he_init(latent_dim, hidden_dim)
        self.bd1 = np.zeros(hidden_dim, dtype=np.float64)
        self.Wd2 = _xavier_init(hidden_dim, input_dim)
        self.bd2 = np.zeros(input_dim, dtype=np.float64)

        # ── Adam state ────────────────────────────────────────────
        self._t = 0
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}

        # ── Per-sample running error stats (for novelty normalization) ──
        self._err_ema   = 0.05   # exponential moving average of reconstruction error
        self._err_std   = 0.05   # EMA standard deviation

    # ── Parameter dict ────────────────────────────────────────────────────────

    def _params(self):
        return {
            "We1": self.We1, "be1": self.be1,
            "We2": self.We2, "be2": self.be2,
            "Wd1": self.Wd1, "bd1": self.bd1,
            "Wd2": self.Wd2, "bd2": self.bd2,
        }

    # ── Forward passes ────────────────────────────────────────────────────────

    def encode(self, X):
        """Encoder forward. X: (batch, input_dim) → latent (batch, latent_dim)."""
        ze1 = X  @ self.We1 + self.be1
        ae1 = _relu(ze1)
        ze2 = ae1 @ self.We2 + self.be2
        ae2 = _relu(ze2)
        return ae2, (X, ze1, ae1, ze2, ae2)

    def decode(self, Z):
        """Decoder forward. Z: (batch, latent_dim) → reconstruction (batch, input_dim)."""
        zd1 = Z  @ self.Wd1 + self.bd1
        ad1 = _relu(zd1)
        zd2 = ad1 @ self.Wd2 + self.bd2
        out = _sigmoid(zd2)
        return out, (Z, zd1, ad1, zd2, out)

    def forward(self, X):
        """Full autoencoder pass. Returns (reconstruction, latent, enc_cache, dec_cache)."""
        latent, enc_cache = self.encode(X)
        recon,  dec_cache = self.decode(latent)
        return recon, latent, enc_cache, dec_cache

    def predict_single(self, x_vec):
        """
        Encode + decode a single 6-dim vector.
        Returns (reconstruction, latent_4dim, mse_error).
        """
        X      = np.array(x_vec, dtype=np.float64).reshape(1, -1)
        recon, latent, _, _ = self.forward(X)
        mse    = float(np.mean((recon[0] - X[0]) ** 2))
        return recon[0], latent[0], mse

    # ── Backward pass ─────────────────────────────────────────────────────────

    def backward(self, X, recon, latent, enc_cache, dec_cache,
                 centroid_targets=None, centroid_weight=0.0):
        """
        Compute gradients for:
          L_total = MSE(recon, X) + centroid_weight * MSE(latent, centroid_target)

        centroid_targets: (batch, latent_dim) — soft target latent positions
                          from K-Means cluster centroids (projected to latent space).
                          None → skip centroid alignment term.
        """
        X_fwd, ze1, ae1, ze2, ae2 = enc_cache
        Z_fwd, zd1, ad1, zd2, out = dec_cache
        batch = float(X.shape[0])

        # ── Decoder gradients (MSE reconstruction loss) ───────────
        d_recon       = (2.0 / batch) * (recon - X) * _sigmoid_grad(zd2)
        dWd2          = ad1.T @ d_recon / batch
        dbd2          = d_recon.mean(axis=0)
        d_ad1         = d_recon @ self.Wd2.T * _relu_grad(zd1)
        dWd1          = Z_fwd.T @ d_ad1 / batch
        dbd1          = d_ad1.mean(axis=0)

        # Gradient flows into latent from decoder
        d_latent      = d_ad1 @ self.Wd1.T

        # ── Centroid alignment term: pulls latent toward cluster centroid ──
        if centroid_targets is not None and centroid_weight > 0.0:
            d_centroid = (2.0 * centroid_weight / batch) * (latent - centroid_targets)
            d_latent   = d_latent + d_centroid

        # ── Encoder gradients ─────────────────────────────────────
        d_ae2         = d_latent * _relu_grad(ze2)
        dWe2          = ae1.T @ d_ae2 / batch
        dbe2          = d_ae2.mean(axis=0)
        d_ae1         = d_ae2 @ self.We2.T * _relu_grad(ze1)
        dWe1          = X_fwd.T @ d_ae1 / batch
        dbe1          = d_ae1.mean(axis=0)

        return {
            "We1": dWe1, "be1": dbe1,
            "We2": dWe2, "be2": dbe2,
            "Wd1": dWd1, "bd1": dbd1,
            "Wd2": dWd2, "bd2": dbd2,
        }

    # ── Adam optimizer ────────────────────────────────────────────────────────

    def _adam_step(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        params = self._params()
        for key in params:
            g = grads[key]
            # Gradient clipping
            norm = np.linalg.norm(g)
            if norm > 3.0:
                g = g * (3.0 / norm)
            self._m[key] = beta1 * self._m[key] + (1.0 - beta1) * g
            self._v[key] = beta2 * self._v[key] + (1.0 - beta2) * (g * g)
            m_hat = self._m[key] / (1.0 - beta1 ** self._t)
            v_hat = self._v[key] / (1.0 - beta2 ** self._t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def train_batch(self, X, centroid_targets=None, centroid_weight=0.0):
        """One training step. Returns MSE reconstruction loss."""
        recon, latent, enc_cache, dec_cache = self.forward(X)
        loss  = float(np.mean((recon - X) ** 2))
        grads = self.backward(X, recon, latent, enc_cache, dec_cache,
                              centroid_targets=centroid_targets,
                              centroid_weight=centroid_weight)
        self._adam_step(grads)
        return loss

    # ── Novelty scoring ───────────────────────────────────────────────────────

    def novelty_score(self, x_vec):
        """
        Compute normalized novelty score for a single NPC's behavior vector.
        Score = 1  → completely novel / anomalous behavior
        Score = 0  → very familiar behavior

        Uses EMA-normalized reconstruction error so the score adapts to
        the current distribution of errors across the population.
        """
        _, _, err = self.predict_single(x_vec)
        # Update running stats with exponential moving average
        alpha = 0.05
        self._err_ema = (1 - alpha) * self._err_ema + alpha * err
        diff          = abs(err - self._err_ema)
        self._err_std = (1 - alpha) * self._err_std + alpha * diff
        # Normalized score (softly bounded 0..1)
        threshold = self._err_ema + max(1e-6, 2.0 * self._err_std)
        score = min(1.0, err / max(1e-6, threshold))
        return float(score), float(err)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path):
        try:
            np.savez_compressed(
                path,
                We1=self.We1, be1=self.be1,
                We2=self.We2, be2=self.be2,
                Wd1=self.Wd1, bd1=self.bd1,
                Wd2=self.Wd2, bd2=self.bd2,
                err_ema=np.array([self._err_ema]),
                err_std=np.array([self._err_std]),
            )
        except Exception as e:
            print(f"  [Autoencoder] Warning: could not save weights: {e}")

    def load(self, path):
        if not os.path.exists(path):
            return False
        try:
            d = np.load(path)
            self.We1 = d["We1"]; self.be1 = d["be1"]
            self.We2 = d["We2"]; self.be2 = d["be2"]
            self.Wd1 = d["Wd1"]; self.bd1 = d["bd1"]
            self.Wd2 = d["Wd2"]; self.bd2 = d["bd2"]
            if "err_ema" in d:
                self._err_ema = float(d["err_ema"][0])
                self._err_std = float(d["err_std"][0])
            # Reset Adam state
            self._t = 0
            self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
            self._v = {k: np.zeros_like(v) for k, v in self._params().items()}
            return True
        except Exception as e:
            print(f"  [Autoencoder] Warning: could not load weights: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
#  BehaviorAutoencoderManager — high-level API for game engine integration
# ═══════════════════════════════════════════════════════════════════════════════

class BehaviorAutoencoderManager:
    """
    Manages the BehaviorAutoencoder lifecycle:
    - Loads / saves persistent weights
    - Trains asynchronously after every clustering cycle
    - Exposes per-NPC novelty scores and latent embeddings
    - Provides stats for the research metrics HUD
    """

    def __init__(self):
        self.enabled         = bool(AUTOENCODER_ENABLED)
        self.net             = BehaviorAutoencoder(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            latent_dim=LATENT_DIM,
            lr=AUTOENCODER_LR,
        )
        self.weight_path      = os.path.abspath(str(AUTOENCODER_WEIGHT_PATH))
        self.epochs           = max(1, int(AUTOENCODER_EPOCHS))
        self.batch_size       = max(4, int(AUTOENCODER_BATCH_SIZE))
        self.centroid_weight  = float(AUTOENCODER_CENTROID_WEIGHT)
        self.novelty_threshold = float(AUTOENCODER_NOVELTY_THRESHOLD)

        # State
        self._lock            = threading.Lock()
        self._training        = False
        self._last_loss       = 0.0
        self._train_count     = 0
        self._last_latent     = {}       # npc_name → latent_vec (4-dim)
        self._last_novelty    = {}       # npc_name → novelty_score (0..1)
        self._last_raw_error  = {}       # npc_name → raw MSE

        if self.enabled:
            loaded = self.net.load(self.weight_path)
            if loaded:
                print("  🔵 Behavior Autoencoder weights loaded from disk")
            else:
                print("  🔵 Behavior Autoencoder initialized (fresh weights)")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_novelty_score(self, npc):
        """
        Return the latest novelty score for an NPC (cached from last training cycle).
        Falls back to a live inference if not yet computed.
        """
        if not self.enabled:
            return 0.0

        name = getattr(npc, "name", None)
        if name and name in self._last_novelty:
            return self._last_novelty[name]

        # Live fallback: encode directly
        try:
            x = self._npc_to_vec(npc)
            score, _ = self.net.novelty_score(x)
            return score
        except Exception:
            return 0.0

    def get_latent_embedding(self, npc):
        """Return latest latent embedding (4-dim) for an NPC, or None."""
        name = getattr(npc, "name", None)
        if name and name in self._last_latent:
            return self._last_latent[name]
        return None

    def get_all_latent_embeddings(self, npcs):
        """
        Return (latent_matrix, valid_npcs) where latent_matrix is (N, LATENT_DIM).
        Only includes NPCs for which embeddings are available.
        If embeddings are not ready yet, returns raw behavior vectors as fallback.
        """
        if not self.enabled:
            return None, []

        vecs   = []
        valid  = []
        for npc in npcs:
            lat = self.get_latent_embedding(npc)
            if lat is not None:
                vecs.append(lat)
                valid.append(npc)

        if len(vecs) == 0:
            return None, []

        return np.array(vecs, dtype=np.float64), valid

    def train_async(self, npcs, cluster_labels=None, cluster_centers_raw=None):
        """
        Trigger async background training after a clustering cycle.

        npcs:                list of NPC objects
        cluster_labels:      list of int cluster IDs (same order as npcs)
        cluster_centers_raw: np.ndarray (n_clusters, 6) — K-Means centroids in raw space
        """
        if not self.enabled or self._training:
            return
        if len(npcs) < 3:
            return

        # Snapshot data for the worker thread (avoid capturing live NPC refs)
        X_snap     = np.array([self._npc_to_vec(npc) for npc in npcs], dtype=np.float64)
        names_snap = [getattr(npc, "name", str(i)) for i, npc in enumerate(npcs)]
        labels_snap    = list(cluster_labels) if cluster_labels is not None else None
        centers_snap   = (np.array(cluster_centers_raw, dtype=np.float64)
                          if cluster_centers_raw is not None else None)

        def _worker():
            with self._lock:
                self._training = True
                try:
                    self._do_training(X_snap, names_snap, labels_snap, centers_snap)
                except Exception as e:
                    print(f"  [Autoencoder] Training error: {e}")
                finally:
                    self._training = False

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    # ── Internal training logic ───────────────────────────────────────────────

    def _do_training(self, X, names, cluster_labels, cluster_centers_raw):
        """
        1. Train the autoencoder on current population behavior vectors
        2. Compute latent embeddings and novelty scores for all NPCs
        3. Optionally use cluster centroid projections as alignment targets
        """
        n = X.shape[0]

        # ── Build centroid target matrix in latent space ───────────────────
        # Strategy: encode each cluster centroid (in raw space) to get a
        # target latent position; then assign that target to all NPCs in
        # the cluster so the latent space clusters match K-Means structure.
        centroid_targets_all = None
        if cluster_labels is not None and cluster_centers_raw is not None:
            try:
                # Encode raw centroids → latent centroid targets
                C               = np.clip(cluster_centers_raw, 0.0, 1.0)
                lat_centroids, _= self.net.encode(C)
                # Build per-sample target matrix
                targets = np.zeros((n, LATENT_DIM), dtype=np.float64)
                for i, lbl in enumerate(cluster_labels):
                    if lbl >= 0 and lbl < len(lat_centroids):
                        targets[i] = lat_centroids[lbl]
                    else:
                        # Outlier: use own encoding as target (no pull)
                        _, (_, _, _, _, ae2) = self.net.encode(X[i:i+1])
                        targets[i] = ae2[0]
                centroid_targets_all = targets
            except Exception:
                centroid_targets_all = None

        # ── Mini-batch training for several epochs ─────────────────────────
        total_loss = 0.0
        steps      = 0
        indices    = np.arange(n)

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            C_shuffled = (centroid_targets_all[indices]
                          if centroid_targets_all is not None else None)

            for start in range(0, n, self.batch_size):
                end     = min(start + self.batch_size, n)
                X_batch = X_shuffled[start:end]
                C_batch = (C_shuffled[start:end]
                           if C_shuffled is not None else None)
                loss    = self.net.train_batch(
                    X_batch,
                    centroid_targets=C_batch,
                    centroid_weight=self.centroid_weight,
                )
                total_loss += loss
                steps      += 1

        self._last_loss = total_loss / max(1, steps)
        self._train_count += 1

        # ── Compute per-NPC novelty scores & latent embeddings ────────────
        try:
            latent_all, _ = self.net.encode(X)
            recon_all, _  = self.net.decode(latent_all)
            errors        = np.mean((recon_all - X) ** 2, axis=1)  # (n,)

            # Normalize errors to [0,1] using population statistics
            err_mean = float(np.mean(errors))
            err_std  = float(np.std(errors)) + 1e-6
            for i, name in enumerate(names):
                raw_err    = float(errors[i])
                # Soft z-score normalization, clamped to [0,1]
                z_score    = (raw_err - err_mean) / (2.0 * err_std)
                nov_score  = float(np.clip(0.5 + z_score, 0.0, 1.0))
                self._last_novelty[name]   = nov_score
                self._last_latent[name]    = latent_all[i].copy()
                self._last_raw_error[name] = raw_err
        except Exception:
            pass

        # ── Save weights ──────────────────────────────────────────────────
        self.net.save(self.weight_path)

    # ── Feature extraction ────────────────────────────────────────────────────

    @staticmethod
    def _npc_to_vec(npc):
        """Convert NPC state to 6-dim behavior vector, clipped to [0,1]."""
        bv    = getattr(npc, "behavior_vector", {})
        needs = getattr(npc, "needs", {})
        vec   = np.array([
            float(bv.get("mood",      0.5)),
            float(needs.get("energy", 0.5)),
            float(needs.get("hunger", 0.3)),
            float(needs.get("social_need", 0.4)),
            float(bv.get("wealth",    0.5)),
            float(bv.get("trust",     0.5)),
        ], dtype=np.float64)
        return np.clip(vec, 0.0, 1.0)

    # ── Stats for HUD / research metrics ─────────────────────────────────────

    def get_stats(self):
        return {
            "enabled":         self.enabled,
            "train_count":     self._train_count,
            "last_loss":       round(self._last_loss, 6),
            "training_active": self._training,
            "n_novelty_scores": len(self._last_novelty),
            "avg_novelty":     (round(float(np.mean(list(self._last_novelty.values()))), 4)
                                if self._last_novelty else 0.0),
            "max_novelty":     (round(float(max(self._last_novelty.values())), 4)
                                if self._last_novelty else 0.0),
        }

    def get_npc_stats(self, npc):
        """Per-NPC stats dict for HUD display."""
        name = getattr(npc, "name", "")
        return {
            "novelty_score": self._last_novelty.get(name, 0.0),
            "latent_norm":   (float(np.linalg.norm(self._last_latent[name]))
                              if name in self._last_latent else 0.0),
            "recon_error":   self._last_raw_error.get(name, 0.0),
        }
