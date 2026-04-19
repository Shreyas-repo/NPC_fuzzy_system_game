"""
Neural Dialogue Network — a lightweight feedforward neural network that learns
from conversation CSV logs and feeds predictions back into the fuzzy decision
system for more human-like NPC behaviour.

Architecture:
  Input (15 features) → Dense(32, ReLU) → Dense(16, ReLU) → Output (8 values)

Output heads:
  [0:6]  — Action weight biases for the fuzzy controller (eat, sleep, socialize, work, flee, guard)
  [6]    — Warmth score (−1..1): how positive/negative the NPC response tone should be
  [7]    — Predicted sentiment shift: expected change in NPC mood after this exchange

Training signal comes from the conversation CSV: the logged sentiment is the
ground truth, and the NPC's state at conversation time provides input features.

The network trains in a background thread every N new rows, saves weights
to disk for cross-session persistence, and exposes a simple `predict()` method
that the fuzzy controller and dialogue generator call each tick / conversation.
"""
import csv
import json
import math
import os
import random
import threading
import time

import numpy as np

# ── Configuration defaults (overridden by config.py imports if available) ──
_DEFAULTS = {
    "NEURAL_ENABLED": True,
    "NEURAL_HIDDEN_1": 32,
    "NEURAL_HIDDEN_2": 16,
    "NEURAL_LR": 0.005,
    "NEURAL_BATCH_SIZE": 32,
    "NEURAL_TRAIN_EVERY_N": 20,
    "NEURAL_EPOCHS_PER_TRAIN": 3,
    "NEURAL_WEIGHT_PATH": "neural_dialogue_weights.npz",
    "NEURAL_WARMTH_INFLUENCE": 0.35,
    "NEURAL_ACTION_INFLUENCE": 0.18,
    "CONVERSATION_CSV_PATH": "conversation_logs.csv",
}

# Try importing from config.py; fall back to defaults.
try:
    from config import (
        CONVERSATION_CSV_PATH,
    )
except ImportError:
    CONVERSATION_CSV_PATH = _DEFAULTS["CONVERSATION_CSV_PATH"]

# Neural-specific config (added to config.py or use defaults).
try:
    from config import NEURAL_ENABLED
except ImportError:
    NEURAL_ENABLED = _DEFAULTS["NEURAL_ENABLED"]
try:
    from config import NEURAL_HIDDEN_1
except ImportError:
    NEURAL_HIDDEN_1 = _DEFAULTS["NEURAL_HIDDEN_1"]
try:
    from config import NEURAL_HIDDEN_2
except ImportError:
    NEURAL_HIDDEN_2 = _DEFAULTS["NEURAL_HIDDEN_2"]
try:
    from config import NEURAL_LR
except ImportError:
    NEURAL_LR = _DEFAULTS["NEURAL_LR"]
try:
    from config import NEURAL_BATCH_SIZE
except ImportError:
    NEURAL_BATCH_SIZE = _DEFAULTS["NEURAL_BATCH_SIZE"]
try:
    from config import NEURAL_TRAIN_EVERY_N
except ImportError:
    NEURAL_TRAIN_EVERY_N = _DEFAULTS["NEURAL_TRAIN_EVERY_N"]
try:
    from config import NEURAL_EPOCHS_PER_TRAIN
except ImportError:
    NEURAL_EPOCHS_PER_TRAIN = _DEFAULTS["NEURAL_EPOCHS_PER_TRAIN"]
try:
    from config import NEURAL_WEIGHT_PATH
except ImportError:
    NEURAL_WEIGHT_PATH = _DEFAULTS["NEURAL_WEIGHT_PATH"]
try:
    from config import NEURAL_WARMTH_INFLUENCE
except ImportError:
    NEURAL_WARMTH_INFLUENCE = _DEFAULTS["NEURAL_WARMTH_INFLUENCE"]
try:
    from config import NEURAL_ACTION_INFLUENCE
except ImportError:
    NEURAL_ACTION_INFLUENCE = _DEFAULTS["NEURAL_ACTION_INFLUENCE"]

INPUT_DIM = 15
OUTPUT_DIM = 8
ACTION_KEYS = ["eat", "sleep", "socialize", "work", "flee", "guard"]


# ═══════════════════════════════════════════════════════════════════════
#  Pure-numpy feedforward network with backpropagation
# ═══════════════════════════════════════════════════════════════════════

def _relu(x):
    return np.maximum(0.0, x)

def _relu_grad(x):
    return (x > 0).astype(np.float64)

def _tanh(x):
    return np.tanh(x)

def _tanh_grad(x):
    t = np.tanh(x)
    return 1.0 - t * t

def _he_init(fan_in, fan_out):
    """He (Kaiming) initialization for ReLU layers."""
    std = math.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out).astype(np.float64) * std


class NeuralDialogueNet:
    """Three-layer feedforward network: Input → H1(ReLU) → H2(ReLU) → Output(tanh)."""

    def __init__(self, h1=NEURAL_HIDDEN_1, h2=NEURAL_HIDDEN_2, lr=NEURAL_LR):
        self.lr = float(lr)
        self.h1 = int(h1)
        self.h2 = int(h2)

        # Layer weights and biases
        self.W1 = _he_init(INPUT_DIM, self.h1)
        self.b1 = np.zeros(self.h1, dtype=np.float64)
        self.W2 = _he_init(self.h1, self.h2)
        self.b2 = np.zeros(self.h2, dtype=np.float64)
        self.W3 = _he_init(self.h2, OUTPUT_DIM)
        self.b3 = np.zeros(OUTPUT_DIM, dtype=np.float64)

        # Adam optimizer state
        self._adam_t = 0
        self._adam_m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._adam_v = {k: np.zeros_like(v) for k, v in self._params().items()}

    def _params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2,
                "b2": self.b2, "W3": self.W3, "b3": self.b3}

    def forward(self, X):
        """
        Forward pass.  X shape: (batch, INPUT_DIM).
        Returns output (batch, OUTPUT_DIM) and cache for backward.
        """
        z1 = X @ self.W1 + self.b1
        a1 = _relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = _relu(z2)
        z3 = a2 @ self.W3 + self.b3
        out = _tanh(z3)  # bounded −1..1
        cache = (X, z1, a1, z2, a2, z3)
        return out, cache

    def predict(self, x_vec):
        """Single-sample prediction.  x_vec: (INPUT_DIM,) → (OUTPUT_DIM,)."""
        X = np.array(x_vec, dtype=np.float64).reshape(1, -1)
        out, _ = self.forward(X)
        return out[0]

    def backward(self, out, cache, targets):
        """
        Backpropagation with MSE loss.
        out:     (batch, OUTPUT_DIM)
        targets: (batch, OUTPUT_DIM)
        Returns dict of gradients.
        """
        X, z1, a1, z2, a2, z3 = cache
        batch = float(X.shape[0])

        # dL/dout for MSE = 2/N * (out - target) * tanh_grad
        d_out = (2.0 / batch) * (out - targets) * _tanh_grad(z3)

        dW3 = a2.T @ d_out / batch
        db3 = d_out.mean(axis=0)

        d_a2 = d_out @ self.W3.T * _relu_grad(z2)
        dW2 = a1.T @ d_a2 / batch
        db2 = d_a2.mean(axis=0)

        d_a1 = d_a2 @ self.W2.T * _relu_grad(z1)
        dW1 = X.T @ d_a1 / batch
        db1 = d_a1.mean(axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    def _adam_step(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam optimizer update."""
        self._adam_t += 1
        params = self._params()
        for key in params:
            g = grads[key]
            self._adam_m[key] = beta1 * self._adam_m[key] + (1.0 - beta1) * g
            self._adam_v[key] = beta2 * self._adam_v[key] + (1.0 - beta2) * (g * g)
            m_hat = self._adam_m[key] / (1.0 - beta1 ** self._adam_t)
            v_hat = self._adam_v[key] / (1.0 - beta2 ** self._adam_t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def train_batch(self, X, Y):
        """One training step.  X: (batch, INPUT_DIM), Y: (batch, OUTPUT_DIM).  Returns loss."""
        out, cache = self.forward(X)
        loss = float(np.mean((out - Y) ** 2))
        grads = self.backward(out, cache, Y)

        # Gradient clipping
        for key in grads:
            norm = np.linalg.norm(grads[key])
            if norm > 5.0:
                grads[key] = grads[key] * (5.0 / norm)

        self._adam_step(grads)
        return loss

    def save(self, path):
        """Save weights to disk."""
        try:
            np.savez_compressed(
                path,
                W1=self.W1, b1=self.b1,
                W2=self.W2, b2=self.b2,
                W3=self.W3, b3=self.b3,
            )
        except Exception as e:
            print(f"  [NeuralNet] Warning: could not save weights: {e}")

    def load(self, path):
        """Load weights from disk if available."""
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path)
            self.W1 = data["W1"]
            self.b1 = data["b1"]
            self.W2 = data["W2"]
            self.b2 = data["b2"]
            self.W3 = data["W3"]
            self.b3 = data["b3"]
            # Reset Adam state for loaded weights
            self._adam_t = 0
            self._adam_m = {k: np.zeros_like(v) for k, v in self._params().items()}
            self._adam_v = {k: np.zeros_like(v) for k, v in self._params().items()}
            return True
        except Exception as e:
            print(f"  [NeuralNet] Warning: could not load weights: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════
#  Feature extraction from NPC state and conversation logs
# ═══════════════════════════════════════════════════════════════════════

# NPC class encoding (ordinal based on social rank)
_CLASS_RANK = {
    "Peasant": 0.0, "Labourer": 0.15, "Traveller": 0.3,
    "Merchant": 0.45, "Blacksmith": 0.5, "Elite": 0.7,
    "Noble": 0.85, "Royal": 1.0,
}


def extract_npc_features(npc, sentiment=0.0, danger_score=0.0, intensity=0.0,
                          word_count=0, topic_score=0.0):
    """
    Build a 15-dim feature vector from NPC state + player message stats.

    Features:
      [0]  mood           (0..1)
      [1]  trust          (0..1)
      [2]  energy         (0..1)
      [3]  hunger         (0..1)
      [4]  social_need    (0..1)
      [5]  wealth         (0..1)
      [6]  class_rank     (0..1)
      [7]  friendliness   (0..1)
      [8]  aggression     (0..1)
      [9]  sociability    (0..1)
      [10] sentiment      (-1..1)
      [11] danger_score   (0..1)
      [12] intensity      (0..1)
      [13] word_count_n   (0..1, normalized)
      [14] topic_score    (0..1)
    """
    bv = npc.behavior_vector if hasattr(npc, "behavior_vector") else {}
    needs = npc.needs if hasattr(npc, "needs") else {}
    personality = npc.personality if hasattr(npc, "personality") else {}
    cls = getattr(npc, "npc_class", "Peasant")

    return np.array([
        float(bv.get("mood", 0.5)),
        float(bv.get("trust", 0.5)),
        float(needs.get("energy", 0.5)),
        float(needs.get("hunger", 0.0)),
        float(needs.get("social_need", 0.0)),
        float(bv.get("wealth", 0.5)),
        _CLASS_RANK.get(cls, 0.3),
        float(personality.get("friendliness", 0.5)),
        float(personality.get("aggression", 0.2)),
        float(personality.get("sociability", 0.5)),
        max(-1.0, min(1.0, float(sentiment))),
        max(0.0, min(1.0, float(danger_score))),
        max(0.0, min(1.0, float(intensity))),
        max(0.0, min(1.0, float(word_count) / 30.0)),
        max(0.0, min(1.0, float(topic_score))),
    ], dtype=np.float64)


def _features_from_csv_row(row):
    """
    Approximate feature vector from a CSV log row.
    Since the CSV doesn't store full NPC state, we reconstruct from available fields.
    """
    sentiment = 0.0
    try:
        sentiment = float(row.get("sentiment", 0.0))
    except (ValueError, TypeError):
        pass

    npc_class = row.get("npc_class", "Peasant")
    class_rank = _CLASS_RANK.get(npc_class, 0.3)

    # Approximate NPC state from sentiment history
    player_msg = row.get("player_message", "")
    npc_response = row.get("npc_response", "")
    word_count = len(player_msg.split()) if player_msg else 0

    # Reconstruct approximate emotion features
    emotion = {}
    try:
        emotion = json.loads(row.get("emotion_json", "{}") or "{}")
    except (json.JSONDecodeError, TypeError):
        pass

    mood = max(0.0, min(1.0, 0.5 + sentiment * 0.3))
    trust = max(0.0, min(1.0, 0.5 + sentiment * 0.2))
    danger = max(0.0, -sentiment * 0.5) if sentiment < 0 else 0.0

    features = np.array([
        mood,                              # mood
        trust,                             # trust
        0.5,                               # energy (unknown, default)
        0.3,                               # hunger (unknown, default)
        0.4,                               # social_need (unknown, default)
        0.5,                               # wealth (unknown, default)
        class_rank,                        # class rank
        0.5,                               # friendliness (unknown)
        0.2,                               # aggression (unknown)
        0.5,                               # sociability (unknown)
        sentiment,                         # sentiment
        danger,                            # danger_score
        min(1.0, word_count / 30.0) * 0.5, # intensity approximation
        min(1.0, word_count / 30.0),       # word_count normalized
        0.3,                               # topic score (unknown)
    ], dtype=np.float64)
    return features, sentiment


def _target_from_csv_row(row, sentiment):
    """
    Build an 8-dim training target from logged data.

    [0:6] — Ideal action weight biases based on sentiment context
    [6]   — Warmth (based on response sentiment)
    [7]   — Sentiment shift
    """
    warmth = max(-1.0, min(1.0, sentiment))

    # Derive ideal action biases from sentiment:
    # Positive sentiment → socialize more, work more
    # Negative sentiment → flee/guard more
    socialize_bias = max(-0.5, min(0.5, sentiment * 0.4))
    work_bias = max(-0.3, min(0.3, sentiment * 0.2))
    flee_bias = max(-0.3, min(0.3, -sentiment * 0.3)) if sentiment < -0.3 else 0.0
    guard_bias = max(-0.2, min(0.3, -sentiment * 0.25)) if sentiment < -0.2 else 0.0

    target = np.array([
        0.0,             # eat bias (neutral)
        0.0,             # sleep bias (neutral)
        socialize_bias,  # socialize bias
        work_bias,       # work bias
        flee_bias,       # flee bias
        guard_bias,      # guard bias
        warmth,          # warmth target
        sentiment * 0.3, # sentiment shift
    ], dtype=np.float64)
    return target


# ═══════════════════════════════════════════════════════════════════════
#  Manager: background training + integration with fuzzy system
# ═══════════════════════════════════════════════════════════════════════

class NeuralDialogueManager:
    """
    Manages the neural network lifecycle:
    - Loads/saves persistent weights
    - Trains in background from CSV conversation logs
    - Provides predictions to FuzzySocialController and DialogueGenerator
    - Tracks training statistics for the research metrics HUD
    """

    def __init__(self):
        self.enabled = bool(NEURAL_ENABLED)
        self.net = NeuralDialogueNet(
            h1=int(NEURAL_HIDDEN_1),
            h2=int(NEURAL_HIDDEN_2),
            lr=float(NEURAL_LR),
        )
        self.weight_path = os.path.abspath(str(NEURAL_WEIGHT_PATH))
        self.csv_path = os.path.abspath(str(CONVERSATION_CSV_PATH))
        self.batch_size = max(4, int(NEURAL_BATCH_SIZE))
        self.train_every_n = max(5, int(NEURAL_TRAIN_EVERY_N))
        self.epochs_per_train = max(1, int(NEURAL_EPOCHS_PER_TRAIN))
        self.warmth_influence = float(NEURAL_WARMTH_INFLUENCE)
        self.action_influence = float(NEURAL_ACTION_INFLUENCE)

        # State tracking
        self._rows_since_train = 0
        self._total_trained = 0
        self._last_loss = 0.0
        self._train_count = 0
        self._training_lock = threading.Lock()
        self._training_in_progress = False
        self._csv_rows_seen = 0

        # Load persisted weights
        if self.enabled:
            loaded = self.net.load(self.weight_path)
            if loaded:
                print("  🧠 Neural dialogue weights loaded from disk")
            else:
                print("  🧠 Neural dialogue network initialized (fresh weights)")
            # Initial training from existing CSV data
            self._schedule_training(force=True)

    # ── Prediction API ──────────────────────────────────────────────

    def predict_for_npc(self, npc, sentiment=0.0, danger_score=0.0,
                         intensity=0.0, word_count=0, topic_score=0.0):
        """
        Get neural network predictions for an NPC + conversation context.

        Returns dict:
          action_biases: dict[str, float]  — fuzzy weight adjustments
          warmth:        float             — response warmth (−1..1)
          sentiment_shift: float           — predicted mood shift
          confidence:    float             — network confidence (0..1)
        """
        if not self.enabled:
            return self._neutral_prediction()

        features = extract_npc_features(
            npc, sentiment=sentiment, danger_score=danger_score,
            intensity=intensity, word_count=word_count, topic_score=topic_score,
        )
        try:
            output = self.net.predict(features)
        except Exception:
            return self._neutral_prediction()

        action_biases = {}
        for i, key in enumerate(ACTION_KEYS):
            action_biases[key] = float(output[i]) * self.action_influence

        warmth = float(output[6]) * self.warmth_influence
        sentiment_shift = float(output[7])

        # Confidence is based on magnitude of outputs (stronger = more confident)
        confidence = min(1.0, float(np.mean(np.abs(output))) * 2.0)

        return {
            "action_biases": action_biases,
            "warmth": warmth,
            "sentiment_shift": sentiment_shift,
            "confidence": confidence,
        }

    def _neutral_prediction(self):
        return {
            "action_biases": {k: 0.0 for k in ACTION_KEYS},
            "warmth": 0.0,
            "sentiment_shift": 0.0,
            "confidence": 0.0,
        }

    # ── Feedback loop: called after each conversation exchange ──────

    def on_exchange(self, npc, player_message, npc_response, sentiment,
                    features_dict=None):
        """
        Called after each player-NPC exchange to feed the training loop.
        Increments row counter and triggers background training when threshold reached.
        """
        if not self.enabled:
            return
        self._rows_since_train += 1
        if self._rows_since_train >= self.train_every_n:
            self._rows_since_train = 0
            self._schedule_training()

    # ── Background training ─────────────────────────────────────────

    def _schedule_training(self, force=False):
        if self._training_in_progress:
            return
        if not force and self._train_count < 1:
            # Skip if no data yet
            pass

        def _worker():
            with self._training_lock:
                self._training_in_progress = True
                try:
                    self._do_training()
                except Exception as e:
                    print(f"  [NeuralNet] Training error: {e}")
                finally:
                    self._training_in_progress = False

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def _do_training(self):
        """Load CSV, build dataset, train for a few epochs, save weights."""
        if not os.path.exists(self.csv_path):
            return

        # Read CSV rows
        rows = []
        try:
            with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
        except Exception:
            return

        if len(rows) < 8:
            return  # Not enough data to train

        # Build training dataset
        X_list = []
        Y_list = []
        for row in rows:
            try:
                features, sentiment = _features_from_csv_row(row)
                target = _target_from_csv_row(row, sentiment)
                X_list.append(features)
                Y_list.append(target)
            except Exception:
                continue

        if len(X_list) < 8:
            return

        X = np.array(X_list, dtype=np.float64)
        Y = np.array(Y_list, dtype=np.float64)
        n = X.shape[0]

        total_loss = 0.0
        steps = 0

        for epoch in range(self.epochs_per_train):
            # Shuffle
            indices = np.random.permutation(n)
            X = X[indices]
            Y = Y[indices]

            # Mini-batch training
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                X_batch = X[start:end]
                Y_batch = Y[start:end]
                loss = self.net.train_batch(X_batch, Y_batch)
                total_loss += loss
                steps += 1

        self._last_loss = total_loss / max(1, steps)
        self._total_trained = n
        self._train_count += 1
        self._csv_rows_seen = len(rows)

        # Save weights
        self.net.save(self.weight_path)

    # ── Stats for HUD / research metrics ────────────────────────────

    def get_stats(self):
        return {
            "enabled": self.enabled,
            "train_count": self._train_count,
            "total_samples": self._total_trained,
            "last_loss": round(self._last_loss, 6),
            "csv_rows": self._csv_rows_seen,
            "training_active": self._training_in_progress,
        }
