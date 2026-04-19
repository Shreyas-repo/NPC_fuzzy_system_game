"""
Emotion / Reaction Vector Database for Unsupervised Learning.

Every player-NPC interaction produces an emotion vector (8 dimensions):
    joy, anger, fear, sadness, trust, surprise, disgust, curiosity

These vectors are stored in the database and periodically clustered
using K-Means to discover emergent emotion patterns. NPCs then use
these patterns to:
    1. Predict likely player sentiment (based on past interactions)
    2. Pre-adjust their mood before conversation
    3. Form opinions about the player based on interaction history

This is the key data structure that makes the NPCs truly learn
from interactions through unsupervised learning.
"""
import json
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import defaultdict
from config import OLLAMA_USE_INTERACTION_EMBEDDINGS


EMOTION_DIMS = ["joy", "anger", "fear", "sadness", "trust", "surprise", "disgust", "curiosity"]
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "emotion_db.json")


class InteractionRecord:
    """A single interaction record in the database."""

    def __init__(self, npc_name, npc_class, player_message, npc_response,
                 emotion_vector, sentiment_score, npc_mood, npc_trust, timestamp=0):
        self.npc_name = npc_name
        self.npc_class = npc_class
        self.player_message = player_message
        self.npc_response = npc_response
        self.emotion_vector = emotion_vector  # dict of 8 floats
        self.sentiment_score = sentiment_score
        self.npc_mood = npc_mood
        self.npc_trust = npc_trust
        self.timestamp = timestamp

    def to_dict(self):
        return {
            "npc_name": self.npc_name,
            "npc_class": self.npc_class,
            "player_message": self.player_message,
            "npc_response": self.npc_response,
            "emotion_vector": self.emotion_vector,
            "sentiment_score": self.sentiment_score,
            "npc_mood": self.npc_mood,
            "npc_trust": self.npc_trust,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(d):
        return InteractionRecord(
            d["npc_name"], d["npc_class"], d["player_message"], d["npc_response"],
            d["emotion_vector"], d["sentiment_score"], d["npc_mood"], d["npc_trust"],
            d.get("timestamp", 0)
        )

    def get_vector(self):
        """Get emotion vector as numpy array."""
        return np.array([self.emotion_vector.get(dim, 0.0) for dim in EMOTION_DIMS])


class EmotionDatabase:
    """
    Persistent database of all player-NPC interaction emotion vectors.
    Uses K-Means to cluster interactions into emotion patterns.
    
    UNSUPERVISED LEARNING:
    The database discovers patterns in how the player interacts with NPCs.
    These patterns influence NPC behavior — if the player has been consistently
    hostile, NPCs pre-adjust their trust. If the player is kind, NPCs become
    more open. This learning is emergent and data-driven, not rule-based.
    """

    def __init__(self, vector_database=None, ollama_client=None):
        self.records = []
        self.emotion_clusters = None
        self.cluster_labels = []
        self.cluster_centers = None
        self.n_clusters = 4
        self.player_profile = None  # Learned player behavior profile
        self.vector_database = vector_database
        self.ollama_client = ollama_client

        # Per-NPC interaction stats (learned from data)
        self.npc_interaction_stats = defaultdict(lambda: {
            "count": 0,
            "avg_sentiment": 0.0,
            "avg_emotion": {dim: 0.0 for dim in EMOTION_DIMS},
            "dominant_emotion": "curiosity",
            "predicted_next_sentiment": 0.0,
        })

        # Global interaction patterns
        self.global_stats = {
            "total_interactions": 0,
            "avg_sentiment": 0.0,
            "emotion_distribution": {dim: 0.0 for dim in EMOTION_DIMS},
            "player_tendency": "neutral",  # learned: friendly, hostile, curious, neutral
        }

        self._load()

    def set_vector_pipeline(self, vector_database=None, ollama_client=None):
        """Attach vector storage + embedding client after construction."""
        if vector_database is not None:
            self.vector_database = vector_database
        if ollama_client is not None:
            self.ollama_client = ollama_client

    def add_interaction(self, npc, player_message, npc_response, emotion_vector,
                       sentiment_score, game_time=0):
        """
        Record a new interaction and update learned patterns.
        This is called after every player-NPC conversation exchange.
        """
        record = InteractionRecord(
            npc_name=npc.name,
            npc_class=npc.npc_class,
            player_message=player_message,
            npc_response=npc_response,
            emotion_vector=emotion_vector,
            sentiment_score=sentiment_score,
            npc_mood=npc.behavior_vector["mood"],
            npc_trust=npc.behavior_vector["trust"],
            timestamp=game_time,
        )
        self.records.append(record)

        self._store_vector_record(record)

        # Update per-NPC stats
        self._update_npc_stats(npc.name, emotion_vector, sentiment_score)

        # Update global stats
        self._update_global_stats(emotion_vector, sentiment_score)

        # Re-cluster periodically
        if len(self.records) >= 5 and len(self.records) % 3 == 0:
            self._cluster_emotions()

        # Update player profile
        if len(self.records) >= 3:
            self._update_player_profile()

        # Auto-save periodically
        if len(self.records) % 10 == 0:
            self._save()

    def _store_vector_record(self, record):
        """Store an embedding-backed interaction record in the vector database."""
        if not self.vector_database:
            return

        try:
            combined = (
                f"npc={record.npc_name} class={record.npc_class} "
                f"player='{record.player_message}' npc_reply='{record.npc_response}' "
                f"sentiment={record.sentiment_score:.3f}"
            )
            embedding = [float(v) for v in record.get_vector().tolist()]

            if OLLAMA_USE_INTERACTION_EMBEDDINGS and self.ollama_client and self.ollama_client.is_available():
                ollama_embedding = self.ollama_client.embed(combined)
                if ollama_embedding:
                    embedding = ollama_embedding

            self.vector_database.add_record(
                {
                    "type": "player_npc_interaction",
                    "npc_name": record.npc_name,
                    "npc_class": record.npc_class,
                    "text": combined,
                    "embedding": embedding,
                    "sentiment": record.sentiment_score,
                    "emotion": record.emotion_vector,
                    "timestamp": record.timestamp,
                }
            )
        except Exception:
            return

    def _update_npc_stats(self, npc_name, emotion_vector, sentiment_score):
        """Update running statistics for a specific NPC."""
        stats = self.npc_interaction_stats[npc_name]
        n = stats["count"]
        stats["count"] = n + 1

        # Running average of sentiment
        stats["avg_sentiment"] = (stats["avg_sentiment"] * n + sentiment_score) / (n + 1)

        # Running average of emotion dimensions
        for dim in EMOTION_DIMS:
            old_avg = stats["avg_emotion"][dim]
            new_val = emotion_vector.get(dim, 0.0)
            stats["avg_emotion"][dim] = (old_avg * n + new_val) / (n + 1)

        # Find dominant emotion
        stats["dominant_emotion"] = max(stats["avg_emotion"], key=stats["avg_emotion"].get)

        # Predict next sentiment (simple exponential moving average)
        alpha = 0.3
        stats["predicted_next_sentiment"] = (
            alpha * sentiment_score + (1 - alpha) * stats["predicted_next_sentiment"]
        )

    def _update_global_stats(self, emotion_vector, sentiment_score):
        """Update global interaction statistics."""
        n = self.global_stats["total_interactions"]
        self.global_stats["total_interactions"] = n + 1

        self.global_stats["avg_sentiment"] = (
            (self.global_stats["avg_sentiment"] * n + sentiment_score) / (n + 1)
        )

        for dim in EMOTION_DIMS:
            old = self.global_stats["emotion_distribution"][dim]
            new = emotion_vector.get(dim, 0.0)
            self.global_stats["emotion_distribution"][dim] = (old * n + new) / (n + 1)

    def _cluster_emotions(self):
        """
        UNSUPERVISED LEARNING: K-Means clustering on emotion vectors.
        Discovers patterns like:
            Cluster 0: Friendly & curious interactions
            Cluster 1: Hostile & angry interactions
            Cluster 2: Fearful & sad interactions
            Cluster 3: Surprised & trusting interactions
        """
        if len(self.records) < self.n_clusters:
            return

        vectors = np.array([r.get_vector() for r in self.records])
        n_clusters = min(self.n_clusters, len(vectors))

        kmeans = KMeans(n_clusters=n_clusters, n_init=5, max_iter=100, random_state=42)
        self.cluster_labels = kmeans.fit_predict(vectors).tolist()
        self.cluster_centers = kmeans.cluster_centers_
        self.emotion_clusters = kmeans

    def _update_player_profile(self):
        """
        UNSUPERVISED LEARNING: Build a player behavior profile from
        accumulated interaction data. This profile is used to pre-adjust
        NPC behavior before conversations.
        """
        if len(self.records) < 3:
            return

        avg_sent = self.global_stats["avg_sentiment"]
        emotions = self.global_stats["emotion_distribution"]

        # Determine player tendency
        if avg_sent > 0.3:
            tendency = "friendly"
        elif avg_sent < -0.3:
            tendency = "hostile"
        elif emotions.get("curiosity", 0) > 0.3:
            tendency = "curious"
        else:
            tendency = "neutral"

        self.global_stats["player_tendency"] = tendency

        # Build profile vector
        recent = self.records[-10:]
        self.player_profile = {
            "tendency": tendency,
            "avg_sentiment": avg_sent,
            "recent_avg_sentiment": sum(r.sentiment_score for r in recent) / len(recent),
            "dominant_emotion": max(emotions, key=emotions.get),
            "interaction_count": len(self.records),
            "emotions": dict(emotions),
        }

    def get_npc_prediction(self, npc_name):
        """
        Get the predicted player behavior for a specific NPC.
        NPCs use this to pre-adjust their mood/trust before talking.
        """
        stats = self.npc_interaction_stats.get(npc_name)
        if not stats or stats["count"] == 0:
            # No history — use global profile
            if self.player_profile:
                return {
                    "predicted_sentiment": self.player_profile["recent_avg_sentiment"],
                    "dominant_emotion": self.player_profile["dominant_emotion"],
                    "tendency": self.player_profile["tendency"],
                }
            return None

        return {
            "predicted_sentiment": stats["predicted_next_sentiment"],
            "dominant_emotion": stats["dominant_emotion"],
            "tendency": "friendly" if stats["avg_sentiment"] > 0.2
                       else "hostile" if stats["avg_sentiment"] < -0.2
                       else "neutral",
            "interaction_count": stats["count"],
        }

    def get_cluster_descriptions(self):
        """Get human-readable descriptions of discovered emotion clusters."""
        if self.cluster_centers is None:
            return {}

        descriptions = {}
        for i, center in enumerate(self.cluster_centers):
            desc_parts = []
            for j, dim in enumerate(EMOTION_DIMS):
                if center[j] > 0.3:
                    intensity = "highly" if center[j] > 0.6 else "moderately"
                    desc_parts.append(f"{intensity} {dim}")
            if not desc_parts:
                desc_parts.append("neutral")

            # Count interactions in this cluster
            count = sum(1 for l in self.cluster_labels if l == i)
            descriptions[i] = {
                "description": " + ".join(desc_parts),
                "count": count,
                "center": center.tolist(),
            }

        return descriptions

    def apply_learned_behavior(self, npcs):
        """
        UNSUPERVISED LEARNING: Apply learned player behavior patterns
        to NPC behavior vectors. This is the feedback loop that makes
        NPCs adapt based on accumulated interaction data.
        """
        if not self.player_profile:
            return

        tendency = self.player_profile["tendency"]

        for npc in npcs:
            prediction = self.get_npc_prediction(npc.name)
            if prediction is None:
                # No data yet — use global tendency
                if tendency == "hostile":
                    # NPCs have heard the player is hostile, slightly guard up
                    npc.behavior_vector["trust"] = max(0.0,
                        npc.behavior_vector["trust"] - 0.005)
                elif tendency == "friendly":
                    # Word has spread the player is nice
                    npc.behavior_vector["trust"] = min(1.0,
                        npc.behavior_vector["trust"] + 0.003)
            else:
                # NPC has personal experience
                pred_sent = prediction["predicted_sentiment"]
                if pred_sent > 0.2:
                    npc.behavior_vector["mood"] = min(1.0,
                        npc.behavior_vector["mood"] + 0.002)
                elif pred_sent < -0.2:
                    npc.behavior_vector["mood"] = max(0.0,
                        npc.behavior_vector["mood"] - 0.002)

    def get_stats_for_ui(self):
        """Get database stats for HUD display."""
        vector_summary = (
            self.vector_database.get_cluster_summary()
            if self.vector_database else
            {"records": 0, "kmeans_clusters": 0, "dbscan_groups": 0}
        )
        return {
            "total_interactions": len(self.records),
            "player_tendency": self.global_stats.get("player_tendency", "unknown"),
            "avg_sentiment": self.global_stats.get("avg_sentiment", 0.0),
            "n_clusters": len(self.get_cluster_descriptions()),
            "emotions": self.global_stats.get("emotion_distribution", {}),
            "vector_records": vector_summary.get("records", 0),
            "vector_kmeans": vector_summary.get("kmeans_clusters", 0),
            "vector_dbscan": vector_summary.get("dbscan_groups", 0),
        }

    def _save(self):
        """Save database to disk."""
        try:
            data = {
                "records": [r.to_dict() for r in self.records[-200:]],  # Keep last 200
                "global_stats": self.global_stats,
                "npc_stats": dict(self.npc_interaction_stats),
            }
            with open(DB_PATH, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save emotion database: {e}")

    def _load(self):
        """Load database from disk."""
        try:
            if os.path.exists(DB_PATH):
                with open(DB_PATH, 'r') as f:
                    data = json.load(f)
                self.records = [InteractionRecord.from_dict(r) for r in data.get("records", [])]
                self.global_stats.update(data.get("global_stats", {}))
                saved_stats = data.get("npc_stats", {})
                for name, stats in saved_stats.items():
                    self.npc_interaction_stats[name] = stats

                # Re-cluster loaded data
                if len(self.records) >= self.n_clusters:
                    self._cluster_emotions()
                    self._update_player_profile()

                print(f"  Loaded emotion database: {len(self.records)} interactions")
        except Exception as e:
            print(f"Warning: Could not load emotion database: {e}")
