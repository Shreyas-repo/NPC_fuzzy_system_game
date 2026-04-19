"""
Persistent vector database for interaction embeddings and unsupervised analysis.
"""
import json
import math
import os
from collections import defaultdict
from collections import Counter

import numpy as np
from sklearn.cluster import DBSCAN, KMeans


VECTOR_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "interaction_vector_db.json",
)


class VectorDatabase:
    """Stores embedding vectors with metadata and clusters them periodically."""

    def __init__(self, path=VECTOR_DB_PATH):
        self.path = path
        self.records = []
        self.kmeans_labels = []
        self.dbscan_labels = []
        self.cluster_centers = None
        self._load()

    def add_record(self, record):
        """Append a vector record and save periodically."""
        embedding = record.get("embedding")
        if not embedding or not isinstance(embedding, list):
            return
        self.records.append(record)
        if len(self.records) % 10 == 0:
            self._save()

    def get_vectors(self):
        raw = []
        for rec in self.records:
            vec = rec.get("embedding")
            if not isinstance(vec, list) or len(vec) == 0:
                continue
            try:
                arr = np.array(vec, dtype=float)
            except Exception:
                continue
            if arr.ndim != 1 or len(arr) == 0:
                continue
            if not np.all(np.isfinite(arr)):
                continue
            raw.append(arr)

        if not raw:
            return np.empty((0, 0))

        # Keep only the dominant embedding length to avoid mixed-shape crashes.
        lengths = [len(v) for v in raw]
        dominant_len, _ = Counter(lengths).most_common(1)[0]
        filtered = [v for v in raw if len(v) == dominant_len]

        return np.vstack(filtered) if filtered else np.empty((0, 0))

    def run_unsupervised(self, k_clusters=5, dbscan_eps=0.55, dbscan_min_samples=3):
        """Run K-Means and DBSCAN over all stored vectors."""
        vectors = self.get_vectors()
        if len(vectors) < 3:
            self.kmeans_labels = []
            self.dbscan_labels = []
            self.cluster_centers = None
            return

        normalized = self._normalize(vectors)

        k = min(k_clusters, len(normalized))
        if k >= 2:
            kmeans = KMeans(n_clusters=k, n_init=5, max_iter=100, random_state=42)
            self.kmeans_labels = kmeans.fit_predict(normalized).tolist()
            self.cluster_centers = kmeans.cluster_centers_
        else:
            self.kmeans_labels = [0 for _ in range(len(normalized))]
            self.cluster_centers = None

        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        self.dbscan_labels = dbscan.fit_predict(normalized).tolist()

    def find_similar(self, embedding, top_k=5):
        """Return top-K nearest records by cosine similarity."""
        if not embedding or len(self.records) == 0:
            return []

        target = np.array(embedding, dtype=float)
        scored = []
        for rec in self.records:
            vec = rec.get("embedding")
            if not vec:
                continue
            score = self._cosine_similarity(target, np.array(vec, dtype=float))
            scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

    def get_npc_profiles(self):
        """Build centroid profile vectors per NPC from stored records."""
        grouped = defaultdict(list)
        for rec in self.records:
            npc_name = rec.get("npc_name")
            vec = rec.get("embedding")
            if npc_name and vec:
                try:
                    arr = np.array(vec, dtype=float)
                except Exception:
                    continue
                if arr.ndim != 1 or len(arr) == 0:
                    continue
                if not np.all(np.isfinite(arr)):
                    continue
                grouped[npc_name].append(arr)

        profiles = {}
        for name, vectors in grouped.items():
            if not vectors:
                continue
            lengths = [len(v) for v in vectors]
            dominant_len, _ = Counter(lengths).most_common(1)[0]
            aligned = [v for v in vectors if len(v) == dominant_len]
            if not aligned:
                continue
            profiles[name] = np.mean(np.vstack(aligned), axis=0)
        return profiles

    def get_cluster_summary(self):
        """Return lightweight summary for HUD/debugging."""
        if not self.kmeans_labels:
            return {"kmeans_clusters": 0, "dbscan_groups": 0, "records": len(self.records)}
        k_count = len(set(self.kmeans_labels))
        d_count = len(set(self.dbscan_labels) - {-1}) if self.dbscan_labels else 0
        return {
            "kmeans_clusters": k_count,
            "dbscan_groups": d_count,
            "records": len(self.records),
        }

    def _normalize(self, vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _cosine_similarity(self, a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return -1.0
        return float(np.dot(a, b) / denom)

    def _save(self):
        try:
            payload = {
                "records": self.records[-1000:],
            }
            with open(self.path, "w") as f:
                json.dump(payload, f)
        except Exception:
            pass

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            self.records = data.get("records", [])
        except Exception:
            self.records = []
