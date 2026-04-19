"""
Unsupervised Learning — K-Means and DBSCAN clustering for NPC behavior.

K-Means: Clusters NPCs by their behavior vectors (mood, energy, hunger, etc.)
          NPCs in the same cluster influence each other's mood.

DBSCAN:  Clusters NPCs by spatial position to detect social gatherings.
         Being near others satisfies social needs; isolation causes loneliness.
"""
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from config import KMEANS_N_CLUSTERS, DBSCAN_EPS, DBSCAN_MIN_SAMPLES


class BehaviorClustering:
    """
    K-Means clustering on NPC behavior vectors.
    Groups NPCs into behavioral archetypes that influence each other.
    """

    def __init__(self, n_clusters=KMEANS_N_CLUSTERS):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = StandardScaler()
        self.labels = []
        self.cluster_centers = None
        self.fitted = False

    def fit_predict(self, behavior_vectors):
        """
        Cluster NPCs by their behavior vectors.
        Returns cluster labels for each NPC.
        """
        if len(behavior_vectors) < self.n_clusters:
            self.labels = list(range(len(behavior_vectors)))
            return self.labels

        # Standardize features
        scaled = self.scaler.fit_transform(behavior_vectors)

        # Fit K-Means
        n_clusters = min(self.n_clusters, len(behavior_vectors))
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=3, max_iter=50, random_state=None)
        self.labels = self.kmeans.fit_predict(scaled).tolist()
        self.cluster_centers = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        self.fitted = True

        return self.labels

    def get_cluster_description(self, cluster_id):
        """Get a human-readable description of a cluster based on its centroid."""
        if not self.fitted or self.cluster_centers is None:
            return "Unknown"
        if cluster_id < 0 or cluster_id >= len(self.cluster_centers):
            return "Outlier"

        center = self.cluster_centers[cluster_id]
        mood, energy, hunger, social, wealth, trust = center

        descriptors = []
        if mood > 0.7:
            descriptors.append("Content")
        elif mood < 0.3:
            descriptors.append("Unhappy")

        if energy > 0.7:
            descriptors.append("Energetic")
        elif energy < 0.3:
            descriptors.append("Tired")

        if hunger > 0.6:
            descriptors.append("Hungry")

        if social > 0.7:
            descriptors.append("Lonely")
        elif social < 0.3:
            descriptors.append("Social")

        if wealth > 0.6:
            descriptors.append("Wealthy")
        elif wealth < 0.3:
            descriptors.append("Poor")

        if trust > 0.7:
            descriptors.append("Trusting")
        elif trust < 0.3:
            descriptors.append("Suspicious")

        return " & ".join(descriptors) if descriptors else "Average"


class SpatialClustering:
    """
    DBSCAN clustering on NPC positions.
    Detects natural social gatherings in the village.
    """

    def __init__(self, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES):
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels = []
        self.n_clusters = 0

    def fit_predict(self, positions):
        """
        Cluster NPCs by their spatial positions.
        Returns cluster labels (-1 for outliers/loners).
        """
        if len(positions) < 2:
            self.labels = [-1] * len(positions)
            return self.labels

        self.labels = self.dbscan.fit_predict(positions).tolist()
        self.n_clusters = len(set(self.labels) - {-1})
        return self.labels

    def get_gathering_info(self, npcs):
        """Get info about each detected gathering."""
        gatherings = {}
        for npc, label in zip(npcs, self.labels):
            if label == -1:
                continue
            if label not in gatherings:
                gatherings[label] = {
                    "npcs": [],
                    "center_x": 0,
                    "center_y": 0,
                }
            gatherings[label]["npcs"].append(npc)
            gatherings[label]["center_x"] += npc.x
            gatherings[label]["center_y"] += npc.y

        for label, info in gatherings.items():
            n = len(info["npcs"])
            info["center_x"] /= n
            info["center_y"] /= n

        return gatherings


class ClusteringEngine:
    """
    Orchestrates both clustering algorithms and applies their effects.
    This is the main unsupervised learning engine.
    """

    def __init__(self):
        self.behavior_clustering = BehaviorClustering()
        self.spatial_clustering = SpatialClustering()
        self.update_timer = 0.0
        self.update_interval = 5.0  # Re-cluster every 5 seconds
        self.behavior_labels = []
        self.spatial_labels = []

    def update(self, dt, npcs, behavior_system):
        """Run clustering algorithms periodically and apply effects."""
        self.update_timer += dt
        if self.update_timer < self.update_interval:
            return
        self.update_timer = 0.0

        if len(npcs) < 2:
            return

        # ─── K-Means on behavior vectors ────────────────────────
        behavior_vectors = behavior_system.get_all_vectors(npcs)
        self.behavior_labels = self.behavior_clustering.fit_predict(behavior_vectors)

        # Assign cluster IDs to NPCs
        for npc, label in zip(npcs, self.behavior_labels):
            npc.cluster_id = label

        # Apply cluster influence (mood propagation)
        behavior_system.apply_cluster_influence(npcs, self.behavior_labels)

        # ─── DBSCAN on spatial positions ────────────────────────
        positions = behavior_system.get_positions(npcs)
        self.spatial_labels = self.spatial_clustering.fit_predict(positions)

        # Apply spatial social effects
        behavior_system.apply_spatial_social_effect(npcs, self.spatial_labels)

    def get_behavior_cluster_info(self):
        """Get descriptions of current behavior clusters."""
        info = {}
        for label in set(self.behavior_labels):
            if label == -1:
                continue
            info[label] = self.behavior_clustering.get_cluster_description(label)
        return info

    def get_gathering_info(self, npcs):
        """Get info about spatial gatherings."""
        return self.spatial_clustering.get_gathering_info(npcs)
