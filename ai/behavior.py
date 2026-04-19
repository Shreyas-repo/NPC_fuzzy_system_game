"""
Behavior vector system — the core data structure for unsupervised learning.
Each NPC has a 6-dimensional behavior vector that is fed into clustering algorithms.
"""
import numpy as np


class BehaviorSystem:
    """Manages NPC behavior vectors and their updates."""

    VECTOR_LABELS = ["mood", "energy", "hunger", "social_need", "wealth", "trust"]

    @staticmethod
    def get_all_vectors(npcs):
        """Extract behavior vectors from all NPCs as a numpy array."""
        if not npcs:
            return np.empty((0, 6))
        return np.array([npc.get_behavior_array() for npc in npcs])

    @staticmethod
    def get_positions(npcs):
        """Extract positions from all NPCs as a numpy array."""
        if not npcs:
            return np.empty((0, 2))
        return np.array([[npc.x, npc.y] for npc in npcs])

    @staticmethod
    def apply_cluster_influence(npcs, cluster_labels, influence_rate=0.02):
        """
        NPCs in the same behavioral cluster influence each other's mood.
        Members drift slightly toward the cluster's average mood.
        This is the KEY unsupervised learning feedback loop.
        """
        if len(npcs) == 0:
            return

        unique_labels = set(cluster_labels)
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise/outliers

            # Get all NPCs in this cluster
            cluster_npcs = [npc for npc, l in zip(npcs, cluster_labels) if l == label]
            if len(cluster_npcs) < 2:
                continue

            # Calculate cluster average mood
            avg_mood = sum(n.behavior_vector["mood"] for n in cluster_npcs) / len(cluster_npcs)
            avg_social = sum(n.needs["social_need"] for n in cluster_npcs) / len(cluster_npcs)

            # Each NPC drifts toward cluster average
            for npc in cluster_npcs:
                mood_diff = avg_mood - npc.behavior_vector["mood"]
                npc.behavior_vector["mood"] += mood_diff * influence_rate

                # Social influence: reduce social need when in a group
                npc.needs["social_need"] = max(0, npc.needs["social_need"] - 0.005)

    @staticmethod
    def apply_spatial_social_effect(npcs, spatial_labels):
        """
        NPCs in the same DBSCAN spatial cluster form social groups.
        Being near others satisfies social needs.
        Being an outlier (-1) increases loneliness.
        """
        for npc, label in zip(npcs, spatial_labels):
            if label == -1:
                # Outlier — lonely
                npc.needs["social_need"] = min(1.0, npc.needs["social_need"] + 0.002)
                npc.social_group = -1
            else:
                # In a group — social needs decrease
                npc.needs["social_need"] = max(0.0, npc.needs["social_need"] - 0.008)
                npc.social_group = label

                # Occasional social interaction
                group_npcs = [n for n, l in zip(npcs, spatial_labels)
                              if l == label and n is not npc]
                if group_npcs and npc.social_group != npc.cluster_id:
                    # Small mood boost from socializing
                    npc.behavior_vector["mood"] = min(1.0, npc.behavior_vector["mood"] + 0.001)

    @staticmethod
    def get_cluster_summary(npcs, cluster_labels):
        """Get a summary of each cluster for UI display."""
        summaries = {}
        unique_labels = set(cluster_labels)
        for label in unique_labels:
            if label == -1:
                continue
            cluster_npcs = [npc for npc, l in zip(npcs, cluster_labels) if l == label]
            if not cluster_npcs:
                continue

            avg_mood = sum(n.behavior_vector["mood"] for n in cluster_npcs) / len(cluster_npcs)
            avg_energy = sum(n.needs["energy"] for n in cluster_npcs) / len(cluster_npcs)
            classes = [n.npc_class for n in cluster_npcs]

            if avg_mood > 0.7:
                mood_label = "Happy"
            elif avg_mood > 0.4:
                mood_label = "Neutral"
            else:
                mood_label = "Unhappy"

            summaries[label] = {
                "size": len(cluster_npcs),
                "avg_mood": avg_mood,
                "avg_energy": avg_energy,
                "mood_label": mood_label,
                "classes": classes,
                "names": [n.name for n in cluster_npcs],
            }
        return summaries
