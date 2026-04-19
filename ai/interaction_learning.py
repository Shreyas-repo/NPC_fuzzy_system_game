"""
Unsupervised interaction learning for NPC-to-NPC adaptation using vector embeddings.
"""
from collections import defaultdict
from config import (
    INTERACTION_SNAPSHOT_BATCH,
    INTERACTION_CLUSTER_INTERVAL,
    OLLAMA_USE_NPC_SNAPSHOT_EMBEDDINGS,
)


class InteractionLearningEngine:
    """Builds NPC interaction embeddings and applies learned social influence."""

    def __init__(self, ollama_client, vector_db):
        self.ollama_client = ollama_client
        self.vector_db = vector_db
        self.update_timer = 0.0
        self.update_interval = 8.0
        self.snapshot_batch_size = INTERACTION_SNAPSHOT_BATCH
        self.cluster_interval = INTERACTION_CLUSTER_INTERVAL
        self.cluster_timer = 0.0
        self.snapshot_cursor = 0

    def update(self, dt, npcs):
        self.update_timer += dt
        self.cluster_timer += dt
        if self.update_timer < self.update_interval:
            return
        self.update_timer = 0.0

        if not self.vector_db:
            return

        if not npcs:
            return

        # Snapshot only a small rotating subset each cycle to avoid update spikes.
        count = min(self.snapshot_batch_size, len(npcs))
        start = self.snapshot_cursor % len(npcs)
        selected = [npcs[(start + i) % len(npcs)] for i in range(count)]
        self.snapshot_cursor = (start + count) % len(npcs)

        for npc in selected:
            snapshot = self._snapshot_text(npc)
            embedding = self._snapshot_vector(npc)
            if OLLAMA_USE_NPC_SNAPSHOT_EMBEDDINGS and self.ollama_client and self.ollama_client.is_available():
                try:
                    ollama_embedding = self.ollama_client.embed(snapshot)
                    if ollama_embedding:
                        embedding = ollama_embedding
                except Exception:
                    pass
            if embedding:
                self.vector_db.add_record(
                    {
                        "type": "npc_snapshot",
                        "npc_name": npc.name,
                        "npc_class": npc.npc_class,
                        "text": snapshot,
                        "embedding": embedding,
                        "sentiment": 0.0,
                        "emotion": {},
                    }
                )

        if self.cluster_timer >= self.cluster_interval:
            self.cluster_timer = 0.0
            self.vector_db.run_unsupervised()
            self._apply_unsupervised_social_influence(npcs)

    def _snapshot_text(self, npc):
        return (
            f"{npc.name} class={npc.npc_class} state={npc.state} "
            f"mood={npc.behavior_vector['mood']:.2f} trust={npc.behavior_vector['trust']:.2f} "
            f"energy={npc.needs['energy']:.2f} hunger={npc.needs['hunger']:.2f} "
            f"social_need={npc.needs['social_need']:.2f}"
        )

    def _snapshot_vector(self, npc):
        """Low-latency embedding from numeric NPC state for unsupervised learning."""
        state_map = {
            "idle": 0.0,
            "walking": 0.2,
            "working": 0.4,
            "socializing": 0.6,
            "sleeping": 0.8,
            "eating": 1.0,
            "talking": 0.5,
        }
        return [
            float(npc.behavior_vector.get("mood", 0.0)),
            float(npc.behavior_vector.get("trust", 0.0)),
            float(npc.needs.get("energy", 0.0)),
            float(npc.needs.get("hunger", 0.0)),
            float(npc.needs.get("social_need", 0.0)),
            float(getattr(npc, "social_rank", 1) / 5.0),
            float(state_map.get(getattr(npc, "state", "idle"), 0.0)),
        ]

    def _apply_unsupervised_social_influence(self, npcs):
        profiles = self.vector_db.get_npc_profiles()
        if not profiles:
            return

        names = [npc.name for npc in npcs if npc.name in profiles]
        if len(names) < 2:
            return

        # Build similarity neighborhoods and nudge trust/mood accordingly.
        neighbors = defaultdict(list)
        for i, name_a in enumerate(names):
            vec_a = profiles[name_a]
            for j, name_b in enumerate(names):
                if i == j:
                    continue
                vec_b = profiles[name_b]
                sim = self._cosine_similarity(vec_a, vec_b)
                if sim > 0.78:
                    neighbors[name_a].append(sim)

        for npc in npcs:
            sims = neighbors.get(npc.name, [])
            if not sims:
                continue
            cohesion = sum(sims) / len(sims)
            npc.behavior_vector["trust"] = min(1.0, npc.behavior_vector["trust"] + 0.004 * cohesion)
            npc.behavior_vector["mood"] = min(1.0, npc.behavior_vector["mood"] + 0.003 * cohesion)
            npc.needs["social_need"] = max(0.0, npc.needs["social_need"] - 0.004 * cohesion)

    def _cosine_similarity(self, a, b):
        a_norm = (a**2).sum() ** 0.5
        b_norm = (b**2).sum() ** 0.5
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float((a * b).sum() / (a_norm * b_norm))
