"""Runtime metrics for research-grade evaluation and paper plots."""
import json
import os
import time


METRIC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "research_metrics.jsonl",
)


class ResearchMetrics:
    def __init__(self):
        self.last_summary = {
            "avg_mood": 0.0,
            "avg_trust": 0.0,
            "conflict_rate": 0.0,
            "social_stability": 0.0,
            "chat_latency": 0.0,
        }

    def update(self, npcs, chat_box=None, weights=None):
        if not npcs:
            return self.last_summary

        n = len(npcs)
        avg_mood = sum(npc.behavior_vector.get("mood", 0.0) for npc in npcs) / n
        avg_trust = sum(npc.behavior_vector.get("trust", 0.0) for npc in npcs) / n

        # Conflict proxy: low trust + angry/fear expressions + hostile state hints.
        conflicts = 0
        for npc in npcs:
            if npc.behavior_vector.get("trust", 0.5) < 0.2:
                conflicts += 1
            emo = getattr(npc, "emotion_display", None)
            if emo and emo[0] in ("😠", "😨", "⚔️"):
                conflicts += 1

        conflict_rate = conflicts / max(1.0, n * 1.5)
        chat_latency = float(getattr(chat_box, "last_response_latency", 0.0) or 0.0)

        social_stability = max(
            0.0,
            min(
                1.0,
                (0.45 * avg_mood) + (0.45 * avg_trust) + (0.1 * (1.0 - conflict_rate)),
            ),
        )

        summary = {
            "avg_mood": avg_mood,
            "avg_trust": avg_trust,
            "conflict_rate": conflict_rate,
            "social_stability": social_stability,
            "chat_latency": chat_latency,
            "weights": dict(weights or {}),
            "ts": time.time(),
        }
        self.last_summary = summary
        self._append(summary)
        return summary

    def objective_score(self):
        s = self.last_summary
        # Multi-objective scalarization for online evolution.
        return (
            0.44 * s.get("social_stability", 0.0)
            + 0.28 * s.get("avg_trust", 0.0)
            + 0.20 * s.get("avg_mood", 0.0)
            + 0.08 * (1.0 - min(1.0, s.get("chat_latency", 0.0) / 6.0))
        )

    def _append(self, row):
        try:
            with open(METRIC_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:
            pass
