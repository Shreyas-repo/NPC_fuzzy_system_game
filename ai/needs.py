"""
Need system for NPCs — hunger, energy, social need.
"""
from config import NEED_DECAY, NEED_RESTORE_RATE


class NeedSystem:
    """Manages NPC needs and their satisfaction."""

    @staticmethod
    def get_most_urgent_need(npc):
        """Get the most pressing need for an NPC."""
        needs_priority = [
            ("energy",      npc.needs["energy"],      0.2, "low"),   # critical if below 0.2
            ("hunger",      npc.needs["hunger"],       0.7, "high"),  # critical if above 0.7
            ("social_need", npc.needs["social_need"],  0.8, "high"),  # critical if above 0.8
        ]

        for name, value, threshold, direction in needs_priority:
            if direction == "low" and value < threshold:
                return name
            elif direction == "high" and value > threshold:
                return name
        return None

    @staticmethod
    def get_satisfaction_score(npc):
        """Get overall satisfaction 0-1."""
        return (
            (1.0 - npc.needs["hunger"]) * 0.3 +
            npc.needs["energy"] * 0.3 +
            (1.0 - npc.needs["social_need"]) * 0.2 +
            npc.behavior_vector["mood"] * 0.2
        )

    @staticmethod
    def get_need_zone(need_name):
        """Get the zone where a need can be satisfied."""
        zone_map = {
            "hunger":      "town_square",
            "energy":      "noble_house_w2",
            "social_need": "town_square",
        }
        return zone_map.get(need_name, "town_square")
