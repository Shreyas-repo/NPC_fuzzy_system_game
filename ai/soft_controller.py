"""Fuzzy controller for explainable NPC action selection (Mamdani-like scoring)."""
import math
from config import FUZZY_ACTION_HYSTERESIS, FUZZY_SWITCH_MARGIN, FUZZY_MEMBERSHIP


class FuzzySocialController:
    def __init__(self):
        self.weights = {
            "eat": 1.0,
            "sleep": 1.0,
            "socialize": 1.0,
            "work": 1.0,
            "flee": 1.0,
            "guard": 1.0,
        }
        self.membership_cfg = dict(FUZZY_MEMBERSHIP)
        self.hysteresis = float(FUZZY_ACTION_HYSTERESIS)
        self.switch_margin = float(FUZZY_SWITCH_MARGIN)

    def set_weights(self, weights):
        for k in self.weights:
            if k in weights:
                self.weights[k] = float(weights[k])

    def get_weights(self):
        return dict(self.weights)

    def recommend(self, npc, npcs, prev_action=None, neural_biases=None, novelty_score=None):
        hunger = float(npc.needs.get("hunger", 0.0))
        energy = float(npc.needs.get("energy", 0.0))
        social = float(npc.needs.get("social_need", 0.0))
        trust = float(npc.behavior_vector.get("trust", 0.5))
        mood = float(npc.behavior_vector.get("mood", 0.5))
        threat = max(0.0, -float(getattr(npc, "last_player_sentiment", 0.0)))
        crowd = self._crowd_density(npc, npcs)

        hunger_high = self._membership_high(hunger, "hunger_high")
        low_energy_high = self._membership_high(1.0 - energy, "low_energy_high")
        social_high = self._membership_high(social, "social_need_high")
        threat_high = self._membership_high(threat, "threat_high")
        low_trust_high = self._membership_high(1.0 - trust, "low_trust_high")
        low_mood_high = self._membership_high(1.0 - mood, "low_mood_high")
        crowd_low = self._membership_low(crowd, "crowd_low")

        scores = {
            "eat": 0.0,
            "sleep": 0.0,
            "socialize": 0.0,
            "work": 0.0,
            "flee": 0.0,
            "guard": 0.0,
        }

        # Mamdani-style fuzzy rules (max-min composition via weighted activations)
        scores["eat"] = max(scores["eat"], min(hunger_high, 0.96))
        scores["sleep"] = max(scores["sleep"], min(low_energy_high, 0.96))
        scores["socialize"] = max(scores["socialize"], min(social_high, 1.0 - threat_high * 0.7))
        scores["work"] = max(scores["work"], min(energy, max(0.0, mood), 1.0 - threat_high * 0.6))

        threat_drive = max(0.0, min(1.0, threat_high * 0.9 + low_trust_high * 0.5 + low_mood_high * 0.3))
        if npc.npc_class in ("Peasant", "Labourer", "Merchant", "Traveller"):
            scores["flee"] = max(scores["flee"], min(1.0, threat_drive + (0.2 * crowd_low)))
        if npc.npc_class in ("Elite", "Royal", "Noble", "Blacksmith"):
            scores["guard"] = max(scores["guard"], min(1.0, threat_drive * 0.95 + 0.2))

        # Apply tunable policy multipliers.
        for k in scores:
            scores[k] *= self.weights[k]

        # ── Neural network bias injection ──
        # Additively blend learned biases from the neural dialogue network.
        if neural_biases and isinstance(neural_biases, dict):
            for k in scores:
                bias = float(neural_biases.get(k, 0.0))
                scores[k] = max(0.0, min(2.0, scores[k] + bias))
        # ── Novelty bias injection (unsupervised autoencoder) ──
        if novelty_score is not None and isinstance(novelty_score, (int, float)):
            # High novelty encourages exploratory actions (socialize, flee)
            if novelty_score > 0.6:
                # Boost socialize and flee modestly, dampen eat/sleep to avoid routine
                scores["socialize"] = max(0.0, min(2.0, scores["socialize"] + 0.2 * novelty_score))
                scores["flee"]       = max(0.0, min(2.0, scores["flee"]       + 0.15 * novelty_score))
                scores["eat"]        = max(0.0, min(2.0, scores["eat"]        - 0.1 * novelty_score))
                scores["sleep"]      = max(0.0, min(2.0, scores["sleep"]      - 0.1 * novelty_score))

        action, confidence, switched = self._stabilized_action(scores, prev_action)
        zone = self._action_to_zone(action, npc)
        return {
            "action": action,
            "zone": zone,
            "scores": scores,
            "confidence": confidence,
            "switched": switched,
            "inputs": {
                "hunger": hunger,
                "energy": energy,
                "social": social,
                "trust": trust,
                "mood": mood,
                "threat": threat,
                "crowd": crowd,
            },
        }

    def _membership_high(self, x, key):
        start, full = self.membership_cfg.get(key, (0.4, 0.8))
        x = max(0.0, min(1.0, float(x)))
        if x <= start:
            return 0.0
        if x >= full:
            return 1.0
        width = max(1e-6, full - start)
        return (x - start) / width

    def _membership_low(self, x, key):
        full_low, end_low = self.membership_cfg.get(key, (0.2, 0.6))
        x = max(0.0, min(1.0, float(x)))
        if x <= full_low:
            return 1.0
        if x >= end_low:
            return 0.0
        width = max(1e-6, end_low - full_low)
        return (end_low - x) / width

    def _stabilized_action(self, scores, prev_action):
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_action, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        switched = best_action != prev_action

        if prev_action in scores:
            scores[prev_action] = scores[prev_action] + self.hysteresis
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            best_action, best_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else 0.0
            if prev_action != best_action and (best_score - second_score) < self.switch_margin:
                best_action = prev_action
                best_score = scores[prev_action]

        switched = best_action != prev_action
        return best_action, float(best_score), switched

    def _action_to_zone(self, action, npc):
        if action == "eat":
            return "town_square"
        if action == "sleep":
            if npc.npc_class == "Noble":
                return "noble_house_w2" if hash(npc.name) % 2 == 0 else "noble_house_e2"
            if npc.npc_class in ("Merchant", "Blacksmith", "Traveller"):
                return "trader_house_2" if hash(npc.name) % 2 == 0 else "trader_house_3"
            if npc.npc_class in ("Labourer", "Peasant"):
                return "peasant_house_2" if hash(npc.name) % 2 == 0 else "peasant_house_3"
            return "castle"
        if action == "socialize":
            return "town_square"
        if action == "guard":
            return "castle"
        if action == "flee":
            return "castle"
        # work
        class_zone = {
            "Merchant": "trader_house_1",
            "Blacksmith": "trader_house_4",
            "Elite": "castle",
            "Royal": "castle",
            "Noble": "noble_house_w1",
            "Traveller": "trader_house_2",
            "Labourer": "wheat_farm_w1",
            "Peasant": "wheat_farm_w2",
        }
        return class_zone.get(npc.npc_class, "town_square")

    def _crowd_density(self, npc, npcs):
        nearby = 0
        for other in npcs:
            if other is npc:
                continue
            dx = other.x - npc.x
            dy = other.y - npc.y
            if dx * dx + dy * dy <= (84 * 84):
                nearby += 1
        return max(0.0, min(1.0, nearby / 6.0))
