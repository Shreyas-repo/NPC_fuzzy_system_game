"""
Social interaction system — NPC-to-NPC relationships and influence propagation.
"""
import json
import os
import random
import math
import hashlib
from config import (
    SOCIAL_CHATTER_HEARING_RADIUS_BASE,
    SOCIAL_CHATTER_HEARING_RADIUS_ALERT,
    SOCIAL_CHATTER_HEAR_PROB_SAME_GROUP,
    SOCIAL_CHATTER_HEAR_PROB_OUTSIDER,
    SOCIAL_CHATTER_TRUST_HIT_PROB,
    SOCIAL_CHATTER_MEMORY_PROB,
    ADAPTIVE_PROFILE_PRESETS,
)


SOCIAL_MEMORY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "npc_social_memory.json",
)


class SocialSystem:
    """Manages NPC-to-NPC social interactions and relationships."""

    def __init__(self):
        self.relationships = {}  # pair_key -> affinity score
        self.interaction_timer = 0.0
        self.interaction_interval = 2.2  # Check for interactions every few seconds
        self.recent_chats = []
        self.max_recent_chats = 160
        self.rumor_knowledge = {}      # npc_name -> set(rumor_id)
        self.rumor_library = {}        # rumor_id -> rumor payload
        self.rumor_counter = 0
        self.npc_profiles = {}         # npc_name -> profile dict
        self.pair_memories = {}        # pair_key -> list[exchange dict]
        self.memory_dirty = False
        self.save_timer = 0.0
        self.save_interval = 12.0
        self.conversation_logger = None
        self._last_npcs = []
        self.hearing_radius_base = float(SOCIAL_CHATTER_HEARING_RADIUS_BASE)
        self.hearing_radius_alert = float(SOCIAL_CHATTER_HEARING_RADIUS_ALERT)
        self.hear_prob_same_group = float(SOCIAL_CHATTER_HEAR_PROB_SAME_GROUP)
        self.hear_prob_outsider = float(SOCIAL_CHATTER_HEAR_PROB_OUTSIDER)
        self.trust_hit_prob = float(SOCIAL_CHATTER_TRUST_HIT_PROB)
        self.memory_prob = float(SOCIAL_CHATTER_MEMORY_PROB)
        self.runtime_profile = "custom"
        self._load_memory()

    def set_adaptation_profile(self, profile_name):
        preset = dict((ADAPTIVE_PROFILE_PRESETS or {}).get(profile_name, {}))
        if not preset:
            return

        self.runtime_profile = str(profile_name)
        self.hearing_radius_base = float(preset.get("social_hearing_radius_base", self.hearing_radius_base))
        self.hearing_radius_alert = float(preset.get("social_hearing_radius_alert", self.hearing_radius_alert))
        self.hear_prob_same_group = float(preset.get("social_hear_prob_same_group", self.hear_prob_same_group))
        self.hear_prob_outsider = float(preset.get("social_hear_prob_outsider", self.hear_prob_outsider))
        self.trust_hit_prob = float(preset.get("social_trust_hit_prob", self.trust_hit_prob))
        self.memory_prob = float(preset.get("social_memory_prob", self.memory_prob))

    def set_conversation_logger(self, logger):
        """Attach optional CSV conversation logger for NPC-to-NPC chatter."""
        self.conversation_logger = logger

    def bootstrap_from_npcs(self, npcs):
        """Ensure all live NPCs have persistent social memory profiles."""
        changed = False
        for npc in npcs:
            self.rumor_knowledge.setdefault(npc.name, set())
            if npc.name not in self.npc_profiles:
                self.npc_profiles[npc.name] = self._new_profile(npc)
                changed = True
            else:
                profile = self.npc_profiles[npc.name]
                self._ensure_profile_identity(npc, profile)
                profile["npc_class"] = npc.npc_class
                profile["last_mood"] = float(npc.behavior_vector.get("mood", 0.5))
                profile["last_trust"] = float(npc.behavior_vector.get("trust", 0.5))

                # Apply persistent affective bias from past life memory.
                npc.behavior_vector["mood"] = max(
                    0.0,
                    min(1.0, npc.behavior_vector.get("mood", 0.5) * 0.7 + profile.get("last_mood", 0.5) * 0.3),
                )
                npc.behavior_vector["trust"] = max(
                    0.0,
                    min(1.0, npc.behavior_vector.get("trust", 0.5) * 0.75 + profile.get("last_trust", 0.5) * 0.25),
                )

        if changed:
            self.memory_dirty = True
            self.save_memory()

    def save_memory(self):
        """Persist social memory to disk."""
        try:
            payload = {
                "relationships": self.relationships,
                "npc_profiles": self.npc_profiles,
                "pair_memories": {
                    k: v[-16:] for k, v in self.pair_memories.items()
                },
            }
            with open(SOCIAL_MEMORY_PATH, "w") as f:
                json.dump(payload, f, indent=2)
            self.memory_dirty = False
        except Exception as exc:
            print(f"Warning: Could not save social memory: {exc}")

    def _load_memory(self):
        if not os.path.exists(SOCIAL_MEMORY_PATH):
            return
        try:
            with open(SOCIAL_MEMORY_PATH, "r") as f:
                payload = json.load(f)
            self.relationships = {
                str(k): float(v)
                for k, v in payload.get("relationships", {}).items()
            }
            self.npc_profiles = dict(payload.get("npc_profiles", {}))
            self.pair_memories = {
                str(k): list(v)
                for k, v in payload.get("pair_memories", {}).items()
            }
            print(f"  Loaded social memory: {len(self.npc_profiles)} NPC profiles")
        except Exception as exc:
            print(f"Warning: Could not load social memory: {exc}")

    def _pair_key(self, npc_a, npc_b):
        left, right = sorted([npc_a.name, npc_b.name])
        return f"{left}|{right}"

    def _stable_seed(self, text):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return int(digest[:12], 16)

    def _pick_many_deterministic(self, items, count, key):
        if not items:
            return []
        if count >= len(items):
            return list(items)
        seed = self._stable_seed(key)
        start = seed % len(items)
        stride = (seed % (len(items) - 1)) + 1
        picked = []
        used = set()
        idx = start
        while len(picked) < count:
            if idx not in used:
                used.add(idx)
                picked.append(items[idx])
            idx = (idx + stride) % len(items)
        return picked

    def _ensure_profile_identity(self, npc, profile):
        style_options = ["plainspoken", "formal", "poetic", "streetwise", "wry", "gentle", "stern", "reflective"]
        humor_options = ["self-deprecating", "dry", "playful", "deadpan", "sarcastic", "awkward", "dark"]
        opener_options = ["listen", "truth be told", "between us", "honestly", "well", "look", "mark me", "to be fair"]
        address_styles = ["respectful", "neutral", "intimate", "teasing"]
        preferred_word_sets = [
            ["steady", "honest", "proper"],
            ["rough", "fair", "plain"],
            ["kind", "gentle", "patient"],
            ["sharp", "clean", "solid"],
            ["grim", "stubborn", "true"],
            ["quiet", "simple", "strong"],
        ]
        taboo_maps = [
            {"stupid": "foolish", "hate": "resent"},
            {"weak": "fragile", "coward": "fearful soul"},
            {"damn": "blasted", "shut up": "hold your tongue"},
            {"liar": "deceiver", "idiot": "dullard"},
        ]
        key = f"{npc.name}|{npc.npc_class}|voice"

        if "voice_style" not in profile:
            profile["voice_style"] = style_options[self._stable_seed(key + "|style") % len(style_options)]
        if "humor_style" not in profile:
            profile["humor_style"] = humor_options[self._stable_seed(key + "|humor") % len(humor_options)]
        if "speech_opener" not in profile:
            profile["speech_opener"] = opener_options[self._stable_seed(key + "|opener") % len(opener_options)]
        if "address_style" not in profile:
            profile["address_style"] = address_styles[self._stable_seed(key + "|address") % len(address_styles)]
        if "catchphrase" not in profile:
            mark = self._stable_seed(key + "|catch") % 100
            profile["catchphrase"] = f"{npc.name.split()[0]} keeps saying: steady now ({mark})."
        if "private_memory" not in profile:
            private_notes = [
                "once hid grain for neighbors",
                "keeps a letter never sent",
                "counts stars when stressed",
                "lost a friend in a bad winter",
                "still remembers a forbidden romance",
                "learned kindness from a stranger",
            ]
            profile["private_memory"] = private_notes[self._stable_seed(key + "|private") % len(private_notes)]
        if "preferred_words" not in profile:
            profile["preferred_words"] = preferred_word_sets[self._stable_seed(key + "|pref") % len(preferred_word_sets)]
        if "taboo_replacements" not in profile:
            profile["taboo_replacements"] = taboo_maps[self._stable_seed(key + "|taboo") % len(taboo_maps)]
        return profile

    def _new_profile(self, npc):
        hardship_pool = {
            "Peasant": ["thin harvest", "cold winter debt", "sick livestock"],
            "Labourer": ["back-breaking shifts", "wage cuts", "injury at work"],
            "Merchant": ["unstable prices", "late caravans", "stolen stock"],
            "Blacksmith": ["ore shortage", "forge repairs", "burn injuries"],
            "Traveller": ["road bandits", "loneliness", "uncertain shelter"],
            "Elite": ["night patrol strain", "political pressure", "public distrust"],
            "Noble": ["estate disputes", "court rivalry", "legacy pressure"],
            "Royal": ["burden of rule", "crown intrigue", "fear of revolt"],
        }
        desire_pool = {
            "Peasant": ["stable bread supply", "a safe family", "land of their own"],
            "Labourer": ["fair wages", "rest and dignity", "a better house"],
            "Merchant": ["market expansion", "trusted partners", "secure trade roads"],
            "Blacksmith": ["masterwork reputation", "steady ore flow", "guild respect"],
            "Traveller": ["belonging", "new stories", "safe passage"],
            "Elite": ["public order", "honor", "loyal troops"],
            "Noble": ["influence", "legacy", "political alliance"],
            "Royal": ["lasting peace", "loyal subjects", "stable succession"],
        }
        experience_pool = {
            "Peasant": ["lost crops one season", "shared food in famine", "helped rebuild a barn"],
            "Labourer": ["worked through storms", "organized workers", "saved a coworker"],
            "Merchant": ["survived a market crash", "won a risky bargain", "funded a shop stall"],
            "Blacksmith": ["forged arms for guards", "repaired town tools", "trained an apprentice"],
            "Traveller": ["crossed distant kingdoms", "escaped border raids", "carried news between towns"],
            "Elite": ["defended the square", "broke up raids", "escorted civilians"],
            "Noble": ["settled peasant disputes", "funded storage granaries", "hosted council talks"],
            "Royal": ["ended a tax riot", "negotiated truce", "ordered famine relief"],
        }

        cls = npc.npc_class
        profile = {
            "npc_class": cls,
            "hardships": self._pick_many_deterministic(
                hardship_pool.get(cls, ["uncertain times"]),
                2,
                f"{npc.name}|{cls}|hardships",
            ),
            "desires": self._pick_many_deterministic(
                desire_pool.get(cls, ["a better tomorrow"]),
                2,
                f"{npc.name}|{cls}|desires",
            ),
            "experiences": self._pick_many_deterministic(
                experience_pool.get(cls, ["lived through hard times"]),
                2,
                f"{npc.name}|{cls}|experiences",
            ),
            "romantic_memory": self._pick_many_deterministic([
                "once loved deeply",
                "still searching for true love",
                "protective of loved ones",
            ], 1, f"{npc.name}|{cls}|romance")[0],
            "bond_bias": {},
            "last_mood": float(npc.behavior_vector.get("mood", 0.5)),
            "last_trust": float(npc.behavior_vector.get("trust", 0.5)),
            "last_topic": "casual",
            "life_events": [],
            "private_memory": "",
        }
        return self._ensure_profile_identity(npc, profile)

    def remember_event(self, npc, event_text, impact=0.0, trust_shift=0.0, mood_shift=0.0, negative=False):
        """Store a persistent life event for one NPC and immediately shape behavior."""
        if npc is None:
            return
        profile = self.npc_profiles.get(npc.name)
        if not profile:
            profile = self._new_profile(npc)
            self.npc_profiles[npc.name] = profile

        events = profile.get("life_events", [])
        events.append(event_text)
        profile["life_events"] = events[-20:]

        experiences = profile.get("experiences", [])
        if event_text not in experiences and not negative:
            experiences.append(event_text)
            profile["experiences"] = experiences[-10:]

        if negative:
            hardships = profile.get("hardships", [])
            if event_text not in hardships:
                hardships.append(event_text)
            profile["hardships"] = hardships[-10:]

        if impact != 0.0:
            npc.behavior_vector["mood"] = max(0.0, min(1.0, npc.behavior_vector.get("mood", 0.5) + impact))
        if trust_shift != 0.0:
            npc.behavior_vector["trust"] = max(0.0, min(1.0, npc.behavior_vector.get("trust", 0.5) + trust_shift))
        if mood_shift != 0.0:
            npc.behavior_vector["mood"] = max(0.0, min(1.0, npc.behavior_vector.get("mood", 0.5) + mood_shift))

        profile["last_mood"] = float(npc.behavior_vector.get("mood", 0.5))
        profile["last_trust"] = float(npc.behavior_vector.get("trust", 0.5))
        self.memory_dirty = True

    def remember_group_event(self, npcs, event_text, trust_shift=0.0, mood_shift=0.0, negative=False):
        """Apply one persistent event across many NPCs."""
        for npc in npcs:
            self.remember_event(
                npc,
                event_text,
                trust_shift=trust_shift,
                mood_shift=mood_shift,
                negative=negative,
            )

    def update(self, dt, npcs):
        """Update social interactions between NPCs."""
        self._last_npcs = list(npcs or [])
        self.save_timer += dt
        if self.memory_dirty and self.save_timer >= self.save_interval:
            self.save_timer = 0.0
            self.save_memory()

        self.interaction_timer += dt
        if self.interaction_timer < self.interaction_interval:
            return
        self.interaction_timer = 0.0

        # Find NPCs near each other
        for i, npc_a in enumerate(npcs):
            if npc_a.state == "talking":
                continue
            for j, npc_b in enumerate(npcs):
                if i >= j:
                    continue
                if npc_b.state == "talking":
                    continue

                dist = math.sqrt((npc_a.x - npc_b.x)**2 + (npc_a.y - npc_b.y)**2)
                if dist < 50:  # Close enough to interact
                    self._social_interaction(npc_a, npc_b)

    def _social_interaction(self, npc_a, npc_b):
        """Process a social interaction between two NPCs."""
        key = self._pair_key(npc_a, npc_b)
        affinity = self.relationships.get(key, 0.5)

        # Social rank difference affects interaction
        rank_a = getattr(npc_a, 'social_rank', 1)
        rank_b = getattr(npc_b, 'social_rank', 1)
        rank_diff = abs(rank_a - rank_b)

        # Similar rank = more friendly interaction
        friendliness = (npc_a.personality["friendliness"] + npc_b.personality["friendliness"]) / 2
        interaction_quality = friendliness - rank_diff * 0.1 + random.uniform(-0.2, 0.2)
        interaction_quality = max(-1, min(1, interaction_quality))

        # Update relationship
        affinity += interaction_quality * 0.05
        affinity = max(0, min(1, affinity))
        self.relationships[key] = affinity
        self.memory_dirty = True

        self._update_profile_feelings(npc_a)
        self._update_profile_feelings(npc_b)

        tone = self._conversation_tone(npc_a, npc_b, affinity, interaction_quality, rank_diff)
        if tone:
            self._perform_conversation(npc_a, npc_b, tone)

        self._share_rumor(npc_a, npc_b, interaction_quality)

        # Mood influence
        if interaction_quality > 0.3:
            npc_a.behavior_vector["mood"] = min(1, npc_a.behavior_vector["mood"] + 0.02)
            npc_b.behavior_vector["mood"] = min(1, npc_b.behavior_vector["mood"] + 0.02)
            # Occasionally show social bubble
            if random.random() < 0.1:
                greetings = ["Hi!", "Hello!", "Good day!", "Hey!", "Well met!"]
                npc_a.show_speech(random.choice(greetings), 2.0)

        elif interaction_quality < -0.3:
            npc_a.behavior_vector["mood"] = max(0, npc_a.behavior_vector["mood"] - 0.02)
            npc_b.behavior_vector["mood"] = max(0, npc_b.behavior_vector["mood"] - 0.02)
            if random.random() < 0.05:
                npc_a.show_emotion("😤", 2.0)

        # Both NPCs get social need reduced
        npc_a.needs["social_need"] = max(0, npc_a.needs["social_need"] - 0.05)
        npc_b.needs["social_need"] = max(0, npc_b.needs["social_need"] - 0.05)

    def _cleanup_rumors(self):
        if not self.rumor_library:
            return
        stale_ids = []
        for rumor_id, rumor in self.rumor_library.items():
            rumor["ttl"] = int(rumor.get("ttl", 0)) - 1
            if rumor["ttl"] <= 0:
                stale_ids.append(rumor_id)
        if not stale_ids:
            return
        stale = set(stale_ids)
        for rumor_id in stale_ids:
            self.rumor_library.pop(rumor_id, None)
        for npc_name, known in self.rumor_knowledge.items():
            self.rumor_knowledge[npc_name] = {rid for rid in known if rid not in stale}

    def _rumor_payload(self, source_npc):
        profile = self.npc_profiles.get(source_npc.name)
        if not profile:
            profile = self._new_profile(source_npc)
            self.npc_profiles[source_npc.name] = profile

        hardship = random.choice(profile.get("hardships", ["hard times"]))
        desire = random.choice(profile.get("desires", ["better days"]))
        event = random.choice(profile.get("life_events", profile.get("experiences", ["an uneasy day"])))
        cls = source_npc.npc_class

        templates = [
            f"News from {cls.lower()} circles: {event} is changing how people work.",
            f"Rumor says {hardship} is getting worse near the market quarter.",
            f"People whisper that leaders may push for {desire} before next week.",
            "Travelers claim night patrol routes changed after recent unrest.",
            "Market talk says grain and trust rise or fall together this season.",
        ]
        text = random.choice(templates)
        self.rumor_counter += 1
        rumor_id = f"r{self.rumor_counter}"
        return rumor_id, {
            "text": text,
            "origin": source_npc.name,
            "credibility": random.uniform(0.45, 0.9),
            "ttl": random.randint(6, 14),
        }

    def _share_rumor(self, npc_a, npc_b, interaction_quality):
        if random.random() > 0.42:
            return

        self.rumor_knowledge.setdefault(npc_a.name, set())
        self.rumor_knowledge.setdefault(npc_b.name, set())

        # Keep rumor graph fresh over time.
        if random.random() < 0.2:
            self._cleanup_rumors()

        know_a = self.rumor_knowledge[npc_a.name]
        know_b = self.rumor_knowledge[npc_b.name]

        # Seed a new rumor if both know nothing yet.
        if not know_a and not know_b:
            rumor_id, rumor = self._rumor_payload(random.choice([npc_a, npc_b]))
            self.rumor_library[rumor_id] = rumor
            random.choice([know_a, know_b]).add(rumor_id)

        speak_from, hear_to = (npc_a, npc_b)
        if random.random() < 0.5:
            speak_from, hear_to = hear_to, speak_from

        known_from = self.rumor_knowledge.get(speak_from.name, set())
        known_to = self.rumor_knowledge.get(hear_to.name, set())
        candidates = [rid for rid in known_from if rid not in known_to and rid in self.rumor_library]

        if not candidates:
            if random.random() < 0.28:
                rumor_id, rumor = self._rumor_payload(speak_from)
                self.rumor_library[rumor_id] = rumor
                self.rumor_knowledge[speak_from.name].add(rumor_id)
                candidates = [rumor_id]
            else:
                return

        rid = random.choice(candidates)
        rumor = self.rumor_library.get(rid)
        if not rumor:
            return

        credibility = float(rumor.get("credibility", 0.6))
        trust_factor = float(hear_to.behavior_vector.get("trust", 0.5))
        friendliness = float(hear_to.personality.get("friendliness", 0.5))
        accept_chance = 0.35 + credibility * 0.35 + max(0.0, interaction_quality) * 0.2 + friendliness * 0.1 - (0.18 * (1.0 - trust_factor))
        if random.random() > max(0.1, min(0.92, accept_chance)):
            self._push_chat(hear_to, "I am not sure I believe that rumor.", "news")
            return

        self.rumor_knowledge[hear_to.name].add(rid)
        short_line = rumor["text"]
        self._push_chat(speak_from, f"I heard this: {short_line}", "news")
        self._push_chat(hear_to, "That is important. I will pass it along.", "news")

        # Rumors subtly shift social behavior and memory.
        mood_delta = 0.012 if "rise" in short_line or "better" in short_line else -0.012
        hear_to.behavior_vector["mood"] = max(0.0, min(1.0, hear_to.behavior_vector.get("mood", 0.5) + mood_delta))
        self.remember_event(
            hear_to,
            f"heard rumor: {short_line[:70]}",
            mood_shift=mood_delta,
            negative=mood_delta < 0,
        )

    def _pressure(self, npc):
        """Pressure is higher when hunger is high, energy is low, and wealth is low."""
        hunger = float(npc.needs.get("hunger", 0.0))
        low_energy = 1.0 - float(npc.needs.get("energy", 1.0))
        low_wealth = 1.0 - float(npc.behavior_vector.get("wealth", 0.5))
        pressure = 0.45 * hunger + 0.25 * low_energy + 0.3 * low_wealth
        return max(0.0, min(1.0, pressure))

    def _lie_drive(self, npc, pressure, rank_diff):
        friendliness = float(npc.personality.get("friendliness", 0.5))
        aggression = float(npc.personality.get("aggression", 0.2))
        insecurity = 1.0 - float(npc.behavior_vector.get("trust", 0.5))
        rank_fear = 0.2 if rank_diff >= 2 else 0.0
        drive = (1.0 - friendliness) * 0.45 + insecurity * 0.3 + pressure * 0.2 + aggression * 0.1 + rank_fear
        return max(0.0, min(1.0, drive))

    def _crime_drive(self, npc, pressure):
        aggression = float(npc.personality.get("aggression", 0.2))
        mood = float(npc.behavior_vector.get("mood", 0.5))
        wealth = float(npc.behavior_vector.get("wealth", 0.5))
        desperation = max(0.0, pressure - 0.45) * 1.2
        drive = aggression * 0.45 + desperation * 0.4 + (1.0 - wealth) * 0.15 + max(0.0, 0.45 - mood) * 0.25
        return max(0.0, min(1.0, drive))

    def _humor_drive(self, npc):
        mood = float(npc.behavior_vector.get("mood", 0.5))
        sociability = float(npc.personality.get("sociability", 0.5))
        friendliness = float(npc.personality.get("friendliness", 0.5))
        return max(0.0, min(1.0, mood * 0.45 + sociability * 0.35 + friendliness * 0.2))

    def _sadness_drive(self, npc, pressure):
        mood = float(npc.behavior_vector.get("mood", 0.5))
        energy = float(npc.needs.get("energy", 0.7))
        return max(0.0, min(1.0, (1.0 - mood) * 0.55 + pressure * 0.3 + (1.0 - energy) * 0.15))

    def _conversation_tone(self, npc_a, npc_b, affinity, interaction_quality, rank_diff):
        """Pick a dialogue tone: casual, deep, lie, crime, humor, or sadness."""
        pressure_a = self._pressure(npc_a)
        pressure_b = self._pressure(npc_b)
        lie_drive = max(
            self._lie_drive(npc_a, pressure_a, rank_diff),
            self._lie_drive(npc_b, pressure_b, rank_diff),
        )
        crime_drive = max(
            self._crime_drive(npc_a, pressure_a),
            self._crime_drive(npc_b, pressure_b),
        )
        humor_drive = max(
            self._humor_drive(npc_a),
            self._humor_drive(npc_b),
        )
        sadness_drive = max(
            self._sadness_drive(npc_a, pressure_a),
            self._sadness_drive(npc_b, pressure_b),
        )
        bond = self._bond_strength(npc_a, npc_b)

        if sadness_drive > 0.66 and random.random() < 0.35:
            return "sadness"
        if humor_drive > 0.58 and interaction_quality >= -0.1 and random.random() < 0.3:
            return "humor"
        if bond > 0.68 and interaction_quality > 0.1 and random.random() < 0.35:
            return "affection"

        if crime_drive > 0.74 and random.random() < 0.5:
            return "crime"
        if lie_drive > 0.64 and random.random() < 0.55:
            return "lie"
        if affinity > 0.62 or interaction_quality > 0.25 or random.random() < 0.35:
            return "deep"
        if random.random() < 0.55:
            return "casual"
        return None

    def _perform_conversation(self, npc_a, npc_b, tone):
        line_a, line_b = self._build_exchange(npc_a, npc_b, tone)
        line_a = self._apply_voice_style(npc_a, npc_b, line_a, tone)
        line_b = self._apply_voice_style(npc_b, npc_a, line_b, tone)
        duration = 3.5 if tone in ("casual", "lie") else 4.6

        npc_a.show_speech(line_a, duration)
        npc_b.show_speech(line_b, duration)
        npc_a.state = "socializing"
        npc_b.state = "socializing"
        npc_a.state_timer = 0.0
        npc_b.state_timer = 0.0

        self._push_chat(npc_a, line_a, tone, target_npc=npc_b)
        self._push_chat(npc_b, line_b, tone, target_npc=npc_a)
        self._record_exchange_memory(npc_a, npc_b, tone, line_a, line_b)

        if tone == "humor":
            npc_a.show_emotion(":)", 2.0)
            npc_b.show_emotion(":D", 2.0)
        elif tone == "sadness":
            npc_a.show_emotion(":(", 2.0)
            npc_b.show_emotion(":(", 2.0)

        if tone == "crime":
            self._maybe_commit_crime(npc_a, npc_b)

    def _peer_title(self, other):
        cls = getattr(other, "npc_class", "")
        if cls == "Royal":
            return "your grace"
        if cls in ("Noble", "Elite"):
            return "sir"
        if cls == "Blacksmith":
            return "smith"
        return "friend"

    def _apply_voice_style(self, npc, other, text, tone):
        profile = self.npc_profiles.get(npc.name)
        if not profile:
            return text

        out = (text or "").strip()
        if not out:
            return out

        style = profile.get("voice_style", "plainspoken")
        opener = profile.get("speech_opener", "well")
        humor_style = profile.get("humor_style", "dry")
        address_style = profile.get("address_style", "neutral")
        catchphrase = profile.get("catchphrase", "")
        preferred_words = profile.get("preferred_words", [])
        taboo_replacements = profile.get("taboo_replacements", {})
        bond = self._bond_strength(npc, other) if other is not None else 0.5

        # Personal taboo filter and vocabulary preference.
        for bad, repl in taboo_replacements.items():
            out = out.replace(bad, repl).replace(bad.title(), repl.title())

        if preferred_words and self._stable_seed(npc.name + out) % 3 == 0:
            out = out.replace("good", preferred_words[0]).replace("hard", preferred_words[-1])

        if style == "formal":
            out = out.replace("don't", "do not").replace("can't", "cannot")
        elif style == "streetwise":
            out = out.replace("them", "'em").replace("going", "goin")
        elif style == "poetic":
            out = out.replace("hard", "heavy").replace("good", "kind")
        elif style == "stern":
            out = out.replace("maybe", "likely")

        if tone == "humor":
            if humor_style in ("deadpan", "dry"):
                out = f"{opener}, {out}"
            elif humor_style in ("playful", "awkward"):
                out = f"{out} (and yes, that was a joke)"
            elif humor_style == "sarcastic":
                out = f"Sure, sure. {out}"
            elif humor_style == "dark":
                out = f"Strange laugh, but {out.lower()}"

        if tone in ("deep", "sadness") and self._stable_seed(npc.name + out) % 4 == 0:
            out = f"{opener}, {out}"

        # Relationship-dependent addressing style.
        if other is not None:
            title = self._peer_title(other)
            if address_style == "respectful" and bond < 0.5:
                out = f"{title}, {out}"
            elif address_style == "intimate" and bond > 0.68:
                out = f"my friend, {out}"
            elif address_style == "teasing" and tone == "humor":
                out = f"oh {title}, {out}"

        if catchphrase and self._stable_seed(npc.name + tone + out) % 13 == 0:
            out = f"{out} {catchphrase}"

        return out

    def _maybe_commit_crime(self, npc_a, npc_b):
        """Occasionally escalate criminal talk into an opportunistic theft event."""
        pressure_a = self._pressure(npc_a)
        pressure_b = self._pressure(npc_b)
        drive_a = self._crime_drive(npc_a, pressure_a)
        drive_b = self._crime_drive(npc_b, pressure_b)

        offender, witness = (npc_a, npc_b) if drive_a >= drive_b else (npc_b, npc_a)
        trigger_chance = max(drive_a, drive_b) * 0.45
        if random.random() >= trigger_chance:
            return

        # Crime event model: offender gains wealth at cost of trust/reputation.
        offender.behavior_vector["wealth"] = min(1.0, offender.behavior_vector.get("wealth", 0.5) + 0.035)
        offender.behavior_vector["trust"] = max(0.0, offender.behavior_vector.get("trust", 0.5) - 0.06)
        offender.behavior_vector["mood"] = min(1.0, offender.behavior_vector.get("mood", 0.5) + 0.03)

        offender.show_emotion("😈", 2.6)
        offender.show_speech("No one saw that. Keep quiet.", 3.6)
        witness.show_emotion("😰", 2.3)

        witness_rank = getattr(witness, "social_rank", 1)
        if witness_rank >= 3 and random.random() < 0.65:
            witness.show_speech("Guard! There was a theft!", 3.0)
            self._push_chat(witness, "Guard! There was a theft!", "crime")
        else:
            witness.show_speech("I saw nothing. I need to stay safe.", 3.0)
            self._push_chat(witness, "I saw nothing. I need to stay safe.", "lie")

    def _normalize_line(self, text):
        cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
        return " ".join(cleaned.split())

    def _recent_pair_signatures(self, npc_a, npc_b, lookback=8):
        pair_key = self._pair_key(npc_a, npc_b)
        recent = self.pair_memories.get(pair_key, [])[-lookback:]
        signatures = set()
        for item in recent:
            signatures.add(self._normalize_line(item.get("line_a", "")))
            signatures.add(self._normalize_line(item.get("line_b", "")))
        return signatures

    def _topic_hint(self, profile_a, profile_b):
        events = profile_a.get("life_events", [])[-3:] + profile_b.get("life_events", [])[-3:]
        merged = " ".join(events).lower()
        if "raid" in merged or "loot" in merged or "stole" in merged:
            return "raid"
        if "tax" in merged or "economy" in merged or "shortage" in merged:
            return "economy"
        if "harvest" in merged or "farm" in merged or "wheat" in merged:
            return "farming"
        if "court" in merged or "guard" in merged or "jail" in merged:
            return "law"
        return "life"

    def _pick_distinct_exchange(self, npc_a, npc_b, candidates):
        if not candidates:
            return ("...", "...")

        blocked = self._recent_pair_signatures(npc_a, npc_b)
        distinct = []
        for left, right in candidates:
            sig_left = self._normalize_line(left)
            sig_right = self._normalize_line(right)
            if sig_left in blocked or sig_right in blocked:
                continue
            distinct.append((left, right))

        pool = distinct if distinct else candidates
        return random.choice(pool)

    def _build_exchange(self, npc_a, npc_b, tone):
        b = npc_b.name
        lower_classes = {"Peasant", "Labourer", "Traveller"}
        upper_classes = {"Elite", "Noble", "Royal"}
        profile_a = self.npc_profiles.get(npc_a.name, self._new_profile(npc_a))
        profile_b = self.npc_profiles.get(npc_b.name, self._new_profile(npc_b))

        hardship_a = random.choice(profile_a.get("hardships", ["hard times"]))
        desire_a = random.choice(profile_a.get("desires", ["a calmer life"]))
        experience_a = random.choice(profile_a.get("experiences", ["lived through uncertainty"]))
        romantic_a = profile_a.get("romantic_memory", "protective of loved ones")
        recent_pair = self.pair_memories.get(self._pair_key(npc_a, npc_b), [])
        recent_topic = recent_pair[-1]["tone"] if recent_pair else None
        topic_hint = self._topic_hint(profile_a, profile_b)
        event_a = random.choice(profile_a.get("life_events", profile_a.get("experiences", ["strange days"])))
        event_b = random.choice(profile_b.get("life_events", profile_b.get("experiences", ["strange days"])))
        hardship_b = random.choice(profile_b.get("hardships", ["hard times"]))
        desire_b = random.choice(profile_b.get("desires", ["a calmer life"]))

        if tone == "humor":
            if npc_a.npc_class in lower_classes:
                lines = [
                    ("I am so poor, even my shadow clocks out at noon.",
                     "Ha! That one was better than court poetry."),
                    ("My purse is so light, the wind taxed it again.",
                     "Ha, true enough. You have sharp wit."),
                    ("I told my boots we are rich now. They laughed first.",
                     "Ha! Fair joke. Keep that spirit."),
                ]
            elif npc_a.npc_class in upper_classes and npc_b.npc_class in lower_classes:
                lines = [
                    ("Tell me a village joke, then. I need a laugh.",
                     "Fine: even the tax ledger blushes when it sees my wages."),
                    ("Go on, amuse me before the next patrol report.",
                     "All right: my roof leaks less than the market promises."),
                ]
            else:
                lines = [
                    ("If stress were coin, we would all be royalty.",
                     "Ha, then the treasury would finally love us."),
                    ("At least jokes are cheaper than medicine.",
                     "True. Laughter feeds the soul when bread is late."),
                ]
            return self._pick_distinct_exchange(npc_a, npc_b, lines)

        if tone == "sadness":
            lines = [
                (f"I am tired of carrying {hardship_a} like it is normal.",
                 "You should not have to carry it alone."),
                (f"After {event_a}, I smile less than I used to.",
                 "I noticed. The village has been heavy lately."),
                ("Some nights I fear this place is forgetting kindness.",
                 "Then we remember it for each other."),
            ]
            return self._pick_distinct_exchange(npc_a, npc_b, lines)

        if tone == "crime":
            lines = [
                (f"{b}, if coin stays thin, I may rob the granary tonight.",
                 "That is a crime. I want no part in it."),
                ("The guard route is weak by the market after dusk.",
                 "You are planning trouble. Think before you do this."),
                ("I could fake a delivery and lift supplies unseen.",
                 "If you get caught, the court will ruin you."),
                (f"After {event_a}, I stopped believing honest work is enough.",
                 "Desperation explains it, but it does not justify it."),
            ]
            return self._pick_distinct_exchange(npc_a, npc_b, lines)

        if tone == "affection":
            lines = [
                (f"After all our struggles, I still carry this: {romantic_a}.",
                 "Then let us protect what we love, not lose it to fear."),
                (f"My desire is simple now: {desire_a}. Maybe with you beside me.",
                 "I know. We keep each other steady through this life."),
                ("When the village feels cruel, your voice feels like home.",
                 "And your honesty keeps me brave."),
                (f"Even after {hardship_a}, I still choose tenderness with you.",
                 "That choice is rarer than gold in this world."),
            ]
            return self._pick_distinct_exchange(npc_a, npc_b, lines)

        if tone == "lie":
            lines = [
                (f"If the watch asks, tell them I worked beside you, {b}.",
                 "That is a lie, but it might keep you out of jail."),
                ("I will say the missing silver was counted wrong.",
                 "Your story is thin. Keep it consistent."),
                ("I never touched the ledger, and that is what I will swear.",
                 "You are gambling on a dangerous lie."),
                (f"If they ask about {event_a}, we tell the same version.",
                 "Then keep your nerves steady or we both fall."),
            ]
            return self._pick_distinct_exchange(npc_a, npc_b, lines)

        if tone == "deep":
            lines = [
                (f"My hardship lately is {hardship_a}, and I still keep going.",
                 "Your resilience matters more than rank."),
                (f"I remember when I {experience_a}. It changed me.",
                 "That memory is why people trust you now."),
                (f"Do you ever think our desires are too small? I just want {desire_a}.",
                 "Small desires keep people human."),
                ("If fear rules us, we become strangers to ourselves.",
                 "Then courage is choosing decency anyway."),
                (f"You carry {hardship_b} quietly. Does anyone really see it?",
                 "Maybe not, but I do. That should count for something."),
                (f"After {event_b}, what do you even hope for now?",
                 f"Honestly? {desire_b}. Nothing grand, just real."),
            ]
            if recent_topic in ("crime", "lie"):
                lines.append((
                    "Last time we strayed too close to wrongdoing. I don't want that path.",
                    "Then we learn from it and choose better now."
                ))
            if topic_hint == "raid":
                lines.append((
                    "The raid changed people. I hear fear in every market voice.",
                    "Fear can harden us, or teach us to protect each other better."
                ))
            if topic_hint == "economy":
                lines.append((
                    "Coin feels tighter every week. Even kind people snap sooner.",
                    "Then we share what we can, before bitterness becomes normal."
                ))
            if topic_hint == "farming":
                lines.append((
                    "When wheat grows well, people talk softer. Have you noticed?",
                    "Yes. Full storage brings patience back into conversation."
                ))
            return self._pick_distinct_exchange(npc_a, npc_b, lines)

        lines = [
            (f"Morning, {b}. Any news from the square?", "Only rumors and rising prices."),
            ("You look tired. Long shift?", "Too long. I need sleep and better pay."),
            ("How is work today?", "Steady enough. Could be worse."),
            (f"I heard about {event_a}. Are you holding up?", "Trying to. One day at a time."),
            ("Any peace in your home this week?", "A little. I'll take that as a blessing."),
            ("Did you eat today, at least?", "Barely, but I am still standing."),
        ]
        return self._pick_distinct_exchange(npc_a, npc_b, lines)

    def _update_profile_feelings(self, npc):
        profile = self.npc_profiles.get(npc.name)
        if not profile:
            profile = self._new_profile(npc)
            self.npc_profiles[npc.name] = profile

        profile["last_mood"] = float(npc.behavior_vector.get("mood", 0.5))
        profile["last_trust"] = float(npc.behavior_vector.get("trust", 0.5))

    def _bond_strength(self, npc_a, npc_b):
        key = self._pair_key(npc_a, npc_b)
        affinity = float(self.relationships.get(key, 0.5))
        profile_a = self.npc_profiles.get(npc_a.name, {})
        bias = float(profile_a.get("bond_bias", {}).get(npc_b.name, 0.0))
        return max(0.0, min(1.0, affinity * 0.8 + bias * 0.2))

    def _record_exchange_memory(self, npc_a, npc_b, tone, line_a, line_b):
        pair_key = self._pair_key(npc_a, npc_b)
        if pair_key not in self.pair_memories:
            self.pair_memories[pair_key] = []

        self.pair_memories[pair_key].append({
            "tone": tone,
            "speaker_a": npc_a.name,
            "line_a": line_a,
            "speaker_b": npc_b.name,
            "line_b": line_b,
        })
        self.pair_memories[pair_key] = self.pair_memories[pair_key][-16:]

        # Update personal memory and bond bias.
        for speaker, other in ((npc_a, npc_b), (npc_b, npc_a)):
            profile = self.npc_profiles.get(speaker.name)
            if not profile:
                profile = self._new_profile(speaker)
                self.npc_profiles[speaker.name] = profile
            profile["last_topic"] = tone
            bond_bias = profile.get("bond_bias", {})
            current = float(bond_bias.get(other.name, 0.0))
            if tone in ("deep", "affection"):
                current += 0.035
            elif tone == "humor":
                current += 0.02
            elif tone == "sadness":
                current += 0.012
            elif tone in ("lie", "crime"):
                current -= 0.03
            bond_bias[other.name] = max(-1.0, min(1.0, current))
            profile["bond_bias"] = bond_bias

        self.memory_dirty = True

    def _push_chat(self, npc, text, tone, target_npc=None):
        social_group = int(getattr(npc, "social_group", -1))
        behavior_cluster = int(getattr(npc, "cluster_id", -1))
        if social_group >= 0:
            cluster_key = f"group-{social_group}"
            cluster_label = f"Group {social_group}"
        elif behavior_cluster >= 0:
            cluster_key = f"behavior-{behavior_cluster}"
            cluster_label = f"Behavior {behavior_cluster}"
        else:
            cluster_key = "isolated"
            cluster_label = "Isolated"

        self.recent_chats.append({
            "speaker": npc.name,
            "npc_class": npc.npc_class,
            "text": text,
            "tone": tone,
            "cluster_key": cluster_key,
            "cluster_label": cluster_label,
            "social_group": social_group,
            "behavior_cluster": behavior_cluster,
        })
        if len(self.recent_chats) > self.max_recent_chats:
            self.recent_chats = self.recent_chats[-self.max_recent_chats:]

        if self.conversation_logger:
            tone_sentiment = {
                "humor": 0.35,
                "affection": 0.45,
                "deep": 0.1,
                "casual": 0.0,
                "news": 0.0,
                "lie": -0.15,
                "crime": -0.4,
                "sadness": -0.25,
            }
            sentiment = float(tone_sentiment.get(tone, 0.0))
            self.conversation_logger.log_exchange(
                npc,
                f"[NPC social/{tone}]",
                str(text),
                sentiment=sentiment,
                emotion={},
                source="npc_social",
            )
            self.conversation_logger.log_social_line(
                speaker=npc,
                text=str(text),
                tone=tone,
                sentiment=sentiment,
                target=target_npc,
                cluster_key=cluster_key,
                cluster_label=cluster_label,
                source="npc_social",
            )

        self._propagate_chatter_to_bystanders(
            speaker=npc,
            text=str(text),
            tone=str(tone),
            target_npc=target_npc,
        )

    def _propagate_chatter_to_bystanders(self, speaker, text, tone, target_npc=None):
        """Nearby NPCs can overhear chatter even without shared cluster membership."""
        if speaker is None or not self._last_npcs:
            return

        base_radius = self.hearing_radius_base
        if tone in ("news", "crime", "lie"):
            base_radius = self.hearing_radius_alert

        tone_shift = {
            "humor": 0.008,
            "affection": 0.01,
            "deep": 0.004,
            "casual": 0.002,
            "news": 0.0,
            "sadness": -0.006,
            "lie": -0.01,
            "crime": -0.02,
        }
        mood_delta = float(tone_shift.get(tone, 0.0))

        for listener in self._last_npcs:
            if listener is speaker or listener is target_npc:
                continue
            if getattr(listener, "state", "") == "talking":
                continue

            dx = float(listener.x - speaker.x)
            dy = float(listener.y - speaker.y)
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > base_radius:
                continue

            # Same social group hears most; outsiders still can hear ambiently.
            same_group = int(getattr(listener, "social_group", -1)) >= 0 and int(getattr(listener, "social_group", -1)) == int(getattr(speaker, "social_group", -2))
            hear_prob = self.hear_prob_same_group if same_group else self.hear_prob_outsider
            hear_prob *= max(0.25, 1.0 - (dist / (base_radius + 1.0)))

            if random.random() > hear_prob:
                continue

            listener.needs["social_need"] = max(0.0, listener.needs.get("social_need", 0.5) - 0.01)
            listener.behavior_vector["mood"] = max(
                0.0,
                min(1.0, listener.behavior_vector.get("mood", 0.5) + mood_delta),
            )

            if tone in ("crime", "lie") and random.random() < self.trust_hit_prob:
                listener.behavior_vector["trust"] = max(0.0, listener.behavior_vector.get("trust", 0.5) - 0.01)

            if random.random() < self.memory_prob:
                snippet = text[:64]
                self.remember_event(
                    listener,
                    f"overheard {tone} chatter: {snippet}",
                    mood_shift=mood_delta,
                    negative=mood_delta < 0,
                )

    def report_player_social_incident(self, npcs, source_npc, incident_type, style, severity, text):
        """Spread gossip from awkward player social moments and nudge village trust."""
        if source_npc is None:
            return

        sev = max(0.0, min(0.4, float(severity)))
        style = str(style or "neutral")
        incident_type = str(incident_type or "silent_prompt")

        tone_by_style = {
            "caring": "sadness",
            "playful": "humor",
            "neutral": "casual",
            "impatient": "sadness",
            "sarcastic": "humor",
            "stern": "lie",
        }
        tone = tone_by_style.get(style, "casual")

        # Source line appears in Village Chatter.
        self._push_chat(source_npc, text, tone)

        # One village gossip line to make the event socially visible.
        gossip_templates = [
            f"The player opened chat with {source_npc.name} and said nothing again.",
            f"People are saying the player acted strange with {source_npc.name}.",
            f"Word spreads fast: {source_npc.name} left that chat annoyed.",
        ]
        gossip_text = random.choice(gossip_templates)
        witness = None
        nearby = []
        for npc in (npcs or []):
            if npc is source_npc:
                continue
            d = math.sqrt((npc.x - source_npc.x) ** 2 + (npc.y - source_npc.y) ** 2)
            if d <= 130:
                nearby.append(npc)
        if nearby:
            witness = random.choice(nearby)
            self._push_chat(witness, gossip_text, "casual")

        # Broader trust impact toward player, stronger near source and for repeats.
        repeat_factor = 1.15 if incident_type == "silent_close" else 1.0
        for npc in (npcs or []):
            dist = math.sqrt((npc.x - source_npc.x) ** 2 + (npc.y - source_npc.y) ** 2)
            proximity = 1.0 if dist < 90 else (0.65 if dist < 180 else 0.35)
            friendliness = float(npc.personality.get("friendliness", 0.5))
            tolerance = 0.75 + friendliness * 0.4
            trust_hit = sev * 0.22 * proximity * repeat_factor / tolerance
            npc.behavior_vector["trust"] = max(0.0, min(1.0, npc.behavior_vector.get("trust", 0.5) - trust_hit))

            if trust_hit > 0.012 and random.random() < 0.22 * proximity:
                self.remember_event(
                    npc,
                    f"noticed awkward player behavior around {source_npc.name}",
                    impact=-0.01 * proximity,
                    trust_shift=-0.02 * proximity,
                    negative=True,
                )

        # Source NPC keeps the strongest memory of this specific moment.
        self.remember_event(
            source_npc,
            f"player interaction felt awkward ({style})",
            impact=-0.03,
            trust_shift=-0.06,
            negative=True,
        )

    def get_recent_chats(self, limit=8):
        """Expose latest NPC-to-NPC conversation lines for HUD rendering."""
        if limit <= 0:
            return []
        return self.recent_chats[-limit:]

    def get_chat_clusters(self, limit=220):
        """Return available chatter cluster buckets seen in recent chat history."""
        chats = self.get_recent_chats(limit=limit)
        buckets = {}
        for item in chats:
            key = item.get("cluster_key", "isolated")
            if key in buckets:
                continue
            buckets[key] = item.get("cluster_label", key.title())
        return buckets

    def get_recent_chats_for_cluster(self, cluster_key, limit=220):
        chats = self.get_recent_chats(limit=limit)
        if cluster_key in (None, "all"):
            return chats
        return [c for c in chats if c.get("cluster_key") == cluster_key]

    def get_recent_chats_for_listener(self, listener, limit=220, hearing_radius=220.0):
        """Return chatter a listener can plausibly hear, regardless of cluster tags."""
        chats = self.get_recent_chats(limit=limit)
        if listener is None:
            return chats

        by_name = {npc.name: npc for npc in self._last_npcs}
        heard = []
        r2 = float(hearing_radius) * float(hearing_radius)
        for item in chats:
            speaker_name = item.get("speaker")
            speaker = by_name.get(speaker_name)
            if speaker is None:
                heard.append(item)
                continue

            dx = float(speaker.x - listener.x)
            dy = float(speaker.y - listener.y)
            if (dx * dx + dy * dy) <= r2:
                heard.append(item)

        return heard

    def get_relationship(self, npc_a, npc_b):
        """Get the relationship score between two NPCs."""
        key = self._pair_key(npc_a, npc_b)
        return self.relationships.get(key, 0.5)
