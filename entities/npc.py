"""
Base NPC class with behavior vector, needs, state machine, and pathfinding.
"""
import pygame
import math
import random
from utils.sprite_assets import SpriteAssets
from config import (
    TILE_SIZE, MAP_WIDTH, MAP_HEIGHT, COLORS,
    NEED_DECAY, NEED_RESTORE_RATE, NPC_SPEED, ZONES
)
from utils.pathfinding import find_path, pixel_to_tile, tile_to_pixel


class NPCState:
    IDLE = "idle"
    WALKING = "walking"
    WORKING = "working"
    SOCIALIZING = "socializing"
    SLEEPING = "sleeping"
    EATING = "eating"
    TALKING_TO_PLAYER = "talking"


class NPC:
    """Base NPC with AI behavior, needs, and pathfinding."""

    def __init__(self, name, npc_class, x, y, tile_map, world):
        self.name = name
        self.npc_class = npc_class
        self.x = float(x)
        self.y = float(y)
        self.tile_map = tile_map
        self.world = world

        self.speed = NPC_SPEED.get(npc_class, 35)
        self.radius = 7
        self.state = NPCState.IDLE
        self.state_timer = 0.0
        self.anim_timer = 0.0

        # ─── Behavior Vector (for unsupervised learning) ────────────
        self.behavior_vector = {
            "mood":        random.uniform(0.5, 0.9),
            "energy":      random.uniform(0.6, 1.0),
            "hunger":      random.uniform(0.0, 0.3),
            "social_need": random.uniform(0.1, 0.5),
            "wealth":      self._initial_wealth(),
            "trust":       0.5,  # trust toward player
        }

        # ─── Needs ──────────────────────────────────────────────────
        self.needs = {
            "hunger":      random.uniform(0.0, 0.3),
            "energy":      random.uniform(0.7, 1.0),
            "social_need": random.uniform(0.3, 0.7),
        }

        # ─── Personality traits (fixed per NPC) ─────────────────────
        self.personality = {
            "friendliness": random.uniform(0.3, 0.9),
            "work_ethic":   random.uniform(0.3, 0.9),
            "sociability":  random.uniform(0.2, 0.8),
            "aggression":   random.uniform(0.0, 0.5),
        }

        # ─── Pathfinding ────────────────────────────────────────────
        self.path = []
        self.path_index = 0
        self.target_position = None
        self.stuck_timer = 0.0
        self.repath_cooldown = 0.0
        self.last_progress_pos = (self.x, self.y)
        self.path_fail_count = 0

        # ─── Daily Routine ──────────────────────────────────────────
        self.routine = []  # Set by subclass
        self.current_routine_index = 0
        self.active_routine_zone = None
        self.active_routine_activity = ""
        self.target_zone_name = None
        self.is_indoors = False
        self.indoor_zone = None
        self.raid_shelter_mode = False

        # ─── Interaction History ────────────────────────────────────
        self.dialogue_history = []
        self.interaction_count = 0
        self.last_player_sentiment = 0.0

        # ─── ML Cluster Assignment ──────────────────────────────────
        self.cluster_id = -1
        self.social_group = -1  # DBSCAN spatial cluster

        # ─── Color ──────────────────────────────────────────────────
        color_key = f"npc_{npc_class.lower()}"
        self.color = COLORS.get(color_key, (150, 150, 150))
        self.outfit_tint = self._compute_outfit_tint()

        # ─── Idle wander timer ──────────────────────────────────────
        self.idle_timer = 0.0
        self.idle_duration = random.uniform(2.0, 5.0)
        self.behavior_pattern = self._assign_behavior_pattern()
        self.pattern_cooldown = random.uniform(1.6, 3.8)
        self.last_social_ping = 0.0

        # ─── Emotion display ───────────────────────────────────────
        self.emotion_display = None  # (emoji_text, timer)
        self.speech_bubble = None    # (text, timer)
        self.soft_action_hint = None

        # Stable home assignment so each NPC has a consistent house.
        self.home_zone = self._assign_home_zone()

    def _initial_wealth(self):
        wealth_map = {
            "Royal": 0.95, "Noble": 0.8, "Elite": 0.7,
            "Merchant": 0.65, "Blacksmith": 0.5,
            "Traveller": 0.4, "Labourer": 0.25, "Peasant": 0.15
        }
        base = wealth_map.get(self.npc_class, 0.3)
        return max(0.0, min(1.0, base + random.uniform(-0.1, 0.1)))

    def _compute_outfit_tint(self):
        """Class-based tint so all NPC classes are visually distinct."""
        palette = {
            "Royal": (228, 190, 72),
            "Noble": (166, 114, 206),
            "Elite": (196, 86, 82),
            "Merchant": (96, 176, 104),
            "Blacksmith": (122, 126, 136),
            "Traveller": (86, 168, 198),
            "Labourer": (188, 148, 84),
            "Peasant": (156, 122, 94),
        }
        base = palette.get(self.npc_class, (150, 150, 150))
        return (
            int((base[0] * 0.7) + (self.color[0] * 0.3)),
            int((base[1] * 0.7) + (self.color[1] * 0.3)),
            int((base[2] * 0.7) + (self.color[2] * 0.3)),
        )

    def get_behavior_array(self):
        """Return behavior vector as a numpy-compatible list."""
        return [
            self.behavior_vector["mood"],
            self.behavior_vector["energy"],
            self.behavior_vector["hunger"],
            self.behavior_vector["social_need"],
            self.behavior_vector["wealth"],
            self.behavior_vector["trust"],
        ]

    def update(self, dt, day_cycle):
        """Update NPC state, needs, and movement."""
        self.anim_timer += dt
        self.state_timer += dt
        self.repath_cooldown = max(0.0, self.repath_cooldown - dt)

        # Update needs decay
        self._update_needs(dt, day_cycle)

        # Update behavior vector from needs
        self.behavior_vector["hunger"] = self.needs["hunger"]
        self.behavior_vector["energy"] = self.needs["energy"]
        self.behavior_vector["social_need"] = self.needs["social_need"]

        # State machine
        if self.state == NPCState.TALKING_TO_PLAYER:
            return  # Frozen while talking

        if self.is_indoors:
            # Indoors NPCs remain in-place and recover needs naturally.
            if self.state != NPCState.SLEEPING and self.raid_shelter_mode:
                self.state = NPCState.SLEEPING
                self.state_timer = 0.0
            return

        if self.state == NPCState.WALKING:
            self._follow_path(dt)
        elif self.state == NPCState.IDLE:
            self.idle_timer += dt
            self.pattern_cooldown = max(0.0, self.pattern_cooldown - dt)
            self.last_social_ping += dt
            self._ambient_social_ping()
            if self.idle_timer >= self.idle_duration:
                self._decide_action(day_cycle)
                self.idle_timer = 0
                self.idle_duration = random.uniform(2.0, 5.0)
        elif self.state == NPCState.WORKING:
            if self.state_timer > random.uniform(8, 15):
                self.state = NPCState.IDLE
                self.state_timer = 0
        elif self.state == NPCState.SOCIALIZING:
            self.needs["social_need"] = max(0, self.needs["social_need"] - 0.01 * dt)
            if self.state_timer > random.uniform(5, 10):
                self.state = NPCState.IDLE
                self.state_timer = 0
        elif self.state == NPCState.EATING:
            self.needs["hunger"] = max(0, self.needs["hunger"] - NEED_RESTORE_RATE * dt)
            if self.state_timer > 5 or self.needs["hunger"] < 0.1:
                self.state = NPCState.IDLE
                self.state_timer = 0
        elif self.state == NPCState.SLEEPING:
            self.needs["energy"] = min(1.0, self.needs["energy"] + NEED_RESTORE_RATE * dt)
            if self.state_timer > 10 or self.needs["energy"] > 0.9:
                self.state = NPCState.IDLE
                self.state_timer = 0

        # Update emotion display timers
        if self.emotion_display:
            self.emotion_display = (self.emotion_display[0], self.emotion_display[1] - dt)
            if self.emotion_display[1] <= 0:
                self.emotion_display = None
        if self.speech_bubble:
            self.speech_bubble = (self.speech_bubble[0], self.speech_bubble[1] - dt)
            if self.speech_bubble[1] <= 0:
                self.speech_bubble = None

    def _update_needs(self, dt, day_cycle):
        """Decay needs over time."""
        self.needs["hunger"] = min(1.0, self.needs["hunger"] + NEED_DECAY["hunger"] * dt)
        if not day_cycle.is_sleep_hours():
            self.needs["energy"] = max(0.0, self.needs["energy"] - NEED_DECAY["energy"] * dt)
        self.needs["social_need"] = min(1.0, self.needs["social_need"] + NEED_DECAY["social_need"] * dt)

        # Mood affected by needs
        avg_need_satisfaction = (
            (1.0 - self.needs["hunger"]) +
            self.needs["energy"] +
            (1.0 - self.needs["social_need"])
        ) / 3.0
        self.behavior_vector["mood"] = (
            self.behavior_vector["mood"] * 0.95 +
            avg_need_satisfaction * 0.05
        )

    def _decide_action(self, day_cycle):
        """Decide what to do based on needs, time, and routine."""
        if self.soft_action_hint:
            hint = self.soft_action_hint
            self.soft_action_hint = None
            zone = hint.get("zone")
            action = hint.get("action", "work")
            if zone:
                self._go_to_zone(zone)
                self.state = NPCState.WALKING
                return
            if action == "sleep":
                self.state = NPCState.SLEEPING
                self.state_timer = 0.0
                return

        # Critical needs override
        if self.needs["hunger"] > 0.7:
            self._go_to_zone("town_square")
            self.state = NPCState.WALKING
            return

        if self.needs["energy"] < 0.2:
            self._go_to_zone(self._get_home_zone())
            self.state = NPCState.WALKING
            return

        # Always prefer explicit learned/default routine over ad-hoc roaming.
        if self._follow_time_routine(day_cycle):
            return

        # Fallback should still be intentional and class-consistent.
        if day_cycle.is_sleep_hours():
            self._go_to_zone(self._get_home_zone())
            self.state = NPCState.WALKING
        else:
            self._go_to_zone("town_square")
            self.state = NPCState.WALKING

    def _get_current_routine_entry(self, current_hour):
        """Return active routine entry for current hour (last entry not after current hour)."""
        if not self.routine:
            return None

        best = None
        hour = int(current_hour)
        for entry in self.routine:
            if entry.hour <= hour and (best is None or entry.hour > best.hour):
                best = entry

        if best is None:
            best = self.routine[-1]
        return best

    def _follow_time_routine(self, day_cycle):
        """Follow scheduled routine zones so NPCs do not roam mindlessly."""
        entry = self._get_current_routine_entry(getattr(day_cycle, "current_hour", 12))
        if entry is None:
            return False

        zone_name = str(getattr(entry, "zone", "") or "")
        if zone_name in ZONES:
            if self.is_indoors and zone_name != self.indoor_zone:
                self._exit_house()
            self.active_routine_zone = zone_name
            self.active_routine_activity = str(getattr(entry, "activity", "") or "")
            self._go_to_zone(zone_name)
            self.state = NPCState.WALKING
            return True
        return False

    def _state_from_routine_activity(self, activity_text, zone_name):
        """Map scheduled activity labels to concrete NPC states."""
        text = str(activity_text or "").lower()

        if any(k in text for k in ("sleep", "rest", "retire", "chambers", "go home", "return home")):
            return NPCState.SLEEPING
        if any(k in text for k in ("lunch", "meal", "eat", "dinner", "breakfast")):
            return NPCState.EATING
        if any(k in text for k in ("social", "visit", "stories", "tales", "prayer", "appearance", "explore")):
            return NPCState.SOCIALIZING
        if any(k in text for k in ("work", "duty", "patrol", "training", "drills", "trade", "forg", "harvest", "field", "chores", "tasks", "business", "court")):
            return NPCState.WORKING

        # Sensible fallback by zone type.
        if zone_name == "town_square":
            return NPCState.SOCIALIZING
        if zone_name and zone_name.startswith(("noble_house_", "peasant_house_", "trader_house_")):
            return NPCState.SLEEPING
        return NPCState.WORKING

    def _assign_behavior_pattern(self):
        """Assign a persistent behavior pattern from personality traits."""
        p = self.personality
        sociability = p.get("sociability", 0.5)
        work_ethic = p.get("work_ethic", 0.5)
        aggression = p.get("aggression", 0.2)
        friendliness = p.get("friendliness", 0.5)

        if aggression > 0.5 and work_ethic > 0.55:
            return "sentinel"
        if sociability > 0.7 and friendliness > 0.55:
            return "socializer"
        if work_ethic > 0.75:
            return "worker"
        if sociability < 0.35 and friendliness < 0.45:
            return "reserved"
        if self.npc_class in ("Traveller", "Merchant"):
            return "wanderer"
        return random.choice(["balanced", "socializer", "worker"])

    def _pattern_zone_candidates(self):
        """Preferred zones for each pattern to make NPC movement look intentional."""
        by_class = {
            "Royal": ["castle", "town_square", "noble_house_e1"],
            "Noble": ["noble_house_w1", "noble_house_e2", "town_square"],
            "Elite": ["castle", "town_square", "trader_house_2", "noble_house_w2"],
            "Merchant": ["trader_house_1", "trader_house_2", "town_square", "market"],
            "Blacksmith": ["trader_house_4", "town_square", "market"],
            "Traveller": ["town_square", "trader_house_1", "trader_house_3", "lake"],
            "Labourer": ["wheat_farm_w1", "wheat_farm_s1", "town_square", "granary_storage"],
            "Peasant": ["wheat_farm_w2", "peasant_house_2", "town_square", "granary_storage"],
        }
        default = by_class.get(self.npc_class, ["town_square"])

        if self.behavior_pattern == "sentinel":
            return ["castle", "town_square", "noble_house_w2"] + default
        if self.behavior_pattern == "socializer":
            return ["town_square", "market", "trader_house_1"] + default
        if self.behavior_pattern == "worker":
            return ["granary_storage", "market", "wheat_farm_w1"] + default
        if self.behavior_pattern == "wanderer":
            return ["town_square", "lake", "market", "trader_house_3"] + default
        if self.behavior_pattern == "reserved":
            return [self._get_home_zone(), "peasant_house_1", "noble_house_e3"] + default
        return default

    def _maybe_pattern_override(self, day_cycle):
        """Occasionally override routine with personality-driven behavior pattern."""
        if self.pattern_cooldown > 0.0:
            return False

        chance = 0.0
        if self.behavior_pattern == "socializer":
            chance = 0.42 if day_cycle.is_evening() else 0.28
        elif self.behavior_pattern == "worker":
            chance = 0.4 if day_cycle.is_work_hours() else 0.18
        elif self.behavior_pattern == "sentinel":
            chance = 0.36
        elif self.behavior_pattern == "wanderer":
            chance = 0.45
        elif self.behavior_pattern == "reserved":
            chance = 0.3 if day_cycle.is_evening() else 0.18
        else:
            chance = 0.22

        if random.random() >= chance:
            return False

        zone_candidates = [z for z in self._pattern_zone_candidates() if z in ZONES]
        if not zone_candidates:
            return False

        target_zone = random.choice(zone_candidates)
        self._go_to_zone(target_zone)
        self.state = NPCState.WALKING
        self.pattern_cooldown = random.uniform(4.0, 9.0)
        return True

    def _ambient_social_ping(self):
        """Small nearby social reactions so all NPCs feel alive outside explicit dialogue."""
        if self.last_social_ping < random.uniform(3.0, 6.0):
            return
        if self.state != NPCState.IDLE:
            return

        nearby = []
        for other in getattr(self, "world_npcs", []):
            if other is self:
                continue
            dx = other.x - self.x
            dy = other.y - self.y
            if (dx * dx + dy * dy) <= (64 * 64):
                nearby.append(other)

        if not nearby:
            self.last_social_ping = 0.0
            return

        other = random.choice(nearby)
        if self.behavior_pattern == "socializer" and random.random() < 0.45:
            self.show_emotion("🙂", 1.6)
            if random.random() < 0.25:
                self.show_speech("Good to see familiar faces.", 1.6)
            self.needs["social_need"] = max(0.0, self.needs["social_need"] - 0.03)
        elif self.behavior_pattern == "reserved" and random.random() < 0.3:
            self.show_emotion("😐", 1.4)
            self.needs["social_need"] = max(0.0, self.needs["social_need"] - 0.01)
        elif self.behavior_pattern == "sentinel" and random.random() < 0.35:
            self.show_emotion("👀", 1.2)
            if other.npc_class in ("Royal", "Noble"):
                self.show_speech("Area looks clear.", 1.3)

        self.last_social_ping = 0.0

    def _get_home_zone(self):
        """Return a class-appropriate home zone for resting/sleep."""
        return self.home_zone

    def _assign_home_zone(self):
        """Pick one deterministic home zone per NPC (stable across decisions)."""
        if self.npc_class == "Noble":
            choices = [
                "noble_house_w1", "noble_house_w2", "noble_house_w3",
                "noble_house_e1", "noble_house_e2", "noble_house_e3",
            ]
        elif self.npc_class in ("Merchant", "Blacksmith", "Traveller"):
            choices = ["trader_house_1", "trader_house_2", "trader_house_3", "trader_house_4"]
        elif self.npc_class in ("Labourer", "Peasant"):
            choices = ["peasant_house_1", "peasant_house_2", "peasant_house_3", "peasant_house_4"]
        elif self.npc_class == "Royal":
            choices = ["castle"]
        else:
            choices = ["peasant_house_2", "trader_house_2", "noble_house_e2"]

        seed = sum(ord(ch) for ch in f"{self.name}|{self.npc_class}")
        return choices[seed % len(choices)]

    def _do_work_routine(self):
        """Work routine — overridden by subclasses."""
        self._wander_near()

    def _do_evening_routine(self):
        """Evening routine — overridden by subclasses."""
        if random.random() < self.personality["sociability"]:
            self._go_to_zone("town_square")
        else:
            self._go_to_zone("town_square")
        self.state = NPCState.WALKING

    def _go_to_zone(self, zone_name):
        """Navigate to a zone."""
        self.target_zone_name = zone_name
        target = self.world.get_zone_random_point(zone_name)
        self._navigate_to(*target)

    def _enter_house(self, zone_name):
        """Mark NPC as indoors (sheltered/sleeping) when reaching home."""
        self.is_indoors = True
        self.indoor_zone = zone_name
        self.path = []
        self.path_index = 0
        self.target_position = None
        self.target_zone_name = zone_name
        self.state = NPCState.SLEEPING
        self.state_timer = 0.0
        center_x, center_y = self.world.get_zone_center(zone_name)
        self.x = float(center_x)
        self.y = float(center_y)

    def _exit_house(self):
        """Bring NPC back outside from indoors state when routine changes."""
        if not self.is_indoors:
            return
        zone = self.indoor_zone
        self.is_indoors = False
        self.indoor_zone = None
        self.raid_shelter_mode = False
        if zone and zone in self.world.buildings:
            ex, ey = self.world.buildings[zone].get_entrance()
            self.x = float(ex)
            self.y = float(ey + TILE_SIZE * 0.9)
        self.state = NPCState.IDLE
        self.state_timer = 0.0

    def _navigate_to(self, target_x, target_y):
        """Calculate path and start walking."""
        start_tile = pixel_to_tile(self.x, self.y)
        end_tile = pixel_to_tile(target_x, target_y)
        long_haul = abs(start_tile[0] - end_tile[0]) + abs(start_tile[1] - end_tile[1]) > 20
        self.path = find_path(start_tile, end_tile, self.tile_map.tiles, prefer_bidirectional=long_haul)
        self.path_index = 0
        self.target_position = (target_x, target_y)
        self.stuck_timer = 0.0
        self.last_progress_pos = (self.x, self.y)
        if self.path:
            self.state = NPCState.WALKING
            self.path_fail_count = 0
        else:
            self.path_fail_count += 1

    def _repath_to_target(self):
        """Recompute path when blocked or stalled to bypass fresh obstacles."""
        if self.target_position is None:
            return False
        if self.repath_cooldown > 0.0:
            return False

        self.repath_cooldown = 0.45
        start_tile = pixel_to_tile(self.x, self.y)
        end_tile = pixel_to_tile(*self.target_position)
        use_bidir = self.path_fail_count >= 1
        new_path = find_path(start_tile, end_tile, self.tile_map.tiles, prefer_bidirectional=use_bidir)
        if not new_path:
            self.path_fail_count += 1
            return False

        self.path = new_path
        self.path_index = 0
        self.state = NPCState.WALKING
        self.stuck_timer = 0.0
        self.last_progress_pos = (self.x, self.y)
        self.path_fail_count = 0
        return True

    def _try_local_detour(self, dir_x, dir_y, dt):
        """Differential side-step steering to slide around temporary blockages."""
        perp = [(-dir_y, dir_x), (dir_y, -dir_x)]
        best = None
        best_d2 = float("inf")
        speed = self.speed * 0.85

        for sx, sy in perp:
            nx = self.x + sx * speed * dt
            ny = self.y + sy * speed * dt
            if not self.tile_map.is_walkable_pixel(nx, ny, radius=12):
                continue
            if self.target_position is None:
                return (nx, ny)
            tx, ty = self.target_position
            d2 = (tx - nx) * (tx - nx) + (ty - ny) * (ty - ny)
            if d2 < best_d2:
                best_d2 = d2
                best = (nx, ny)

        return best

    def _follow_path(self, dt):
        """Follow the calculated path."""
        if not self.path or self.path_index >= len(self.path):
            self.state = NPCState.IDLE
            self.path = []
            self.path_index = 0
            # Determine what to do at destination
            if self.target_position:
                self._arrive_at_destination()
                self.target_position = None
            return

        target_tile = self.path[self.path_index]
        target_x, target_y = tile_to_pixel(*target_tile)

        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.sqrt(dx * dx + dy * dy)
        prev_x, prev_y = self.x, self.y

        if dist < 4:
            self.path_index += 1
        else:
            dx /= dist
            dy /= dist
            next_x = self.x + dx * self.speed * dt
            next_y = self.y + dy * self.speed * dt
            if self.tile_map.is_walkable_pixel(next_x, next_y, radius=12):
                self.x = next_x
                self.y = next_y
            else:
                detour = self._try_local_detour(dx, dy, dt)
                if detour is not None:
                    self.x, self.y = detour
                elif not self._repath_to_target():
                    # Could not detour/repath right now; remain walking and retry.
                    self.stuck_timer += dt

        moved = math.sqrt((self.x - prev_x) ** 2 + (self.y - prev_y) ** 2)
        if moved < 0.25 and dist > 6:
            self.stuck_timer += dt
        else:
            self.stuck_timer = max(0.0, self.stuck_timer - dt * 1.5)

        # If state-machine walking gets stuck, force a smarter reroute.
        if self.stuck_timer > 0.9:
            if self._repath_to_target():
                self.stuck_timer = 0.0
            else:
                # Last-resort: clear stale path and re-decide next tick.
                self.path = []
                self.path_index = 0
                self.state = NPCState.IDLE

    def _arrive_at_destination(self):
        """Called when NPC arrives at their target zone."""
        # Check what zone we're in
        zone_name, _ = self.world.get_building_at(self.x, self.y)
        if self.target_zone_name:
            zone_name = self.target_zone_name
        if zone_name is None and "town_square" in ZONES:
            tx, ty = int(self.x // TILE_SIZE), int(self.y // TILE_SIZE)
            zx, zy, zw, zh = ZONES["town_square"]
            if zx <= tx < zx + zw and zy <= ty < zy + zh:
                zone_name = "town_square"

        # Prefer explicit scheduled activity to avoid generic/mindless post-arrival behavior.
        routine_state = self._state_from_routine_activity(self.active_routine_activity, zone_name)
        if zone_name and zone_name.startswith(("noble_house_", "peasant_house_", "trader_house_")):
            if self.raid_shelter_mode or routine_state == NPCState.SLEEPING:
                self._enter_house(zone_name)
                return

        self.state = routine_state
        self.state_timer = 0

    def _wander_near(self):
        """Wander to a nearby random position."""
        for _ in range(10):
            ox = random.randint(-3, 3)
            oy = random.randint(-3, 3)
            tx, ty = pixel_to_tile(self.x, self.y)
            nx, ny = tx + ox, ty + oy
            if self.tile_map.is_walkable(nx, ny):
                target_x, target_y = tile_to_pixel(nx, ny)
                self._navigate_to(target_x, target_y)
                return

    def show_emotion(self, text, duration=3.0):
        """Show an emotion indicator above the NPC."""
        self.emotion_display = (text, duration)

    def show_speech(self, text, duration=4.0):
        """Show a speech bubble above the NPC."""
        self.speech_bubble = (text, duration)

    def react_to_threat(self, reaction_type, npcs=None):
        """Apply a class-based immediate reaction to a player threat."""
        self.behavior_vector["trust"] = max(0.0, self.behavior_vector["trust"] - 0.18)
        self.behavior_vector["mood"] = max(0.0, self.behavior_vector["mood"] - 0.12)

        if reaction_type == "flee_to_guard":
            self.show_emotion("😨", 3.0)
            self.show_speech("Help! Thief!", 3.0)
            self._go_to_zone("castle")
            self.state = NPCState.WALKING
            self._alert_nearby(npcs)
            return

        if reaction_type == "call_for_help":
            self.show_emotion("😠", 3.0)
            self.show_speech("Guards! To me!", 3.0)
            self._go_to_zone("castle")
            self.state = NPCState.WALKING
            self._alert_nearby(npcs)
            return

        if reaction_type == "fight":
            self.show_emotion("⚔️", 3.0)
            self.show_speech("Stand down, now!", 3.0)
            self.state = NPCState.WORKING
            self.state_timer = 0.0
            return

        # Default: freeze/helpless
        self.show_emotion("😰", 3.0)
        self.show_speech("Please don't hurt me...", 3.0)
        self.state = NPCState.IDLE
        self.state_timer = 0.0

    def _alert_nearby(self, npcs):
        """Alert nearby NPCs so the world visibly reacts to danger."""
        if not npcs:
            return
        for other in npcs:
            if other is self:
                continue
            dx = other.x - self.x
            dy = other.y - self.y
            if (dx * dx + dy * dy) <= (130 * 130):
                if other.npc_class in ("Elite", "Royal", "Noble"):
                    other.show_speech("I'm on it!", 2.2)
                    other.show_emotion("⚔️", 2.5)
                    other._go_to_zone("castle")
                    other.state = NPCState.WALKING
                else:
                    other.show_speech("Help!", 1.8)
                    other.show_emotion("😨", 2.0)

    def apply_sentiment_effect(self, sentiment_score):
        """Apply the effect of player interaction sentiment on NPC behavior."""
        self.last_player_sentiment = sentiment_score
        self.interaction_count += 1

        # Positive interaction: mood up, trust up
        if sentiment_score > 0.1:
            self.behavior_vector["mood"] = min(1.0, self.behavior_vector["mood"] + sentiment_score * 0.2)
            self.behavior_vector["trust"] = min(1.0, self.behavior_vector["trust"] + sentiment_score * 0.15)
            self.needs["social_need"] = max(0.0, self.needs["social_need"] - 0.2)
            if sentiment_score > 0.5:
                self.show_emotion("😊", 3.0)
            else:
                self.show_emotion("🙂", 2.0)

        # Negative interaction: mood down, trust down
        elif sentiment_score < -0.1:
            self.behavior_vector["mood"] = max(0.0, self.behavior_vector["mood"] + sentiment_score * 0.2)
            self.behavior_vector["trust"] = max(0.0, self.behavior_vector["trust"] + sentiment_score * 0.15)
            if sentiment_score < -0.5:
                self.show_emotion("😠", 3.0)
                # Very negative: may refuse to talk
                if self.behavior_vector["trust"] < 0.2:
                    self.show_speech("Leave me alone!", 3.0)
            else:
                self.show_emotion("😐", 2.0)

        # Neutral
        else:
            self.show_emotion("🤔", 2.0)

    def _class_motion_profile(self):
        """Per-class movement feel so NPC classes are visually distinct while walking/idling."""
        profile = {
            "Royal": {"cadence": 0.78, "walk_amp": 1.0, "idle_amp": 0.6},
            "Noble": {"cadence": 0.9, "walk_amp": 1.3, "idle_amp": 0.8},
            "Elite": {"cadence": 1.18, "walk_amp": 2.0, "idle_amp": 0.4},
            "Merchant": {"cadence": 1.0, "walk_amp": 1.5, "idle_amp": 0.7},
            "Blacksmith": {"cadence": 0.82, "walk_amp": 1.2, "idle_amp": 0.5},
            "Traveller": {"cadence": 1.25, "walk_amp": 2.2, "idle_amp": 0.9},
            "Labourer": {"cadence": 0.86, "walk_amp": 1.7, "idle_amp": 0.4},
            "Peasant": {"cadence": 0.95, "walk_amp": 1.4, "idle_amp": 0.6},
        }
        return profile.get(self.npc_class, {"cadence": 1.0, "walk_amp": 1.5, "idle_amp": 0.6})

    def _motion_animation_time(self):
        profile = self._class_motion_profile()
        return self.anim_timer * profile["cadence"]

    def _motion_bob(self):
        profile = self._class_motion_profile()
        if self.state == NPCState.WALKING:
            return int(math.sin(self.anim_timer * 8.0 * profile["cadence"]) * profile["walk_amp"])
        return int(math.sin(self.anim_timer * 2.0 * profile["cadence"]) * profile["idle_amp"])

    def render(self, surface, camera):
        """Render the NPC."""
        if self.is_indoors:
            return

        cam_rect = camera.get_visible_rect()
        sx = int(self.x - cam_rect.x)
        sy = int(self.y - cam_rect.y)

        # Don't render if off-screen
        if sx < -30 or sx > cam_rect.width + 30 or sy < -30 or sy > cam_rect.height + 30:
            return

        # Class-specific gait and idle posture motion.
        bob = self._motion_bob()
        anim_time = self._motion_animation_time()

        direction = "down"
        if self.path and 0 <= self.path_index < len(self.path):
            tx, ty = tile_to_pixel(*self.path[self.path_index])
            dx = tx - self.x
            dy = ty - self.y
            if abs(dx) > abs(dy):
                direction = "right" if dx > 0 else "left"
            else:
                direction = "down" if dy > 0 else "up"

        sprite = SpriteAssets.get().get_character(
            self.npc_class,
            direction,
            self.state == NPCState.WALKING,
            anim_time,
            tint=self.outfit_tint,
            tile_size=36,
            variant_seed=self.name,
        )
        if sprite is not None:
            shadow_w = 16
            pygame.draw.ellipse(surface, (0, 0, 0, 58), (sx - shadow_w // 2, sy + 4, shadow_w, 6))
            surface.blit(sprite, (sx - sprite.get_width() // 2, sy - sprite.get_height() + 9 + bob))

            # Overlay class icon on top of sprite so class identity remains explicit.
            self._draw_class_icon(surface, sx, sy - 2 + bob)

            # State indicator dot
            state_colors = {
                NPCState.IDLE: (200, 200, 200),
                NPCState.WALKING: (100, 200, 100),
                NPCState.WORKING: (200, 200, 50),
                NPCState.SOCIALIZING: (100, 150, 255),
                NPCState.EATING: (255, 150, 50),
                NPCState.SLEEPING: (100, 100, 200),
                NPCState.TALKING_TO_PLAYER: (255, 100, 255),
            }
            dot_color = state_colors.get(self.state, (200, 200, 200))
            pygame.draw.circle(surface, dot_color, (sx + 10, sy - 11 + bob), 3)

            font = pygame.font.SysFont("Arial", 10, bold=True)
            name_surf = font.render(self.name, True, (255, 255, 255))
            surface.blit(name_surf, (sx - name_surf.get_width() // 2, sy - self.radius - 16 + bob))

            # Mood bar
            bar_w = 20
            bar_h = 2
            bar_x = sx - bar_w // 2
            bar_y = sy - self.radius - 19 + bob
            mood = self.behavior_vector["mood"]
            pygame.draw.rect(surface, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h))
            mood_color = (int(255 * (1 - mood)), int(255 * mood), 50)
            pygame.draw.rect(surface, mood_color, (bar_x, bar_y, int(bar_w * mood), bar_h))

            if self.emotion_display:
                efont = pygame.font.SysFont("Segoe UI Emoji", 16)
                esurf = efont.render(self.emotion_display[0], True, (255, 255, 255))
                surface.blit(esurf, (sx - esurf.get_width() // 2, sy - 35 + bob))

            if self.speech_bubble:
                self._draw_speech_bubble(surface, sx, sy - 40 + bob, self.speech_bubble[0])
            return

        # Shadow
        shadow_surf = pygame.Surface((14, 6), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, 50), (0, 0, 14, 6))
        surface.blit(shadow_surf, (sx - 7, sy + 4))

        # Sleeping indicator
        if self.state == NPCState.SLEEPING:
            font = pygame.font.SysFont("Arial", 10)
            zzz = font.render("zzZ", True, (180, 180, 255))
            surface.blit(zzz, (sx + 8, sy - 20))

        # Character model: torso + head + tiny feet for less blob-like visuals
        outline = tuple(max(0, c - 40) for c in self.color)
        torso_w = self.radius + 5
        torso_h = self.radius + 6
        torso_rect = pygame.Rect(sx - torso_w // 2, sy - torso_h // 2 + bob - 1, torso_w, torso_h)
        pygame.draw.ellipse(surface, outline, torso_rect.inflate(2, 2))
        pygame.draw.ellipse(surface, self.color, torso_rect)

        head_color = tuple(min(255, c + 24) for c in self.color)
        head_y = sy - self.radius - 5 + bob
        pygame.draw.circle(surface, outline, (sx, head_y), 5)
        pygame.draw.circle(surface, head_color, (sx, head_y), 4)

        # Feet
        pygame.draw.circle(surface, (60, 48, 38), (sx - 3, sy + 3 + bob), 2)
        pygame.draw.circle(surface, (60, 48, 38), (sx + 3, sy + 3 + bob), 2)

        # Class icon/accent on torso
        self._draw_class_icon(surface, sx, sy - 2 + bob)

        # State indicator dot
        state_colors = {
            NPCState.IDLE: (200, 200, 200),
            NPCState.WALKING: (100, 200, 100),
            NPCState.WORKING: (200, 200, 50),
            NPCState.SOCIALIZING: (100, 150, 255),
            NPCState.EATING: (255, 150, 50),
            NPCState.SLEEPING: (100, 100, 200),
            NPCState.TALKING_TO_PLAYER: (255, 100, 255),
        }
        dot_color = state_colors.get(self.state, (200, 200, 200))
        pygame.draw.circle(surface, dot_color, (sx + self.radius + 2, sy - self.radius - 2 + bob), 3)

        # Name label
        font = pygame.font.SysFont("Arial", 10, bold=True)
        name_surf = font.render(self.name, True, (255, 255, 255))
        surface.blit(name_surf, (sx - name_surf.get_width() // 2, sy - self.radius - 14 + bob))

        # Mood bar
        bar_w = 20
        bar_h = 2
        bar_x = sx - bar_w // 2
        bar_y = sy - self.radius - 17 + bob
        mood = self.behavior_vector["mood"]
        pygame.draw.rect(surface, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h))
        mood_color = (
            int(255 * (1 - mood)),
            int(255 * mood),
            50
        )
        pygame.draw.rect(surface, mood_color, (bar_x, bar_y, int(bar_w * mood), bar_h))

        # Emotion display
        if self.emotion_display:
            efont = pygame.font.SysFont("Segoe UI Emoji", 16)
            esurf = efont.render(self.emotion_display[0], True, (255, 255, 255))
            surface.blit(esurf, (sx - esurf.get_width() // 2, sy - 35 + bob))

        # Speech bubble
        if self.speech_bubble:
            self._draw_speech_bubble(surface, sx, sy - 40 + bob, self.speech_bubble[0])

    def _draw_class_icon(self, surface, cx, cy):
        """Draw a small icon on the NPC body to indicate class."""
        # Default: small diamond
        size = 3
        pygame.draw.polygon(surface, (255, 255, 255, 180), [
            (cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)
        ])

    def _draw_speech_bubble(self, surface, x, y, text):
        """Draw a speech bubble above the NPC."""
        font = pygame.font.SysFont("Arial", 10)
        text_surf = font.render(text, True, (30, 30, 30))
        bw = text_surf.get_width() + 10
        bh = text_surf.get_height() + 6

        bubble_surf = pygame.Surface((bw, bh + 6), pygame.SRCALPHA)
        pygame.draw.rect(bubble_surf, (240, 240, 240, 230), (0, 0, bw, bh), border_radius=4)
        pygame.draw.rect(bubble_surf, (180, 180, 180), (0, 0, bw, bh), 1, border_radius=4)
        # Tail
        pygame.draw.polygon(bubble_surf, (240, 240, 240, 230), [
            (bw // 2 - 4, bh), (bw // 2, bh + 5), (bw // 2 + 4, bh)
        ])
        bubble_surf.blit(text_surf, (5, 3))

        surface.blit(bubble_surf, (x - bw // 2, y - bh - 5))

    def get_info_dict(self):
        """Get NPC info for UI display."""
        return {
            "name": self.name,
            "class": self.npc_class,
            "state": self.state,
            "mood": self.behavior_vector["mood"],
            "energy": self.needs["energy"],
            "hunger": self.needs["hunger"],
            "social": self.needs["social_need"],
            "trust": self.behavior_vector["trust"],
            "cluster": self.cluster_id,
        }
