"""
Daily routine engine — NPCs follow time-based schedules that adapt over time.
Uses a simplified Self-Organizing Map approach: NPCs observe where others go
at certain times and may adjust their own routines accordingly.
"""
import random

try:
    from config import ZONES
except ModuleNotFoundError:
    import os
    import sys

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from config import ZONES


class RoutineEntry:
    """A single routine entry: go to a zone at a certain hour."""

    def __init__(self, hour, zone, activity, priority=1):
        self.hour = hour          # 0-23
        self.zone = zone          # zone name from config
        self.activity = activity  # description
        self.priority = priority  # higher = more important
        self.times_followed = 0
        self.times_skipped = 0


class RoutineEngine:
    """
    Manages NPC daily routines and their adaptation over time.
    
    UNSUPERVISED LEARNING:
    NPCs observe where other NPCs go at certain times and may adapt
    their routines to match popular patterns. This creates emergent
    social behavior — e.g., if many NPCs go to the tavern at 7 PM,
    others will gradually learn to do the same.
    """

    def __init__(self):
        # Track where NPCs go at each hour (for learning)
        self.hourly_zone_visits = {}  # {hour: {zone: count}}
        self.observation_timer = 0.0
        self.adaptation_timer = 0.0
        self.adaptation_interval = 30.0  # Adapt routines every 30 seconds

    def get_current_routine_zone(self, npc, current_hour):
        """Get the zone an NPC should go to at the current hour."""
        if not npc.routine:
            return None

        # Find the routine entry closest to (but not after) current hour
        best_entry = None
        for entry in npc.routine:
            if entry.hour <= current_hour:
                if best_entry is None or entry.hour > best_entry.hour:
                    best_entry = entry

        # If no entry found (before first routine time), use last entry
        if best_entry is None and npc.routine:
            best_entry = npc.routine[-1]

        return best_entry

    def observe_npcs(self, npcs, current_hour):
        """
        Record where NPCs are at the current hour.
        This data feeds the routine adaptation algorithm.
        """
        hour_key = int(current_hour)
        if hour_key not in self.hourly_zone_visits:
            self.hourly_zone_visits[hour_key] = {}

        for npc in npcs:
            # Determine which zone the NPC is closest to
            closest_zone = self._get_npc_zone(npc)
            if closest_zone:
                visits = self.hourly_zone_visits[hour_key]
                visits[closest_zone] = visits.get(closest_zone, 0) + 1

    def adapt_routines(self, npcs, dt):
        """
        UNSUPERVISED LEARNING: Adapt NPC routines based on observed patterns.
        
        NPCs with higher sociability are more likely to follow the crowd.
        NPCs with higher work_ethic stick to their default routines.
        """
        self.adaptation_timer += dt
        if self.adaptation_timer < self.adaptation_interval:
            return
        self.adaptation_timer = 0.0

        if not self.hourly_zone_visits:
            return

        for npc in npcs:
            # Only adapt occasionally based on personality
            adapt_chance = npc.personality["sociability"] * 0.3
            if random.random() > adapt_chance:
                continue

            # Pick a random hour from observed data
            observed_hours = list(self.hourly_zone_visits.keys())
            if not observed_hours:
                continue

            hour = random.choice(observed_hours)
            zone_counts = self.hourly_zone_visits[hour]
            if not zone_counts:
                continue

            # Find the most popular zone at this hour
            popular_zone = max(zone_counts, key=zone_counts.get)

            # Check if this NPC already goes there
            already_goes = any(
                e.hour == hour and e.zone == popular_zone
                for e in npc.routine
            )
            if already_goes:
                continue

            # Maybe adapt: add or modify a routine entry
            if random.random() < 0.2:  # 20% chance to actually adapt
                # Find existing entry at this hour
                existing = None
                for entry in npc.routine:
                    if entry.hour == hour:
                        existing = entry
                        break

                if existing and existing.priority < 3:
                    # Modify existing low-priority entry
                    existing.zone = popular_zone
                    existing.activity = f"Following the crowd to {popular_zone}"
                elif len(npc.routine) < 10:
                    # Add new entry
                    npc.routine.append(RoutineEntry(
                        hour, popular_zone,
                        f"Learned: Visit {popular_zone}",
                        priority=0  # very low priority, easily overridden
                    ))

    def _get_npc_zone(self, npc):
        """Determine which zone an NPC is closest to."""
        from config import TILE_SIZE

        best_zone = None
        best_dist = float('inf')

        for zone_name, (zx, zy, zw, zh) in ZONES.items():
            cx = (zx + zw / 2) * TILE_SIZE
            cy = (zy + zh / 2) * TILE_SIZE
            dist = abs(npc.x - cx) + abs(npc.y - cy)
            if dist < best_dist:
                best_dist = dist
                best_zone = zone_name

        return best_zone if best_dist < 200 else None

    def assign_default_routines(self, npcs):
        """Assign default daily routines based on NPC class."""
        routine_templates = {
            "Royal": [
                RoutineEntry(6, "castle", "Morning court", 5),
                RoutineEntry(10, "town_square", "Public appearance", 4),
                RoutineEntry(12, "castle", "Lunch at court", 3),
                RoutineEntry(14, "castle", "Afternoon affairs", 4),
                RoutineEntry(18, "castle", "Evening dinner", 3),
                RoutineEntry(21, "castle", "Retire to chambers", 5),
            ],
            "Noble": [
                RoutineEntry(7, "noble_house_w1", "Morning at home", 3),
                RoutineEntry(10, "town_square", "Social visit", 2),
                RoutineEntry(12, "noble_house_e1", "Midday meal", 2),
                RoutineEntry(14, "noble_house_w2", "Estate business", 3),
                RoutineEntry(18, "town_square", "Evening social", 2),
                RoutineEntry(21, "noble_house_e2", "Return home", 3),
            ],
            "Elite": [
                RoutineEntry(5, "castle", "Morning training", 5),
                RoutineEntry(8, "town_square", "Patrol start", 4),
                RoutineEntry(11, "noble_house_w3", "District patrol", 3),
                RoutineEntry(13, "town_square", "Lunch break", 2),
                RoutineEntry(14, "castle", "Guard duty", 4),
                RoutineEntry(18, "castle", "Evening drills", 3),
                RoutineEntry(21, "castle", "Rest", 3),
            ],
            "Merchant": [
                RoutineEntry(6, "trader_house_1", "Open shop", 5),
                RoutineEntry(12, "town_square", "Lunch", 2),
                RoutineEntry(13, "trader_house_2", "Afternoon trade", 4),
                RoutineEntry(17, "trader_house_3", "Close shop", 3),
                RoutineEntry(18, "town_square", "Evening social", 2),
                RoutineEntry(21, "trader_house_2", "Go home", 3),
            ],
            "Blacksmith": [
                RoutineEntry(5, "trader_house_4", "Start work", 5),
                RoutineEntry(12, "town_square", "Lunch break", 2),
                RoutineEntry(13, "trader_house_4", "Afternoon work", 5),
                RoutineEntry(18, "town_square", "Evening break", 2),
                RoutineEntry(21, "trader_house_3", "Rest", 3),
            ],
            "Traveller": [
                RoutineEntry(7, "trader_house_1", "Morning near traders", 2),
                RoutineEntry(9, "trader_house_2", "Browse district", 1),
                RoutineEntry(11, "town_square", "Explore village", 1),
                RoutineEntry(13, "town_square", "Lunch & stories", 2),
                RoutineEntry(15, "noble_house_e3", "Visit nobles", 1),
                RoutineEntry(18, "town_square", "Evening tales", 2),
                RoutineEntry(21, "trader_house_2", "Return to lodging", 3),
            ],
            "Labourer": [
                RoutineEntry(5, "peasant_house_1", "Morning tasks", 4),
                RoutineEntry(8, "wheat_farm_w1", "Field labor", 5),
                RoutineEntry(10, "wheat_farm_w2", "Continue harvesting", 5),
                RoutineEntry(12, "town_square", "Lunch", 2),
                RoutineEntry(13, "wheat_farm_s1", "Deliver grain", 4),
                RoutineEntry(16, "wheat_farm_s2", "Late field chores", 4),
                RoutineEntry(19, "town_square", "Evening rest", 2),
                RoutineEntry(21, "peasant_house_3", "Sleep", 4),
            ],
            "Peasant": [
                RoutineEntry(5, "peasant_house_1", "Early chores", 3),
                RoutineEntry(8, "town_square", "Fetch supplies", 2),
                RoutineEntry(9, "wheat_farm_w1", "Work the wheat rows", 5),
                RoutineEntry(11, "wheat_farm_w2", "Irrigate and weed", 5),
                RoutineEntry(12, "town_square", "Midday meal", 2),
                RoutineEntry(14, "wheat_farm_s1", "Afternoon harvest", 5),
                RoutineEntry(16, "wheat_farm_s2", "Store grain bundles", 4),
                RoutineEntry(17, "town_square", "Evening prayer", 1),
                RoutineEntry(19, "town_square", "Socialize", 1),
                RoutineEntry(20, "peasant_house_4", "Sleep", 4),
            ],
        }

        for npc in npcs:
            template = routine_templates.get(npc.npc_class, routine_templates["Peasant"])
            # Deep copy with slight random variation
            npc.routine = []
            for entry in template:
                hour_variation = random.choice([-1, 0, 0, 0, 1])
                new_hour = max(0, min(23, entry.hour + hour_variation))
                npc.routine.append(RoutineEntry(
                    new_hour, entry.zone, entry.activity, entry.priority
                ))
