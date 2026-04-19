"""
All 8 NPC subclasses with unique routines, personalities, and dialogue.
"""
import random
from entities.npc import NPC, NPCState


# ─── Names ───────────────────────────────────────────────────────────
NAMES = {
    "Royal":     ["King Aldric", "Queen Seraphina"],
    "Noble":     ["Lord Cedric", "Lady Isolde", "Lord Percival"],
    "Elite":     ["Captain Gareth", "Knight Rowan", "Guard Helena"],
    "Merchant":  ["Trader Elara", "Merchant Tobias", "Vendor Mirela", "Shopkeep Boris"],
    "Blacksmith": ["Smith Ragnar", "Forgemaster Hilda"],
    "Traveller": ["Wanderer Kai", "Pilgrim Yara", "Explorer Thane"],
    "Labourer":  ["Worker Amos", "Builder Petra", "Farmer Giles", "Hauler Bron", "Field Hand Ivy"],
    "Peasant":   ["Old Marta", "Young Felix", "Widow Greta", "Beggar Tom", "Milkmaid Rosa", "Shepherd Nils"],
}


class RoyalNPC(NPC):
    """Rulers of the village. High authority, grand demeanor."""

    def __init__(self, name, x, y, tile_map, world):
        super().__init__(name, "Royal", x, y, tile_map, world)
        self.personality["friendliness"] = random.uniform(0.4, 0.7)
        self.personality["aggression"] = random.uniform(0.1, 0.3)
        self.social_rank = 5

    def _do_work_routine(self):
        zone = random.choice(["castle", "town_square"])
        self._go_to_zone(zone)
        self.state = NPCState.WALKING

    def _do_evening_routine(self):
        self._go_to_zone("castle")
        self.state = NPCState.WALKING

    def _draw_class_icon(self, surface, cx, cy):
        import pygame
        # Crown icon
        pygame.draw.polygon(surface, (255, 215, 0), [
            (cx - 4, cy - 2), (cx - 3, cy - 5), (cx, cy - 3),
            (cx + 3, cy - 5), (cx + 4, cy - 2)
        ])


class NobleNPC(NPC):
    """Wealthy aristocrats. Social and political role."""

    def __init__(self, name, x, y, tile_map, world):
        super().__init__(name, "Noble", x, y, tile_map, world)
        self.personality["sociability"] = random.uniform(0.6, 0.9)
        self.social_rank = 4

    def _do_work_routine(self):
        zone = random.choice(["noble_house_w1", "noble_house_e1", "town_square"])
        self._go_to_zone(zone)
        self.state = NPCState.WALKING

    def _do_evening_routine(self):
        if random.random() < 0.6:
            self._go_to_zone("town_square")
        else:
            self._go_to_zone("noble_house_e2")
        self.state = NPCState.WALKING

    def _draw_class_icon(self, surface, cx, cy):
        import pygame
        # Shield icon
        pygame.draw.rect(surface, (220, 200, 255), (cx - 3, cy - 3, 6, 5))
        pygame.draw.polygon(surface, (220, 200, 255), [(cx - 3, cy + 2), (cx, cy + 5), (cx + 3, cy + 2)])


class EliteNPC(NPC):
    """Military guards and knights. Patrol and protect."""

    def __init__(self, name, x, y, tile_map, world):
        super().__init__(name, "Elite", x, y, tile_map, world)
        self.personality["aggression"] = random.uniform(0.3, 0.6)
        self.personality["work_ethic"] = random.uniform(0.7, 1.0)
        self.social_rank = 4
        self.patrol_zones = ["castle", "town_square", "noble_house_w2", "trader_house_2"]
        self.patrol_index = 0

    def _do_work_routine(self):
        zone = self.patrol_zones[self.patrol_index % len(self.patrol_zones)]
        self.patrol_index += 1
        self._go_to_zone(zone)
        self.state = NPCState.WALKING

    def _do_evening_routine(self):
        if random.random() < 0.4:
            self._go_to_zone("town_square")
        else:
            self._go_to_zone("castle")
        self.state = NPCState.WALKING

    def _draw_class_icon(self, surface, cx, cy):
        import pygame
        # Sword icon
        pygame.draw.line(surface, (200, 200, 220), (cx, cy - 5), (cx, cy + 3), 2)
        pygame.draw.line(surface, (200, 200, 220), (cx - 3, cy - 1), (cx + 3, cy - 1), 2)


class MerchantNPC(NPC):
    """Traders and shopkeepers. Focused on commerce."""

    def __init__(self, name, x, y, tile_map, world):
        super().__init__(name, "Merchant", x, y, tile_map, world)
        self.personality["friendliness"] = random.uniform(0.6, 0.9)
        self.social_rank = 3

    def _do_work_routine(self):
        self._go_to_zone("trader_house_1")
        self.state = NPCState.WALKING

    def _do_evening_routine(self):
        zone = random.choice(["trader_house_2", "trader_house_3", "town_square"])
        self._go_to_zone(zone)
        self.state = NPCState.WALKING

    def _draw_class_icon(self, surface, cx, cy):
        import pygame
        # Coin icon
        pygame.draw.circle(surface, (255, 215, 0), (cx, cy), 3)
        pygame.draw.circle(surface, (200, 170, 0), (cx, cy), 3, 1)


class BlacksmithNPC(NPC):
    """Forgers and metalworkers. Strong and steady."""

    def __init__(self, name, x, y, tile_map, world):
        super().__init__(name, "Blacksmith", x, y, tile_map, world)
        self.personality["work_ethic"] = random.uniform(0.8, 1.0)
        self.personality["sociability"] = random.uniform(0.2, 0.5)
        self.social_rank = 3

    def _do_work_routine(self):
        self._go_to_zone("trader_house_4")
        self.state = NPCState.WALKING

    def _do_evening_routine(self):
        if random.random() < 0.5:
            self._go_to_zone("town_square")
        else:
            self._go_to_zone("trader_house_4")
        self.state = NPCState.WALKING

    def _draw_class_icon(self, surface, cx, cy):
        import pygame
        # Hammer icon
        pygame.draw.rect(surface, (180, 180, 190), (cx - 1, cy - 4, 2, 7))
        pygame.draw.rect(surface, (180, 180, 190), (cx - 3, cy - 4, 6, 3))


class TravellerNPC(NPC):
    """Wanderers, pilgrims, and knowledge-seekers."""

    def __init__(self, name, x, y, tile_map, world):
        super().__init__(name, "Traveller", x, y, tile_map, world)
        self.personality["sociability"] = random.uniform(0.5, 0.8)
        self.personality["friendliness"] = random.uniform(0.5, 0.9)
        self.social_rank = 2

    def _do_work_routine(self):
        zone = random.choice(["trader_house_1", "trader_house_2", "town_square", "noble_house_e3"])
        self._go_to_zone(zone)
        self.state = NPCState.WALKING

    def _do_evening_routine(self):
        zone = random.choice(["town_square", "trader_house_2"])
        self._go_to_zone(zone)
        self.state = NPCState.WALKING

    def _draw_class_icon(self, surface, cx, cy):
        import pygame
        # Backpack/compass icon
        pygame.draw.circle(surface, (100, 200, 220), (cx, cy), 3, 1)
        pygame.draw.line(surface, (100, 200, 220), (cx, cy - 3), (cx, cy + 1), 1)


class LabourerNPC(NPC):
    """Workers, farmers, builders. The backbone of the village."""

    def __init__(self, name, x, y, tile_map, world):
        super().__init__(name, "Labourer", x, y, tile_map, world)
        self.personality["work_ethic"] = random.uniform(0.6, 0.9)
        self.social_rank = 1

    def _do_work_routine(self):
        if random.random() < 0.75:
            zone = random.choice(["wheat_farm_w1", "wheat_farm_w2", "wheat_farm_s1", "wheat_farm_s2"])
        else:
            zone = random.choice(["peasant_house_1", "peasant_house_2", "town_square"])
        self._go_to_zone(zone)
        self.state = NPCState.WALKING

    def _do_evening_routine(self):
        if random.random() < 0.4:
            self._go_to_zone("town_square")
        else:
            self._go_to_zone("town_square")
        self.state = NPCState.WALKING

    def _draw_class_icon(self, surface, cx, cy):
        import pygame
        # Pickaxe/shovel
        pygame.draw.line(surface, (200, 180, 130), (cx - 3, cy - 3), (cx + 3, cy + 3), 2)
        pygame.draw.line(surface, (200, 180, 130), (cx + 1, cy - 3), (cx + 4, cy - 1), 2)


class PeasantNPC(NPC):
    """The common folk. Subsistence living."""

    def __init__(self, name, x, y, tile_map, world):
        super().__init__(name, "Peasant", x, y, tile_map, world)
        self.personality["friendliness"] = random.uniform(0.4, 0.8)
        self.social_rank = 0

    def _do_work_routine(self):
        if random.random() < 0.85:
            zone = random.choice(["wheat_farm_w1", "wheat_farm_w2", "wheat_farm_s1", "wheat_farm_s2"])
        else:
            zone = random.choice(["peasant_house_1", "peasant_house_2", "town_square"])
        self._go_to_zone(zone)
        self.state = NPCState.WALKING

    def _do_evening_routine(self):
        if random.random() < 0.3:
            self._go_to_zone("town_square")
        else:
            self._go_to_zone("peasant_house_4")
        self.state = NPCState.WALKING

    def _draw_class_icon(self, surface, cx, cy):
        import pygame
        # Simple dot
        pygame.draw.circle(surface, (200, 180, 140), (cx, cy), 2)


# ─── NPC class registry ─────────────────────────────────────────────
NPC_CLASSES = {
    "Royal": RoyalNPC,
    "Noble": NobleNPC,
    "Elite": EliteNPC,
    "Merchant": MerchantNPC,
    "Blacksmith": BlacksmithNPC,
    "Traveller": TravellerNPC,
    "Labourer": LabourerNPC,
    "Peasant": PeasantNPC,
}


def create_all_npcs(tile_map, world):
    """Create all NPCs according to config counts."""
    from config import NPC_COUNTS, ZONES, TILE_SIZE
    npcs = []

    # Spawn zones for each class
    spawn_zones = {
        "Royal":     ["castle"],
        "Noble":     ["noble_house_w1", "noble_house_w2", "noble_house_e1", "noble_house_e2"],
        "Elite":     ["castle"],
        "Merchant":  ["trader_house_1", "trader_house_2", "trader_house_3"],
        "Blacksmith": ["trader_house_4"],
        "Traveller": ["trader_house_1", "trader_house_2", "trader_house_3"],
        "Labourer":  ["peasant_house_1", "peasant_house_2", "peasant_house_3"],
        "Peasant":   ["peasant_house_1", "peasant_house_2", "peasant_house_3", "peasant_house_4"],
    }

    for npc_class, count in NPC_COUNTS.items():
        cls = NPC_CLASSES[npc_class]
        names = NAMES.get(npc_class, [f"{npc_class} {i}" for i in range(count)])

        for i in range(count):
            name = names[i % len(names)]
            if i >= len(names):
                name = f"{name} II"

            # Pick spawn location
            zones = spawn_zones.get(npc_class, ["town_square"])
            zone_name = random.choice(zones)
            sx, sy = world.get_zone_random_point(zone_name)

            npc = cls(name, sx, sy, tile_map, world)
            npcs.append(npc)

    return npcs
