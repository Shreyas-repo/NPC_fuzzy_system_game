"""
World generator for the clean reference town plan.
"""
import random
import pygame
import math
from config import TILE_SIZE, MAP_WIDTH, MAP_HEIGHT, ZONES, TILE_STONE, TILE_WATER, TILE_FARM
from game.tile_map import TileMap
from game.buildings import create_buildings
from utils.sprite_assets import SpriteAssets


class World:
    """The complete village world: terrain + simple block buildings."""

    def __init__(self):
        self.tile_map = TileMap()
        self.buildings = create_buildings()
        self.decorations = []
        self._place_walls()
        self._generate_decorations()

    def _place_walls(self):
        """Block movement through building/lake blocks without adding extra visuals."""
        structure_zones = {"castle", "lake"}
        for zone_name in ZONES:
            if zone_name.startswith(("noble_house_", "peasant_house_", "trader_house_")):
                structure_zones.add(zone_name)

        for zone_name, (zx, zy, zw, zh) in ZONES.items():
            if zone_name not in structure_zones:
                continue
            for x in range(zx, zx + zw):
                for y in range(zy, zy + zh):
                    if 0 <= x < MAP_WIDTH and 0 <= y < MAP_HEIGHT:
                        self.tile_map.blocked_tiles.add((x, y))

        self.tile_map._render_map_surface()

    def render(self, surface, camera, day_cycle=None, farm_system=None, npcs=None):
        """Render terrain and block buildings."""
        self.tile_map.render(surface, camera)

        if farm_system is not None:
            farm_system.render(surface, camera)

        cam_rect = camera.get_visible_rect()
        for dec_type, dx, dy in self.decorations:
            sx = dx - cam_rect.x
            sy = dy - cam_rect.y
            if not (-TILE_SIZE * 2 <= sx <= cam_rect.width + TILE_SIZE * 2 and
                    -TILE_SIZE * 2 <= sy <= cam_rect.height + TILE_SIZE * 2):
                continue
            if dec_type == "lamp":
                self._draw_lamp(surface, sx, sy, day_cycle)

        occupied_homes = self._occupied_home_zones(npcs)
        for name, building in self.buildings.items():
            bx = building.pixel_x - cam_rect.x
            by = building.pixel_y - cam_rect.y
            if (bx + building.pixel_w > 0 and bx < cam_rect.width and
                by + building.pixel_h > 0 and by < cam_rect.height):
                building.render(surface, camera)
                if name in occupied_homes:
                    self._draw_occupied_house_hint(surface, building, bx, by, day_cycle)

    def _occupied_home_zones(self, npcs):
        """Collect house zones with NPCs currently inside (sleeping/sheltering)."""
        zones = set()
        if not npcs:
            return zones
        for npc in npcs:
            if getattr(npc, "is_indoors", False):
                zone = getattr(npc, "indoor_zone", None)
                if zone and zone in self.buildings:
                    zones.add(zone)
        return zones

    def _draw_occupied_house_hint(self, surface, building, screen_x, screen_y, day_cycle=None):
        """Render warm window lights to indicate occupied interiors."""
        is_nightish = day_cycle is not None and day_cycle.day_phase in ("dusk", "night")
        glow_alpha = 72
        if is_nightish:
            t = pygame.time.get_ticks() / 1000.0
            pulse = 0.5 + 0.5 * math.sin(t * 2.2)
            glow_alpha = int(98 + (42 * pulse))

        win_w = max(8, min(14, building.pixel_w // 7))
        win_h = max(8, min(12, building.pixel_h // 7))
        pad_x = max(10, building.pixel_w // 5)
        top_y = screen_y + max(10, building.pixel_h // 5)

        windows = [
            (screen_x + pad_x, top_y),
            (screen_x + building.pixel_w - pad_x - win_w, top_y),
        ]

        for wx, wy in windows:
            pygame.draw.rect(surface, (246, 214, 140), (wx, wy, win_w, win_h), border_radius=2)
            pygame.draw.rect(surface, (116, 84, 48), (wx, wy, win_w, win_h), 1, border_radius=2)

            glow = pygame.Surface((win_w + 18, win_h + 18), pygame.SRCALPHA)
            pygame.draw.ellipse(glow, (255, 208, 128, glow_alpha), (0, 0, win_w + 18, win_h + 18))
            surface.blit(glow, (wx - 9, wy - 9), special_flags=pygame.BLEND_RGBA_ADD)

    def _is_safe_decoration_tile(self, tx, ty, occupied=None, avoid_road=True):
        if not (0 <= tx < MAP_WIDTH and 0 <= ty < MAP_HEIGHT):
            return False
        if occupied is not None and (tx, ty) in occupied:
            return False
        if (tx, ty) in self.tile_map.blocked_tiles:
            return False
        tile = self.tile_map.tiles[ty][tx]
        if tile == TILE_WATER:
            return False
        if tile == TILE_FARM:
            return False
        if avoid_road and tile == TILE_STONE:
            return False
        return True

    def _generate_decorations(self):
        """Add lamps beside roads."""
        self.decorations = []
        occupied = set()

        road_x = MAP_WIDTH // 2
        road_y = MAP_HEIGHT // 2

        # Lamps run parallel to both road arms, placed on grass beside the road.
        for x in range(4, MAP_WIDTH - 4, 6):
            for y in (road_y - 3, road_y + 2):
                if self._is_safe_decoration_tile(x, y, occupied, avoid_road=True):
                    self.decorations.append(("lamp", x * TILE_SIZE + TILE_SIZE // 2, y * TILE_SIZE + TILE_SIZE // 2))
                    occupied.add((x, y))

        for y in range(5, MAP_HEIGHT - 4, 6):
            for x in (road_x - 3, road_x + 2):
                if self._is_safe_decoration_tile(x, y, occupied, avoid_road=True):
                    self.decorations.append(("lamp", x * TILE_SIZE + TILE_SIZE // 2, y * TILE_SIZE + TILE_SIZE // 2))
                    occupied.add((x, y))

    def _draw_lamp(self, surface, x, y, day_cycle=None):
        pole_h = 22
        pygame.draw.rect(surface, (70, 64, 56), (x - 1, y - pole_h, 3, pole_h))
        pygame.draw.rect(surface, (58, 52, 44), (x - 3, y - 2, 7, 3))
        pygame.draw.rect(surface, (94, 86, 72), (x - 4, y - pole_h - 6, 8, 6))

        is_nightish = day_cycle is not None and day_cycle.day_phase in ("dusk", "night")
        if is_nightish:
            glow_radius = 34
            glow = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            for r in range(glow_radius, 5, -4):
                alpha = max(8, int(96 * (r / glow_radius)))
                pygame.draw.circle(glow, (255, 220, 140, alpha), (glow_radius, glow_radius), r)
            surface.blit(glow, (x - glow_radius, y - pole_h - glow_radius // 2), special_flags=pygame.BLEND_RGBA_ADD)
            pygame.draw.circle(surface, (255, 240, 170), (x, y - pole_h - 3), 3)
            pygame.draw.circle(surface, (255, 255, 220), (x, y - pole_h - 4), 1)
        else:
            # Dim bulb visible in daytime so lamp heads still read clearly.
            pygame.draw.circle(surface, (170, 156, 120), (x, y - pole_h - 3), 2)

    def get_building_at(self, world_x, world_y):
        """Get the building at a world position, or None."""
        for name, building in self.buildings.items():
            if building.rect.collidepoint(world_x, world_y):
                return name, building
        return None, None

    def get_zone_center(self, zone_name):
        """Get the center pixel position of a zone."""
        if zone_name in ZONES:
            zx, zy, zw, zh = ZONES[zone_name]
            return (zx + zw // 2) * TILE_SIZE, (zy + zh // 2) * TILE_SIZE
        return MAP_WIDTH * TILE_SIZE // 2, MAP_HEIGHT * TILE_SIZE // 2

    def get_zone_random_point(self, zone_name):
        """Get a random walkable point near a zone."""
        if zone_name in ZONES:
            zx, zy, zw, zh = ZONES[zone_name]
            if zone_name.startswith("wheat_farm_"):
                for _ in range(20):
                    rx = random.randint(zx, zx + zw - 1)
                    ry = random.randint(zy, zy + zh - 1)
                    if self.tile_map.is_walkable(rx, ry):
                        return rx * TILE_SIZE + TILE_SIZE // 2, ry * TILE_SIZE + TILE_SIZE // 2
            if zone_name == "granary_storage":
                for _ in range(20):
                    rx = random.randint(zx, zx + zw - 1)
                    ry = random.randint(zy, zy + zh - 1)
                    if self.tile_map.is_walkable(rx, ry):
                        return rx * TILE_SIZE + TILE_SIZE // 2, ry * TILE_SIZE + TILE_SIZE // 2
            for _ in range(20):
                rx = random.randint(zx - 1, zx + zw)
                ry = random.randint(zy + zh, zy + zh + 2)
                if self.tile_map.is_walkable(rx, ry):
                    return rx * TILE_SIZE + TILE_SIZE // 2, ry * TILE_SIZE + TILE_SIZE // 2
            cx = (zx + zw // 2) * TILE_SIZE
            cy = (zy + zh + 1) * TILE_SIZE
            return cx, cy
        return MAP_WIDTH * TILE_SIZE // 2, MAP_HEIGHT * TILE_SIZE // 2

    def get_nearby_enterable_building(self, world_x, world_y, max_distance=52):
        """Return closest enterable building near the player position."""
        enterable = {"castle"}
        for zone_name in ZONES:
            if zone_name.startswith(("noble_house_", "peasant_house_", "trader_house_")):
                enterable.add(zone_name)

        best = None
        best_dist_sq = max_distance * max_distance
        for name, building in self.buildings.items():
            if name not in enterable:
                continue
            ex, ey = building.get_entrance()
            dx = ex - world_x
            dy = ey - world_y
            dist_sq = dx * dx + dy * dy
            if dist_sq <= best_dist_sq:
                best = (name, building)
                best_dist_sq = dist_sq
        return best
