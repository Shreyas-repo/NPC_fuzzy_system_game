"""
Tile-based map system for the village.
"""
import pygame
import random
from utils.sprite_assets import SpriteAssets
from config import (
    TILE_SIZE, MAP_WIDTH, MAP_HEIGHT,
    TILE_GRASS, TILE_DIRT, TILE_STONE, TILE_WATER, TILE_WALL, TILE_FLOOR,
    TILE_FARM, TILE_MARKET, WALKABLE_TILES, COLORS, ZONES
)


class TileMap:
    """Manages the tile grid and renders terrain."""

    def __init__(self):
        self.tiles = [[TILE_GRASS for _ in range(MAP_WIDTH)] for _ in range(MAP_HEIGHT)]
        self.blocked_tiles = set()
        self.tile_surfaces = {}
        self._build_tile_surfaces()
        self._generate()
        # Pre-render the full map to a surface for performance
        self.map_surface = pygame.Surface((MAP_WIDTH * TILE_SIZE, MAP_HEIGHT * TILE_SIZE))
        self._render_map_surface()

    def _build_tile_surfaces(self):
        """Create cached tile surfaces for each type with rich procedural detail."""
        sprites = SpriteAssets.get()
        tile_colors = {
            TILE_GRASS: COLORS["grass"],
            TILE_DIRT:  COLORS["dirt_path"],
            TILE_STONE: COLORS["stone_road"],
            TILE_WATER: COLORS["water"],
            TILE_WALL:  COLORS["wall_stone"],
            TILE_FLOOR: COLORS["wall_wood"],
            TILE_FARM:  (90, 150, 60),
            TILE_MARKET: (180, 160, 120),
        }

        sprite_key_for_tile = {
            TILE_GRASS: "grass",
            TILE_DIRT: "dirt",
            TILE_STONE: "stone",
            TILE_WATER: "water",
            TILE_WALL: "wall",
            TILE_FLOOR: "floor",
            TILE_FARM: "farm",
            TILE_MARKET: "market",
        }

        for tile_type, color in tile_colors.items():
            sprite_key = sprite_key_for_tile.get(tile_type)
            # Prefer internet textures for core terrain while keeping procedural fallback.
            use_atlas = tile_type in (TILE_GRASS, TILE_STONE, TILE_WATER)
            sprite_surf = sprites.get_tile(sprite_key, TILE_SIZE) if (sprite_key and use_atlas) else None
            if sprite_surf is not None:
                if tile_type == TILE_STONE:
                    # Keep the road visibly gray even when source texture is warm-toned.
                    road = sprite_surf.copy()
                    tint = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                    tint.fill((130, 130, 130, 70))
                    road.blit(tint, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                    # Add gravel-like speckles so roads read as stoney pathway.
                    for _ in range(24):
                        px = random.randint(1, TILE_SIZE - 2)
                        py = random.randint(1, TILE_SIZE - 2)
                        pebble = random.choice([(106, 106, 106), (138, 138, 138), (166, 166, 166)])
                        road.set_at((px, py), pebble)
                    for _ in range(8):
                        px = random.randint(2, TILE_SIZE - 3)
                        py = random.randint(2, TILE_SIZE - 3)
                        pygame.draw.circle(road, random.choice([(98, 98, 98), (152, 152, 152)]), (px, py), 1)
                    sprite_surf = road
                self.tile_surfaces[tile_type] = sprite_surf
                continue

            # --- Rich procedural fallbacks ---
            surf = pygame.Surface((TILE_SIZE, TILE_SIZE))
            surf.fill(color)

            if tile_type == TILE_GRASS:
                self._proc_grass(surf, color)
            elif tile_type == TILE_DIRT:
                self._proc_dirt(surf, color)
            elif tile_type == TILE_WATER:
                self._proc_water(surf, color)
            elif tile_type == TILE_FARM:
                self._proc_farm(surf)
            elif tile_type == TILE_MARKET:
                self._proc_market(surf, color)
            elif tile_type == TILE_STONE:
                self._proc_stone_road(surf, color)

            self.tile_surfaces[tile_type] = surf

    # ─── Procedural tile detail generators ────────────────────────────

    def _proc_grass(self, surf, base):
        """Lush grass with blade variations, subtle color shifts, and wildflower dots."""
        TS = TILE_SIZE
        # Subtle color variation patches
        for _ in range(6):
            px = random.randint(0, TS - 6)
            py = random.randint(0, TS - 6)
            w = random.randint(4, 10)
            h = random.randint(4, 10)
            shift = random.randint(-12, 12)
            patch_color = (
                max(0, min(255, base[0] + shift)),
                max(0, min(255, base[1] + shift)),
                max(0, min(255, base[2] + shift)),
            )
            pygame.draw.rect(surf, patch_color, (px, py, w, h))

        # Grass blade strokes
        for _ in range(18):
            bx = random.randint(1, TS - 2)
            by = random.randint(2, TS - 1)
            blade_len = random.randint(2, 5)
            green = random.randint(180, 230)
            blade_color = (max(0, green - 80), green, max(0, green - 120))
            pygame.draw.line(surf, blade_color, (bx, by), (bx + random.randint(-1, 1), by - blade_len), 1)

        # Occasional wildflower dots
        if random.random() < 0.35:
            fx = random.randint(3, TS - 4)
            fy = random.randint(3, TS - 4)
            flower_colors = [(255, 220, 80), (240, 120, 120), (180, 140, 255), (255, 180, 200)]
            pygame.draw.circle(surf, random.choice(flower_colors), (fx, fy), 1)

    def _proc_dirt(self, surf, base):
        """Earth-tone speckling with small cracks and footprint impressions."""
        TS = TILE_SIZE
        # Earth tone speckles
        for _ in range(30):
            px = random.randint(0, TS - 1)
            py = random.randint(0, TS - 1)
            shift = random.randint(-20, 20)
            speck_color = (
                max(0, min(255, base[0] + shift)),
                max(0, min(255, base[1] + shift - 10)),
                max(0, min(255, base[2] + shift - 15)),
            )
            surf.set_at((px, py), speck_color)

        # Small pebbles
        for _ in range(5):
            px = random.randint(2, TS - 3)
            py = random.randint(2, TS - 3)
            pebble_color = (
                base[0] + random.randint(-30, -10),
                base[1] + random.randint(-25, -5),
                base[2] + random.randint(-20, 5),
            )
            pebble_color = tuple(max(0, min(255, c)) for c in pebble_color)
            pygame.draw.circle(surf, pebble_color, (px, py), 1)

        # Subtle crack lines
        if random.random() < 0.4:
            cx = random.randint(4, TS - 8)
            cy = random.randint(4, TS - 8)
            crack_len = random.randint(4, 10)
            dark = tuple(max(0, c - 35) for c in base)
            pygame.draw.line(surf, dark, (cx, cy), (cx + random.randint(-3, 5), cy + crack_len), 1)

    def _proc_water(self, surf, base):
        """Water with shimmer highlights and depth variation."""
        TS = TILE_SIZE
        deep = COLORS["water_deep"]
        # Depth gradient patches
        for _ in range(4):
            px = random.randint(0, TS - 8)
            py = random.randint(0, TS - 8)
            w = random.randint(6, 14)
            h = random.randint(4, 10)
            blend = random.uniform(0.3, 0.7)
            patch = (
                int(base[0] * (1 - blend) + deep[0] * blend),
                int(base[1] * (1 - blend) + deep[1] * blend),
                int(base[2] * (1 - blend) + deep[2] * blend),
            )
            pygame.draw.rect(surf, patch, (px, py, w, h))

        # Shimmer highlight stripes
        for i in range(3):
            y = random.randint(3, TS - 4)
            x1 = random.randint(2, TS // 2)
            x2 = x1 + random.randint(4, 12)
            highlight = (min(255, base[0] + 50), min(255, base[1] + 60), min(255, base[2] + 40))
            pygame.draw.line(surf, highlight, (x1, y), (min(TS - 2, x2), y), 1)

        # Subtle ripple dots
        for _ in range(4):
            rx = random.randint(2, TS - 3)
            ry = random.randint(2, TS - 3)
            surf.set_at((rx, ry), (min(255, base[0] + 30), min(255, base[1] + 40), min(255, base[2] + 25)))

    def _proc_farm(self, surf):
        """Farm tiles with crop row lines and alternating green shades."""
        TS = TILE_SIZE
        light_green = (110, 170, 70)
        dark_green = (70, 130, 50)
        soil = (120, 90, 55)

        # Alternating crop rows
        row_h = max(2, TS // 6)
        for y in range(0, TS, row_h):
            row_color = light_green if (y // row_h) % 2 == 0 else dark_green
            pygame.draw.rect(surf, row_color, (0, y, TS, row_h))
            # Soil line between rows
            pygame.draw.line(surf, soil, (0, y), (TS, y), 1)

        # Seed/crop dots
        for _ in range(10):
            cx = random.randint(2, TS - 3)
            cy = random.randint(2, TS - 3)
            crop_green = (random.randint(80, 140), random.randint(160, 210), random.randint(40, 80))
            pygame.draw.circle(surf, crop_green, (cx, cy), 1)

    def _proc_market(self, surf, base):
        """Cobblestone checkerboard with subtle wear marks."""
        TS = TILE_SIZE
        stone_size = max(4, TS // 5)
        for y in range(0, TS, stone_size):
            for x in range(0, TS, stone_size):
                checker = ((x // stone_size) + (y // stone_size)) % 2
                if checker == 0:
                    c = (base[0] - 12, base[1] - 10, base[2] - 8)
                else:
                    c = (base[0] + 8, base[1] + 6, base[2] + 4)
                c = tuple(max(0, min(255, v)) for v in c)
                pygame.draw.rect(surf, c, (x, y, stone_size, stone_size))
                # Mortar line
                mortar = tuple(max(0, v - 30) for v in base)
                pygame.draw.rect(surf, mortar, (x, y, stone_size, stone_size), 1)

        # Wear marks
        for _ in range(3):
            wx = random.randint(2, TS - 3)
            wy = random.randint(2, TS - 3)
            wear_color = tuple(max(0, v - 20) for v in base)
            surf.set_at((wx, wy), wear_color)

    def _proc_stone_road(self, surf, base):
        """Enhanced stone road with pebbles and mortar lines."""
        TS = TILE_SIZE
        # Base with slight variation
        for _ in range(30):
            px = random.randint(0, TS - 1)
            py = random.randint(0, TS - 1)
            shift = random.randint(-15, 15)
            speck = tuple(max(0, min(255, c + shift)) for c in base)
            surf.set_at((px, py), speck)

        # Pebble shapes
        for _ in range(8):
            px = random.randint(2, TS - 3)
            py = random.randint(2, TS - 3)
            pebble = random.choice([(130, 130, 130), (155, 155, 155), (115, 115, 115)])
            pygame.draw.circle(surf, pebble, (px, py), random.choice([1, 2]))

        # Mortar/joint lines
        mid = TS // 2
        mortar = tuple(max(0, c - 25) for c in base)
        pygame.draw.line(surf, mortar, (0, mid), (TS, mid), 1)
        pygame.draw.line(surf, mortar, (mid, 0), (mid, TS), 1)

    def _generate(self):
        """Generate the base terrain layout."""
        # Grass everywhere by default
        # Fixed lake in the top-right, matching the requested town plan.
        lake_x, lake_y, lake_w, lake_h = ZONES.get("lake", (52, 0, 28, 16))
        for y in range(lake_y, min(lake_y + lake_h, MAP_HEIGHT)):
            for x in range(lake_x, min(lake_x + lake_w, MAP_WIDTH)):
                self.tiles[y][x] = TILE_WATER

        # Wheat farms for economy simulation.
        for zone_name, (zx, zy, zw, zh) in ZONES.items():
            if zone_name.startswith("wheat_farm_"):
                self._fill_zone(zx, zy, zw, zh, TILE_FARM)
            elif zone_name == "granary_storage":
                self._fill_zone(zx, zy, zw, zh, TILE_MARKET)

        # Lay down dirt paths connecting zones
        self._lay_paths()

    def _fill_zone(self, x, y, w, h, tile_type):
        for ty in range(y, min(y + h, MAP_HEIGHT)):
            for tx in range(x, min(x + w, MAP_WIDTH)):
                self.tiles[ty][tx] = tile_type

    def _lay_paths(self):
        """Create the single gray cross-road from the reference layout."""
        structure_zones = {"castle", "lake"}
        for zone_name in ZONES:
            if zone_name.startswith(("noble_house_", "peasant_house_", "trader_house_")):
                structure_zones.add(zone_name)

        def blocked(tx, ty):
            for name, (zx, zy, zw, zh) in ZONES.items():
                if name not in structure_zones:
                    continue
                if zx <= tx < zx + zw and zy <= ty < zy + zh:
                    return True
            return False

        road_y = MAP_HEIGHT // 2
        road_x = MAP_WIDTH // 2

        # Main horizontal road
        for x in range(MAP_WIDTH):
            for dy in range(-1, 1):
                y = road_y + dy
                if 0 <= y < MAP_HEIGHT and self.tiles[y][x] != TILE_WATER and not blocked(x, y):
                    self.tiles[y][x] = TILE_STONE

        # Main vertical road
        for y in range(MAP_HEIGHT):
            for dx in range(-1, 1):
                x = road_x + dx
                if 0 <= x < MAP_WIDTH and self.tiles[y][x] != TILE_WATER and not blocked(x, y):
                    self.tiles[y][x] = TILE_STONE

        # Footpaths from buildings to nearest road
        self._lay_footpaths(road_x, road_y, blocked)

    def _lay_footpaths(self, road_x, road_y, blocked):
        """Draw dirt footpaths from each house/castle entrance to the nearest main road."""
        house_zones = []
        for zone_name, (zx, zy, zw, zh) in ZONES.items():
            if zone_name.startswith(("noble_house_", "peasant_house_", "trader_house_")):
                house_zones.append((zone_name, zx, zy, zw, zh))
            elif zone_name == "castle":
                house_zones.append((zone_name, zx, zy, zw, zh))

        for zone_name, zx, zy, zw, zh in house_zones:
            # Entrance is at the bottom center of the building
            door_x = zx + zw // 2
            door_y = zy + zh  # just below the building

            # Determine nearest road axis
            dist_to_h_road = abs(door_y - road_y)
            dist_to_v_road = abs(door_x - road_x)

            # Draw a footpath: first go vertically to the horizontal road,
            # or horizontally to the vertical road, whichever is closer.
            if dist_to_h_road <= dist_to_v_road:
                # Walk vertically from door to horizontal road, then horizontally if needed
                self._draw_dirt_line_v(door_x, door_y, road_y, blocked)
            else:
                # Walk horizontally from door to vertical road, then vertically if needed
                self._draw_dirt_line_h(door_x, door_y, road_x, blocked)

            # Also add a short vertical stub from door downward (1-2 tiles) so
            # the entrance always visually connects to a path.
            for dy in range(1, 3):
                ty = zy + zh - 1 + dy
                if 0 <= ty < MAP_HEIGHT and 0 <= door_x < MAP_WIDTH:
                    if self.tiles[ty][door_x] not in (TILE_WATER, TILE_STONE, TILE_WALL):
                        if not blocked(door_x, ty):
                            self.tiles[ty][door_x] = TILE_DIRT

    def _draw_dirt_line_v(self, tx, from_y, to_y, blocked):
        """Draw a vertical dirt footpath from from_y toward to_y at column tx."""
        if from_y == to_y:
            return
        step = 1 if to_y > from_y else -1
        y = from_y
        while y != to_y:
            if 0 <= tx < MAP_WIDTH and 0 <= y < MAP_HEIGHT:
                if self.tiles[y][tx] not in (TILE_WATER, TILE_STONE, TILE_WALL):
                    if not blocked(tx, y):
                        self.tiles[y][tx] = TILE_DIRT
            y += step

    def _draw_dirt_line_h(self, from_x, ty, to_x, blocked):
        """Draw a horizontal dirt footpath from from_x toward to_x at row ty."""
        if from_x == to_x:
            return
        step = 1 if to_x > from_x else -1
        x = from_x
        while x != to_x:
            if 0 <= x < MAP_WIDTH and 0 <= ty < MAP_HEIGHT:
                if self.tiles[ty][x] not in (TILE_WATER, TILE_STONE, TILE_WALL):
                    if not blocked(x, ty):
                        self.tiles[ty][x] = TILE_DIRT
            x += step

    def _render_map_surface(self):
        """Pre-render the entire map to a surface."""
        for y in range(MAP_HEIGHT):
            for x in range(MAP_WIDTH):
                tile = self.tiles[y][x]
                px, py_pos = x * TILE_SIZE, y * TILE_SIZE
                self.map_surface.blit(self.tile_surfaces[tile], (px, py_pos))

    def render(self, surface, camera):
        """Render visible portion of the map."""
        # Calculate visible area
        cam_rect = camera.get_visible_rect()
        # Blit the pre-rendered map with camera offset
        surface.blit(self.map_surface, (-cam_rect.x, -cam_rect.y))

    def is_walkable(self, tile_x, tile_y):
        """Check if a tile position is walkable."""
        if 0 <= tile_x < MAP_WIDTH and 0 <= tile_y < MAP_HEIGHT:
            return self.tiles[tile_y][tile_x] in WALKABLE_TILES and (tile_x, tile_y) not in self.blocked_tiles
        return False

    def is_walkable_pixel(self, px, py, radius=10):
        """Check walkability around a pixel point using a circular-ish footprint."""
        sample_points = [
            (px, py),
            (px - radius, py),
            (px + radius, py),
            (px, py - radius),
            (px, py + radius),
            (px - radius * 0.7, py - radius * 0.7),
            (px + radius * 0.7, py - radius * 0.7),
            (px - radius * 0.7, py + radius * 0.7),
            (px + radius * 0.7, py + radius * 0.7),
        ]
        for sx, sy in sample_points:
            tx = int(sx // TILE_SIZE)
            ty = int(sy // TILE_SIZE)
            if not self.is_walkable(tx, ty):
                return False
        return True

    def get_tile(self, tile_x, tile_y):
        """Get tile type at position."""
        if 0 <= tile_x < MAP_WIDTH and 0 <= tile_y < MAP_HEIGHT:
            return self.tiles[tile_y][tile_x]
        return TILE_WALL  # Out of bounds = wall
