"""Farming mechanics with staged crop lifecycle and NPC farm labor behavior."""
import random
import pygame

from config import TILE_SIZE, ZONES


class FarmSystem:
    """Simulates tilling, sowing, growing, harvesting, and storage delivery."""

    STAGE_UNTILLED = 0
    STAGE_TILLED = 1
    STAGE_SOWN = 2
    STAGE_GROWING = 3
    STAGE_RIPE = 4
    STAGE_HARVESTED = 5

    STAGE_NAMES = {
        STAGE_UNTILLED: "untilled",
        STAGE_TILLED: "tilled",
        STAGE_SOWN: "sown",
        STAGE_GROWING: "growing",
        STAGE_RIPE: "ripe",
        STAGE_HARVESTED: "harvested",
    }

    def __init__(self, world):
        self.world = world
        self.farm_zones = sorted([z for z in ZONES if z.startswith("wheat_farm_")])
        self.storage_zone = "granary_storage" if "granary_storage" in ZONES else "town_square"

        self.plots = {}  # (tx, ty) -> {stage, timer}
        self.worker_progress = {}  # npc_name -> {plot, action, progress}
        self.worker_carry = {}  # npc_name -> wheat bundles carried

        self.growth_sown_to_growing = 14.0
        self.growth_growing_to_ripe = 24.0

        self._stage_surfaces = self._build_stage_surfaces()
        self._init_plots()

    def _init_plots(self):
        for zone_name in self.farm_zones:
            zx, zy, zw, zh = ZONES[zone_name]
            for tx in range(zx, zx + zw):
                for ty in range(zy, zy + zh):
                    if self.world.tile_map.is_walkable(tx, ty):
                        self.plots[(tx, ty)] = {
                            "stage": random.choice([self.STAGE_UNTILLED, self.STAGE_TILLED]),
                            "timer": random.uniform(0.0, 4.0),
                        }

    def _build_stage_surfaces(self):
        surfaces = {}

        # Stage 0: untilled dry earth.
        s0 = pygame.Surface((TILE_SIZE, TILE_SIZE))
        s0.fill((118, 86, 54))
        for _ in range(28):
            x = random.randint(0, TILE_SIZE - 1)
            y = random.randint(0, TILE_SIZE - 1)
            s0.set_at((x, y), (98, 66, 40))
        surfaces[self.STAGE_UNTILLED] = s0

        # Stage 1: tilled furrows.
        s1 = pygame.Surface((TILE_SIZE, TILE_SIZE))
        s1.fill((102, 72, 44))
        for y in range(2, TILE_SIZE, 5):
            pygame.draw.line(s1, (78, 54, 34), (0, y), (TILE_SIZE, y), 2)
        surfaces[self.STAGE_TILLED] = s1

        # Stage 2: sown seeds.
        s2 = s1.copy()
        for _ in range(24):
            x = random.randint(1, TILE_SIZE - 2)
            y = random.randint(1, TILE_SIZE - 2)
            s2.set_at((x, y), (190, 156, 96))
        surfaces[self.STAGE_SOWN] = s2

        # Stage 3: growing sprouts.
        s3 = s2.copy()
        for _ in range(26):
            x = random.randint(2, TILE_SIZE - 3)
            y = random.randint(6, TILE_SIZE - 2)
            pygame.draw.line(s3, (80, 168, 78), (x, y), (x, y - random.randint(2, 4)), 1)
        surfaces[self.STAGE_GROWING] = s3

        # Stage 4: ripe wheat.
        s4 = pygame.Surface((TILE_SIZE, TILE_SIZE))
        s4.fill((130, 98, 54))
        for _ in range(36):
            x = random.randint(2, TILE_SIZE - 3)
            y = random.randint(6, TILE_SIZE - 2)
            h = random.randint(3, 6)
            pygame.draw.line(s4, (218, 184, 72), (x, y), (x, y - h), 1)
            pygame.draw.circle(s4, (232, 198, 84), (x, max(1, y - h)), 1)
        surfaces[self.STAGE_RIPE] = s4

        # Stage 5: harvested stubble.
        s5 = pygame.Surface((TILE_SIZE, TILE_SIZE))
        s5.fill((116, 84, 50))
        for _ in range(20):
            x = random.randint(1, TILE_SIZE - 2)
            y = random.randint(8, TILE_SIZE - 2)
            pygame.draw.line(s5, (178, 140, 66), (x, y), (x, y - 1), 1)
        surfaces[self.STAGE_HARVESTED] = s5

        return surfaces

    def update(self, dt, npcs, economy_system=None):
        self._advance_growth(dt)

        if not npcs:
            return

        workers = [n for n in npcs if n.npc_class in ("Peasant", "Labourer")]
        for npc in workers:
            self.worker_carry.setdefault(npc.name, 0.0)
            self._update_worker(npc, dt, economy_system)

    def _advance_growth(self, dt):
        for plot in self.plots.values():
            stage = plot["stage"]
            if stage == self.STAGE_SOWN:
                plot["timer"] += dt
                if plot["timer"] >= self.growth_sown_to_growing:
                    plot["stage"] = self.STAGE_GROWING
                    plot["timer"] = 0.0
            elif stage == self.STAGE_GROWING:
                plot["timer"] += dt
                if plot["timer"] >= self.growth_growing_to_ripe:
                    plot["stage"] = self.STAGE_RIPE
                    plot["timer"] = 0.0

    def _npc_tile(self, npc):
        return int(npc.x // TILE_SIZE), int(npc.y // TILE_SIZE)

    def _is_in_zone(self, tile_pos, zone_name):
        tx, ty = tile_pos
        zx, zy, zw, zh = ZONES[zone_name]
        return zx <= tx < zx + zw and zy <= ty < zy + zh

    def _in_any_farm_zone(self, tile_pos):
        return any(self._is_in_zone(tile_pos, z) for z in self.farm_zones)

    def _nearest_plot_for_action(self, npc, wanted_stages):
        nx, ny = self._npc_tile(npc)
        candidates = []
        for (tx, ty), info in self.plots.items():
            if info["stage"] not in wanted_stages:
                continue
            d = abs(tx - nx) + abs(ty - ny)
            candidates.append((d, tx, ty))
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0])
        _, tx, ty = candidates[0]
        return tx, ty

    def _choose_action(self, npc):
        carried = self.worker_carry.get(npc.name, 0.0)
        if carried >= 2.0:
            return "deliver", None

        target = self._nearest_plot_for_action(npc, {self.STAGE_RIPE})
        if target:
            return "harvest", target

        target = self._nearest_plot_for_action(npc, {self.STAGE_UNTILLED, self.STAGE_HARVESTED})
        if target:
            return "till", target

        target = self._nearest_plot_for_action(npc, {self.STAGE_TILLED})
        if target:
            return "sow", target

        return "idle", None

    def _start_action(self, npc, action, plot):
        self.worker_progress[npc.name] = {
            "action": action,
            "plot": plot,
            "progress": 0.0,
            "duration": random.uniform(2.6, 4.2),
        }

        if action == "deliver":
            npc.show_speech("Taking harvest to storage.", 2.2)
            npc._go_to_zone(self.storage_zone)
            npc.state = "walking"
            return

        if plot is None:
            return

        px = plot[0] * TILE_SIZE + TILE_SIZE // 2
        py = plot[1] * TILE_SIZE + TILE_SIZE // 2
        npc._navigate_to(px, py)
        npc.state = "walking"

    def _finish_action(self, npc, action, plot, economy_system):
        if action == "deliver":
            bundles = self.worker_carry.get(npc.name, 0.0)
            if bundles <= 0.0:
                return
            self.worker_carry[npc.name] = 0.0
            npc.show_speech("Harvest delivered to storage.", 2.3)
            if economy_system and id(npc) in economy_system.accounts:
                acc = economy_system.accounts[id(npc)]
                acc["grain"] += bundles * 1.6
                wage = bundles * 0.75
                acc["coin"] += wage
                economy_system.ledger["grain_output"] = economy_system.ledger.get("grain_output", 0.0) + bundles * 1.6
                economy_system.ledger["market_turnover"] = economy_system.ledger.get("market_turnover", 0.0) + wage * 0.3
                if hasattr(economy_system, "_event"):
                    economy_system._event(f"{npc.name} delivered {bundles:.1f} bundles to storage")
            return

        if not plot or plot not in self.plots:
            return

        p = self.plots[plot]
        if action == "till" and p["stage"] in (self.STAGE_UNTILLED, self.STAGE_HARVESTED):
            p["stage"] = self.STAGE_TILLED
            p["timer"] = 0.0
            npc.show_speech("Tilling the soil.", 1.8)
        elif action == "sow" and p["stage"] == self.STAGE_TILLED:
            p["stage"] = self.STAGE_SOWN
            p["timer"] = 0.0
            npc.show_speech("Sowing wheat seeds.", 1.8)
        elif action == "harvest" and p["stage"] == self.STAGE_RIPE:
            p["stage"] = self.STAGE_HARVESTED
            p["timer"] = 0.0
            self.worker_carry[npc.name] = self.worker_carry.get(npc.name, 0.0) + random.uniform(0.8, 1.2)
            npc.show_speech("Harvesting ripe wheat.", 1.9)
            npc.show_emotion("🌾", 2.0)

    def _update_worker(self, npc, dt, economy_system):
        name = npc.name
        current = self.worker_progress.get(name)
        tile = self._npc_tile(npc)

        if current and current["action"] == "deliver":
            if self._is_in_zone(tile, self.storage_zone):
                self._finish_action(npc, "deliver", None, economy_system)
                self.worker_progress.pop(name, None)
                npc.state = "idle"
            return

        if not self._in_any_farm_zone(tile):
            # If not currently farming, let routine system move worker naturally.
            if not current and npc.state in ("idle", "socializing"):
                if random.random() < 0.25:
                    npc._go_to_zone(random.choice(self.farm_zones))
                    npc.state = "walking"
            return

        if current is None:
            action, plot = self._choose_action(npc)
            if action != "idle":
                self._start_action(npc, action, plot)
            return

        action = current["action"]
        plot = current["plot"]
        if action in ("till", "sow", "harvest"):
            if plot is None:
                self.worker_progress.pop(name, None)
                return
            target_x = plot[0] * TILE_SIZE + TILE_SIZE // 2
            target_y = plot[1] * TILE_SIZE + TILE_SIZE // 2
            dist_sq = (npc.x - target_x) ** 2 + (npc.y - target_y) ** 2
            if dist_sq > (18 * 18):
                npc.state = "walking"
                return

            npc.state = "working"
            current["progress"] += dt
            if current["progress"] >= current["duration"]:
                self._finish_action(npc, action, plot, economy_system)
                self.worker_progress.pop(name, None)
                npc.state = "idle"

    def render(self, surface, camera):
        cam_rect = camera.get_visible_rect()
        tile_min_x = max(0, int(cam_rect.x // TILE_SIZE) - 1)
        tile_min_y = max(0, int(cam_rect.y // TILE_SIZE) - 1)
        tile_max_x = int((cam_rect.x + cam_rect.width) // TILE_SIZE) + 2
        tile_max_y = int((cam_rect.y + cam_rect.height) // TILE_SIZE) + 2

        for ty in range(tile_min_y, tile_max_y):
            for tx in range(tile_min_x, tile_max_x):
                key = (tx, ty)
                plot = self.plots.get(key)
                if not plot:
                    continue
                stage = int(plot["stage"])
                tile_surf = self._stage_surfaces.get(stage)
                if tile_surf is None:
                    continue
                sx = tx * TILE_SIZE - cam_rect.x
                sy = ty * TILE_SIZE - cam_rect.y
                surface.blit(tile_surf, (sx, sy))

    def get_stats_for_ui(self):
        counts = {k: 0 for k in self.STAGE_NAMES.values()}
        for info in self.plots.values():
            counts[self.STAGE_NAMES.get(info["stage"], "untilled")] += 1
        return {
            "plots": len(self.plots),
            "stage_counts": counts,
            "storage_zone": self.storage_zone,
        }
