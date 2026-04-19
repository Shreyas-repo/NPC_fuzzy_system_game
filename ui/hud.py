"""
HUD (Heads-Up Display) — time, NPC info, minimap, controls hint.
"""
import pygame
import math
from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, COLORS, TILE_SIZE,
    MAP_WIDTH, MAP_HEIGHT, NPC_COUNTS, HUD_CHATTER_HEARING_RADIUS,
    ADAPTIVE_PROFILE_PRESETS,
)


class HUD:
    """Heads-up display showing game info, minimap, and NPC details."""

    def __init__(self):
        self.font = None
        self.small_font = None
        self.title_font = None
        self.hovered_npc = None
        self.show_minimap = True
        self.economy_system = None
        self.chatter_paused = False
        self.chatter_scroll = 0
        self.chatter_snapshot = []
        self.chatter_cluster_mode = "auto"  # auto | all | explicit cluster key
        self.hovered_player_ref = None
        self.hovered_npcs_ref = []
        self.chatter_hearing_radius = float(HUD_CHATTER_HEARING_RADIUS)
        self.runtime_profile = "custom"

        # Minimap
        self.minimap_size = 150
        self.minimap_x = SCREEN_WIDTH - self.minimap_size - 10
        self.minimap_y = 10
        self.minimap_scale_x = self.minimap_size / (MAP_WIDTH * TILE_SIZE)
        self.minimap_scale_y = self.minimap_size / (MAP_HEIGHT * TILE_SIZE)

    def init_fonts(self):
        self.font = pygame.font.SysFont("Arial", 13)
        self.small_font = pygame.font.SysFont("Arial", 11)
        self.title_font = pygame.font.SysFont("Arial", 15, bold=True)

    def update(self, player, npcs, camera):
        """Update HUD state — find hovered NPC, etc."""
        self.hovered_player_ref = player
        self.hovered_npcs_ref = npcs
        mouse_x, mouse_y = pygame.mouse.get_pos()
        world_x, world_y = camera.screen_to_world(mouse_x, mouse_y)

        self.hovered_npc = None
        for npc in npcs:
            dist = math.sqrt((world_x - npc.x) ** 2 + (world_y - npc.y) ** 2)
            if dist < 20:
                self.hovered_npc = npc
                break

    def render(
        self,
        surface,
        day_cycle,
        player,
        npcs,
        camera,
        clustering_engine=None,
        social_system=None,
        economy_system=None,
        policy_tuner=None,
        research_metrics=None,
    ):
        """Render all HUD elements."""
        if not self.font:
            return

        self.economy_system = economy_system

        self._render_time(surface, day_cycle)
        self._render_controls_hint(surface, camera)
        if self.show_minimap:
            self._render_minimap(surface, player, npcs, camera)
        if self.hovered_npc:
            self._render_npc_tooltip(surface)
        self._render_npc_count(surface, npcs)
        if clustering_engine:
            self._render_cluster_info(surface, clustering_engine)
        if social_system:
            self._render_social_chatter(surface, social_system)
        if economy_system:
            self._render_economy_panel(surface, economy_system)
        if policy_tuner is not None and research_metrics is not None:
            self._render_soft_compute_status(surface, policy_tuner, research_metrics)

    def _render_soft_compute_status(self, surface, policy_tuner, research_metrics):
        """Render live fuzzy/evolution convergence diagnostics."""
        panel_w = 360
        panel_h = 78
        x = SCREEN_WIDTH - panel_w - 10
        y = self.minimap_y + self.minimap_size + 34

        bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        pygame.draw.rect(bg, (15, 20, 28, 220), (0, 0, panel_w, panel_h), border_radius=8)
        pygame.draw.rect(bg, (148, 188, 116), (0, 0, panel_w, panel_h), 1, border_radius=8)

        title = self.small_font.render("Soft Computing Live", True, (168, 226, 136))
        bg.blit(title, (10, 7))

        gen = int(getattr(policy_tuner, "generation", 0))
        eval_idx = int(getattr(policy_tuner, "eval_index", 0))
        pop_size = int(getattr(policy_tuner, "population_size", 0))
        best_score = float(getattr(policy_tuner, "best_score", 0.0))

        metrics = getattr(research_metrics, "last_summary", {}) or {}
        stability = float(metrics.get("social_stability", 0.0))
        trust = float(metrics.get("avg_trust", 0.0))
        conflict = float(metrics.get("conflict_rate", 0.0))

        line1 = f"Gen {gen} | Candidate {eval_idx + 1}/{max(1, pop_size)} | Best {best_score:.3f}"
        line2 = f"Stability {stability:.3f} | Trust {trust:.3f} | Conflict {conflict:.3f}"

        bg.blit(self.small_font.render(line1, True, COLORS["ui_text"]), (10, 28))
        bg.blit(self.small_font.render(line2, True, COLORS["ui_text_dim"]), (10, 45))

        surface.blit(bg, (x, y))

    def toggle_chatter_pause(self, social_system):
        """Pause/resume village chatter feed; when paused, freeze visible history."""
        self.chatter_paused = not self.chatter_paused
        if self.chatter_paused and social_system is not None:
            self.chatter_snapshot = social_system.get_recent_chats(limit=220)
            self.chatter_scroll = 0
        if not self.chatter_paused:
            self.chatter_scroll = 0

    def scroll_chatter(self, delta):
        """Scroll paused chatter history. Positive delta moves older."""
        self.chatter_scroll = max(0, self.chatter_scroll + int(delta))

    def cycle_chatter_cluster_mode(self, delta, player, npcs, social_system):
        """Cycle chatter view between AUTO, ALL, and explicit cluster buckets."""
        if social_system is None:
            return
        buckets = social_system.get_chat_clusters(limit=220)
        keys = sorted(buckets.keys())
        modes = ["auto", "all"] + keys
        if not modes:
            return

        current = self.chatter_cluster_mode if self.chatter_cluster_mode in modes else "auto"
        idx = modes.index(current)
        idx = (idx + int(delta)) % len(modes)
        self.chatter_cluster_mode = modes[idx]
        self.chatter_scroll = 0

    def set_adaptation_profile(self, profile_name):
        preset = dict((ADAPTIVE_PROFILE_PRESETS or {}).get(profile_name, {}))
        if not preset:
            return
        self.runtime_profile = str(profile_name)
        self.chatter_hearing_radius = float(preset.get("hud_hearing_radius", self.chatter_hearing_radius))

    def _cluster_key_from_npc(self, npc):
        if npc is None:
            return "isolated"
        social_group = int(getattr(npc, "social_group", -1))
        behavior_cluster = int(getattr(npc, "cluster_id", -1))
        if social_group >= 0:
            return f"group-{social_group}"
        if behavior_cluster >= 0:
            return f"behavior-{behavior_cluster}"
        return "isolated"

    def _nearest_cluster_key(self, player, npcs):
        if player is None or not npcs:
            return "isolated"
        nearest = min(npcs, key=lambda n: (n.x - player.x) * (n.x - player.x) + (n.y - player.y) * (n.y - player.y))
        return self._cluster_key_from_npc(nearest)

    def _resolve_chatter_cluster(self, player, npcs, social_system):
        buckets = social_system.get_chat_clusters(limit=220)
        keys = sorted(buckets.keys())
        if self.chatter_cluster_mode == "all":
            return "all", "All Clusters"
        if self.chatter_cluster_mode == "auto":
            key = self._nearest_cluster_key(player, npcs)
            label = buckets.get(key, "Nearby Cluster")
            return key, f"{label} (auto)"
        key = self.chatter_cluster_mode
        if key in buckets:
            return key, buckets[key]
        self.chatter_cluster_mode = "auto"
        key = self._nearest_cluster_key(player, npcs)
        return key, f"{buckets.get(key, 'Nearby Cluster')} (auto)"

    def _wrap_text(self, text, max_width):
        words = text.split()
        lines = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if self.small_font.render(candidate, True, (255, 255, 255)).get_width() <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def _render_social_chatter(self, surface, social_system):
        """Render visible NPC-to-NPC chat feed (including deep/deceptive/crime talk)."""
        live = social_system.get_recent_chats_for_listener(
            self.hovered_player_ref,
            limit=220,
            hearing_radius=self.chatter_hearing_radius,
        )
        chats = self.chatter_snapshot if self.chatter_paused else live
        if not chats:
            return

        cluster_key, cluster_label = self._resolve_chatter_cluster(self.hovered_player_ref, self.hovered_npcs_ref, social_system)
        if cluster_key != "all":
            chats = [c for c in chats if c.get("cluster_key") == cluster_key]
        if not chats:
            return

        panel_w = 560
        panel_h = 210
        x = SCREEN_WIDTH - panel_w - 10
        y = SCREEN_HEIGHT - panel_h - 36

        bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        pygame.draw.rect(bg, (16, 20, 30, 220), (0, 0, panel_w, panel_h), border_radius=8)
        pygame.draw.rect(bg, (82, 132, 190), (0, 0, panel_w, panel_h), 1, border_radius=8)

        pause_suffix = " [PAUSED]" if self.chatter_paused else ""
        title_text = f"Village Chatter • {cluster_label}{pause_suffix}"
        title = self.small_font.render(title_text, True, COLORS["ui_accent"])
        bg.blit(title, (10, 8))

        tone_color = {
            "casual": COLORS["ui_text_dim"],
            "deep": (140, 205, 255),
            "humor": (255, 228, 140),
            "sadness": (170, 185, 220),
            "news": (156, 232, 178),
            "lie": COLORS["ui_warning"],
            "crime": COLORS["ui_negative"],
        }

        line_y = 30
        max_text_w = panel_w - 20
        visible_items = 10
        max_scroll = max(0, len(chats) - visible_items)
        self.chatter_scroll = min(self.chatter_scroll, max_scroll)
        end_idx = len(chats) - self.chatter_scroll
        start_idx = max(0, end_idx - visible_items)

        for item in chats[start_idx:end_idx]:
            speaker = item.get("speaker", "NPC")
            text = item.get("text", "...")
            tone = item.get("tone", "casual")
            color = tone_color.get(tone, COLORS["ui_text_dim"])

            raw = f"{speaker}: {text}"
            wrapped = self._wrap_text(raw, max_text_w)
            for line in wrapped[:2]:
                if line_y > panel_h - 16:
                    break
                line_surf = self.small_font.render(line, True, color)
                bg.blit(line_surf, (10, line_y))
                line_y += 14
            if line_y > panel_h - 16:
                break

        hint_text = "C: pause | PgUp/PgDn: scroll | [:prev ]:next cluster"
        hint_surf = self.small_font.render(hint_text, True, COLORS["ui_text_dim"])
        bg.blit(hint_surf, (10, panel_h - 16))

        surface.blit(bg, (x, y))

    def _render_economy_panel(self, surface, economy_system):
        stats = economy_system.get_stats_for_ui()
        events = economy_system.get_recent_events(limit=2)

        panel_w = 330
        panel_h = 122
        x = 10
        y = SCREEN_HEIGHT - panel_h - 38

        bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        pygame.draw.rect(bg, (15, 20, 28, 220), (0, 0, panel_w, panel_h), border_radius=8)
        pygame.draw.rect(bg, (110, 168, 112), (0, 0, panel_w, panel_h), 1, border_radius=8)

        title = self.small_font.render("Village Economy", True, (152, 226, 136))
        bg.blit(title, (10, 7))

        line1 = f"Treasury {stats['treasury']:.1f}c | Tax {stats['last_tax_rate'] * 100:.0f}% | Barter {stats['barter_count']}"
        line2 = f"Wheat {stats['grain_output']:.1f} | Turnover {stats['market_turnover']:.1f} | Farms {stats['owned_farms']}/{stats['farm_count']}"
        line3 = f"Collectors {stats.get('tax_collector_count', 0)} | Corrupt hits {stats.get('corrupt_collections', 0)} | Skim {stats.get('corruption_skim_last_round', 0.0):.1f}"

        bg.blit(self.small_font.render(line1, True, COLORS["ui_text"]), (10, 25))
        bg.blit(self.small_font.render(line2, True, COLORS["ui_text"]), (10, 41))
        bg.blit(self.small_font.render(line3, True, COLORS["ui_text_dim"]), (10, 55))

        ey = 73
        for event in events:
            ev = self._wrap_text(event, panel_w - 20)[0]
            bg.blit(self.small_font.render(f"- {ev}", True, COLORS["ui_text_dim"]), (10, ey))
            ey += 14

        surface.blit(bg, (x, y))

    def _render_time(self, surface, day_cycle):
        """Render time and day info."""
        bg = pygame.Surface((182, 58), pygame.SRCALPHA)
        pygame.draw.rect(bg, (14, 18, 30, 225), (0, 0, 182, 58), border_radius=8)
        pygame.draw.rect(bg, (82, 132, 190), (0, 0, 182, 58), 1, border_radius=8)
        pygame.draw.rect(bg, (255, 255, 255, 20), (1, 1, 180, 18), border_radius=8)

        time_str = day_cycle.time_string
        day_str = f"Day {day_cycle.day_count}"
        phase = day_cycle.day_phase.capitalize()

        phase_colors = {
            "Dawn": (255, 180, 100),
            "Day": (255, 255, 180),
            "Dusk": (255, 140, 80),
            "Night": (140, 140, 220),
        }
        phase_color = phase_colors.get(phase, COLORS["ui_text"])

        time_surf = self.title_font.render(f"⏰ {time_str}", True, COLORS["ui_text"])
        day_surf = self.small_font.render(f"{day_str} • {phase}", True, phase_color)

        bg.blit(time_surf, (12, 8))
        bg.blit(day_surf, (12, 34))
        surface.blit(bg, (10, 10))

    def _render_controls_hint(self, surface, camera):
        """Render keyboard controls hint."""
        if camera.spectator_mode:
            hints = "WASD: Move | +/-: Zoom | Click: Follow NPC | TAB: Exit Spectator | 1-5: Speed | C/PgUp/PgDn/[ ]: Chatter"
        else:
            hints = "WASD: Move | E: Talk to NPC | TAB: Spectator Mode | C/PgUp/PgDn/[ ]: Chatter | ESC: Menu"

        hint_surf = self.small_font.render(hints, True, COLORS["ui_text_dim"])
        bg = pygame.Surface((hint_surf.get_width() + 16, 22), pygame.SRCALPHA)
        pygame.draw.rect(bg, (20, 20, 35, 160), (0, 0, bg.get_width(), 22), border_radius=4)
        bg.blit(hint_surf, (8, 4))
        surface.blit(bg, (SCREEN_WIDTH // 2 - bg.get_width() // 2, SCREEN_HEIGHT - 30))

    def _render_minimap(self, surface, player, npcs, camera):
        """Render minimap showing NPC positions."""
        panel_w = self.minimap_size + 8
        panel_h = self.minimap_size + 26
        mm = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        pygame.draw.rect(mm, (12, 16, 24, 220), (0, 0, panel_w, panel_h), border_radius=8)
        pygame.draw.rect(mm, (66, 108, 160), (0, 0, panel_w, panel_h), 1, border_radius=8)
        pygame.draw.rect(mm, (255, 255, 255, 20), (1, 1, panel_w - 2, 14), border_radius=8)

        # Draw terrain (simplified)
        mm_inner = pygame.Surface((self.minimap_size, self.minimap_size))
        mm_inner.fill((40, 60, 35))

        # NPC dots
        for npc in npcs:
            mx = int(npc.x * self.minimap_scale_x)
            my = int(npc.y * self.minimap_scale_y)
            mx = max(1, min(self.minimap_size - 1, mx))
            my = max(1, min(self.minimap_size - 1, my))
            pygame.draw.circle(mm_inner, npc.color, (mx, my), 2)

        # Player dot
        px = int(player.x * self.minimap_scale_x)
        py_pos = int(player.y * self.minimap_scale_y)
        px = max(1, min(self.minimap_size - 1, px))
        py_pos = max(1, min(self.minimap_size - 1, py_pos))
        pygame.draw.circle(mm_inner, COLORS["player"], (px, py_pos), 3)
        pygame.draw.circle(mm_inner, (255, 255, 255), (px, py_pos), 3, 1)

        # Camera viewport rectangle
        cam_rect = camera.get_visible_rect()
        vx = int(cam_rect.x * self.minimap_scale_x)
        vy = int(cam_rect.y * self.minimap_scale_y)
        vw = max(4, int(cam_rect.width * self.minimap_scale_x))
        vh = max(4, int(cam_rect.height * self.minimap_scale_y))
        pygame.draw.rect(mm_inner, (255, 255, 255, 100), (vx, vy, vw, vh), 1)

        mm.blit(mm_inner, (4, 18))
        surface.blit(mm, (self.minimap_x - 4, self.minimap_y - 8))

        # Minimap label
        label = self.small_font.render("Minimap", True, COLORS["ui_text_dim"])
        surface.blit(label, (self.minimap_x + self.minimap_size // 2 - label.get_width() // 2,
                     self.minimap_y + self.minimap_size + 12))

    def _render_npc_tooltip(self, surface):
        """Render tooltip for hovered NPC."""
        npc = self.hovered_npc
        info = npc.get_info_dict()
        econ_line = None
        econ_color = COLORS["ui_text_dim"]
        if self.economy_system is not None:
            econ = self.economy_system.get_npc_economic_status(npc)
            econ_line = f"Economy: {econ.get('label', 'unknown')}"
            tag = econ.get("color", "neutral")
            if tag == "positive":
                econ_color = COLORS["ui_positive"]
            elif tag == "negative":
                econ_color = COLORS["ui_negative"]
            elif tag == "warning":
                econ_color = COLORS["ui_warning"]

        lines = [
            f"{info['name']} ({info['class']})",
            f"State: {info['state']}",
            f"Mood: {'█' * int(info['mood'] * 10)}{'░' * (10 - int(info['mood'] * 10))} {info['mood']:.0%}",
            f"Energy: {'█' * int(info['energy'] * 10)}{'░' * (10 - int(info['energy'] * 10))} {info['energy']:.0%}",
            f"Hunger: {'█' * int(info['hunger'] * 10)}{'░' * (10 - int(info['hunger'] * 10))} {info['hunger']:.0%}",
            f"Trust: {'█' * int(info['trust'] * 10)}{'░' * (10 - int(info['trust'] * 10))} {info['trust']:.0%}",
            f"Cluster: {info['cluster']}",
        ]
        if econ_line:
            lines.append(econ_line)

        max_width = max(self.font.render(l, True, (0, 0, 0)).get_width() for l in lines)
        tooltip_w = max_width + 20
        tooltip_h = len(lines) * 18 + 10

        mouse_x, mouse_y = pygame.mouse.get_pos()
        tx = min(mouse_x + 15, SCREEN_WIDTH - tooltip_w - 5)
        ty = max(5, mouse_y - tooltip_h - 5)

        bg = pygame.Surface((tooltip_w, tooltip_h), pygame.SRCALPHA)
        pygame.draw.rect(bg, (25, 25, 40, 230), (0, 0, tooltip_w, tooltip_h), border_radius=6)
        pygame.draw.rect(bg, npc.color, (0, 0, tooltip_w, tooltip_h), 2, border_radius=6)

        y = 5
        for i, line in enumerate(lines):
            color = COLORS["ui_text"] if i > 0 else npc.color
            if econ_line and i == len(lines) - 1:
                color = econ_color
            text_surf = self.font.render(line, True, color)
            bg.blit(text_surf, (10, y))
            y += 18

        surface.blit(bg, (tx, ty))

    def _render_npc_count(self, surface, npcs):
        """Render NPC count by class."""
        y = 70
        bg = pygame.Surface((146, 24 + len(NPC_COUNTS) * 16), pygame.SRCALPHA)
        pygame.draw.rect(bg, (14, 18, 30, 210), (0, 0, bg.get_width(), bg.get_height()), border_radius=6)
        pygame.draw.rect(bg, (82, 132, 190), (0, 0, bg.get_width(), bg.get_height()), 1, border_radius=6)

        title = self.small_font.render("NPCs in Village", True, COLORS["ui_text_dim"])
        bg.blit(title, (10, 5))

        by = 24
        class_counts = {}
        for npc in npcs:
            class_counts[npc.npc_class] = class_counts.get(npc.npc_class, 0) + 1

        for cls, count in sorted(class_counts.items()):
            color_key = f"npc_{cls.lower()}"
            color = COLORS.get(color_key, (150, 150, 150))
            pygame.draw.circle(bg, color, (14, by + 6), 4)
            text = self.small_font.render(f"{cls}: {count}", True, COLORS["ui_text"])
            bg.blit(text, (24, by))
            by += 16

        surface.blit(bg, (10, y))

    def _render_cluster_info(self, surface, clustering_engine):
        """Render behavior cluster info."""
        cluster_info = clustering_engine.get_behavior_cluster_info()
        if not cluster_info:
            return

        y_start = SCREEN_HEIGHT - 300
        bg_h = 20 + len(cluster_info) * 16
        bg = pygame.Surface((200, bg_h), pygame.SRCALPHA)
        pygame.draw.rect(bg, (20, 20, 35, 180), (0, 0, 200, bg_h), border_radius=4)

        title = self.small_font.render("Behavior Clusters", True, COLORS["ui_accent"])
        bg.blit(title, (8, 4))

        by = 22
        cluster_colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255),
        ]
        for label, desc in cluster_info.items():
            color = cluster_colors[label % len(cluster_colors)]
            pygame.draw.circle(bg, color, (14, by + 6), 4)
            text = self.small_font.render(f"C{label}: {desc}", True, COLORS["ui_text"])
            bg.blit(text, (24, by))
            by += 16

        surface.blit(bg, (10, y_start))
