"""
Spectator mode UI overlay.
"""
import pygame
import math
from config import SCREEN_WIDTH, SCREEN_HEIGHT, COLORS


class SpectatorUI:
    """Overlay UI for spectator mode."""

    def __init__(self):
        self.font = None
        self.small_font = None
        self.title_font = None
        self.following_npc = None
        self.show_clusters = True
        self.time_speed = 1.0

    def init_fonts(self):
        self.font = pygame.font.SysFont("Arial", 13)
        self.small_font = pygame.font.SysFont("Arial", 11)
        self.title_font = pygame.font.SysFont("Arial", 18, bold=True)

    def handle_click(self, mouse_x, mouse_y, npcs, camera):
        """Handle mouse click to select NPC to follow."""
        world_x, world_y = camera.screen_to_world(mouse_x, mouse_y)

        closest_npc = None
        closest_dist = 30  # max click distance

        for npc in npcs:
            dist = math.sqrt((world_x - npc.x) ** 2 + (world_y - npc.y) ** 2)
            if dist < closest_dist:
                closest_dist = dist
                closest_npc = npc

        if closest_npc:
            self.following_npc = closest_npc
            camera.follow_npc(closest_npc)
        else:
            self.following_npc = None
            camera.stop_following_npc()

        return closest_npc

    def render(self, surface, camera, day_cycle, npcs, clustering_engine=None):
        """Render spectator mode overlay."""
        if not self.font:
            return

        # "SPECTATOR MODE" banner
        banner = self.title_font.render("👁 SPECTATOR MODE", True, COLORS["ui_accent"])
        bg = pygame.Surface((banner.get_width() + 20, 30), pygame.SRCALPHA)
        pygame.draw.rect(bg, (20, 20, 40, 200), (0, 0, bg.get_width(), 30), border_radius=6)
        pygame.draw.rect(bg, COLORS["ui_accent"], (0, 0, bg.get_width(), 30), 2, border_radius=6)
        bg.blit(banner, (10, 4))
        surface.blit(bg, (SCREEN_WIDTH // 2 - bg.get_width() // 2, 10))

        # Time speed controls
        speed_text = f"Speed: {day_cycle.speed_multiplier:.1f}x (Keys 1-5)"
        speed_surf = self.font.render(speed_text, True, COLORS["ui_text"])
        surface.blit(speed_surf, (SCREEN_WIDTH // 2 - speed_surf.get_width() // 2, 48))

        raid_hint = self.small_font.render("Press R to start a raid", True, COLORS["ui_text_dim"])
        surface.blit(raid_hint, (SCREEN_WIDTH // 2 - raid_hint.get_width() // 2, 64))

        # Zoom info
        zoom_text = f"Zoom: {camera.zoom:.1f}x (+/- to change)"
        zoom_surf = self.small_font.render(zoom_text, True, COLORS["ui_text_dim"])
        surface.blit(zoom_surf, (SCREEN_WIDTH // 2 - zoom_surf.get_width() // 2, 80))

        # Following NPC info
        if self.following_npc:
            self._render_following_panel(surface)

        # Gathering visualization
        if clustering_engine and self.show_clusters:
            self._render_gathering_circles(surface, npcs, clustering_engine, camera)

    def _render_following_panel(self, surface):
        """Render detailed info about the followed NPC."""
        npc = self.following_npc
        if not npc:
            return

        panel_w = 250
        panel_h = 200
        panel_x = 10
        panel_y = SCREEN_HEIGHT // 2 - panel_h // 2

        bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        pygame.draw.rect(bg, (25, 25, 40, 220), (0, 0, panel_w, panel_h), border_radius=8)
        pygame.draw.rect(bg, npc.color, (0, 0, panel_w, panel_h), 2, border_radius=8)

        y = 10
        # Name and class
        name_surf = self.title_font.render(f"📌 {npc.name}", True, npc.color)
        bg.blit(name_surf, (10, y)); y += 24

        class_surf = self.font.render(f"Class: {npc.npc_class}", True, COLORS["ui_text"])
        bg.blit(class_surf, (10, y)); y += 18

        state_surf = self.font.render(f"State: {npc.state}", True, COLORS["ui_text"])
        bg.blit(state_surf, (10, y)); y += 22

        # Bars
        bars = [
            ("Mood", npc.behavior_vector["mood"], (100, 200, 100)),
            ("Energy", npc.needs["energy"], (100, 150, 255)),
            ("Hunger", npc.needs["hunger"], (255, 150, 50)),
            ("Social", npc.needs["social_need"], (200, 100, 255)),
            ("Trust", npc.behavior_vector["trust"], (255, 215, 0)),
            ("Wealth", npc.behavior_vector["wealth"], (255, 200, 100)),
        ]

        for label, value, color in bars:
            label_surf = self.small_font.render(f"{label}:", True, COLORS["ui_text_dim"])
            bg.blit(label_surf, (10, y))

            bar_x = 70
            bar_w = 140
            bar_h = 8
            pygame.draw.rect(bg, (50, 50, 60), (bar_x, y + 2, bar_w, bar_h), border_radius=3)
            pygame.draw.rect(bg, color, (bar_x, y + 2, int(bar_w * value), bar_h), border_radius=3)

            val_surf = self.small_font.render(f"{value:.0%}", True, COLORS["ui_text_dim"])
            bg.blit(val_surf, (bar_x + bar_w + 5, y))
            y += 16

        # Cluster info
        cluster_surf = self.small_font.render(f"Behavior Cluster: {npc.cluster_id}", True, COLORS["ui_text_dim"])
        bg.blit(cluster_surf, (10, y)); y += 14
        social_surf = self.small_font.render(f"Social Group: {npc.social_group}", True, COLORS["ui_text_dim"])
        bg.blit(social_surf, (10, y))

        surface.blit(bg, (panel_x, panel_y))

        # "Click elsewhere to unfollow" hint
        hint = self.small_font.render("Click elsewhere to unfollow", True, COLORS["ui_text_dim"])
        surface.blit(hint, (panel_x + panel_w // 2 - hint.get_width() // 2,
                            panel_y + panel_h + 5))

    def _render_gathering_circles(self, surface, npcs, clustering_engine, camera):
        """Draw circles around spatial gatherings detected by DBSCAN."""
        gatherings = clustering_engine.get_gathering_info(npcs)

        gathering_colors = [
            (255, 100, 100, 40), (100, 255, 100, 40), (100, 100, 255, 40),
            (255, 255, 100, 40), (255, 100, 255, 40), (100, 255, 255, 40),
        ]

        for label, info in gatherings.items():
            sx, sy = camera.world_to_screen(info["center_x"], info["center_y"])
            radius = max(30, len(info["npcs"]) * 20)
            color = gathering_colors[label % len(gathering_colors)]

            # Draw gathering circle
            circle_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(circle_surf, color, (radius, radius), radius)
            pygame.draw.circle(circle_surf, (color[0], color[1], color[2], 100),
                               (radius, radius), radius, 2)
            surface.blit(circle_surf, (sx - radius, sy - radius))

            # Label
            label_text = f"Gathering ({len(info['npcs'])} NPCs)"
            label_surf = self.small_font.render(label_text, True,
                                                 (color[0], color[1], color[2]))
            surface.blit(label_surf, (sx - label_surf.get_width() // 2, sy - radius - 15))
