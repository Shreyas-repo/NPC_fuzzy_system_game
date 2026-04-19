"""
Camera system — follows the player or allows free spectator movement.
"""
import math
import pygame
from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, WORLD_WIDTH, WORLD_HEIGHT,
    CAMERA_LERP_SPEED, SPECTATOR_SPEED,
    SPECTATOR_ZOOM_MIN, SPECTATOR_ZOOM_MAX, SPECTATOR_ZOOM_STEP
)


class Camera:
    """Camera that can follow a target or move freely in spectator mode."""

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.zoom = 1.0
        self.target = None
        self.spectator_mode = False
        self.following_npc = None  # NPC to follow in spectator mode

    @property
    def view_width(self):
        return int(SCREEN_WIDTH / self.zoom)

    @property
    def view_height(self):
        return int(SCREEN_HEIGHT / self.zoom)

    def get_visible_rect(self):
        """Get the world-space rectangle that's currently visible."""
        return pygame.Rect(
            int(self.x - self.view_width // 2),
            int(self.y - self.view_height // 2),
            self.view_width,
            self.view_height
        )

    def set_target(self, entity):
        """Set an entity for the camera to follow."""
        self.target = entity

    def follow_npc(self, npc):
        """Follow an NPC in spectator mode."""
        self.following_npc = npc

    def stop_following_npc(self):
        self.following_npc = None

    def update(self, dt, keys=None):
        """Update camera position."""
        if self.spectator_mode:
            if self.following_npc:
                # Smoothly follow the NPC with frame-rate independent lerp
                target_x = self.following_npc.x
                target_y = self.following_npc.y
                factor = 1.0 - math.exp(-CAMERA_LERP_SPEED * 18.0 * dt)
                self.x += (target_x - self.x) * factor
                self.y += (target_y - self.y) * factor
            elif keys:
                # Free movement with WASD
                speed = SPECTATOR_SPEED * dt / self.zoom
                if keys[pygame.K_w] or keys[pygame.K_UP]:
                    self.y -= speed
                if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                    self.y += speed
                if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                    self.x -= speed
                if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                    self.x += speed
        else:
            # Follow player with frame-rate independent smoothing
            if self.target:
                target_x = self.target.x
                target_y = self.target.y
                factor = 1.0 - math.exp(-CAMERA_LERP_SPEED * 12.0 * dt)
                self.x += (target_x - self.x) * factor
                self.y += (target_y - self.y) * factor

        # Clamp to world bounds
        half_w = self.view_width // 2
        half_h = self.view_height // 2
        self.x = max(half_w, min(WORLD_WIDTH - half_w, self.x))
        self.y = max(half_h, min(WORLD_HEIGHT - half_h, self.y))

    def handle_zoom(self, direction):
        """Zoom in (+1) or out (-1). Only works in spectator mode."""
        if self.spectator_mode:
            self.zoom += direction * SPECTATOR_ZOOM_STEP
            self.zoom = max(SPECTATOR_ZOOM_MIN, min(SPECTATOR_ZOOM_MAX, self.zoom))

    def toggle_spectator(self):
        """Toggle between player follow and spectator mode."""
        self.spectator_mode = not self.spectator_mode
        if not self.spectator_mode:
            self.zoom = 1.0
            self.following_npc = None

    def world_to_screen(self, world_x, world_y):
        """Convert world coordinates to screen coordinates."""
        cam_rect = self.get_visible_rect()
        sx = (world_x - cam_rect.x) * self.zoom
        sy = (world_y - cam_rect.y) * self.zoom
        return int(sx), int(sy)

    def screen_to_world(self, screen_x, screen_y):
        """Convert screen coordinates to world coordinates."""
        cam_rect = self.get_visible_rect()
        wx = screen_x / self.zoom + cam_rect.x
        wy = screen_y / self.zoom + cam_rect.y
        return wx, wy
