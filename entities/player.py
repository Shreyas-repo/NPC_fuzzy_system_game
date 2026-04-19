"""
Player character entity.
"""
import pygame
import math
from utils.sprite_assets import SpriteAssets
from config import (
    PLAYER_SPEED, TILE_SIZE, COLORS, INTERACTION_RADIUS,
    MAP_WIDTH, MAP_HEIGHT, WALKABLE_TILES
)


class Player:
    """The player-controlled character."""

    def __init__(self, x, y, tile_map):
        self.x = float(x)
        self.y = float(y)
        self.tile_map = tile_map
        self.speed = PLAYER_SPEED
        self.radius = 12
        self.direction = "down"  # up, down, left, right
        self.moving = False
        self.anim_timer = 0.0
        self._name_font = None
        self._get_name_font()

    def _get_name_font(self):
        if self._name_font is None:
            self._name_font = pygame.font.SysFont("Arial", 12, bold=True)
        return self._name_font

    def update(self, dt, keys):
        """Update player position based on input."""
        dx, dy = 0, 0
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            dy = -1
            self.direction = "up"
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            dy = 1
            self.direction = "down"
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            dx = -1
            self.direction = "left"
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            dx = 1
            self.direction = "right"

        self.moving = dx != 0 or dy != 0

        if self.moving:
            # Normalize diagonal movement
            length = math.sqrt(dx * dx + dy * dy)
            dx /= length
            dy /= length

            new_x = self.x + dx * self.speed * dt
            new_y = self.y + dy * self.speed * dt

            # Check collision
            tile_x = int(new_x // TILE_SIZE)
            tile_y = int(new_y // TILE_SIZE)

            # Check X movement
            if self.tile_map.is_walkable_pixel(new_x, self.y, radius=14):
                self.x = new_x
            # Check Y movement
            if self.tile_map.is_walkable_pixel(self.x, new_y, radius=14):
                self.y = new_y

            # Clamp to world bounds
            self.x = max(TILE_SIZE, min(MAP_WIDTH * TILE_SIZE - TILE_SIZE, self.x))
            self.y = max(TILE_SIZE, min(MAP_HEIGHT * TILE_SIZE - TILE_SIZE, self.y))

            self.anim_timer += dt

    def render(self, surface, camera):
        """Render the player character."""
        cam_rect = camera.get_visible_rect()
        sx = int(self.x - cam_rect.x)
        sy = int(self.y - cam_rect.y)

        sprite = SpriteAssets.get().get_character(
            "player",
            self.direction,
            self.moving,
            self.anim_timer,
            tint=None,
            tile_size=56,
        )
        if sprite is not None:
            # Shadow
            pygame.draw.ellipse(surface, (0, 0, 0, 70), (sx - 12, sy + 5, 24, 8))
            surface.blit(sprite, (sx - sprite.get_width() // 2, sy - sprite.get_height() + 10))
            font = self._get_name_font()
            label = font.render("You", True, (255, 255, 255))
            surface.blit(label, (sx - label.get_width() // 2, sy - self.radius - 28))
            return

        # Shadow
        pygame.draw.ellipse(surface, (0, 0, 0, 60), (sx - 11, sy + 4, 22, 8))

        # Body
        body_color = COLORS["player"]
        outline_color = COLORS["player_outline"]

        # Walking bob animation
        bob = 0
        if self.moving:
            bob = int(math.sin(self.anim_timer * 10) * 2)

        # Body (circle)
        pygame.draw.circle(surface, outline_color, (sx, sy - 5 + bob), self.radius + 1)
        pygame.draw.circle(surface, body_color, (sx, sy - 5 + bob), self.radius)

        # Direction indicator (small triangle)
        dir_offsets = {
            "up":    (0, -self.radius - 4),
            "down":  (0, self.radius + 2),
            "left":  (-self.radius - 4, 0),
            "right": (self.radius + 4, 0),
        }
        dox, doy = dir_offsets[self.direction]
        tip_x = sx + dox
        tip_y = sy - 5 + bob + doy

        if self.direction in ("up", "down"):
            points = [
                (tip_x, tip_y),
                (tip_x - 5, tip_y + (5 if self.direction == "up" else -5)),
                (tip_x + 5, tip_y + (5 if self.direction == "up" else -5)),
            ]
        else:
            points = [
                (tip_x, tip_y),
                (tip_x + (-5 if self.direction == "right" else 5), tip_y - 5),
                (tip_x + (-5 if self.direction == "right" else 5), tip_y + 5),
            ]
        pygame.draw.polygon(surface, (255, 255, 255), points)

        # Name label
        font = self._get_name_font()
        label = font.render("You", True, (255, 255, 255))
        surface.blit(label, (sx - label.get_width() // 2, sy - self.radius - 20 + bob))

    def get_nearby_npc(self, npcs):
        """Find the closest NPC within interaction radius."""
        closest = None
        closest_dist = INTERACTION_RADIUS
        for npc in npcs:
            dist = math.sqrt((self.x - npc.x) ** 2 + (self.y - npc.y) ** 2)
            if dist < closest_dist:
                closest_dist = dist
                closest = npc
        return closest
