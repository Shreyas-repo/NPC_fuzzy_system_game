"""
Reference-layout building blocks.
"""
import pygame
from config import TILE_SIZE, ZONES
from utils.sprite_assets import SpriteAssets


class BlockBuilding:
    """Simple colored rectangle with centered label."""

    def __init__(self, name, zone_key, fill_color, label, has_door=False):
        self.name = name
        self.zone_key = zone_key
        self.fill_color = fill_color
        self.label = label
        self.has_door = has_door

        zx, zy, zw, zh = ZONES[zone_key]
        self.tile_x, self.tile_y = zx, zy
        self.tile_w, self.tile_h = zw, zh
        self.pixel_x = zx * TILE_SIZE
        self.pixel_y = zy * TILE_SIZE
        self.pixel_w = zw * TILE_SIZE
        self.pixel_h = zh * TILE_SIZE

        self.rect = pygame.Rect(self.pixel_x, self.pixel_y, self.pixel_w, self.pixel_h)
        self.surface = self._create_surface()

    def _create_surface(self):
        surf = pygame.Surface((self.pixel_w, self.pixel_h), pygame.SRCALPHA)
        surf.fill(self.fill_color)

        font_size = max(16, min(64, self.pixel_h // 3))
        font = pygame.font.SysFont("arial", font_size, bold=True)
        text = font.render(self.label, True, (0, 0, 0))
        text_x = (self.pixel_w - text.get_width()) // 2
        text_y = (self.pixel_h - text.get_height()) // 2
        surf.blit(text, (text_x, text_y))
        return surf

    def render(self, surface, camera):
        cam_rect = camera.get_visible_rect()
        screen_x = self.pixel_x - cam_rect.x
        screen_y = self.pixel_y - cam_rect.y
        surface.blit(self.surface, (screen_x, screen_y))

    def get_entrance(self):
        return (
            self.pixel_x + self.pixel_w // 2,
            self.pixel_y + self.pixel_h - TILE_SIZE // 2,
        )

    def get_interior_point(self):
        return (
            self.pixel_x + self.pixel_w // 2,
            self.pixel_y + self.pixel_h // 2,
        )


class House3Building(BlockBuilding):
    """Class house rendered with the shared house3 model."""

    def _create_surface(self):
        surf = pygame.Surface((self.pixel_w, self.pixel_h), pygame.SRCALPHA)
        # Keep sprite size consistent across all class houses.
        house_w = min(self.pixel_w - 8, TILE_SIZE * 4)
        house_h = min(self.pixel_h - 8, TILE_SIZE * 4)
        house = SpriteAssets.get().get_house("house3", house_w, house_h)
        if house is not None:
            hx = (self.pixel_w - house_w) // 2
            hy = (self.pixel_h - house_h) // 2
            surf.blit(house, (hx, hy))
            return surf

        # Fallback if texture cannot be loaded.
        surf.fill(self.fill_color)
        return surf


class CastleModelBuilding(BlockBuilding):
    """Castle block that uses downloaded castle model artwork."""

    def _create_surface(self):
        surf = pygame.Surface((self.pixel_w, self.pixel_h), pygame.SRCALPHA)
        model = SpriteAssets.get().get_house("castle", self.pixel_w - 8, self.pixel_h - 8)
        if model is not None:
            mx = (self.pixel_w - model.get_width()) // 2
            my = (self.pixel_h - model.get_height()) // 2
            surf.blit(model, (mx, my))
            return surf
        return super()._create_surface()


def create_buildings():
    """Create only the blocks shown in the reference layout."""
    buildings = {}

    buildings["castle"] = CastleModelBuilding("Castle", "castle", (220, 0, 0), "CASTLE")

    for zone_name in sorted(ZONES):
        if zone_name.startswith("noble_house_"):
            buildings[zone_name] = House3Building("Nobles", zone_name, (184, 102, 217), "nobles")
        elif zone_name.startswith("peasant_house_"):
            buildings[zone_name] = House3Building("Peasant", zone_name, (123, 211, 84), "PEASANT")
        elif zone_name.startswith("trader_house_"):
            buildings[zone_name] = House3Building("Traders", zone_name, (255, 122, 32), "TRADERS")

    return buildings
