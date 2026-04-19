"""Sprite asset loader supporting per-entity PNG files with atlas fallback."""
import os
import pygame


ASSET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "sprites", "tiny16"
)

ENTITY_ASSET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "entities"
)


class SpriteAssets:
    _instance = None

    def __init__(self):
        self.basictiles = None
        self.characters = None
        self._tile_cache = {}
        self._char_cache = {}
        self._dec_cache = {}
        self._house_cache = {}
        self._entity_cache = {}
        self._load()

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = SpriteAssets()
        return cls._instance

    def _load(self):
        basic_path = os.path.join(ASSET_DIR, "basictiles.png")
        chars_path = os.path.join(ASSET_DIR, "characters.png")
        if os.path.exists(basic_path):
            self.basictiles = pygame.image.load(basic_path).convert_alpha()
        if os.path.exists(chars_path):
            self.characters = pygame.image.load(chars_path).convert_alpha()

    def _cut(self, sheet, tx, ty, tw=16, th=16):
        if sheet is None:
            return None
        rect = pygame.Rect(tx * tw, ty * th, tw, th)
        if rect.right > sheet.get_width() or rect.bottom > sheet.get_height():
            return None
        surf = pygame.Surface((tw, th), pygame.SRCALPHA)
        surf.blit(sheet, (0, 0), rect)
        return surf

    def _load_entity_image(self, category, key):
        cache_key = (category, key)
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]

        for ext in ("png", "jpg", "jpeg", "webp"):
            path = os.path.join(ENTITY_ASSET_DIR, category, f"{key}.{ext}")
            if os.path.exists(path):
                img = pygame.image.load(path).convert_alpha()
                if category == "houses":
                    img = self._clean_house_texture(img)
                self._entity_cache[cache_key] = img
                return img

        self._entity_cache[cache_key] = None
        return None

    def _clean_house_texture(self, img):
        """Remove edge background and trim transparent margins from house textures."""
        if img is None:
            return None
        w, h = img.get_size()
        if w <= 2 or h <= 2:
            return img

        # Use average corner color as guessed background (works for boxed JPG/PNG assets).
        corners = [
            img.get_at((0, 0)),
            img.get_at((w - 1, 0)),
            img.get_at((0, h - 1)),
            img.get_at((w - 1, h - 1)),
        ]
        bg = (
            int(sum(c.r for c in corners) / 4),
            int(sum(c.g for c in corners) / 4),
            int(sum(c.b for c in corners) / 4),
        )

        cleaned = img.copy()
        tolerance = 42

        def near_bg(col):
            return (
                abs(col.r - bg[0]) <= tolerance and
                abs(col.g - bg[1]) <= tolerance and
                abs(col.b - bg[2]) <= tolerance
            )

        # Remove only border-connected background to avoid erasing interior details.
        stack = []
        visited = set()

        for x in range(w):
            stack.append((x, 0))
            stack.append((x, h - 1))
        for y in range(h):
            stack.append((0, y))
            stack.append((w - 1, y))

        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            if not (0 <= x < w and 0 <= y < h):
                continue
            c = cleaned.get_at((x, y))
            if not near_bg(c):
                continue

            cleaned.set_at((x, y), (c.r, c.g, c.b, 0))
            stack.append((x + 1, y))
            stack.append((x - 1, y))
            stack.append((x, y + 1))
            stack.append((x, y - 1))

        # Crop to non-transparent content when possible.
        mask = pygame.mask.from_surface(cleaned)
        rects = mask.get_bounding_rects()
        if rects:
            return cleaned.subsurface(rects[0]).copy()
        return cleaned

    def _stylize_castle_texture(self, img):
        """Tone-map castle model so it matches muted medieval world palette."""
        if img is None:
            return None

        w, h = img.get_size()
        if w <= 0 or h <= 0:
            return img

        styled = img.copy()

        # Subtle desaturation + cool stone tint + vertical lighting gradient.
        for y in range(h):
            row_t = y / float(max(1, h - 1))
            # Slightly brighter top, darker base to anchor building to terrain.
            shade = 1.05 - (row_t * 0.2)
            for x in range(w):
                px = styled.get_at((x, y))
                if px.a == 0:
                    continue

                lum = 0.299 * px.r + 0.587 * px.g + 0.114 * px.b
                # Pull extreme saturation down for painterly consistency.
                r = (px.r * 0.62) + (lum * 0.38)
                g = (px.g * 0.62) + (lum * 0.38)
                b = (px.b * 0.62) + (lum * 0.38)

                # Nudge toward neutral cool-gray stone palette.
                r = (r * 0.86) + 18
                g = (g * 0.9) + 20
                b = (b * 0.98) + 24

                r = max(0, min(255, int(r * shade)))
                g = max(0, min(255, int(g * shade)))
                b = max(0, min(255, int(b * shade)))
                styled.set_at((x, y), (r, g, b, px.a))

        # Add soft edge vignette so silhouette reads better against grass/roads.
        edge = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(edge, (0, 0, 0, 36), (0, 0, w, h), width=max(2, w // 28))
        styled.blit(edge, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)

        return styled

    def get_house(self, key, width, height):
        cache_key = (key, width, height)
        if cache_key in self._house_cache:
            return self._house_cache[cache_key]

        src = self._load_entity_image("houses", key)
        if src is None:
            return None

        if key == "castle":
            src = self._stylize_castle_texture(src)

        scaled = pygame.transform.scale(src, (width, height))
        self._house_cache[cache_key] = scaled
        return scaled

    def get_tile(self, key, tile_size):
        cache_key = (key, tile_size)
        if cache_key in self._tile_cache:
            return self._tile_cache[cache_key]

        single_src = self._load_entity_image("tiles", key)
        if single_src is not None:
            scaled = pygame.transform.scale(single_src, (tile_size, tile_size))
            self._tile_cache[cache_key] = scaled
            return scaled

        coords = {
            "grass": (1, 2),
            "dirt": (0, 1),
            "stone": (0, 0),
            "water": (5, 2),
            "wall": (1, 0),
            "floor": (0, 13),
            "farm": (4, 11),
            "market": (2, 8),
        }
        if key not in coords:
            return None

        src = self._cut(self.basictiles, coords[key][0], coords[key][1])
        if src is None:
            return None
        scaled = pygame.transform.scale(src, (tile_size, tile_size))
        self._tile_cache[cache_key] = scaled
        return scaled

    def get_decoration(self, key, tile_size):
        return self.get_decoration_variant(key, tile_size, variant=None)

    def _trim_transparent_bounds(self, img, padding=1):
        """Trim fully transparent margins around a sprite cell."""
        if img is None:
            return None

        mask = pygame.mask.from_surface(img)
        rects = mask.get_bounding_rects()
        if not rects:
            return img

        bounds = rects[0].copy()
        for rect in rects[1:]:
            bounds.union_ip(rect)

        if padding > 0:
            bounds.inflate_ip(padding * 2, padding * 2)
            bounds.x = max(0, bounds.x)
            bounds.y = max(0, bounds.y)
            bounds.width = min(img.get_width() - bounds.x, bounds.width)
            bounds.height = min(img.get_height() - bounds.y, bounds.height)

        if bounds.width <= 0 or bounds.height <= 0:
            return img
        return img.subsurface(bounds).copy()

    def _strip_border_background(self, img, tolerance=30):
        """Make border-connected corner-colored background transparent."""
        if img is None:
            return None

        w, h = img.get_size()
        if w <= 2 or h <= 2:
            return img

        cleaned = img.copy()
        corners = [
            cleaned.get_at((0, 0)),
            cleaned.get_at((w - 1, 0)),
            cleaned.get_at((0, h - 1)),
            cleaned.get_at((w - 1, h - 1)),
        ]

        def near_corner_color(col):
            for c in corners:
                if (
                    abs(col.r - c.r) <= tolerance and
                    abs(col.g - c.g) <= tolerance and
                    abs(col.b - c.b) <= tolerance
                ):
                    return True
            return False

        stack = []
        visited = set()

        for x in range(w):
            stack.append((x, 0))
            stack.append((x, h - 1))
        for y in range(h):
            stack.append((0, y))
            stack.append((w - 1, y))

        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            if not (0 <= x < w and 0 <= y < h):
                continue

            px = cleaned.get_at((x, y))
            if px.a == 0 or not near_corner_color(px):
                continue

            cleaned.set_at((x, y), (px.r, px.g, px.b, 0))
            stack.append((x + 1, y))
            stack.append((x - 1, y))
            stack.append((x, y + 1))
            stack.append((x, y - 1))

        return cleaned

    def _extract_tree_variants(self, src):
        """Extract 6 tree variants from a spritesheet, assumed to be a 3x2 grid."""
        variants = []
        if src is None:
            return variants

        cols, rows = 3, 2
        cell_w = src.get_width() // cols
        cell_h = src.get_height() // rows
        if cell_w <= 0 or cell_h <= 0:
            return variants

        for row in range(rows):
            for col in range(cols):
                rect = pygame.Rect(col * cell_w, row * cell_h, cell_w, cell_h)
                if rect.right > src.get_width() or rect.bottom > src.get_height():
                    continue
                tile = pygame.Surface((cell_w, cell_h), pygame.SRCALPHA)
                tile.blit(src, (0, 0), rect)
                tile = self._strip_border_background(tile)
                tile = self._trim_transparent_bounds(tile, padding=1)
                variants.append(tile)
        return variants

    def get_decoration_variant(self, key, tile_size, variant=None):
        cache_key = (key, tile_size, variant)
        if cache_key in self._dec_cache:
            return self._dec_cache[cache_key]

        single_src = self._load_entity_image("things", key)
        if single_src is not None:
            size = int(tile_size * (1.0 if key != "tree" else 1.25))
            scaled = pygame.transform.scale(single_src, (size, size))
            self._dec_cache[cache_key] = scaled
            return scaled

        if key == "tree":
            sheet_src = self._load_entity_image("things", "trees")
            variants = self._extract_tree_variants(sheet_src)
            if variants:
                selected_index = 0 if variant is None else int(variant) % len(variants)
                tree_src = variants[selected_index]
                size = int(tile_size * 1.25)
                scaled = pygame.transform.scale(tree_src, (size, size))
                self._dec_cache[cache_key] = scaled
                return scaled

        # 16x16 source tiles from basictiles atlas.
        coords = {
            "tree": (3, 2),
            "bush": (2, 2),
            "rock": (4, 0),
        }
        src = self._cut(self.basictiles, *coords.get(key, (99, 99)))
        if src is None:
            return None

        size = int(tile_size * (1.0 if key != "tree" else 1.25))
        scaled = pygame.transform.scale(src, (size, size))
        self._dec_cache[cache_key] = scaled
        return scaled

    def get_character(
        self,
        role,
        direction,
        moving,
        anim_timer,
        tint=None,
        tile_size=32,
        variant_seed=None,
    ):
        """Return a scaled character sprite from characters sheet.

        role selects a character column-group and direction selects row.
        """
        single_name = str(role).strip().lower().replace(" ", "_")
        single_src = self._load_entity_image("characters", single_name)
        if single_src is None and role != "player":
            single_src = self._load_entity_image("characters", "npc")
        if single_src is None and role == "player":
            single_src = self._load_entity_image("characters", "player")

        if single_src is not None:
            cache_key = ("single", role, tint, tile_size, variant_seed)
            if cache_key in self._char_cache:
                return self._char_cache[cache_key]
            src = single_src.copy()
            if tint is not None:
                tint_surf = pygame.Surface(src.get_size(), pygame.SRCALPHA)
                tint_surf.fill((tint[0], tint[1], tint[2], 60))
                src.blit(tint_surf, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
            scaled = pygame.transform.scale(src, (tile_size, tile_size))
            self._char_cache[cache_key] = scaled
            return scaled

        if self.characters is None:
            return None

        group_map = {
            "player": 0,
            "Royal": 1,
            "Noble": 0,
            "Elite": 2,
            "Merchant": 1,
            "Blacksmith": 2,
            "Traveller": 1,
            "Labourer": 0,
            "Peasant": 2,
        }

        groups_available = max(1, self.characters.get_width() // (16 * 3))
        base_group = group_map.get(role, 0) % groups_available
        if role == "player":
            group = base_group
        else:
            seed_text = str(variant_seed) if variant_seed is not None else str(role)
            variant = abs(hash(seed_text)) % groups_available
            group = (base_group + variant) % groups_available
        group_start = group * 3

        # Tiny16 character sheet uses 3 columns per character and 4 directional rows.
        row_map = {"down": 0, "left": 1, "right": 2, "up": 3}
        row = row_map.get(direction, 0)
        frame_col = group_start + (1 if moving and int(anim_timer * 8) % 2 == 0 else 0)

        cache_key = (role, direction, moving, int(anim_timer * 8) % 2, tint, tile_size, variant_seed)
        if cache_key in self._char_cache:
            return self._char_cache[cache_key]

        src = self._cut(self.characters, frame_col, row)
        if src is None:
            return None

        if tint is not None:
            tint_surf = pygame.Surface(src.get_size(), pygame.SRCALPHA)
            tint_surf.fill((tint[0], tint[1], tint[2], 60))
            src = src.copy()
            src.blit(tint_surf, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

        scaled = pygame.transform.scale(src, (tile_size, tile_size))
        self._char_cache[cache_key] = scaled
        return scaled
