# Single PNG Entity Assets

Put one PNG per entity in these folders.

## Folder structure
- `assets/entities/houses/`
- `assets/entities/characters/`
- `assets/entities/things/`
- `assets/entities/tiles/`

## Required/recognized file names

### Houses
- `main_house.png` -> used by `Building`
- `neighborhood_block.png` -> used by `NeighborhoodBlock`

### Characters
- `player.png` -> player sprite
- `npc.png` -> fallback for all NPCs
- Role-specific optional files (lowercase, spaces as `_`):
  - `royal.png`
  - `noble.png`
  - `elite.png`
  - `merchant.png`
  - `blacksmith.png`
  - `traveller.png`
  - `labourer.png`
  - `peasant.png`

### Things (decor)
- `tree.png`
- `trees.png` -> 6-tree spritesheet (3 columns x 2 rows), used for all world trees
- `bush.png`
- `rock.png`

### Tiles
- `grass.png`
- `dirt.png`
- `stone.png`
- `water.png`
- `wall.png`
- `floor.png`
- `farm.png`
- `market.png`

## Notes
- If a file is missing, the game falls back to the old spritesheet/procedural visuals.
- Keep PNGs square where possible for best scaling.
- Characters use a single PNG (no directional spritesheet required).
