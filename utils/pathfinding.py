"""
A* Pathfinding on a tile grid.
"""
import heapq
from config import TILE_SIZE, MAP_WIDTH, MAP_HEIGHT, WALKABLE_TILES


def heuristic(a, b):
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(pos, tile_map):
    """Get walkable neighboring tiles (4-directional)."""
    x, y = pos
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT:
            if tile_map[ny][nx] in WALKABLE_TILES:
                neighbors.append((nx, ny))
    return neighbors


def astar(start_tile, end_tile, tile_map):
    """
    A* pathfinding from start_tile to end_tile.
    Returns a list of (tile_x, tile_y) positions forming the path,
    or an empty list if no path found.
    """
    if start_tile == end_tile:
        return [start_tile]

    # Clamp to map bounds
    sx = max(0, min(start_tile[0], MAP_WIDTH - 1))
    sy = max(0, min(start_tile[1], MAP_HEIGHT - 1))
    ex = max(0, min(end_tile[0], MAP_WIDTH - 1))
    ey = max(0, min(end_tile[1], MAP_HEIGHT - 1))
    start_tile = (sx, sy)
    end_tile = (ex, ey)

    # If destination is not walkable, find nearest walkable tile
    if tile_map[ey][ex] not in WALKABLE_TILES:
        best = None
        best_dist = float('inf')
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                cx, cy = ex + dx, ey + dy
                if 0 <= cx < MAP_WIDTH and 0 <= cy < MAP_HEIGHT:
                    if tile_map[cy][cx] in WALKABLE_TILES:
                        d = abs(dx) + abs(dy)
                        if d < best_dist:
                            best_dist = d
                            best = (cx, cy)
        if best is None:
            return []
        end_tile = best

    open_set = []
    heapq.heappush(open_set, (0, start_tile))
    came_from = {}
    g_score = {start_tile: 0}
    f_score = {start_tile: heuristic(start_tile, end_tile)}
    closed_set = set()

    max_iterations = 2000  # prevent freezing on large maps

    iterations = 0
    while open_set and iterations < max_iterations:
        iterations += 1
        _, current = heapq.heappop(open_set)

        if current == end_tile:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        if current in closed_set:
            continue
        closed_set.add(current)

        for neighbor in get_neighbors(current, tile_map):
            if neighbor in closed_set:
                continue

            tentative_g = g_score[current] + 1

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, end_tile)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []  # No path found


def bidirectional_astar(start_tile, end_tile, tile_map):
    """Bidirectional A* search to reduce expansion on longer paths."""
    if start_tile == end_tile:
        return [start_tile]

    sx = max(0, min(start_tile[0], MAP_WIDTH - 1))
    sy = max(0, min(start_tile[1], MAP_HEIGHT - 1))
    ex = max(0, min(end_tile[0], MAP_WIDTH - 1))
    ey = max(0, min(end_tile[1], MAP_HEIGHT - 1))
    start = (sx, sy)
    goal = (ex, ey)

    if tile_map[goal[1]][goal[0]] not in WALKABLE_TILES:
        best = None
        best_dist = float("inf")
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                cx, cy = goal[0] + dx, goal[1] + dy
                if 0 <= cx < MAP_WIDTH and 0 <= cy < MAP_HEIGHT:
                    if tile_map[cy][cx] in WALKABLE_TILES:
                        d = abs(dx) + abs(dy)
                        if d < best_dist:
                            best_dist = d
                            best = (cx, cy)
        if best is None:
            return []
        goal = best

    f_open = []
    b_open = []
    heapq.heappush(f_open, (0, start))
    heapq.heappush(b_open, (0, goal))

    f_parent = {}
    b_parent = {}
    f_g = {start: 0}
    b_g = {goal: 0}
    f_closed = set()
    b_closed = set()

    meet = None
    iterations = 0
    max_iterations = 2600

    while f_open and b_open and iterations < max_iterations:
        iterations += 1

        _, f_cur = heapq.heappop(f_open)
        if f_cur in f_closed:
            continue
        f_closed.add(f_cur)
        if f_cur in b_closed:
            meet = f_cur
            break

        for nb in get_neighbors(f_cur, tile_map):
            if nb in f_closed:
                continue
            ng = f_g[f_cur] + 1
            if ng < f_g.get(nb, float("inf")):
                f_parent[nb] = f_cur
                f_g[nb] = ng
                heapq.heappush(f_open, (ng + heuristic(nb, goal), nb))

        _, b_cur = heapq.heappop(b_open)
        if b_cur in b_closed:
            continue
        b_closed.add(b_cur)
        if b_cur in f_closed:
            meet = b_cur
            break

        for nb in get_neighbors(b_cur, tile_map):
            if nb in b_closed:
                continue
            ng = b_g[b_cur] + 1
            if ng < b_g.get(nb, float("inf")):
                b_parent[nb] = b_cur
                b_g[nb] = ng
                heapq.heappush(b_open, (ng + heuristic(nb, start), nb))

    if meet is None:
        return []

    # Reconstruct start -> meet
    left = [meet]
    cur = meet
    while cur in f_parent:
        cur = f_parent[cur]
        left.append(cur)
    left.reverse()

    # Reconstruct meet -> goal
    right = []
    cur = meet
    while cur in b_parent:
        cur = b_parent[cur]
        right.append(cur)

    return left + right


def find_path(start_tile, end_tile, tile_map, prefer_bidirectional=False):
    """Try multiple path strategies for resilient navigation."""
    if prefer_bidirectional:
        path = bidirectional_astar(start_tile, end_tile, tile_map)
        if path:
            return path
        return astar(start_tile, end_tile, tile_map)

    path = astar(start_tile, end_tile, tile_map)
    if path:
        return path
    return bidirectional_astar(start_tile, end_tile, tile_map)


def pixel_to_tile(px, py):
    """Convert pixel coordinates to tile coordinates."""
    return int(px // TILE_SIZE), int(py // TILE_SIZE)


def tile_to_pixel(tx, ty):
    """Convert tile coordinates to pixel center coordinates."""
    return tx * TILE_SIZE + TILE_SIZE // 2, ty * TILE_SIZE + TILE_SIZE // 2
