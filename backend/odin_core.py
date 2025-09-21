# odin_core.py
# Backend AI logic for ODIN project

import numpy as np

from typing import List, Tuple, Set, Dict, Optional

class Odin:
    def __init__(self):
        self.log = []

    def predict(self, trajectories):
        """Pick the safest trajectory based on minimum collision risk."""
        scores = [self.evaluate_traj(t) for t in trajectories]
        best_idx = np.argmin(scores)
        self.log.append({
            "trajectories": [t.tolist() for t in trajectories],
            "scores": scores,
            "chosen_index": best_idx
        })
        return best_idx, scores

    def evaluate_traj(self, traj):
        """Dummy evaluation: risk = distance to obstacle (smaller is riskier)."""
        obstacle = np.array([5, 5])
        dist = np.linalg.norm(traj - obstacle, axis=1).min()
        return 1 / (dist + 0.1)  # Avoid div/0


class QLearningPathPlanner:
    """
    Simple grid-world Q-learning planner from Earth(start) to Moon(goal) avoiding hazard obstacles.
    Grid coordinates are (x,y,z) with x in [0,width-1], y in [0,height-1], z in [0,depth-1].
    """

    def __init__(self, width: int = 60, height: int = 30, depth: int = 20):
        self.width = width
        self.height = height
        self.depth = depth
        self.actions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]  # E, W, N, S, U, D
        self.num_actions = len(self.actions)

    def _state_index(self, x: int, y: int, z: int) -> int:
        return z * self.width * self.height + y * self.width + x

    def _neighbors(self, x: int, y: int, z: int) -> List[Tuple[int,int,int,int]]:
        out = []
        for ai,(dx,dy,dz) in enumerate(self.actions):
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < self.width and 0 <= ny < self.height and 0 <= nz < self.depth:
                out.append((nx, ny, nz, ai))
        return out

    def train(self,
              start: Tuple[int,int,int],
              goal: Tuple[int,int,int],
              obstacles: Set[Tuple[int,int,int]],
              episodes: int = 500,
              alpha: float = 0.4,
              gamma: float = 0.95,
              epsilon: float = 0.2,
              max_steps: int = 200) -> List[Tuple[int,int,int]]:
        num_states = self.width * self.height * self.depth
        Q = np.zeros((num_states, self.num_actions), dtype=np.float32)

        def step(x: int, y: int, z: int, action_index: int) -> Tuple[int,int,int,float,bool]:
            dx, dy, dz = self.actions[action_index]
            nx, ny, nz = x + dx, y + dy, z + dz
            if not (0 <= nx < self.width and 0 <= ny < self.height and 0 <= nz < self.depth):
                return x, y, z, -2.0, False  # wall penalty
            if (nx, ny, nz) in obstacles:
                return nx, ny, nz, -3.0, False  # obstacle penalty
            if (nx, ny, nz) == goal:
                return nx, ny, nz, +10.0, True  # goal reward
            # small progress reward toward goal on x-axis
            reward = 0.1 if nx > x else -0.05
            return nx, ny, nz, reward, False

        rng = np.random.default_rng()
        for _ in range(episodes):
            x, y, z = start
            for _ in range(max_steps):
                s = self._state_index(x, y, z)
                if rng.random() < epsilon:
                    a = rng.integers(0, self.num_actions)
                else:
                    a = int(np.argmax(Q[s]))
                nx, ny, nz, r, done = step(x, y, z, a)
                ns = self._state_index(nx, ny, nz)
                Q[s, a] = (1 - alpha) * Q[s, a] + alpha * (r + gamma * np.max(Q[ns]))
                x, y, z = nx, ny, nz
                if done:
                    break

        # Derive greedy path
        path = [start]
        visited = set([start])
        x, y, z = start
        for _ in range(self.width * 3):
            if (x, y, z) == goal:
                break
            s = self._state_index(x, y, z)
            a = int(np.argmax(Q[s]))
            dx, dy, dz = self.actions[a]
            nx, ny, nz = x + dx, y + dy, z + dz
            if not (0 <= nx < self.width and 0 <= ny < self.height and 0 <= nz < self.depth):
                break
            if (nx, ny, nz) in visited:
                # fallback: try any neighbor not visited
                moved = False
                for _,_,_,ai in self._neighbors(x, y, z):
                    dx2, dy2, dz2 = self.actions[ai]
                    tx, ty, tz = x + dx2, y + dy2, z + dz2
                    if (tx, ty, tz) not in visited and 0 <= tx < self.width and 0 <= ty < self.height and 0 <= tz < self.depth:
                        nx, ny, nz = tx, ty, tz
                        moved = True
                        break
                if not moved:
                    break
            x, y, z = nx, ny, nz
            path.append((x, y, z))
            visited.add((x, y, z))
        return path

    def plan_path(self, hazards: List[Dict], grid_height_mid: int = 15, grid_depth_mid: int = 10) -> Dict:
        # Map Earth at (0, mid, mid), Moon at (width-1, mid, mid)
        start = (0, grid_height_mid, grid_depth_mid)
        goal = (self.width - 1, grid_height_mid, grid_depth_mid)

        # Project hazards along x-axis fraction using timestamp ordering
        obstacles: Set[Tuple[int,int,int]] = set()
        sorted_h = sorted(hazards, key=lambda h: h.get("timestamp", ""))
        for i, h in enumerate(sorted_h[: min(50, len(sorted_h))]):
            frac = (i + 1) / (len(sorted_h) + 1)
            cx = max(1, min(self.width - 2, int(frac * (self.width - 1))))
            cy = grid_height_mid
            cz = grid_depth_mid
            spread = 2 if float(h.get("severity", 0.5)) < 0.6 else 4
            for dx in range(-spread, spread + 1):
                for dy in range(-spread, spread + 1):
                    for dz in range(-1, 2):  # Limited Z spread for realistic 3D space
                        tx, ty, tz = cx + dx, cy + dy, cz + dz
                        if 0 <= tx < self.width and 0 <= ty < self.height and 0 <= tz < self.depth:
                            obstacles.add((tx, ty, tz))

        path = self.train(start, goal, obstacles)

        # Convert grid to world coordinates (arbitrary units)
        scale = 10.0
        world_path = [(float(x) * scale, float(y) * scale, float(z) * scale) for (x, y, z) in path]
        earth = {"name": "Earth", "pos": (0.0, float(grid_height_mid) * scale, float(grid_depth_mid) * scale)}
        moon = {"name": "Moon", "pos": (float(self.width - 1) * scale, float(grid_height_mid) * scale, float(grid_depth_mid) * scale)}
        obstacle_points = [(float(x) * scale, float(y) * scale, float(z) * scale) for (x, y, z) in obstacles]

        return {
            "earth": earth,
            "moon": moon,
            "obstacles": obstacle_points,
            "path": world_path,
        }


class AStarPathfinder:
    """Grid-based A* pathfinder for baseline safe routing."""
    def __init__(self, width: int = 60, height: int = 30, depth: int = 20):
        self.width = width
        self.height = height
        self.depth = depth
        self.moves = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

    def _heuristic(self, a: Tuple[int,int,int], b: Tuple[int,int,int]) -> float:
        return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])

    def plan(self, start: Tuple[int,int,int], goal: Tuple[int,int,int], blocked: Set[Tuple[int,int,int]]) -> List[Tuple[int,int,int]]:
        import heapq
        open_heap: List[Tuple[float, Tuple[int,int,int]]] = []
        heapq.heappush(open_heap, (0.0, start))
        came_from: Dict[Tuple[int,int,int], Optional[Tuple[int,int,int]]] = {start: None}
        g: Dict[Tuple[int,int,int], float] = {start: 0.0}

        while open_heap:
            _f, current = heapq.heappop(open_heap)
            if current == goal:
                break
            cx, cy, cz = current
            for dx, dy, dz in self.moves:
                nx, ny, nz = cx + dx, cy + dy, cz + dz
                if not (0 <= nx < self.width and 0 <= ny < self.height and 0 <= nz < self.depth):
                    continue
                if (nx, ny, nz) in blocked:
                    continue
                tentative = g[current] + 1.0
                if tentative < g.get((nx, ny, nz), float('inf')):
                    g[(nx, ny, nz)] = tentative
                    f = tentative + self._heuristic((nx, ny, nz), goal)
                    came_from[(nx, ny, nz)] = current
                    heapq.heappush(open_heap, (f, (nx, ny, nz)))

        # Reconstruct
        if goal not in came_from:
            return [start]
        path: List[Tuple[int,int,int]] = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = came_from.get(cur)
        path.reverse()
        return path


class RLAdjuster:
    """Lightweight RL adjuster that replans from current node when a new hazard appears."""
    def __init__(self, width: int = 60, height: int = 30, depth: int = 20):
        self.width = width
        self.height = height
        self.depth = depth
        self.q = QLearningPathPlanner(width, height, depth)
        self.astar = AStarPathfinder(width, height, depth)

    def adjust_path(self,
                    current: Tuple[int,int,int],
                    goal: Tuple[int,int,int],
                    blocked: Set[Tuple[int,int,int]]) -> List[Tuple[int,int,int]]:
        # Try quick A* first for efficiency
        path = self.astar.plan(current, goal, blocked)
        if len(path) > 1:
            return path
        # fallback to a brief Q-learning session for robustness
        return self.q.train(current, goal, blocked, episodes=200, max_steps=200)
