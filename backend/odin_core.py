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
    Grid coordinates are (x,y) with x in [0,width-1], y in [0,height-1].
    """

    def __init__(self, width: int = 60, height: int = 30):
        self.width = width
        self.height = height
        self.actions = [(1,0), (-1,0), (0,1), (0,-1)]  # E, W, N, S
        self.num_actions = len(self.actions)

    def _state_index(self, x: int, y: int) -> int:
        return y * self.width + x

    def _neighbors(self, x: int, y: int) -> List[Tuple[int,int,int]]:
        out = []
        for ai,(dx,dy) in enumerate(self.actions):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                out.append((nx, ny, ai))
        return out

    def train(self,
              start: Tuple[int,int],
              goal: Tuple[int,int],
              obstacles: Set[Tuple[int,int]],
              episodes: int = 500,
              alpha: float = 0.4,
              gamma: float = 0.95,
              epsilon: float = 0.2,
              max_steps: int = 200) -> List[Tuple[int,int]]:
        num_states = self.width * self.height
        Q = np.zeros((num_states, self.num_actions), dtype=np.float32)

        def step(x: int, y: int, action_index: int) -> Tuple[int,int,float,bool]:
            dx, dy = self.actions[action_index]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                return x, y, -2.0, False  # wall penalty
            if (nx, ny) in obstacles:
                return nx, ny, -3.0, False  # obstacle penalty
            if (nx, ny) == goal:
                return nx, ny, +10.0, True  # goal reward
            # small progress reward toward goal on x-axis
            reward = 0.1 if nx > x else -0.05
            return nx, ny, reward, False

        rng = np.random.default_rng()
        for _ in range(episodes):
            x, y = start
            for _ in range(max_steps):
                s = self._state_index(x, y)
                if rng.random() < epsilon:
                    a = rng.integers(0, self.num_actions)
                else:
                    a = int(np.argmax(Q[s]))
                nx, ny, r, done = step(x, y, a)
                ns = self._state_index(nx, ny)
                Q[s, a] = (1 - alpha) * Q[s, a] + alpha * (r + gamma * np.max(Q[ns]))
                x, y = nx, ny
                if done:
                    break

        # Derive greedy path
        path = [start]
        visited = set([start])
        x, y = start
        for _ in range(self.width * 3):
            if (x, y) == goal:
                break
            s = self._state_index(x, y)
            a = int(np.argmax(Q[s]))
            dx, dy = self.actions[a]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                break
            if (nx, ny) in visited:
                # fallback: try any neighbor not visited
                moved = False
                for _,_,ai in self._neighbors(x, y):
                    dx2, dy2 = self.actions[ai]
                    tx, ty = x + dx2, y + dy2
                    if (tx, ty) not in visited and 0 <= tx < self.width and 0 <= ty < self.height:
                        nx, ny = tx, ty
                        moved = True
                        break
                if not moved:
                    break
            x, y = nx, ny
            path.append((x, y))
            visited.add((x, y))
        return path

    def plan_path(self, hazards: List[Dict], grid_height_mid: int = 15) -> Dict:
        # Map Earth at (0, mid), Moon at (width-1, mid)
        start = (0, grid_height_mid)
        goal = (self.width - 1, grid_height_mid)

        # Project hazards along x-axis fraction using timestamp ordering
        obstacles: Set[Tuple[int,int]] = set()
        sorted_h = sorted(hazards, key=lambda h: h.get("timestamp", ""))
        for i, h in enumerate(sorted_h[: min(50, len(sorted_h))]):
            frac = (i + 1) / (len(sorted_h) + 1)
            cx = max(1, min(self.width - 2, int(frac * (self.width - 1))))
            cy = grid_height_mid
            spread = 2 if float(h.get("severity", 0.5)) < 0.6 else 4
            for dx in range(-spread, spread + 1):
                for dy in range(-spread, spread + 1):
                    tx, ty = cx + dx, cy + dy
                    if 0 <= tx < self.width and 0 <= ty < self.height:
                        obstacles.add((tx, ty))

        path = self.train(start, goal, obstacles)

        # Convert grid to world coordinates (arbitrary units)
        scale = 10.0
        world_path = [(float(x) * scale, float(y) * scale) for (x, y) in path]
        earth = {"name": "Earth", "pos": (0.0, float(grid_height_mid) * scale)}
        moon = {"name": "Moon", "pos": (float(self.width - 1) * scale, float(grid_height_mid) * scale)}
        obstacle_points = [(float(x) * scale, float(y) * scale) for (x, y) in obstacles]

        return {
            "earth": earth,
            "moon": moon,
            "obstacles": obstacle_points,
            "path": world_path,
        }


class AStarPathfinder:
    """Grid-based A* pathfinder for baseline safe routing."""
    def __init__(self, width: int = 60, height: int = 30):
        self.width = width
        self.height = height
        self.moves = [(1,0), (-1,0), (0,1), (0,-1)]

    def _heuristic(self, a: Tuple[int,int], b: Tuple[int,int]) -> float:
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def plan(self, start: Tuple[int,int], goal: Tuple[int,int], blocked: Set[Tuple[int,int]]) -> List[Tuple[int,int]]:
        import heapq
        open_heap: List[Tuple[float, Tuple[int,int]]] = []
        heapq.heappush(open_heap, (0.0, start))
        came_from: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {start: None}
        g: Dict[Tuple[int,int], float] = {start: 0.0}

        while open_heap:
            _f, current = heapq.heappop(open_heap)
            if current == goal:
                break
            cx, cy = current
            for dx, dy in self.moves:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if (nx, ny) in blocked:
                    continue
                tentative = g[current] + 1.0
                if tentative < g.get((nx, ny), float('inf')):
                    g[(nx, ny)] = tentative
                    f = tentative + self._heuristic((nx, ny), goal)
                    came_from[(nx, ny)] = current
                    heapq.heappush(open_heap, (f, (nx, ny)))

        # Reconstruct
        if goal not in came_from:
            return [start]
        path: List[Tuple[int,int]] = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = came_from.get(cur)
        path.reverse()
        return path


class RLAdjuster:
    """Lightweight RL adjuster that replans from current node when a new hazard appears."""
    def __init__(self, width: int = 60, height: int = 30):
        self.width = width
        self.height = height
        self.q = QLearningPathPlanner(width, height)
        self.astar = AStarPathfinder(width, height)

    def adjust_path(self,
                    current: Tuple[int,int],
                    goal: Tuple[int,int],
                    blocked: Set[Tuple[int,int]]) -> List[Tuple[int,int]]:
        # Try quick A* first for efficiency
        path = self.astar.plan(current, goal, blocked)
        if len(path) > 1:
            return path
        # fallback to a brief Q-learning session for robustness
        return self.q.train(current, goal, blocked, episodes=200, max_steps=200)
