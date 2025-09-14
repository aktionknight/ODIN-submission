# odin_core.py
# Backend AI logic for ODIN project

import numpy as np

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
