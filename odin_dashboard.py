import streamlit as st
import numpy as np
import plotly.graph_objects as go
import json

# -------------------------------
# ODIN Prototype (Simplified)
# -------------------------------
class Odin:
    def __init__(self):
        self.log = []

    def predict(self, trajectories):
        """Pick the safest trajectory based on minimum collision risk."""
        scores = [self.evaluate_traj(t) for t in trajectories]
        best_idx = np.argmin(scores)
        self.log.append({
            "trajectories": trajectories.tolist(),
            "scores": scores,
            "chosen_index": best_idx
        })
        return best_idx

    def evaluate_traj(self, traj):
        """Dummy evaluation: risk = distance to obstacle (smaller is riskier)."""
        obstacle = np.array([5, 5])
        dist = np.linalg.norm(traj - obstacle, axis=1).min()
        return 1 / (dist + 0.1)  # Avoid division by zero


# -------------------------------
# Streamlit Dashboard
# -------------------------------
st.title("🛰️ ODIN Interactive Prototype Dashboard")

# Generate trajectories
t = np.linspace(0, 10, 100)
trajectories = [
    np.column_stack([t, np.sin(t) + i]) for i in range(3)
]

# Run ODIN AI
odin = Odin()
chosen_idx = odin.predict(np.array(trajectories, dtype=object))

# Save logs
with open("odin_decision_log.json", "w") as f:
    json.dump(odin.log, f, indent=2)

# Plotly Figure
fig = go.Figure()

# Add trajectories
for idx, traj in enumerate(trajectories):
    fig.add_trace(go.Scatter(
        x=traj[:, 0], y=traj[:, 1],
        mode="lines",
        name=f"Trajectory {idx+1}",
        line=dict(color="blue" if idx != chosen_idx else "green", width=4 if idx == chosen_idx else 2, dash="solid" if idx == chosen_idx else "dot")
    ))

# Add obstacle
fig.add_trace(go.Scatter(
    x=[5], y=[5],
    mode="markers",
    marker=dict(color="red", size=12, symbol="x"),
    name="Obstacle"
))

# Layout
fig.update_layout(
    title="Trajectory Selection (Interactive)",
    xaxis_title="X Position",
    yaxis_title="Y Position",
    template="plotly_dark",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

# Show in Streamlit
st.plotly_chart(fig, use_container_width=True)

# Show decision log
st.subheader("📜 ODIN Decision Log")
st.json(odin.log)
