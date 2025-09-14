import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys
import os

# Import backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from odin_core import Odin

# Title
st.title("🛰️ ODIN - Optimal Dynamic Interplanetary Navigator")

# Generate sample trajectories
t = np.linspace(0, 10, 100)
trajectories = [
    np.column_stack([t, np.sin(t) + i]) for i in range(3)
]

# Run backend AI
odin = Odin()
chosen_idx, scores = odin.predict(trajectories)

# Plot interactive graph
fig = go.Figure()

for idx, traj in enumerate(trajectories):
    fig.add_trace(go.Scatter(
        x=traj[:, 0], y=traj[:, 1],
        mode="lines",
        name=f"Trajectory {idx+1} (Score={scores[idx]:.2f})",
        line=dict(
            color="green" if idx == chosen_idx else "blue",
            width=4 if idx == chosen_idx else 2,
            dash="solid" if idx == chosen_idx else "dot"
        )
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
    template="plotly_dark"
)

# Display
st.plotly_chart(fig, use_container_width=True)

# Show decision log
st.subheader("📜 ODIN Decision Log")
st.json(odin.log)
