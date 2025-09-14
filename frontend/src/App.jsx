import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import Controls from "./components/Controls";
import Logs from "./components/Logs";

function App() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch("http://localhost:8000/trajectories")
      .then(res => res.json())
      .then(setData);
  }, []);

  if (!data) return <div className="text-white text-center mt-20">Loading ODIN...</div>;

  const { trajectories, chosen_index, scores, log } = data;

  const plotData = trajectories.map((traj, i) => ({
    x: traj.map(p => p[0]),
    y: traj.map(p => p[1]),
    type: "scatter",
    mode: "lines",
    name: `Trajectory ${i+1} (Score: ${scores[i].toFixed(2)})`,
    line: {
      color: i === chosen_index ? "lime" : "cyan",
      width: i === chosen_index ? 4 : 2,
      dash: i === chosen_index ? "solid" : "dot"
    }
  }));

  return (
    <div className="bg-black min-h-screen text-white p-6">
      <h1 className="text-4xl font-bold text-center mb-6">🛰️ ODIN Navigator</h1>

      <Plot
        data={plotData}
        layout={{
          title: "Optimal Trajectory Selection",
          paper_bgcolor: "black",
          plot_bgcolor: "black",
          font: { color: "white" },
          xaxis: { title: "X Position" },
          yaxis: { title: "Y Position" }
        }}
        style={{ width: "100%", height: "600px" }}
      />

      <Controls />
      <Logs log={log} />
    </div>
  );
}

export default App;
