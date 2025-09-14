import React from "react";
import Plot from "react-plotly.js";

interface TrajectoryChartProps {
  trajectories: number[][][];
  chosenIndex: number;
  scores: number[];
}

const TrajectoryChart: React.FC<TrajectoryChartProps> = ({ trajectories, chosenIndex, scores }) => {
  const plotData = trajectories.map((traj, i) => ({
    x: traj.map((p) => p[0]),
    y: traj.map((p) => p[1]),
    type: "scatter",
    mode: "lines",
    name: `Trajectory ${i + 1} (Score: ${scores[i].toFixed(2)})`,
    line: {
      color: i === chosenIndex ? "lime" : "cyan",
      width: i === chosenIndex ? 4 : 2,
      dash: i === chosenIndex ? "solid" : "dot",
    },
  }));

  // Add obstacle point (same as backend)
  plotData.push({
    x: [5],
    y: [5],
    type: "scatter",
    mode: "markers",
    name: "Obstacle",
    marker: { color: "red", size: 12, symbol: "x" },
  });

  return (
    <div className="bg-gray-900 p-4 rounded-lg shadow-lg">
      <h2 className="text-xl font-semibold mb-3">🚀 Trajectory Visualization</h2>
      <Plot
        data={plotData}
        layout={{
          title: "Optimal Trajectory Selection",
          paper_bgcolor: "black",
          plot_bgcolor: "black",
          font: { color: "white" },
          xaxis: { title: "X Position", gridcolor: "#444" },
          yaxis: { title: "Y Position", gridcolor: "#444" },
          autosize: true,
        }}
        style={{ width: "100%", height: "600px" }}
        config={{ responsive: true }}
      />
    </div>
  );
};

export default TrajectoryChart;
