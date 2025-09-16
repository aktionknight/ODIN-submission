import React, { useEffect, useMemo, useState, useRef } from "react";
import Plot from "react-plotly.js";
import Controls from "./components/Controls";
import Logs from "./components/Logs";

function App() {
  const [data, setData] = useState(null);
  const [sim, setSim] = useState(null);
  const [animIndex, setAnimIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [pauseMsg, setPauseMsg] = useState("");
  const pauseTimerRef = useRef(null);
  const tickTimerRef = useRef(null);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/trajectories")
      .then(res => res.json())
      .then(setData);
  }, []);

  const clearTimers = () => {
    clearInterval(pauseTimerRef.current);
    clearInterval(tickTimerRef.current);
  };

  const handleSimulate = async () => {
    setIsLoading(true);
    clearTimers();
    try {
      const res = await fetch("http://127.0.0.1:8000/simulate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) });
      const json = await res.json();
      setSim(json);
      setAnimIndex(0);
      const delayHours = json?.rl?.constraints?.suggestedDelayHours || 0;
      const secondsPerHour = json?.rl?.timeScale?.simSecondsPerHour ?? 0.5;
      if (delayHours > 0) {
        await runPause(`${delayHours}h launch delay due to CME`, Math.max(1, Math.floor(delayHours * secondsPerHour)));
      }
      startTicker(json);
    } finally {
      setIsLoading(false);
    }
  };

  const runPause = (label, seconds) => {
    return new Promise((resolve) => {
      let remaining = Math.max(0, Math.floor(seconds));
      setPauseMsg(`${label} — ${remaining}s`);
      clearInterval(pauseTimerRef.current);
      pauseTimerRef.current = setInterval(() => {
        remaining -= 1;
        if (remaining <= 0) {
          clearInterval(pauseTimerRef.current);
          setPauseMsg("");
          resolve();
        } else {
          setPauseMsg(`${label} — ${remaining}s`);
        }
      }, 1000);
    });
  };

  const startTicker = (json) => {
    const pauses = (json?.rl?.pauses || []).slice().sort((a,b) => a.index - b.index);
    const secondsPerHour = json?.rl?.timeScale?.simSecondsPerHour ?? 0.5;
    clearInterval(tickTimerRef.current);
    tickTimerRef.current = setInterval(async () => {
      const nextPause = pauses.find(p => p.index === animIndex);
      if (nextPause && !pauseMsg) {
        clearInterval(tickTimerRef.current);
        await runPause(`${nextPause.reason} pause`, Math.max(1, nextPause.durationSeconds ?? Math.floor((nextPause.durationHours || 1) * secondsPerHour)));
        startTicker(json);
        return;
      }
      setAnimIndex((i) => {
        const next = i + 1;
        return next >= (json?.rl?.path?.length || 1) ? (json?.rl?.path?.length || 1) - 1 : next;
      });
    }, 200);
  };

  const bestPlan = useMemo(() => {
    if (!sim) return null;
    return sim.plans[sim.bestIndex];
  }, [sim]);

  useEffect(() => {
    return () => clearTimers();
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

  const constraints = sim?.rl?.constraints || {};

  return (
    <div className="bg-black min-h-screen text-white p-6">
      <h1 className="text-4xl font-bold text-center mb-6">🛰️ ODIN Navigator</h1>

      <Plot
        data={(function() {
          const base = [...plotData];
          if (sim?.rl) {
            const rlPath = sim.rl.path || [];
            const baseline = sim.rl.baseline || [];
            const hazards = sim.rl.obstaclesHazards || [];
            const debris = sim.rl.obstaclesDebris || [];
            const earth = sim.rl.earth?.pos || [0,0];
            const moon = sim.rl.moon?.pos || [0,0];

            if (baseline.length) {
              base.push({ x: baseline.map(p => p[0]), y: baseline.map(p => p[1]), type: "scatter", mode: "lines", name: "Baseline (A*)", line: { color: "deepskyblue", width: 2, dash: "dot" } });
            }
            if (rlPath.length) {
              base.push({ x: rlPath.map(p => p[0]), y: rlPath.map(p => p[1]), type: "scatter", mode: "lines", name: "Adjusted (RL)", line: { color: "orange", width: 3 } });
              const idx = Math.min(animIndex, Math.max(0, rlPath.length - 1));
              base.push({ x: [rlPath[idx][0]], y: [rlPath[idx][1]], type: "scatter", mode: "markers", name: "Spacecraft", marker: { color: "yellow", size: 14, symbol: "triangle-up" } });
            }
            base.push({ x: [earth[0]], y: [earth[1]], type: "scatter", mode: "markers+text", name: "Earth", text: ["Earth"], textposition: "top center", marker: { color: "dodgerblue", size: 16, symbol: "circle" }});
            base.push({ x: [moon[0]], y: [moon[1]], type: "scatter", mode: "markers+text", name: "Moon", text: ["Moon"], textposition: "top center", marker: { color: "lightgray", size: 14, symbol: "circle" }});

            if (hazards.length) {
              base.push({ x: hazards.map(p => p[0]), y: hazards.map(p => p[1]), type: "scatter", mode: "markers", name: "Solar Hazards", marker: { color: "red", size: 6, symbol: "x" }});
            }
            if (debris.length) {
              base.push({ x: debris.map(p => p[0]), y: debris.map(p => p[1]), type: "scatter", mode: "markers", name: "Debris", marker: { color: "orange", size: 5, symbol: "square" }});
            }
          }
          return base;
        })()}
        layout={{
          title: "ODIN: Baseline A* and RL Adjustment",
          paper_bgcolor: "black",
          plot_bgcolor: "black",
          font: { color: "white" },
          xaxis: { title: "X" },
          yaxis: { title: "Y", scaleanchor: "x", scaleratio: 1 }
        }}
        style={{ width: "100%", height: "600px" }}
      />

      <Controls onSimulate={handleSimulate} isLoading={isLoading} />
      {sim && (
        <div className="mt-4 text-sm text-gray-300 space-y-1">
          <div>Timeframe: {sim.timeframe.start} → {sim.timeframe.end}</div>
          <div>Chosen Plan: {sim.plans[sim.bestIndex].name}</div>
          <div>Hazards: {sim.hazards.length} | Debris points: {sim.rl?.obstaclesDebris?.length || 0}</div>
          <div>Baseline length: {sim.rl?.baseline?.length || 0} | Adjusted length: {sim.rl?.path?.length || 0}</div>
          <div>Constraints: delay={constraints.suggestedDelayHours || 0}h, flarePause={constraints.flarePauseHours || 0}h, KpMax={constraints.kpMax || 0}, dv×nearEarth={constraints.dvMultiplierNearEarth || 1}</div>
          {sim?.rl?.debug && <div>Space-Track: called={String(sim.rl.debug.spacetrackCalled)} count={sim.rl.debug.tleCount}</div>}
          {pauseMsg && <div className="mt-2 text-yellow-300">{pauseMsg}</div>}
        </div>
      )}
      <Logs log={log} />
    </div>
  );
}

export default App;
