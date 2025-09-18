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
    <div className="bg-gray-900 min-h-screen text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <h1 className="text-3xl font-bold text-center text-red-500">🛰️ ODIN Navigator</h1>
      </header>

      {/* Main Layout */}
      <div className="flex h-screen">
        {/* Left Sidebar - API Logs */}
        <div className="w-80 bg-gray-800 border-r border-gray-700 p-4 overflow-y-auto">
          <h2 className="text-lg font-semibold mb-4 text-blue-400">📡 API Status</h2>
          <div className="space-y-3 text-sm">
            <div className="bg-gray-700 p-3 rounded">
              <div className="text-green-400 font-medium">DONKI APIs</div>
              <div className="text-gray-300">✅ CME Analysis</div>
              <div className="text-gray-300">✅ Solar Flares</div>
              <div className="text-gray-300">✅ Geomagnetic Storms</div>
            </div>
            <div className="bg-gray-700 p-3 rounded">
              <div className="text-orange-400 font-medium">Space-Track</div>
              <div className="text-gray-300">
                {sim?.rl?.debug?.spacetrackCalled ? "✅ Connected" : "❌ No Credentials"}
              </div>
              <div className="text-gray-300">TLE Count: {sim?.rl?.debug?.tleCount || 0}</div>
            </div>
            {sim && (
              <div className="bg-gray-700 p-3 rounded">
                <div className="text-purple-400 font-medium">Simulation Data</div>
                <div className="text-gray-300">Hazards: {sim.hazards.length}</div>
                <div className="text-gray-300">Debris: {sim.rl?.obstaclesDebris?.length || 0}</div>
                <div className="text-gray-300">Timeframe: {sim.timeframe?.start?.split('T')[0]} → {sim.timeframe?.end?.split('T')[0]}</div>
              </div>
            )}
          </div>
        </div>

        {/* Center - Simulation */}
        <div className="flex-1 flex flex-col">
          <div className="p-6">
            <Plot
              data={(function() {
                const base = [];
                if (sim?.trajectory_options) {
                  // Add multiple trajectory options
                  sim.trajectory_options.forEach((traj, i) => {
                    base.push({
                      x: traj.path.map(p => p[0]),
                      y: traj.path.map(p => p[1]),
                      type: "scatter",
                      mode: "lines",
                      name: traj.name,
                      line: {
                        color: traj.color,
                        width: traj.width,
                        dash: traj.style
                      },
                      opacity: traj.is_best ? 1.0 : 0.7
                    });
                  });
                  
                  // Add spacecraft animation on best trajectory
                  const bestTraj = sim.trajectory_options.find(t => t.is_best);
                  if (bestTraj && bestTraj.path.length > 0) {
                    const idx = Math.min(animIndex, Math.max(0, bestTraj.path.length - 1));
                    base.push({
                      x: [bestTraj.path[idx][0]],
                      y: [bestTraj.path[idx][1]],
                      type: "scatter",
                      mode: "markers",
                      name: "Spacecraft",
                      marker: { color: "yellow", size: 16, symbol: "triangle-up" }
                    });
                  }
                } else if (sim?.rl) {
                  // Fallback to old single trajectory display
                  const rlPath = sim.rl.path || [];
                  const baseline = sim.rl.baseline || [];
                  
                  if (baseline.length) {
                    base.push({ x: baseline.map(p => p[0]), y: baseline.map(p => p[1]), type: "scatter", mode: "lines", name: "Baseline (A*)", line: { color: "deepskyblue", width: 2, dash: "dot" } });
                  }
                  if (rlPath.length) {
                    base.push({ x: rlPath.map(p => p[0]), y: rlPath.map(p => p[1]), type: "scatter", mode: "lines", name: "Adjusted (RL)", line: { color: "orange", width: 3 } });
                    const idx = Math.min(animIndex, Math.max(0, rlPath.length - 1));
                    base.push({ x: [rlPath[idx][0]], y: [rlPath[idx][1]], type: "scatter", mode: "markers", name: "Spacecraft", marker: { color: "yellow", size: 14, symbol: "triangle-up" } });
                  }
                } else {
                  // Fallback to original plotData
                  base.push(...plotData);
                }
                
                // Add Earth and Moon
                const earth = sim?.rl?.earth?.pos || [0,0];
                const moon = sim?.rl?.moon?.pos || [0,0];
                base.push({ x: [earth[0]], y: [earth[1]], type: "scatter", mode: "markers+text", name: "Earth", text: ["Earth"], textposition: "top center", marker: { color: "dodgerblue", size: 16, symbol: "circle" }});
                base.push({ x: [moon[0]], y: [moon[1]], type: "scatter", mode: "markers+text", name: "Moon", text: ["Moon"], textposition: "top center", marker: { color: "lightgray", size: 14, symbol: "circle" }});

                // Add hazards
                const hazards = sim?.rl?.obstaclesHazards || [];
                const debris = sim?.rl?.obstaclesDebris || [];
                if (hazards.length) {
                  base.push({ x: hazards.map(p => p[0]), y: hazards.map(p => p[1]), type: "scatter", mode: "markers", name: "Solar Hazards", marker: { color: "red", size: 6, symbol: "x" }});
                }
                if (debris.length) {
                  base.push({ x: debris.map(p => p[0]), y: debris.map(p => p[1]), type: "scatter", mode: "markers", name: "Debris", marker: { color: "orange", size: 5, symbol: "square" }});
                }
                
                return base;
              })()}
        layout={{
                title: "Spacecraft Navigation Simulation",
          paper_bgcolor: "black",
          plot_bgcolor: "black",
          font: { color: "white" },
                xaxis: { title: "Distance (km)", gridcolor: "#444" },
                yaxis: { title: "Altitude (km)", gridcolor: "#444", scaleanchor: "x", scaleratio: 1 },
                legend: { x: 0.02, y: 0.98, bgcolor: "rgba(0,0,0,0.5)" }
              }}
              style={{ width: "100%", height: "500px" }}
            />
          </div>
          
          {/* Controls */}
          <div className="px-6 pb-4">
            <Controls onSimulate={handleSimulate} isLoading={isLoading} />
          </div>
        </div>

        {/* Right Sidebar - AI Copilot */}
        <div className="w-80 bg-gray-800 border-l border-gray-700 p-4 flex flex-col">
          <h2 className="text-lg font-semibold mb-4 text-green-400">🤖 AI Copilot</h2>
          
          {/* Mission Status */}
          {sim && (
            <div className="bg-gray-700 p-4 rounded mb-4">
              <div className="text-yellow-400 font-medium mb-2">Mission Analysis</div>
              <div className="text-sm space-y-1">
                <div>Plan: {sim.plans[sim.bestIndex].name}</div>
                <div>Path Length: {sim.rl?.path?.length || 0} nodes</div>
                <div>Baseline: {sim.rl?.baseline?.length || 0} nodes</div>
              </div>
            </div>
          )}
          
          {/* Constraints */}
          <div className="bg-gray-700 p-4 rounded mb-4">
            <div className="text-blue-400 font-medium mb-2">Constraints</div>
            <div className="text-sm space-y-1">
              <div>Launch Delay: {constraints.suggestedDelayHours || 0}h</div>
              <div>Flare Pause: {constraints.flarePauseHours || 0}h</div>
              <div>Kp Max: {constraints.kpMax || 0}</div>
              <div>Δv Multiplier: {constraints.dvMultiplierNearEarth || 1}×</div>
            </div>
          </div>

          {/* AI Recommendations */}
          <div className="bg-gray-700 p-4 rounded mb-4">
            <div className="text-purple-400 font-medium mb-2">AI Recommendations</div>
            <div className="text-sm text-gray-300">
              {sim?.rl?.ai_analysis ? (
                <div className="space-y-2">
                  <div className="text-yellow-400 font-medium">Threat Level: {sim.rl.ai_analysis.threat_analysis?.threat_level || "Unknown"}</div>
                  <div className="text-green-400 font-medium">Trajectory: {sim.rl.ai_analysis.trajectory_evaluation?.recommended_trajectory || "Unknown"}</div>
                  <div className="text-blue-400 font-medium">Confidence: {sim.rl.ai_analysis.trajectory_evaluation?.confidence_level || "Unknown"}</div>
                  {sim.rl.ai_analysis.ai_recommendations?.immediate_actions?.length > 0 && (
                    <div>
                      <div className="text-red-400 font-medium">Immediate Actions:</div>
                      {sim.rl.ai_analysis.ai_recommendations.immediate_actions.slice(0, 2).map((action, i) => (
                        <div key={i}>• {action}</div>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div>Click "Simulate" to generate AI recommendations</div>
              )}
            </div>
          </div>

          {/* AI Co-Pilot Logs */}
          <div className="bg-gray-900 border border-gray-600 rounded flex-1 flex flex-col min-h-0">
            <div className="flex items-center justify-between p-3 border-b border-gray-600 flex-shrink-0">
              <h3 className="text-sm font-semibold text-blue-400 flex items-center">
                <span className="mr-2">&gt;_</span>
                AI CO-PILOT LOGS
              </h3>
              <div className="flex space-x-2">
                <button className="bg-blue-600 hover:bg-blue-700 text-white text-xs px-2 py-1 rounded">
                  GENERATE AI LOG
                </button>
                <button className="bg-gray-600 hover:bg-gray-700 text-white text-xs px-2 py-1 rounded">
                  CLEAR
                </button>
              </div>
            </div>
            
            <div className="flex-1 overflow-y-auto p-3 space-y-2 min-h-0">
              {sim?.rl?.ai_analysis?.decision_logs ? (
                sim.rl.ai_analysis.decision_logs.slice(-8).map((logEntry, i) => {
                  // Parse log entry to determine type and styling
                  const isWarning = logEntry.includes('⚠️') || logEntry.includes('WARNING');
                  const isError = logEntry.includes('❌') || logEntry.includes('ERROR');
                  const isSuccess = logEntry.includes('✅') || logEntry.includes('SUCCESS');
                  const isInfo = logEntry.includes('🚀') || logEntry.includes('INFO');
                  
                  let barColor = 'bg-gray-500';
                  let iconColor = 'text-gray-400';
                  let icon = 'ⓘ';
                  
                  if (isWarning) {
                    barColor = 'bg-orange-500';
                    iconColor = 'text-orange-400';
                    icon = '!';
                  } else if (isError) {
                    barColor = 'bg-red-500';
                    iconColor = 'text-red-400';
                    icon = '!';
                  } else if (isSuccess) {
                    barColor = 'bg-green-500';
                    iconColor = 'text-green-400';
                    icon = '✓';
                  } else if (isInfo) {
                    barColor = 'bg-blue-500';
                    iconColor = 'text-blue-400';
                    icon = 'ⓘ';
                  }
                  
                  return (
                    <div key={i} className="flex items-start space-x-3 p-2 bg-gray-800 rounded">
                      <div className={`w-1 h-12 ${barColor} rounded-full flex-shrink-0`}></div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            <span className={`${iconColor} text-sm`}>{icon}</span>
                            <span className="text-blue-400 text-sm font-medium">ODIN-AI</span>
                          </div>
                          <span className="text-gray-500 text-xs">T+{i * 15}:{String(i * 30).padStart(2, '0')}</span>
                        </div>
                        <div className="text-gray-300 text-xs mt-1 leading-relaxed">
                          {logEntry.replace(/\[.*?\]/g, '').trim()}
                        </div>
                      </div>
                    </div>
                  );
                })
              ) : (
                <div className="text-gray-500 text-sm text-center py-8">
                  No AI logs available. Run simulation to generate logs.
                </div>
              )}
            </div>
          </div>

          {/* Mission Pause Alert */}
          {pauseMsg && (
            <div className="bg-yellow-900 border border-yellow-600 p-3 rounded mt-4">
              <div className="text-yellow-300 font-medium">⚠️ Mission Pause</div>
              <div className="text-sm text-yellow-200">{pauseMsg}</div>
            </div>
          )}
        </div>
      </div>

      {/* Bottom - Scrollable ODIN Logs */}
      <div className="bg-gray-800 border-t border-gray-700 h-32 overflow-y-auto">
        <div className="p-4">
          <h3 className="text-sm font-semibold text-gray-400 mb-2">📋 System Status & Trajectory Options</h3>
          {sim?.trajectory_options ? (
            <div className="space-y-1 text-xs text-gray-300">
              <div className="text-green-400">✅ Multiple trajectory options generated</div>
              <div className="text-blue-400">📊 Best trajectory: {sim.trajectory_options.find(t => t.is_best)?.name || "Unknown"}</div>
              <div className="text-yellow-400">🛰️ Spacecraft following optimal path</div>
              {sim.rl?.ai_analysis?.decision_logs && (
                <div className="text-purple-400">🤖 AI analysis complete: {sim.rl.ai_analysis.decision_logs.length} decisions logged</div>
              )}
            </div>
          ) : (
      <Logs log={log} />
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
