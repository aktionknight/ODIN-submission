from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from odin_core import Odin, QLearningPathPlanner, AStarPathfinder, RLAdjuster
import uvicorn
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import datetime as dt
import api2
import os, sys
# Ensure project root is importable when running from backend/
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from odin_prototype import TrajectoryEngine, AIStrategyCopilot, DecisionEngine

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

odin = Odin()

@app.get("/")
def root():
    return {"message": "ODIN backend running 🚀"}


if __name__ == "__main__":
    port = 8000
    print(f"🚀 Server running at: http://127.0.0.1:{port}")
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)

@app.get("/trajectories")
def get_trajectories():
    t = np.linspace(0, 10, 100)

    # Convert trajectories to regular lists
    trajectories = [np.column_stack([t, np.sin(t) + i]).tolist() for i in range(3)]

    chosen_idx, scores = odin.predict([np.array(traj) for traj in trajectories])

    # Ensure JSON serializable
    return {
        "trajectories": trajectories,
        "chosen_index": int(chosen_idx),                # convert numpy.int64 → int
        "scores": [float(s) for s in scores],           # convert numpy.float → float
        "log": [str(entry) for entry in odin.log]       # ensure log is serializable
    }


class SimulateRequest(BaseModel):
    startTime: Optional[str] = None  # ISO string
    endTime: Optional[str] = None    # ISO string
    windowHours: Optional[int] = 72


@app.post("/simulate")
def simulate(req: SimulateRequest):
    # Determine timeframe
    now = dt.datetime.utcnow()
    if req.startTime and req.endTime:
        try:
            start = dt.datetime.fromisoformat(req.startTime.replace("Z", "+00:00")).replace(tzinfo=None)
            end = dt.datetime.fromisoformat(req.endTime.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            start = now - dt.timedelta(hours=req.windowHours or 72)
            end = now
    else:
        # Random window between 2012-01-01 and 2018-12-31
        start_bound = dt.datetime(2012,1,1)
        end_bound = dt.datetime(2018,12,31,23,59,59)
        total_seconds = int((end_bound - start_bound).total_seconds())
        offset = int(np.random.randint(0, max(1, total_seconds - (req.windowHours or 72) * 3600)))
        start = start_bound + dt.timedelta(seconds=offset)
        end = start + dt.timedelta(hours=req.windowHours or 72)

    # Pull space weather data via api2 and map to simplified hazards
    # Use CMEAnalysis API within selected timeframe
    cme_analysis = api2.fetch_donki_cme_analysis(start, end)
    flares = api2.fetch_donki_flare(start, end)
    gsts = api2.fetch_donki_gst(start, end)
    tle_debris = api2.fetch_spacetrack_tle(start, end, limit=200)

    hazards: List[Dict[str, Any]] = []

    def parse_time(s: Optional[str]) -> Optional[dt.datetime]:
        if not s:
            return None
        try:
            return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            return None

    for c in cme_analysis[:120]:
        ts = parse_time(c.get("time21_5") or c.get("associatedCMEstartTime"))
        if ts and start <= ts <= end:
            speed = c.get("speed") or 0
            sev = float(speed) / 2000.0 if speed else 0.4
            sev = max(0.0, min(1.0, sev))
            hazards.append({
                "timestamp": ts.isoformat() + "Z",
                "hazard_type": "CME",
                "severity": sev,
                "description": c.get("type") or "CME"
            })

    for f in flares[:200]:
        ts = parse_time(f.get("peakTime") or f.get("beginTime"))
        if ts and start <= ts <= end:
            class_type = f.get("classType") or "M1.0"
            # crude severity mapping
            sev = 0.3
            if class_type.startswith("X"): sev = 0.9
            elif class_type.startswith("M"): sev = 0.6
            elif class_type.startswith("C"): sev = 0.3
            hazards.append({
                "timestamp": ts.isoformat() + "Z",
                "hazard_type": "SolarFlare",
                "severity": sev,
                "description": class_type
            })

    # Debris as hazards (time-based using EPOCH)
    spacetrack_called = False
    if isinstance(tle_debris, list) and len(tle_debris) > 0:
        spacetrack_called = True
        for d in tle_debris[:300]:
            ts = parse_time(d.get("EPOCH"))
            if ts and start <= ts <= end:
                # crude severity from perigee/apogee or object type
                try:
                    perigee = float(d.get("PERIGEE") or 0)
                except Exception:
                    perigee = 0.0
                sev = 0.4 + min(0.6, perigee / 2000.0)
                hazards.append({
                    "timestamp": ts.isoformat() + "Z",
                    "hazard_type": "DebrisConjunction",
                    "severity": max(0.1, min(1.0, sev)),
                    "description": d.get("OBJECT_NAME") or "Debris"
                })

    # GST: track max KpIndex within timeframe for Δv multiplier near Earth
    kp_values = []
    for g in gsts[:200]:
        arr = g.get("allKpIndex") or []
        for item in arr:
            kp = item.get("kpIndex") if isinstance(item, dict) else None
            if kp is not None:
                try:
                    kp_values.append(float(kp))
                except Exception:
                    pass
    kp_max = max(kp_values) if kp_values else 0.0

    # Build trajectories using prototype engines
    engine = TrajectoryEngine()
    baseline = engine.baseline_plan()

    # Convert hazards to minimal objects for AI copilot heuristic
    # We only need type and severity
    simplified_haz = []
    for h in hazards:
        simplified_haz.append(type("HZ", (), {
            "hazard_type": h["hazard_type"],
            "severity": h["severity"]
        }))

    ai = AIStrategyCopilot()
    proposals = [p for p,_ in ai.propose_alternatives(baseline, simplified_haz)]

    dec = DecisionEngine()
    best_plan, report = dec.select_best(baseline, proposals)

    all_plans = [baseline] + proposals
    best_index = next(i for i,p in enumerate(all_plans) if p.name == best_plan.name)

    def plan_to_dict(p):
        return {
            "name": p.name,
            "delta_v": p.delta_v,
            "time_of_flight": p.time_of_flight,
            "estimated_radiation": p.estimated_radiation,
            "nodes": p.nodes,
        }

    # Simple animation timeline: one step per node
    animation = {
        "indices": list(range(len(best_plan.nodes)))
    }

    # Construct hazard shapes (DONKI CME as circular regions)
    # map hazards along midline; more severe → larger radius
    grid_w, grid_h = 60, 30
    mid = grid_h // 2
    def hazard_obstacle_set(hzs):
        blocked = set()
        duration = (end - start).total_seconds() or 1.0
        for h in hzs[:200]:
            ts_s = h.get("timestamp")
            ts = None
            try:
                if isinstance(ts_s, str):
                    ts = dt.datetime.fromisoformat(ts_s.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                ts = None
            if not ts:
                continue
            # place along x by timestamp fraction within the chosen timeframe
            frac = (ts - start).total_seconds() / duration
            if not np.isfinite(frac):
                continue
            frac = max(0.0, min(1.0, float(frac)))
            cx = max(1, min(grid_w - 2, int(frac * (grid_w - 1))))
            # deterministically jitter y based on timestamp to avoid all hazards on exact midline
            seed = int(ts.timestamp()) % 100000
            rng = np.random.default_rng(seed)
            jitter = int((rng.random() - 0.5) * 8)  # -4..+4
            cy = max(1, min(grid_h - 2, mid + jitter))
            r = 2 if float(h.get("severity", 0.5)) < 0.6 else 4
            for x in range(cx - r - 1, cx + r + 2):
                for y in range(cy - r - 1, cy + r + 2):
                    if 0 <= x < grid_w and 0 <= y < grid_h:
                        if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                            blocked.add((x, y))
        return blocked

    hazards_solar = [h for h in hazards if h.get("hazard_type") != "DebrisConjunction"]
    hazards_debris = [h for h in hazards if h.get("hazard_type") == "DebrisConjunction"]

    blocked_haz = hazard_obstacle_set(hazards_solar)
    blocked_debris = hazard_obstacle_set(hazards_debris)
    blocked = blocked_haz | blocked_debris
    start_xy, goal_xy = (0, mid), (grid_w - 1, mid)

    # Baseline A* path
    astar = AStarPathfinder(width=grid_w, height=grid_h)
    baseline_path = astar.plan(start_xy, goal_xy, blocked)

    # RL adjuster to replan from mid-point to goal (no artificial mid-route hazard)
    adjuster = RLAdjuster(width=grid_w, height=grid_h)
    if len(baseline_path) > 3:
        current = baseline_path[len(baseline_path)//2]
    else:
        current = start_xy
    adjusted_tail = adjuster.adjust_path(current, goal_xy, blocked)
    adjusted_path = baseline_path[:len(baseline_path)//2] + adjusted_tail

    # Convert grid coords to world
    scale = 10.0
    def worldify(seq):
        return [(float(x) * scale, float(y) * scale) for (x, y) in seq]
    # Apply mission constraints: CME/FLR can induce time and Δv penalties
    # If any high-severity CME overlaps start epoch, suggest launch delay hours
    high_cme_present = any(h["hazard_type"] == "CME" and h["severity"] >= 0.7 for h in hazards)
    suggested_delay_hours = 8 if high_cme_present else 0
    # Flares: if present in window, add pause penalty
    flare_present = any(h["hazard_type"] == "SolarFlare" for h in hazards)
    flare_pause_hours = 1 if flare_present else 0
    # GST: Kp >= 7 → burn multiplier near Earth
    dv_multiplier_near_earth = 1.2 if kp_max >= 7 else 1.0

    # Build explicit pause events from actual flare timestamps mapped to path indices
    sim_seconds_per_hour = 0.5  # speed up: 1 sim hour shows in 0.5s on screen
    pauses = []
    if len(adjusted_path) >= 2:
        total_len = len(adjusted_path)
        for f in flares[:8]:
            ts = parse_time(f.get("peakTime") or f.get("beginTime"))
            if not ts:
                continue
            if not (start <= ts <= end):
                continue
            frac = (ts - start).total_seconds() / ((end - start).total_seconds() or 1.0)
            frac = max(0.0, min(1.0, float(frac)))
            idx = max(1, min(total_len - 2, int(frac * (total_len - 1))))
            pauses.append({
                "index": int(idx),
                "reason": f.get("classType") or "SolarFlare",
                "durationHours": 1,
                "durationSeconds": int(1 * sim_seconds_per_hour)
            })

    rl = {
        "earth": {"name": "Earth", "pos": (0.0, float(mid) * scale)},
        "moon": {"name": "Moon", "pos": (float(grid_w - 1) * scale, float(mid) * scale)},
        "obstaclesHazards": [(float(x) * scale, float(y) * scale) for (x, y) in blocked_haz],
        "obstaclesDebris": [(float(x) * scale, float(y) * scale) for (x, y) in blocked_debris],
        "path": worldify(adjusted_path),
        "baseline": worldify(baseline_path),
        "constraints": {
            "suggestedDelayHours": suggested_delay_hours,
            "flarePauseHours": flare_pause_hours,
            "kpMax": kp_max,
            "dvMultiplierNearEarth": dv_multiplier_near_earth
        },
        "pauses": pauses,
        "timeScale": {"simSecondsPerHour": sim_seconds_per_hour},
        "debug": {"spacetrackCalled": spacetrack_called, "tleCount": len(tle_debris) if isinstance(tle_debris, list) else 0}
    }

    return {
        "timeframe": {"start": start.isoformat() + "Z", "end": end.isoformat() + "Z"},
        "hazards": hazards,
        "plans": [plan_to_dict(p) for p in all_plans],
        "bestIndex": int(best_index),
        "report": report,
        "animation": animation,
        "rl": rl
    }
