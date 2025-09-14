"""
ODIN Prototype (Single-file) - odin_prototype.py

Purpose:
A runnable, professional-taste prototype for the hackathon that demonstrates
- ingesting (mocked) historical space-weather + debris data
- creating a baseline Earth->Moon trajectory using simple orbital mechanics (Hohmann-like estimates)
- detecting hazards and generating alternate routes (AI copilot is stubbed but shows interface)
- a decision engine to weigh options by delta-v, time, and estimated crew radiation exposure
- human-readable logs and a matplotlib visualization of trajectories and hazard events

How to run:
- Requires: Python 3.8+, numpy, matplotlib
- Run: python odin_prototype.py
- Output: console logs + saved PNG visualization files in the working directory

This is an intentionally self-contained prototype; API calls and real orbital models are stubbed
so you can run it locally as a demonstration. Replace stubbed parts with real data/APIs and
more accurate orbital libraries (poliastro/astropy) for production.

Author: Generated for user hackathon prototype
"""

import math
import random
import datetime
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# --------------------------- Utilities & Data Models ---------------------------

@dataclass
class HazardEvent:
    timestamp: datetime.datetime
    hazard_type: str  # 'CME', 'SolarFlare', 'DebrisConjunction'
    severity: float   # 0..1
    description: str

@dataclass
class TrajectoryPlan:
    name: str
    delta_v: float  # m/s
    time_of_flight: float  # hours
    estimated_radiation: float  # arbitrary units (0..1)
    nodes: List[Tuple[float, float]]  # simplified x,y positions for plotting
    metadata: Dict

# --------------------------- Data Ingestor (Mock) ---------------------------
class DataIngestor:
    """Mocked historical data loader. In production, replace with NASA/NOAA/CelesTrak API calls.
    The loader picks a random timestamp between Jan 1, 2012 and Dec 31, 2018 and generates
    hazard events around that time to simulate real ingestion.
    """
    def __init__(self):
        self.start = datetime.datetime(2012,1,1)
        self.end = datetime.datetime(2018,12,31,23,59,59)

    def random_timestamp(self) -> datetime.datetime:
        span = (self.end - self.start).total_seconds()
        offset = random.random() * span
        return self.start + datetime.timedelta(seconds=offset)

    def fetch_historical_hazards(self, ts: datetime.datetime, window_hours:int=72) -> List[HazardEvent]:
        # Create some mock hazard events around ts within window_hours
        events = []
        n = random.randint(1,4)
        for i in range(n):
            dt = datetime.timedelta(hours=random.uniform(-window_hours/2, window_hours/2))
            t = ts + dt
            typ = random.choices(['CME','SolarFlare','DebrisConjunction'], weights=[0.4,0.4,0.2])[0]
            severity = round(random.random(),3)
            desc = f"Simulated {typ} severity {severity:.2f}"
            events.append(HazardEvent(timestamp=t,hazard_type=typ,severity=severity,description=desc))
        events.sort(key=lambda e: e.timestamp)
        print(f"[DataIngestor] Fetched {len(events)} hazard events around {ts.isoformat()}")
        return events

# --------------------------- Trajectory Engine ---------------------------
class TrajectoryEngine:
    """Simplified engine that computes baseline and alternate trajectory 'plans' using
    classical approximate formulas. This is a conceptual prototype — for real mission
    design, use poliastro, GMAT, or professional astrodynamics tooling.
    """
    # Constants
    MU_EARTH = 3.986004418e14  # Earth's gravitational parameter, m^3/s^2
    R_EARTH = 6371e3  # m
    R_MOON_ORBIT = 384400e3  # average orbital radius of the Moon from Earth, m

    def __init__(self, start_alt_km: float = 200.0, target_alt_km: float = 100.0):
        self.r0 = self.R_EARTH + start_alt_km*1000.0
        # We model the 'destination orbit' as circular at lunar orbital radius + moon alt (simplified)
        self.r_moon = self.R_MOON_ORBIT + target_alt_km*1000.0

    def hohmann_delta_v(self, r1: float, r2: float) -> Tuple[float,float,float]:
        """Return (dv1,dv2,total_dv) in m/s for a Hohmann-like transfer between circular orbits r1->r2
        Note: Using patched-conic simplification around Earth only. Time-of-flight is half the transfer ellipse period.
        """
        mu = self.MU_EARTH
        v1 = math.sqrt(mu / r1)
        v2 = math.sqrt(mu / r2)
        a_trans = 0.5 * (r1 + r2)
        v_perigee = math.sqrt(mu * (2/r1 - 1/a_trans))
        v_apogee = math.sqrt(mu * (2/r2 - 1/a_trans))
        dv1 = abs(v_perigee - v1)
        dv2 = abs(v2 - v_apogee)
        total = dv1 + dv2
        # time of flight (half the period of ellipse)
        tof = math.pi * math.sqrt(a_trans**3 / mu)  # seconds
        return dv1, dv2, total, tof

    def baseline_plan(self) -> TrajectoryPlan:
        dv1,dv2,total, tof = self.hohmann_delta_v(self.r0, self.r_moon)
        tof_hours = tof / 3600.0
        # naive radiation estimate: proportional to time-of-flight and small baseline
        radiation = min(1.0, 0.2 + 0.001 * tof_hours)
        nodes = self._sample_line(self.r0, self.r_moon, 40)
        plan = TrajectoryPlan(
            name='Baseline Hohmann-like',
            delta_v=total,
            time_of_flight=tof_hours,
            estimated_radiation=radiation,
            nodes=nodes,
            metadata={'dv1':dv1,'dv2':dv2}
        )
        print(f"[TrajectoryEngine] Baseline plan: Δv={total:.1f} m/s, TOF={tof_hours:.1f} hr")
        return plan

    def alternate_higher_inclination(self, extra_delta_v_factor: float = 1.15) -> TrajectoryPlan:
        # Creates an alternate that trades fuel for lower radiation by adding more maneuvering (longer path)
        base = self.baseline_plan()
        total = base.delta_v * extra_delta_v_factor
        tof = base.time_of_flight * 1.2
        radiation = max(0.0, base.estimated_radiation * 0.25)  # assume protective geometry reduces exposure
        nodes = self._sample_curved(self.r0, self.r_moon, 40, curvature=0.4)
        return TrajectoryPlan(
            name='Alternate: Higher-Inclination / Safer',
            delta_v=total,
            time_of_flight=tof,
            estimated_radiation=radiation,
            nodes=nodes,
            metadata={'reason':'avoid-sun-aspect,pass-behind-earth'}
        )

    def alternate_delay_launch(self, delay_hours: float = 6.0) -> TrajectoryPlan:
        # Delay reduces radiation by avoiding a transient event but increases mission time
        base = self.baseline_plan()
        total = base.delta_v * 1.02
        tof = base.time_of_flight + delay_hours
        radiation = max(0.0, base.estimated_radiation * 0.5)
        nodes = self._sample_line(self.r0, self.r_moon, 40)
        return TrajectoryPlan(
            name=f'Alternate: Delay Launch {int(delay_hours)}h',
            delta_v=total,
            time_of_flight=tof,
            estimated_radiation=radiation,
            nodes=nodes,
            metadata={'delay_hours':delay_hours}
        )

    def _sample_line(self, r1: float, r2: float, n:int) -> List[Tuple[float,float]]:
        # 1D radial line projected onto x-axis for simplified plotting
        xs = np.linspace(r1, r2, n)
        return [(float(x), 0.0) for x in xs]

    def _sample_curved(self, r1: float, r2: float, n:int, curvature: float = 0.3) -> List[Tuple[float,float]]:
        xs = np.linspace(0.0, 1.0, n)
        rs = r1 + (r2 - r1) * xs
        ys = np.sin(xs * math.pi) * (r2 - r1) * curvature
        return [(float(rs[i]), float(ys[i])) for i in range(n)]

# --------------------------- AI Strategy Co-Pilot (Stub) ---------------------------
class AIStrategyCopilot:
    """This module simulates a generative AI co-pilot. In production this would call an LLM
    (with mission context) to propose alternate trajectories and natural-language explanations.
    Here, we return precomputed alternate TrajectoryPlan objects and a narrative.
    """
    def __init__(self):
        pass

    def propose_alternatives(self, baseline: TrajectoryPlan, hazards: List[HazardEvent]) -> List[Tuple[TrajectoryPlan,str]]:
        # Simple heuristics to propose alternatives based on hazard types/severity
        engine = TrajectoryEngine()
        proposals = []
        # If a CME or SolarFlare with severity > 0.5, propose higher inclination and delay
        major = any(h.hazard_type in ('CME','SolarFlare') and h.severity > 0.5 for h in hazards)
        if major:
            alt1 = engine.alternate_higher_inclination(extra_delta_v_factor=1.2)
            text1 = f"Detected major solar event; propose higher-inclination trajectory to reduce exposure. Expected Δv {alt1.delta_v:.1f} m/s, +{alt1.time_of_flight - baseline.time_of_flight:.1f} hr." 
            proposals.append((alt1,text1))
            alt2 = engine.alternate_delay_launch(delay_hours=8.0)
            text2 = f"Or delay launch by 8 hours to allow solar conditions to settle. Estimated radiation reduction {baseline.estimated_radiation - alt2.estimated_radiation:.3f}."
            proposals.append((alt2,text2))
        else:
            # Low severity: minor adjustment
            alt = engine.alternate_delay_launch(delay_hours=2.0)
            text = f"Minor event detected; small launch delay (2h) reduces exposure modestly. Δv change small." 
            proposals.append((alt,text))
        print(f"[AIStrategyCopilot] Proposed {len(proposals)} alternatives")
        return proposals

# --------------------------- Decision Engine ---------------------------
class DecisionEngine:
    """Evaluates candidate trajectory plans and selects the best option by scoring.
    Scoring uses weighted multi-objective function: minimize delta-v, time, radiation.
    Weights are mission-configurable.
    """
    def __init__(self, w_dv:float=0.4, w_time:float=0.2, w_rad:float=0.4):
        self.w_dv = w_dv
        self.w_time = w_time
        self.w_rad = w_rad

    def score(self, plan: TrajectoryPlan, baseline: TrajectoryPlan) -> float:
        # Normalize with respect to baseline to keep dimensionless
        dv_norm = plan.delta_v / max(1.0, baseline.delta_v)
        time_norm = plan.time_of_flight / max(1.0, baseline.time_of_flight)
        rad_norm = plan.estimated_radiation / max(1e-6, baseline.estimated_radiation)
        # lower is better; we convert to a score where higher is better by subtracting from 1
        score = 1.0 - (self.w_dv * (dv_norm - 1.0 if dv_norm>1 else dv_norm - 1.0) +
                       self.w_time * (time_norm - 1.0 if time_norm>1 else time_norm - 1.0) +
                       self.w_rad * (rad_norm - 1.0 if rad_norm>1 else rad_norm - 1.0))
        # Simplify: compute penalty for increases vs baseline
        penalty = self.w_dv * (dv_norm - 1.0) + self.w_time * (time_norm - 1.0) + self.w_rad * (rad_norm - 1.0)
        score = max(0.0, 1.0 - penalty)
        return score

    def select_best(self, baseline: TrajectoryPlan, candidates: List[TrajectoryPlan]) -> Tuple[TrajectoryPlan, Dict]:
        all_plans = [baseline] + candidates
        scored = []
        for p in all_plans:
            s = self.score(p, baseline)
            scored.append((p,s))
        scored.sort(key=lambda x: x[1], reverse=True)
        best, bestscore = scored[0]
        report = {
            'ranked': [(p.name,p.delta_v,p.time_of_flight,p.estimated_radiation,round(s,3)) for p,s in scored],
            'chosen': best.name,
            'score': round(bestscore,3)
        }
        print(f"[DecisionEngine] Selected: {best.name} with score {bestscore:.3f}")
        return best, report

# --------------------------- Logger / Explainability ---------------------------
class ExplainableLogger:
    def __init__(self):
        self.entries = []

    def log(self, message: str, extras: Optional[Dict]=None):
        ts = datetime.datetime.utcnow().isoformat()
        entry = {'timestamp':ts, 'message':message, 'extras':extras or {}}
        self.entries.append(entry)
        print(f"[LOG {ts}] {message}")

    def save_json(self, filename:str = 'odin_decision_log.json'):
        with open(filename,'w') as f:
            json.dump(self.entries,f,indent=2,default=str)
        print(f"[ExplainableLogger] Saved logs to {filename}")

# --------------------------- Dashboard / Visualization ---------------------------
class Dashboard:
    def plot_trajectories(self, baseline:TrajectoryPlan, alternatives:List[TrajectoryPlan], hazards:List[HazardEvent], out_png:str='odin_trajectories.png'):
        plt.figure(figsize=(10,6))
        # Plot baseline
        for plan,style in [(baseline,'-k'),] + [(alt,'--r') for alt in alternatives]:
            xs = [node[0]/1e6 for node in plan.nodes]  # convert to Mm for plotting
            ys = [node[1]/1e6 for node in plan.nodes]
            plt.plot(xs, ys, style, label=f"{plan.name} (Δv={plan.delta_v:.0f} m/s, tof={plan.time_of_flight:.1f}h)")
        # Plot hazards as vertical markers at approximate radial positions (simplified)
        for h in hazards:
            # map hazard time into an approximate radial fraction along baseline nodes
            frac = 0.5 + 0.5 * math.tanh((h.timestamp - datetime.datetime.utcnow()).total_seconds() / (3600*24))
            idx = int(frac * (len(baseline.nodes)-1))
            rx = baseline.nodes[idx][0]/1e6
            plt.scatter([rx],[0], s=80, marker='x')
            plt.text(rx,0.02, f"{h.hazard_type} s={h.severity:.2f}", rotation=0)
        plt.xlabel('Radial distance from Earth center (Mm)')
        plt.ylabel('Cross-track distance (Mm)')
        plt.title('ODIN Prototype: Baseline vs Alternate Trajectories (simplified)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        print(f"[Dashboard] Saved visualization to {out_png}")

# --------------------------- Main orchestration ---------------------------

def run_simulation():
    logger = ExplainableLogger()
    ingestor = DataIngestor()
    ts = ingestor.random_timestamp()
    logger.log(f"Initializing ODIN simulation with historical timestamp {ts.isoformat()}")

    hazards = ingestor.fetch_historical_hazards(ts)
    for h in hazards:
        logger.log(f"Ingested hazard: {h.hazard_type} severity={h.severity}", extras=asdict(h))

    engine = TrajectoryEngine()
    baseline = engine.baseline_plan()
    logger.log("Computed baseline trajectory", extras={'delta_v':baseline.delta_v,'tof_hr':baseline.time_of_flight})

    ai = AIStrategyCopilot()
    proposals_with_text = ai.propose_alternatives(baseline, hazards)
    candidates = [p for p,_t in proposals_with_text]
    for p,t in proposals_with_text:
        logger.log(f"AI proposed: {p.name}", extras={'explanation':t, 'delta_v':p.delta_v, 'tof':p.time_of_flight})

    dec = DecisionEngine()
    best, report = dec.select_best(baseline, candidates)
    logger.log(f"Decision outcome: chosen plan {best.name}", extras=report)

    # Produce a human-readable decision log entry
    human_readable = f"Hazards detected: {[h.hazard_type+':'+str(h.severity) for h in hazards]}. Baseline Δv={baseline.delta_v:.1f} m/s. Chosen: {best.name}. Rationale: highest composite score (see report)."
    logger.log(human_readable)

    dashboard = Dashboard()
    dashboard.plot_trajectories(baseline, candidates, hazards)

    logger.save_json()

    print('\nSimulation complete. Files generated: odin_decision_log.json, odin_trajectories.png')

if __name__ == '__main__':
    run_simulation()
