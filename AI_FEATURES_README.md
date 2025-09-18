# ODIN AI Strategy Co-Pilot Features

## Overview
ODIN now includes a comprehensive AI strategy co-pilot powered by Google Gemini LLM that provides intelligent threat analysis, trajectory evaluation, and human-readable decision logs.

## Key Features

### 1. Threat Analysis (`analyze_threats`)
- **Real-time threat assessment** using space weather data from NASA DONKI APIs
- **Categorizes threats** by type (CME, Solar Flares, Debris Conjunctions)
- **Severity scoring** based on threat characteristics
- **Crew safety risk evaluation** with specific recommendations
- **Launch delay recommendations** when threats are critical

### 2. Trajectory Evaluation (`evaluate_trajectory_options`)
- **Compares baseline A* path** vs **RL-adjusted trajectory**
- **Trade-off analysis** including:
  - Fuel efficiency (ΔV cost)
  - Time to destination
  - Safety benefits
  - Risk reduction percentages
- **Confidence scoring** for recommendations
- **Mission success probability** estimation

### 3. Decision Logging (`generate_decision_log`)
- **Human-readable logs** explaining every decision
- **Timestamped entries** with clear reasoning
- **Trade-off explanations** (e.g., "+6 hours travel time, -90% radiation exposure")
- **Mission status updates** throughout the flight
- **Justification for trajectory choices**

### 4. AI Recommendations (`generate_ai_recommendations`)
- **Immediate actions** based on threat level
- **Mission modifications** (delays, route changes)
- **Safety measures** for crew protection
- **Monitoring requirements** for ongoing threats
- **Contingency plans** for emergency situations

## Integration

### Backend Integration
The AI analysis is automatically triggered in the `/simulate` endpoint:

```python
# AI Analysis using Gemini LLM
ai_analysis = analyze_mission_with_ai(
    hazards=hazards,
    timeframe={"start": start.isoformat() + "Z", "end": end.isoformat() + "Z"},
    baseline_path=baseline_path,
    adjusted_path=adjusted_path,
    constraints=constraints
)
```

### Frontend Display
The frontend now shows:
- **AI Recommendations panel** with threat level, trajectory choice, and confidence
- **AI Decision Logs** replacing basic system logs
- **Real-time threat analysis** in the right sidebar
- **Immediate actions** when threats are detected

## Environment Setup

### Required Environment Variables
Add to your `.env` file:

```bash
# Google Gemini API Key (required for AI features)
GEMINI_API_KEY=your_gemini_api_key_here

# Space-Track API Credentials (for debris data)
SPACETRACK_USER=your_username_here
SPACETRACK_PASS=your_password_here

# API Logging (optional)
ODIN_API_LOG=1
```

### Getting a Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file

## Example AI Decision Logs

```
[2025-01-18 12:30:15 UTC] 🚀 ODIN MISSION INITIATED
[2025-01-18 12:30:15 UTC] Mission Status: IN_PROGRESS
[2025-01-18 12:30:16 UTC] ⚠️  THREAT ASSESSMENT: HIGH level detected
[2025-01-18 12:30:16 UTC] Primary concerns: CME, X-class Solar Flare
[2025-01-18 12:30:16 UTC] 👥 Crew safety risk: HIGH
[2025-01-18 12:30:16 UTC] ⏰ LAUNCH DELAY REQUIRED: 8 hours due to HIGH threats
[2025-01-18 12:30:16 UTC] Rationale: Crew safety takes priority over mission timeline
[2025-01-18 12:30:17 UTC] 🛰️  TRAJECTORY SELECTED: ADJUSTED (Confidence: HIGH)
[2025-01-18 12:30:17 UTC] 📊 TRADE-OFF ANALYSIS:
[2025-01-18 12:30:17 UTC]   • Fuel cost: +150.0 m/s ΔV for safety
[2025-01-18 12:30:17 UTC]   • Time cost: +2.5 hours for safety
[2025-01-18 12:30:17 UTC] 🛡️  Safety benefit: Avoids 12 high-severity hazards
[2025-01-18 12:30:17 UTC] Risk reduction: 85% hazard exposure reduction
[2025-01-18 12:30:17 UTC] 💭 DECISION JUSTIFICATION: Adjusted trajectory provides optimal balance of safety and efficiency given detected CME and solar flare activity
[2025-01-18 12:30:17 UTC] 🎯 Mission success probability: 92%
[2025-01-18 12:30:17 UTC] ✅ FINAL RECOMMENDATION: Proceed with ADJUSTED trajectory
[2025-01-18 12:30:17 UTC] ODIN decision-making complete. Mission parameters optimized for safety and efficiency.
```

## AI Strategy Co-Pilot Capabilities

### Threat Detection & Response
- **CME Analysis**: Evaluates Coronal Mass Ejections for radiation risk
- **Solar Flare Assessment**: Analyzes flare intensity and timing
- **Debris Conjunction**: Identifies orbital debris threats
- **Geomagnetic Storm Impact**: Considers Kp index effects on navigation

### Decision Making Process
1. **Threat Assessment**: Analyze all detected hazards
2. **Trajectory Comparison**: Evaluate baseline vs adjusted paths
3. **Trade-off Analysis**: Quantify fuel, time, and safety impacts
4. **Recommendation Generation**: Select optimal trajectory
5. **Justification Logging**: Explain reasoning clearly

### Resilience Features
- **Adaptive Planning**: Adjusts to real-time threat changes
- **Safety Prioritization**: Crew safety over mission timeline
- **Transparent Decision Making**: Clear explanations for all choices
- **Contingency Planning**: Prepared for emergency scenarios

## Error Handling
The AI system includes robust error handling:
- **Fallback recommendations** if AI analysis fails
- **Graceful degradation** to basic trajectory planning
- **Clear error logging** for debugging
- **Manual review flags** when AI confidence is low

## Performance Considerations
- **Async processing** for AI analysis
- **Caching** of threat analysis results
- **Timeout handling** for API calls
- **Efficient logging** with configurable verbosity

This AI integration transforms ODIN from a basic trajectory planner into an intelligent, adaptive mission planning system that can handle complex space weather scenarios with human-readable explanations.
