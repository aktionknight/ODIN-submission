import google.generativeai as genai
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ODINGeminiCopilot:
    """
    ODIN AI Strategy Co-pilot using Google Gemini LLM
    Analyzes space weather threats and generates human-readable decision logs
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini API with API key"""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = None
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("Gemini API initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini API: {e}")
                self.model = None
        else:
            logger.warning("GEMINI_API_KEY not found, using fallback analysis")
        
        # System prompt for ODIN AI behavior
        self.system_prompt = """
You are ODIN (Orbital Decision Intelligence Navigator), an advanced AI system for spacecraft navigation and mission planning. Your role is to:

1. ANALYZE space weather threats and hazards in real-time
2. EVALUATE trajectory options considering mission-critical factors
3. GENERATE clear, human-readable logs explaining decisions and trade-offs
4. DEMONSTRATE resilience and adaptability in dynamic space conditions

Key principles:
- Prioritize crew safety above all else
- Consider fuel efficiency (ΔV cost), time to destination, and radiation exposure
- Provide specific, actionable recommendations
- Explain trade-offs clearly (e.g., "+6 hours travel time, -90% radiation exposure")
- Use technical but accessible language
- Be decisive and confident in recommendations

Always format your responses as structured JSON with clear sections for analysis, recommendations, and justifications.
"""

    def _fallback_threat_analysis(self, hazards: List[Dict]) -> Dict[str, Any]:
        """Fallback threat analysis when Gemini API is not available"""
        high_severity_count = len([h for h in hazards if h.get("severity", 0) >= 0.7])
        total_hazards = len(hazards)
        
        if high_severity_count > 5:
            threat_level = "CRITICAL"
        elif high_severity_count > 2:
            threat_level = "HIGH"
        elif total_hazards > 10:
            threat_level = "MODERATE"
        else:
            threat_level = "LOW"
            
        return {
            "threat_level": threat_level,
            "primary_concerns": [f"{total_hazards} hazards detected", f"{high_severity_count} high-severity"],
            "crew_safety_risk": "HIGH" if threat_level in ["CRITICAL", "HIGH"] else "MODERATE",
            "mission_impact": "SEVERE" if threat_level == "CRITICAL" else "SIGNIFICANT" if threat_level == "HIGH" else "MODERATE",
            "recommended_actions": ["Proceed with enhanced monitoring", "Implement safety protocols"],
            "launch_delay_required": threat_level in ["CRITICAL", "HIGH"],
            "delay_hours": 8 if threat_level == "CRITICAL" else 4 if threat_level == "HIGH" else 0,
            "radiation_exposure_risk": "HIGH" if threat_level in ["CRITICAL", "HIGH"] else "MODERATE"
        }

    def analyze_threats(self, hazards: List[Dict], timeframe: Dict) -> Dict[str, Any]:
        """
        Analyze detected threats and their potential impact on mission
        
        Args:
            hazards: List of hazard events from space weather APIs
            timeframe: Mission timeframe with start/end dates
            
        Returns:
            Threat analysis with severity assessment and recommendations
        """
        # If Gemini is not available, use fallback analysis
        if not self.model:
            return self._fallback_threat_analysis(hazards)
            
        try:
            # Categorize threats by type and severity
            cme_threats = [h for h in hazards if h.get("hazard_type") == "CME"]
            flare_threats = [h for h in hazards if h.get("hazard_type") == "SolarFlare"]
            debris_threats = [h for h in hazards if h.get("hazard_type") == "DebrisConjunction"]
            
            # Count high-severity threats
            high_severity_cme = len([h for h in cme_threats if h.get("severity", 0) >= 0.7])
            x_class_flares = len([h for h in flare_threats if h.get("description", "").startswith("X")])
            
            prompt = f"""
Analyze the following space weather threats detected during mission timeframe {timeframe.get('start')} to {timeframe.get('end')}:

CME THREATS: {len(cme_threats)} total, {high_severity_cme} high-severity (≥0.7)
- Details: {json.dumps(cme_threats[:5], indent=2)}

SOLAR FLARE THREATS: {len(flare_threats)} total, {x_class_flares} X-class
- Details: {json.dumps(flare_threats[:5], indent=2)}

DEBRIS CONJUNCTIONS: {len(debris_threats)} total
- Details: {json.dumps(debris_threats[:5], indent=2)}

Provide a threat analysis in this JSON format:
{{
    "threat_level": "LOW|MODERATE|HIGH|CRITICAL",
    "primary_concerns": ["list of main threats"],
    "crew_safety_risk": "LOW|MODERATE|HIGH|CRITICAL",
    "mission_impact": "MINIMAL|MODERATE|SIGNIFICANT|SEVERE",
    "recommended_actions": ["specific actions to take"],
    "launch_delay_required": true/false,
    "delay_hours": number,
    "radiation_exposure_risk": "LOW|MODERATE|HIGH|CRITICAL"
}}
"""
            
            response = self.model.generate_content(self.system_prompt + prompt)
            analysis = json.loads(response.text)
            
            logger.info(f"Threat analysis completed: {analysis.get('threat_level')} threat level")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in threat analysis: {e}")
            # Provide fallback analysis based on hazard data
            high_severity_count = len([h for h in hazards if h.get("severity", 0) >= 0.7])
            total_hazards = len(hazards)
            
            if high_severity_count > 5:
                threat_level = "CRITICAL"
            elif high_severity_count > 2:
                threat_level = "HIGH"
            elif total_hazards > 10:
                threat_level = "MODERATE"
            else:
                threat_level = "LOW"
                
            return {
                "threat_level": threat_level,
                "primary_concerns": [f"{total_hazards} hazards detected", f"{high_severity_count} high-severity"],
                "crew_safety_risk": "HIGH" if threat_level in ["CRITICAL", "HIGH"] else "MODERATE",
                "mission_impact": "SEVERE" if threat_level == "CRITICAL" else "SIGNIFICANT" if threat_level == "HIGH" else "MODERATE",
                "recommended_actions": ["Proceed with enhanced monitoring", "Implement safety protocols"],
                "launch_delay_required": threat_level in ["CRITICAL", "HIGH"],
                "delay_hours": 8 if threat_level == "CRITICAL" else 4 if threat_level == "HIGH" else 0,
            "radiation_exposure_risk": "HIGH" if threat_level in ["CRITICAL", "HIGH"] else "MODERATE"
        }

    def _fallback_trajectory_evaluation(self, baseline_path: List, adjusted_path: List, hazards: List[Dict]) -> Dict[str, Any]:
        """Fallback trajectory evaluation when Gemini API is not available"""
        baseline_length = len(baseline_path)
        adjusted_length = len(adjusted_path)
        path_difference = adjusted_length - baseline_length
        
        baseline_dv = baseline_length * 100
        adjusted_dv = adjusted_length * 100
        dv_difference = adjusted_dv - baseline_dv
        
        baseline_time = baseline_length * 0.5
        adjusted_time = adjusted_length * 0.5
        time_difference = adjusted_time - baseline_time
        
        is_adjusted_better = len(adjusted_path) > len(baseline_path) and len(hazards) > 5
        confidence = "HIGH" if is_adjusted_better else "MODERATE"
        
        return {
            "recommended_trajectory": "ADJUSTED" if is_adjusted_better else "BASELINE",
            "confidence_level": confidence,
            "trade_off_analysis": {
                "fuel_cost": {
                    "baseline": baseline_dv, 
                    "adjusted": adjusted_dv, 
                    "difference": dv_difference, 
                    "impact": f"Adjusted path uses {dv_difference:+.1f} m/s more fuel for safety"
                },
                "time_cost": {
                    "baseline": baseline_time, 
                    "adjusted": adjusted_time, 
                    "difference": time_difference, 
                    "impact": f"Adjusted path takes {time_difference:+.1f} hours longer"
                },
                "safety_benefit": f"Avoids {len(hazards)} detected hazards",
                "risk_reduction": f"{min(90, len(hazards) * 10)}% hazard exposure reduction"
            },
            "justification": "Adjusted trajectory recommended for hazard avoidance" if is_adjusted_better else "Baseline trajectory sufficient for current threat level",
            "alternative_considerations": ["Monitor for new threats", "Prepare contingency plans"],
            "mission_success_probability": f"{85 + min(10, len(hazards))}%"
        }

    def evaluate_trajectory_options(self, baseline_path: List, adjusted_path: List, 
                                  hazards: List[Dict], constraints: Dict) -> Dict[str, Any]:
        """
        Evaluate trajectory options and provide detailed trade-off analysis
        
        Args:
            baseline_path: Original A* baseline trajectory
            adjusted_path: RL-adjusted trajectory
            hazards: Detected space weather hazards
            constraints: Mission constraints (delay, Δv multiplier, etc.)
            
        Returns:
            Detailed evaluation with trade-offs and recommendations
        """
        # If Gemini is not available, use fallback analysis
        if not self.model:
            return self._fallback_trajectory_evaluation(baseline_path, adjusted_path, hazards)
            
        try:
            # Calculate basic metrics
            baseline_length = len(baseline_path)
            adjusted_length = len(adjusted_path)
            path_difference = adjusted_length - baseline_length
            
            # Estimate fuel costs (simplified)
            baseline_dv = baseline_length * 100  # Simplified ΔV calculation
            adjusted_dv = adjusted_length * 100
            dv_difference = adjusted_dv - baseline_dv
            
            # Estimate time costs
            baseline_time = baseline_length * 0.5  # hours
            adjusted_time = adjusted_length * 0.5
            time_difference = adjusted_time - baseline_time
            
            prompt = f"""
Evaluate these two trajectory options for a spacecraft mission:

BASELINE TRAJECTORY (A* Pathfinding):
- Path length: {baseline_length} nodes
- Estimated ΔV: {baseline_dv:.1f} m/s
- Estimated time: {baseline_time:.1f} hours
- Direct, fuel-efficient route

ADJUSTED TRAJECTORY (RL Hazard Avoidance):
- Path length: {adjusted_length} nodes
- Estimated ΔV: {adjusted_dv:.1f} m/s
- Estimated time: {adjusted_time:.1f} hours
- Avoids {len(hazards)} detected hazards

MISSION CONSTRAINTS:
- Launch delay: {constraints.get('suggestedDelayHours', 0)} hours
- Flare pause: {constraints.get('flarePauseHours', 0)} hours
- Kp index max: {constraints.get('kpMax', 0)}
- ΔV multiplier near Earth: {constraints.get('dvMultiplierNearEarth', 1.0)}x

Provide evaluation in this JSON format:
{{
    "recommended_trajectory": "BASELINE|ADJUSTED",
    "confidence_level": "LOW|MODERATE|HIGH",
    "trade_off_analysis": {{
        "fuel_cost": {{
            "baseline": {baseline_dv},
            "adjusted": {adjusted_dv},
            "difference": {dv_difference},
            "impact": "description of fuel impact"
        }},
        "time_cost": {{
            "baseline": {baseline_time},
            "adjusted": {adjusted_time},
            "difference": {time_difference},
            "impact": "description of time impact"
        }},
        "safety_benefit": "description of safety improvements",
        "risk_reduction": "percentage or description of hazard avoidance"
    }},
    "justification": "detailed explanation of recommendation",
    "alternative_considerations": ["other factors to consider"],
    "mission_success_probability": "percentage estimate"
}}
"""
            
            response = self.model.generate_content(self.system_prompt + prompt)
            evaluation = json.loads(response.text)
            
            logger.info(f"Trajectory evaluation completed: {evaluation.get('recommended_trajectory')} recommended")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error in trajectory evaluation: {e}")
            # Provide fallback analysis based on path characteristics
            is_adjusted_better = len(adjusted_path) > len(baseline_path) and len(hazards) > 5
            confidence = "HIGH" if is_adjusted_better else "MODERATE"
            
            return {
                "recommended_trajectory": "ADJUSTED" if is_adjusted_better else "BASELINE",
                "confidence_level": confidence,
                "trade_off_analysis": {
                    "fuel_cost": {
                        "baseline": baseline_dv, 
                        "adjusted": adjusted_dv, 
                        "difference": dv_difference, 
                        "impact": f"Adjusted path uses {dv_difference:+.1f} m/s more fuel for safety"
                    },
                    "time_cost": {
                        "baseline": baseline_time, 
                        "adjusted": adjusted_time, 
                        "difference": time_difference, 
                        "impact": f"Adjusted path takes {time_difference:+.1f} hours longer"
                    },
                    "safety_benefit": f"Avoids {len(hazards)} detected hazards",
                    "risk_reduction": f"{min(90, len(hazards) * 10)}% hazard exposure reduction"
                },
                "justification": "Adjusted trajectory recommended for hazard avoidance" if is_adjusted_better else "Baseline trajectory sufficient for current threat level",
                "alternative_considerations": ["Monitor for new threats", "Prepare contingency plans"],
                "mission_success_probability": f"{85 + min(10, len(hazards))}%"
            }

    def generate_decision_log(self, threat_analysis: Dict, trajectory_evaluation: Dict, 
                            mission_status: str = "IN_PROGRESS") -> List[str]:
        """
        Generate human-readable decision logs explaining ODIN's reasoning
        
        Args:
            threat_analysis: Results from threat analysis
            trajectory_evaluation: Results from trajectory evaluation
            mission_status: Current mission status
            
        Returns:
            List of formatted log entries
        """
        try:
            logs = []
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            
            # Mission start log
            logs.append(f"[{timestamp}] 🚀 ODIN MISSION INITIATED")
            logs.append(f"[{timestamp}] Mission Status: {mission_status}")
            
            # Threat assessment log
            threat_level = threat_analysis.get("threat_level", "UNKNOWN")
            primary_concerns = threat_analysis.get("primary_concerns", [])
            
            logs.append(f"[{timestamp}] ⚠️  THREAT ASSESSMENT: {threat_level} level detected")
            if primary_concerns:
                logs.append(f"[{timestamp}] Primary concerns: {', '.join(primary_concerns)}")
            
            # Crew safety log
            safety_risk = threat_analysis.get("crew_safety_risk", "UNKNOWN")
            logs.append(f"[{timestamp}] 👥 Crew safety risk: {safety_risk}")
            
            # Launch delay decision
            if threat_analysis.get("launch_delay_required", False):
                delay_hours = threat_analysis.get("delay_hours", 0)
                logs.append(f"[{timestamp}] ⏰ LAUNCH DELAY REQUIRED: {delay_hours} hours due to {threat_level} threats")
                logs.append(f"[{timestamp}] Rationale: Crew safety takes priority over mission timeline")
            else:
                logs.append(f"[{timestamp}] ✅ Launch window clear - proceeding as scheduled")
            
            # Trajectory decision log
            recommended = trajectory_evaluation.get("recommended_trajectory", "UNKNOWN")
            confidence = trajectory_evaluation.get("confidence_level", "UNKNOWN")
            
            logs.append(f"[{timestamp}] 🛰️  TRAJECTORY SELECTED: {recommended} (Confidence: {confidence})")
            
            # Trade-off analysis log
            trade_offs = trajectory_evaluation.get("trade_off_analysis", {})
            fuel_diff = trade_offs.get("fuel_cost", {}).get("difference", 0)
            time_diff = trade_offs.get("time_cost", {}).get("difference", 0)
            
            if fuel_diff != 0 or time_diff != 0:
                logs.append(f"[{timestamp}] 📊 TRADE-OFF ANALYSIS:")
                if fuel_diff > 0:
                    logs.append(f"[{timestamp}]   • Fuel cost: +{fuel_diff:.1f} m/s ΔV for safety")
                elif fuel_diff < 0:
                    logs.append(f"[{timestamp}]   • Fuel savings: {abs(fuel_diff):.1f} m/s ΔV")
                
                if time_diff > 0:
                    logs.append(f"[{timestamp}]   • Time cost: +{time_diff:.1f} hours for safety")
                elif time_diff < 0:
                    logs.append(f"[{timestamp}]   • Time savings: {abs(time_diff):.1f} hours")
            
            # Safety benefit log
            safety_benefit = trade_offs.get("safety_benefit", "Unknown")
            risk_reduction = trade_offs.get("risk_reduction", "Unknown")
            logs.append(f"[{timestamp}] 🛡️  Safety benefit: {safety_benefit}")
            logs.append(f"[{timestamp}] Risk reduction: {risk_reduction}")
            
            # Justification log
            justification = trajectory_evaluation.get("justification", "No justification provided")
            logs.append(f"[{timestamp}] 💭 DECISION JUSTIFICATION: {justification}")
            
            # Mission success probability
            success_prob = trajectory_evaluation.get("mission_success_probability", "Unknown")
            logs.append(f"[{timestamp}] 🎯 Mission success probability: {success_prob}")
            
            # Alternative considerations
            alternatives = trajectory_evaluation.get("alternative_considerations", [])
            if alternatives:
                logs.append(f"[{timestamp}] 🔄 Alternative considerations:")
                for alt in alternatives:
                    logs.append(f"[{timestamp}]   • {alt}")
            
            # Final recommendation
            logs.append(f"[{timestamp}] ✅ FINAL RECOMMENDATION: Proceed with {recommended} trajectory")
            logs.append(f"[{timestamp}] ODIN decision-making complete. Mission parameters optimized for safety and efficiency.")
            
            logger.info(f"Generated {len(logs)} decision log entries")
            return logs
            
        except Exception as e:
            logger.error(f"Error generating decision logs: {e}")
            return [
                f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}] ❌ Error generating decision logs: {str(e)}",
                f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}] ⚠️  Manual review required"
            ]

    def generate_ai_recommendations(self, threat_analysis: Dict, trajectory_evaluation: Dict) -> Dict[str, Any]:
        """
        Generate AI recommendations for mission planning
        
        Args:
            threat_analysis: Results from threat analysis
            trajectory_evaluation: Results from trajectory evaluation
            
        Returns:
            Structured AI recommendations
        """
        try:
            recommendations = {
                "immediate_actions": [],
                "mission_modifications": [],
                "safety_measures": [],
                "monitoring_requirements": [],
                "contingency_plans": []
            }
            
            # Immediate actions based on threat level
            threat_level = threat_analysis.get("threat_level", "UNKNOWN")
            if threat_level in ["HIGH", "CRITICAL"]:
                recommendations["immediate_actions"].append("Implement enhanced radiation shielding")
                recommendations["immediate_actions"].append("Activate emergency communication protocols")
                recommendations["immediate_actions"].append("Prepare for potential mission abort")
            
            # Mission modifications
            if threat_analysis.get("launch_delay_required", False):
                delay_hours = threat_analysis.get("delay_hours", 0)
                recommendations["mission_modifications"].append(f"Delay launch by {delay_hours} hours")
            
            # Safety measures
            safety_risk = threat_analysis.get("crew_safety_risk", "UNKNOWN")
            if safety_risk in ["HIGH", "CRITICAL"]:
                recommendations["safety_measures"].append("Increase crew radiation monitoring frequency")
                recommendations["safety_measures"].append("Prepare emergency medical protocols")
            
            # Monitoring requirements
            recommendations["monitoring_requirements"].append("Continuous space weather monitoring")
            recommendations["monitoring_requirements"].append("Real-time trajectory adjustment capability")
            
            # Contingency plans
            recommended_trajectory = trajectory_evaluation.get("recommended_trajectory", "UNKNOWN")
            if recommended_trajectory == "ADJUSTED":
                recommendations["contingency_plans"].append("Prepare alternative trajectory options")
                recommendations["contingency_plans"].append("Maintain fuel reserves for emergency maneuvers")
            
            logger.info(f"Generated AI recommendations with {len(recommendations['immediate_actions'])} immediate actions")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating AI recommendations: {e}")
            return {
                "immediate_actions": ["Manual review required"],
                "mission_modifications": ["Error in analysis"],
                "safety_measures": ["Proceed with caution"],
                "monitoring_requirements": ["Enhanced monitoring recommended"],
                "contingency_plans": ["Prepare for manual intervention"]
            }

# Convenience function for easy integration
def analyze_mission_with_ai(hazards: List[Dict], timeframe: Dict, 
                          baseline_path: List, adjusted_path: List, 
                          constraints: Dict, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Complete AI analysis pipeline for mission planning
    
    Args:
        hazards: List of detected hazards
        timeframe: Mission timeframe
        baseline_path: Baseline trajectory
        adjusted_path: Adjusted trajectory
        constraints: Mission constraints
        api_key: Gemini API key (optional)
        
    Returns:
        Complete AI analysis results
    """
    try:
        # Initialize AI copilot
        copilot = ODINGeminiCopilot(api_key)
        
        # Perform threat analysis
        threat_analysis = copilot.analyze_threats(hazards, timeframe)
        
        # Evaluate trajectory options
        trajectory_evaluation = copilot.evaluate_trajectory_options(
            baseline_path, adjusted_path, hazards, constraints
        )
        
        # Generate decision logs
        decision_logs = copilot.generate_decision_log(threat_analysis, trajectory_evaluation)
        
        # Generate AI recommendations
        ai_recommendations = copilot.generate_ai_recommendations(threat_analysis, trajectory_evaluation)
        
        return {
            "threat_analysis": threat_analysis,
            "trajectory_evaluation": trajectory_evaluation,
            "decision_logs": decision_logs,
            "ai_recommendations": ai_recommendations,
            "ai_status": "SUCCESS"
        }
        
    except Exception as e:
        logger.error(f"Error in AI analysis pipeline: {e}")
        return {
            "threat_analysis": {"threat_level": "UNKNOWN", "error": str(e)},
            "trajectory_evaluation": {"recommended_trajectory": "UNKNOWN", "error": str(e)},
            "decision_logs": [f"AI analysis failed: {str(e)}"],
            "ai_recommendations": {"immediate_actions": ["Manual review required"]},
            "ai_status": "ERROR"
        }
