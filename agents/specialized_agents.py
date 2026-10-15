from agents.state import AgentState

def tumor_agent_node(state: AgentState) -> Dict[str, Any]:
    patient_state = state["patient_state"]
    tumor_burden, _, _, progression, _ = patient_state
    alert = tumor_burden >= 0.7 or progression == 1.0
    
    return {
        "tumor_analysis": {
            "agent": "Tumor_Monitor",
            "tumor_burden": tumor_burden,
            "progression": bool(progression),
            "alert": alert,
            "summary": "Tumeur critique ou en progression" if alert else "Charge tumorale sous contrôle"
        }
    }

def toxicity_agent_node(state: AgentState) -> Dict[str, Any]:
    patient_state = state["patient_state"]
    _, toxicity, _, _, _ = patient_state
    alert = toxicity >= 0.7
    
    return {
        "toxicity_analysis": {
            "agent": "Toxicity_Monitor",
            "toxicity": toxicity,
            "alert": alert,
            "summary": "Toxicité critique (Effets secondaires graves)" if alert else "Toxicité acceptable"
        }
    }