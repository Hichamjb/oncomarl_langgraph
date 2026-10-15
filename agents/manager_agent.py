from agents.state import AgentState

def manager_node(state: AgentState) -> Dict[str, Any]:
    toxicity_info = state["toxicity_analysis"]
    model_info = state["model_recommendation"]
    
    final_action = model_info["recommended_action"]
    override = False

    # Règle de sécurité : Si la toxicité est critique mais que le modèle RL conseille de continuer
    if toxicity_info["alert"] and final_action == "CONTINUE_CURRENT_PATHWAY":
        final_action = "MULTIDISCIPLINARY_REASSESSMENT"
        override = True

    return {
        "final_decision": final_action,
        "manager_override": override
    }