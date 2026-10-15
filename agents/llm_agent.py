import ollama
from agents.state import AgentState

def llm_consultant_node(state: AgentState) -> Dict[str, Any]:
    patient_state = state["patient_state"]
    tumor_info = state["tumor_analysis"]
    toxicity_info = state["toxicity_analysis"]
    model_info = state["model_recommendation"]

    prompt = f"""
    Vous êtes un oncologue consultant au sein d'un comité pluridisciplinaire.
    
    Données du patient : {patient_state}
    - Surveillance Tumeur : {tumor_info['summary']}
    - Surveillance Toxicité : {toxicity_info['summary']}
    - Proposition du Modèle RL : {model_info['recommended_action']}

    Donnez un avis clinique synthétique en 2 phrases expliquant si la proposition du modèle RL est adaptée ou risquée.
    """
    
    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        opinion = response["message"]["content"]
    except Exception:
        opinion = "Analyse LLM non disponible (Vérifiez le service Ollama)."

    return {
        "llm_consultation": {
            "agent": "LLM_Consultant",
            "clinical_opinion": opinion
        }
    }