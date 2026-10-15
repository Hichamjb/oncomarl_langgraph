from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    patient_state: List[float]               # [tumor_burden, toxicity, uncertainty, progression, visit]
    tumor_analysis: Optional[Dict[str, Any]]
    toxicity_analysis: Optional[Dict[str, Any]]
    model_recommendation: Optional[Dict[str, Any]]
    llm_consultation: Optional[Dict[str, Any]]
    final_decision: Optional[str]
    manager_override: Optional[bool]