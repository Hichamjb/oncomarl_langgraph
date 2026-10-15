import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from agents.state import AgentState

ACTIONS = {
    0: "CONTINUE_CURRENT_PATHWAY",
    1: "REASSESS_CURRENT_PATHWAY",
    2: "SWITCH_PATHWAY",
    3: "MULTIDISCIPLINARY_REASSESSMENT",
}

class QNetwork(nn.Module):
    def __init__(self, state_dim=5, action_dim=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.network(x)

class RLModelAgentNode:
    def __init__(self, model_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model = QNetwork(checkpoint["state_dim"], checkpoint["action_dim"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        patient_state = state["patient_state"]
        state_tensor = torch.tensor(patient_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()[0]
        
        best_idx = int(np.argmax(q_values))
        return {
            "model_recommendation": {
                "agent": "RL_Model_Agent",
                "recommended_action": ACTIONS[best_idx],
                "action_id": best_idx,
                "q_values": {ACTIONS[i]: float(q_values[i]) for i in range(len(ACTIONS))}
            }
        }