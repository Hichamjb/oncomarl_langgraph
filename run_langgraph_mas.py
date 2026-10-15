from pathlib import Path
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.specialized_agents import tumor_agent_node, toxicity_agent_node
from agents.model_agent import RLModelAgentNode
from agents.llm_agent import llm_consultant_node
from agents.manager_agent import manager_node

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "oncomarl_v2" / "double_dqn_v2.pt"

# 1. Instanciation du nœud basé sur un fichier de modèle
rl_agent_node = RLModelAgentNode(model_path=MODEL_PATH)

# 2. Déclaration du graphe d'états
builder = StateGraph(AgentState)

# 3. Ajout des nœuds (Agents)
builder.add_node("tumor_monitor", tumor_agent_node)
builder.add_node("toxicity_monitor", toxicity_agent_node)
builder.add_node("rl_model_agent", rl_agent_node)
builder.add_node("llm_consultant", llm_consultant_node)
builder.add_node("manager_agent", manager_node)

# 4. Connexion des arêtes (Workflow)
# Le point d'entrée lance en parallèle le suivi Tumeur et Toxicité
builder.set_entry_point("tumor_monitor")
builder.add_edge("tumor_monitor", "toxicity_monitor")

# Ensuite, exécution de l'agent modèle RL puis du LLM
builder.add_edge("toxicity_monitor", "rl_model_agent")
builder.add_edge("rl_model_agent", "llm_consultant")

# Enfin, l'agent Manager consolide toutes les analyses
builder.add_edge("llm_consultant", "manager_agent")
builder.add_edge("manager_agent", END)

# Compile le graphe
graph = builder.compile()

# 5. Exécution du workflow sur un cas patient de test
# Format : [Charge tumorale, Toxicité, Incertitude, Progression, Étape de visite]
test_patient_state = [0.75, 0.85, 0.30, 1.0, 0.50]

print("=" * 60)
print("EXÉCUTION DU SYSTÈME MULTI-AGENTS (LANGGRAPH)")
print("=" * 60)

initial_input = {"patient_state": test_patient_state}
final_state = graph.invoke(initial_input)

# Affichage des résultats
print(f"État du Patient        : {final_state['patient_state']}")
print(f"Agent Tumeur           : {final_state['tumor_analysis']['summary']}")
print(f"Agent Toxicité         : {final_state['toxicity_analysis']['summary']}")
print(f"Agent Modèle RL (DQN)  : {final_state['model_recommendation']['recommended_action']}")
print(f"Agent LLM (Consultant) : {final_state['llm_consultation']['clinical_opinion']}")
print("-" * 60)
print(f">> DÉCISION FINALE MANAGER : {final_state['final_decision']}")
if final_state["manager_override"]:
    print(">> NOTE : Le Manager a appliqué une règle de sécurité pour modifier l'action du modèle RL.")
print("=" * 60)