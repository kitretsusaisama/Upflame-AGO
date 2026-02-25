from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, replace
import copy
import uuid

@dataclass
class AgentState:
    id: str
    name: str
    status: str
    current_task: Optional[str] = None
    inventory: List[str] = field(default_factory=list)
    location: str = "home"

@dataclass
class EnvironmentState:
    timestamp: float
    resources: Dict[str, Any]
    active_agents: Dict[str, AgentState]
    active_tasks: List[str]
    global_status: str

class WorldModel:
    def __init__(self):
        self.history: List[EnvironmentState] = []
        self.current_state: Optional[EnvironmentState] = None

    def initialize(self, agents: List[AgentState], resources: Dict[str, Any]):
        self.current_state = EnvironmentState(
            timestamp=0.0,
            resources=copy.deepcopy(resources),
            active_agents={a.id: copy.deepcopy(a) for a in agents},
            active_tasks=[],
            global_status="initialized"
        )
        self.history.append(self.current_state)

    def update(self, action: Dict[str, Any], observation: Dict[str, Any]) -> EnvironmentState:
        if self.current_state is None:
            raise ValueError("WorldModel not initialized")

        # Create next state
        new_state = EnvironmentState(
            timestamp=self.current_state.timestamp + 1.0,
            resources=copy.deepcopy(self.current_state.resources),
            active_agents=copy.deepcopy(self.current_state.active_agents),
            active_tasks=copy.deepcopy(self.current_state.active_tasks),
            global_status="running"
        )

        # Apply action effects
        agent_id = action.get("agent_id")
        if agent_id and agent_id in new_state.active_agents:
            agent = new_state.active_agents[agent_id]

            action_type = action.get("type")
            if action_type == "move":
                target = action.get("target")
                if target:
                    agent.location = target
            elif action_type == "take_task":
                task_id = action.get("task_id")
                if task_id:
                    agent.current_task = task_id
                    if task_id not in new_state.active_tasks:
                        new_state.active_tasks.append(task_id)
            elif action_type == "complete_task":
                task_id = action.get("task_id")
                if task_id:
                    agent.current_task = None
                    if task_id in new_state.active_tasks:
                        new_state.active_tasks.remove(task_id)

        # Apply observation updates (ground truth correction)
        if "resources" in observation:
            new_state.resources.update(observation["resources"])

        self.current_state = new_state
        self.history.append(new_state)
        return new_state

    def predict(self, action: Dict[str, Any]) -> EnvironmentState:
        # Prediction logic (returns hypothetical state)
        pass
