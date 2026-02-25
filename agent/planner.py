from typing import List, Dict, Any
from dataclasses import dataclass, field
import uuid

@dataclass
class Task:
    id: str
    description: str
    status: str = "pending"
    required_tools: List[str] = field(default_factory=list)
    assigned_agent_id: str = None

class Planner:
    def __init__(self, model=None):
        self.model = model

    def decompose(self, goal: str) -> List[Task]:
        tasks = []
        if "test" in goal.lower():
            tasks.append(Task(id=str(uuid.uuid4()), description="Run system diagnostics", required_tools=["python"]))
            tasks.append(Task(id=str(uuid.uuid4()), description="Verify memory integrity", required_tools=["python"]))
        else:
            tasks.append(Task(id=str(uuid.uuid4()), description=f"Execute: {goal}", required_tools=["search", "python"]))

        return tasks

    def replan(self, failed_task: Task, error_message: str) -> List[Task]:
        return [Task(id=str(uuid.uuid4()), description=f"Retry: {failed_task.description}", required_tools=failed_task.required_tools)]
