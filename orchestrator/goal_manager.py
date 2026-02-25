from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import uuid
import time

@dataclass
class Goal:
    id: str
    description: str
    priority: int = 1
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None

class GoalManager:
    def __init__(self):
        self.goals: List[Goal] = []

    def add_goal(self, description: str, priority: int = 1) -> Goal:
        goal = Goal(id=str(uuid.uuid4()), description=description, priority=priority)
        self.goals.append(goal)
        self.goals.sort(key=lambda x: x.priority, reverse=True)
        return goal

    def get_next_goal(self) -> Optional[Goal]:
        for goal in self.goals:
            if goal.status == "pending":
                return goal
        return None

    def update_status(self, goal_id: str, status: str):
        for goal in self.goals:
            if goal.id == goal_id:
                goal.status = status
                break
