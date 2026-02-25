from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import uuid
import random

@dataclass
class Team:
    id: str
    agents: List[str]
    objective: str
    status: str = "forming"
    score: float = 0.0

class TeamBuilder:
    def __init__(self, agent_pool: List[str]):
        self.agent_pool = agent_pool
        self.active_teams: List[Team] = []

    def create_team(self, objective: str, size: int = 1) -> Optional[Team]:
        available = [a for a in self.agent_pool if not self.is_agent_busy(a)]

        if len(available) < size:
            return None

        assigned = random.sample(available, size)
        team = Team(id=str(uuid.uuid4()), agents=assigned, objective=objective)
        self.active_teams.append(team)
        return team

    def disband_team(self, team_id: str):
        self.active_teams = [t for t in self.active_teams if t.id != team_id]

    def is_agent_busy(self, agent_id: str) -> bool:
        for team in self.active_teams:
            if agent_id in team.agents:
                return True
        return False
