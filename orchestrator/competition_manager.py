from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import uuid

@dataclass
class Competition:
    id: str
    goal: str
    teams: List[str]
    scores: Dict[str, float] = field(default_factory=dict)
    winner: Optional[str] = None
    status: str = "active"

class CompetitionManager:
    def __init__(self):
        self.competitions: List[Competition] = []

    def start_competition(self, goal: str, team_ids: List[str]) -> Competition:
        comp = Competition(id=str(uuid.uuid4()), goal=goal, teams=team_ids)
        self.competitions.append(comp)
        return comp

    def update_score(self, competition_id: str, team_id: str, score: float):
        for comp in self.competitions:
            if comp.id == competition_id:
                comp.scores[team_id] = score
                break

    def resolve_competition(self, competition_id: str) -> Optional[str]:
        for comp in self.competitions:
            if comp.id == competition_id:
                if not comp.scores:
                    return None
                winner_id = max(comp.scores, key=comp.scores.get)
                comp.winner = winner_id
                comp.status = "finished"
                return winner_id
        return None
