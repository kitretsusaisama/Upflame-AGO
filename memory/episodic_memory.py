from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import uuid
import time

@dataclass
class EpisodeStep:
    state: Dict[str, Any] = field(default_factory=dict)
    action: Dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    next_state: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class Episode:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: List[EpisodeStep] = field(default_factory=list)
    total_reward: float = 0.0

class EpisodicMemory:
    def __init__(self):
        self.episodes: List[Episode] = []

    def add_episode(self, episode: Episode):
        self.episodes.append(episode)

    def get_recent_episodes(self, limit: int = 10) -> List[Episode]:
        return self.episodes[-limit:]

    def sample_experiences(self, batch_size: int = 32) -> List[EpisodeStep]:
        # Implement experience replay sampling logic here
        # For now, just return latest steps
        steps = []
        for ep in reversed(self.episodes):
            steps.extend(ep.steps)
            if len(steps) >= batch_size:
                break
        return steps[:batch_size]
