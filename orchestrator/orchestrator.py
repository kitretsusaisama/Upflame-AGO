import time
import logging
from typing import List, Dict, Any
from upflame_ago.world_model.state_model import WorldModel, AgentState
from upflame_ago.memory.episodic_memory import EpisodicMemory, Episode, EpisodeStep
from upflame_ago.memory.vector_memory import VectorMemory
from upflame_ago.agent.agent_runtime import AgentRuntime
from upflame_ago.agent.planner import Planner
from upflame_ago.orchestrator.goal_manager import GoalManager
from upflame_ago.orchestrator.team_builder import TeamBuilder
from upflame_ago.orchestrator.competition_manager import CompetitionManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self):
        logger.info("Initializing UpFlame-AGO Orchestrator...")

        # 1. Initialize Memory and World Model
        self.vector_memory = VectorMemory()
        self.episodic_memory = EpisodicMemory()
        self.world_model = WorldModel()

        # 2. Initialize Core Agents
        self.agent_pool = []
        initial_agents = [
            AgentState(id="agent_1", name="Alpha", status="idle"),
            AgentState(id="agent_2", name="Beta", status="idle"),
            AgentState(id="agent_3", name="Gamma", status="idle"),
            AgentState(id="agent_4", name="Delta", status="idle")
        ]
        self.world_model.initialize(initial_agents, resources={"cpu": 100, "memory": 1024})

        # Create runtime for each agent
        self.agents = {}
        for agent_state in initial_agents:
            self.agents[agent_state.id] = AgentRuntime(agent_state.id, agent_state.name, self.world_model)
            self.agent_pool.append(agent_state.id)

        # 3. Initialize Managers
        self.goal_manager = GoalManager()
        self.team_builder = TeamBuilder(self.agent_pool)
        self.competition_manager = CompetitionManager()
        self.planner = Planner()

    def run_loop(self, max_steps=10):
        """
        Main orchestration loop.
        """
        logger.info("Starting Orchestrator Loop...")
        step = 0
        while step < max_steps:
            step += 1
            logger.info(f"--- Step {step} ---")

            # 1. Goal Selection
            goal = self.goal_manager.get_next_goal()
            if not goal:
                logger.info("No pending goals. Adding default goals.")
                self.goal_manager.add_goal("Analyze system logs")
                self.goal_manager.add_goal("Optimize database queries")
                continue

            logger.info(f"Processing Goal: {goal.description}")
            self.goal_manager.update_status(goal.id, "in_progress")

            # 2. Task Decomposition
            tasks = self.planner.decompose(goal.description)
            logger.info(f"Decomposed into {len(tasks)} tasks.")

            # 3. Team Formation (NvsN competition or single team)
            # For this demo, we create two teams to compete on the first task
            task = tasks[0]
            team_a = self.team_builder.create_team(task.description, size=1)
            team_b = self.team_builder.create_team(task.description, size=1)

            if team_a and team_b:
                logger.info(f"Formed competing teams: Team A ({team_a.agents}) vs Team B ({team_b.agents})")
                comp = self.competition_manager.start_competition(task.description, [team_a.id, team_b.id])

                # 4. Execution (Parallel in real system, sequential here)
                for team, team_name in [(team_a, "A"), (team_b, "B")]:
                    agent_id = team.agents[0]
                    agent = self.agents[agent_id]

                    logger.info(f"Team {team_name} executing...")
                    result = agent.execute_task(task.description)

                    # 5. Evaluation & Reward
                    score = self.evaluate_result(result)
                    self.competition_manager.update_score(comp.id, team.id, score)
                    logger.info(f"Team {team_name} score: {score}")

                    # Store episode
                    episode = Episode() # Populate with steps from result
                    self.episodic_memory.add_episode(episode)

                # 6. Resolve Competition
                winner_id = self.competition_manager.resolve_competition(comp.id)
                winner_team = "A" if winner_id == team_a.id else "B"
                logger.info(f"Competition winner: Team {winner_team}")

                # Cleanup
                self.team_builder.disband_team(team_a.id)
                self.team_builder.disband_team(team_b.id)

                self.goal_manager.update_status(goal.id, "completed")

            else:
                logger.info("Not enough agents for competition. Skipping.")
                self.goal_manager.update_status(goal.id, "pending") # Retry later

            time.sleep(1)

    def evaluate_result(self, result: Dict[str, Any]) -> float:
        # Mock evaluation logic
        # In RL, this would come from the Reward Model
        if result.get("status") == "success":
            return 1.0
        return 0.0

def main():
    orch = Orchestrator()
    orch.run_loop(max_steps=5)

if __name__ == "__main__":
    main()
