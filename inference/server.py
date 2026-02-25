from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
from upflame_ago.model.config import UpFlameAGOConfig
from upflame_ago.model.architecture import UpFlameAGOForCausalLM
from upflame_ago.agent.planner import Planner
from upflame_ago.agent.agent_runtime import AgentRuntime
from upflame_ago.world_model.state_model import WorldModel, AgentState

app = FastAPI(title="UpFlame-AGO Inference API")

model = None
planner = None
agent_runtime = None
world_model = None

@app.on_event("startup")
async def startup_event():
    global model, planner, agent_runtime, world_model
    print("Loading model...")
    config = UpFlameAGOConfig(vocab_size=65536, hidden_size=512, num_hidden_layers=4)
    model = UpFlameAGOForCausalLM(config)
    model.eval()

    print("Initializing components...")
    planner = Planner(model)
    world_model = WorldModel()
    world_model.initialize([AgentState(id="api_agent", name="API_Agent", status="idle")], {})
    agent_runtime = AgentRuntime("api_agent", "API_Agent", world_model)
    print("Server ready.")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50

class PlanRequest(BaseModel):
    goal: str

class AgentActionRequest(BaseModel):
    task: str

@app.post("/generate")
async def generate(req: GenerateRequest):
    return {"text": f"Generated response for: {req.prompt} [Model Output Placeholder]"}

@app.post("/plan")
async def plan(req: PlanRequest):
    if not planner:
        raise HTTPException(status_code=500, detail="Planner not initialized")

    tasks = planner.decompose(req.goal)
    return {"tasks": [t.__dict__ for t in tasks]}

@app.post("/agent/action")
async def agent_action(req: AgentActionRequest):
    if not agent_runtime:
        raise HTTPException(status_code=500, detail="Agent runtime not initialized")

    result = agent_runtime.execute_task(req.task)
    return result

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
