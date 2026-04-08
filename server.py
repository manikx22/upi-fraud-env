"""
FastAPI server exposing UPI Fraud OpenEnv via HTTP API.
All endpoints follow the OpenEnv spec: /reset, /step, /state, /grade, /tasks.
"""

import uuid
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import UPIFraudEnv, Action, TurnObservation, Reward
from tasks import TASKS, grade_task
from scenarios import SCENARIOS, ActionType

app = FastAPI(
    title="UPI Fraud Prevention OpenEnv",
    description="OpenEnv-compliant environment for UPI fraud prevention and recovery tasks",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: dict[str, UPIFraudEnv] = {}


# ─── Request / Response models ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int = 42

class StepRequest(BaseModel):
    session_id: str
    action: Action

class StepResponse(BaseModel):
    observation: TurnObservation
    reward: Reward
    done: bool
    info: dict

class GradeRequest(BaseModel):
    task_id: str
    actions: list[Action]


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "UPI Fraud Prevention OpenEnv",
        "version": "1.0.0",
        "tasks": [t.id for t in TASKS],
        "action_space": [a.value for a in ActionType],
        "endpoints": ["/reset", "/step", "/state/{id}", "/grade", "/tasks", "/scenarios"],
    }

@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(_sessions)}

@app.get("/tasks")
def list_tasks():
    return [
        {
            "id": t.id,
            "name": t.name,
            "description": t.description,
            "difficulty": t.difficulty,
            "success_threshold": t.success_threshold,
        }
        for t in TASKS
    ]

@app.get("/scenarios")
def list_scenarios():
    return [
        {
            "id": s.id,
            "scam_type": s.scam_type.value,
            "title": s.title,
            "difficulty": s.difficulty,
            "amount_at_risk": s.amount_at_risk,
            "n_turns": len(s.turns),
            "tags": s.tags,
        }
        for s in SCENARIOS
    ]

@app.get("/actions")
def list_actions():
    return {
        "all_actions": [a.value for a in ActionType],
        "safe": [
            "decline_request", "hang_up", "block_contact", "ignore",
            "call_bank_helpline", "raise_upi_dispute", "report_to_npci",
            "file_cybercrime", "verify_upi_id", "check_qr_code",
            "call_official_number", "share_with_family", "take_screenshot", "do_nothing",
        ],
        "dangerous": ["share_otp", "share_pin", "pay_scammer", "scan_qr"],
    }

@app.post("/reset")
def reset(req: ResetRequest = Body(default_factory=ResetRequest)):
    session_id = str(uuid.uuid4())
    env = UPIFraudEnv(task_id=req.task_id, seed=req.seed)
    obs = env.reset()
    _sessions[session_id] = env
    return {"session_id": session_id, "observation": obs}

@app.post("/step")
def step(req: StepRequest) -> StepResponse:
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(404, f"Session '{req.session_id}' not found. Call /reset first.")
    try:
        obs, reward, done, info = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return StepResponse(observation=obs, reward=reward, done=done, info=info)

@app.get("/state/{session_id}")
def state(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    return env.state()

@app.post("/grade")
def grade(req: GradeRequest):
    try:
        return grade_task(req.task_id, req.actions)
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]
        return {"deleted": True}
    raise HTTPException(404, "Session not found")
