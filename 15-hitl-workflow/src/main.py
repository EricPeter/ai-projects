import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any

import anthropic
from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import BaseModel

from .approval_store import create as create_approval, decide, get as get_approval, list_pending
from .uncertainty import assess

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

_workflows: dict[str, dict[str, Any]] = {}
_websockets: dict[str, list[WebSocket]] = {}


app = FastAPI(title="Human-in-the-Loop Workflow")


async def notify(workflow_id: str, event: dict):
    for ws in _websockets.get(workflow_id, []):
        try:
            await ws.send_json(event)
        except Exception:
            pass


def _plan_task(task: str) -> str:
    msg = client.messages.create(
        model=MODEL, max_tokens=512,
        messages=[{"role": "user", "content": f"Create a concise action plan for this task: {task}"}],
    )
    return msg.content[0].text


def _execute_plan(task: str, plan: str, feedback: str = "") -> str:
    user_content = f"Task: {task}\nPlan:\n{plan}"
    if feedback:
        user_content += f"\nHuman feedback: {feedback}"
    msg = client.messages.create(
        model=MODEL, max_tokens=1024,
        messages=[{"role": "user", "content": user_content}],
    )
    return msg.content[0].text


class StartRequest(BaseModel):
    task: str


class DecisionRequest(BaseModel):
    decided_by: str = "human"
    feedback: str = ""


@app.post("/workflows")
async def start_workflow(body: StartRequest):
    wf_id = str(uuid.uuid4())
    plan = _plan_task(body.task)
    uncertainty = assess(body.task, plan)

    audit = [
        {"step": "plan", "output": plan},
        {"step": "uncertainty_check", "should_pause": uncertainty.should_pause, "reason": uncertainty.reason},
    ]

    if uncertainty.should_pause:
        create_approval(wf_id, plan)
        _workflows[wf_id] = {"id": wf_id, "task": body.task, "plan": plan, "status": "awaiting_approval", "audit": audit}
        await notify(wf_id, {"event": "paused", "reason": uncertainty.reason})
    else:
        result = _execute_plan(body.task, plan)
        audit.append({"step": "execute", "output": result})
        _workflows[wf_id] = {"id": wf_id, "task": body.task, "plan": plan, "status": "completed", "result": result, "audit": audit}
        await notify(wf_id, {"event": "completed"})

    return {"workflow_id": wf_id, "status": _workflows[wf_id]["status"]}


@app.get("/workflows/{wf_id}")
def get_workflow(wf_id: str):
    wf = _workflows.get(wf_id)
    if not wf:
        raise HTTPException(404, "Workflow not found")
    return wf


@app.post("/workflows/{wf_id}/approve")
async def approve(wf_id: str, body: DecisionRequest):
    wf = _workflows.get(wf_id)
    if not wf or wf["status"] != "awaiting_approval":
        raise HTTPException(400, "Workflow not awaiting approval")
    decide(wf_id, "approved", body.decided_by, body.feedback)
    result = _execute_plan(wf["task"], wf["plan"], body.feedback)
    wf["status"] = "completed"
    wf["result"] = result
    wf["audit"].append({"step": "approved_by", "user": body.decided_by})
    wf["audit"].append({"step": "execute", "output": result})
    await notify(wf_id, {"event": "completed"})
    return {"status": "completed", "result": result}


@app.post("/workflows/{wf_id}/reject")
async def reject(wf_id: str, body: DecisionRequest):
    wf = _workflows.get(wf_id)
    if not wf or wf["status"] != "awaiting_approval":
        raise HTTPException(400, "Workflow not awaiting approval")
    decide(wf_id, "rejected", body.decided_by, body.feedback)
    wf["status"] = "rejected"
    wf["audit"].append({"step": "rejected_by", "user": body.decided_by, "feedback": body.feedback})
    await notify(wf_id, {"event": "rejected"})
    return {"status": "rejected"}


@app.get("/workflows/pending")
def get_pending():
    return {"pending": list_pending()}


@app.get("/workflows/{wf_id}/audit")
def get_audit(wf_id: str):
    wf = _workflows.get(wf_id)
    if not wf:
        raise HTTPException(404)
    return {"audit": wf.get("audit", [])}


@app.websocket("/ws/{wf_id}")
async def websocket_endpoint(websocket: WebSocket, wf_id: str):
    await websocket.accept()
    _websockets.setdefault(wf_id, []).append(websocket)
    try:
        while True:
            await asyncio.sleep(30)
    except Exception:
        _websockets[wf_id].remove(websocket)


@app.get("/health")
def health():
    return {"status": "ok"}
