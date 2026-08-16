import uuid
from contextlib import asynccontextmanager

import anthropic
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .auth import (
    LoginRequest, RegisterRequest, create_access_token,
    get_current_user, get_db, pwd_context,
)
from .database import AuditLog, Tenant, UsageRecord, User, init_db
from .rate_limiter import check_rate_limit

ANTHROPIC_CLIENT = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="Multi-Tenant SaaS Agent", lifespan=lifespan)


@app.post("/auth/register")
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(400, "Email already registered")

    tenant = Tenant(id=str(uuid.uuid4()), name=body.company_name)
    user = User(
        id=str(uuid.uuid4()),
        tenant_id=tenant.id,
        email=body.email,
        hashed_password=pwd_context.hash(body.password),
    )
    db.add(tenant)
    db.add(user)
    await db.commit()
    return {"user_id": user.id, "tenant_id": tenant.id, "token": create_access_token(user.id, tenant.id)}


@app.post("/auth/login")
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if not user or not pwd_context.verify(body.password, user.hashed_password):
        raise HTTPException(401, "Invalid credentials")
    return {"token": create_access_token(user.id, user.tenant_id)}


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(body: ChatRequest, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    tenant = await db.get(Tenant, user.tenant_id)
    check_rate_limit(tenant.id, tenant.plan)

    msg = ANTHROPIC_CLIENT.messages.create(
        model=MODEL, max_tokens=1024,
        system=tenant.system_prompt,
        messages=[{"role": "user", "content": body.message}],
    )
    response_text = msg.content[0].text
    cost = (msg.usage.input_tokens / 1e6) * 3.0 + (msg.usage.output_tokens / 1e6) * 15.0

    db.add(UsageRecord(tenant_id=tenant.id, user_id=user.id,
                       input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens, cost_usd=cost))
    db.add(AuditLog(tenant_id=tenant.id, user_id=user.id, action="chat",
                    detail=body.message[:200]))
    await db.commit()

    return {"response": response_text, "tokens": {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens}}


@app.get("/usage")
async def usage(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(UsageRecord).where(UsageRecord.tenant_id == user.tenant_id)
    )
    records = result.scalars().all()
    total_cost = sum(r.cost_usd for r in records)
    return {"tenant_id": user.tenant_id, "total_requests": len(records), "total_cost_usd": round(total_cost, 6)}


@app.get("/health")
def health():
    return {"status": "ok"}
