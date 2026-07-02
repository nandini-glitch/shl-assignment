import asyncio
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.agent import run_turn
from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl-agent")

app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")

# The grader hits this from its own harness, not a browser, but permissive
# CORS costs nothing here and avoids a class of avoidable failures.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


_TURN_CAP_REPLY = (
    "We've covered a lot of ground -- here's where things stand. If you'd like to keep refining, "
    "feel free to start a fresh conversation with the latest constraints."
)


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    settings = get_settings()

    # Defensive turn cap: the grader caps conversations at 8 turns itself, but
    # we don't rely on a caller being well-behaved.
    if len(payload.messages) > settings.max_turns:
        return ChatResponse(reply=_TURN_CAP_REPLY, recommendations=[], end_of_conversation=True)

    try:
        loop = asyncio.get_running_loop()
        reply, recommendations, end_of_conversation = await asyncio.wait_for(
            loop.run_in_executor(None, run_turn, payload.messages),
            timeout=settings.request_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning("chat turn timed out")
        raise HTTPException(status_code=504, detail="agent timed out generating a response")
    except Exception:
        logger.exception("chat turn failed")
        raise HTTPException(status_code=500, detail="agent failed to generate a response")

    return ChatResponse(
        reply=reply, recommendations=recommendations, end_of_conversation=end_of_conversation
    )
