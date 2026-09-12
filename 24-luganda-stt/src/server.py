from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import (
    FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect,
)
from pydantic import BaseModel

from .config import settings
from .pipeline import PipelineResult, SegmentResult, STTPipeline, WordToken
from .streaming_pipeline import InterimEvent, StreamingLive, TurnEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_pipeline: STTPipeline | None = None
_live: StreamingLive | None = None
_live_lock = asyncio.Lock()


async def get_live() -> StreamingLive:
    """Lazy-load the streaming models on first WebSocket connection."""
    global _live
    async with _live_lock:
        if _live is None:
            logger.info("Loading streaming (live) models...")
            live = StreamingLive(settings)
            await asyncio.to_thread(live.load)
            _live = live
            logger.info("Streaming models ready.")
    return _live


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    mode = "fast"
    logger.info("Loading pipeline (mode=%s)...", mode)
    _pipeline = STTPipeline(settings, mode=mode)
    _pipeline.load()
    logger.info("Pipeline ready.")
    yield
    _pipeline = None


app = FastAPI(
    title="Luganda STT + Translation",
    description=(
        "End-to-end Luganda speech pipeline: preprocessing → VAD → diarization → "
        "Canary ASR + NGPU-LM + GPU-PB word boosting → spell check → "
        "punctuation / ITN → translation."
    ),
    version="0.2.0",
    lifespan=lifespan,
)


# ── Response models ───────────────────────────────────────────────────────────

class WordOut(BaseModel):
    word: str
    start: float
    end: float
    confidence: float


class SegmentOut(BaseModel):
    speaker: str
    start: float
    end: float
    transcript: str
    transcript_corrected: str
    transcript_final: str
    translation: str
    confidence: float
    words: list[WordOut]
    boosted_words_hit: int


class TranscribeResponse(BaseModel):
    segments: list[SegmentOut]
    full_transcript: str
    full_translation: str
    num_speakers: int
    duration: float
    language: str
    mode: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    boosted_words: int
    spell_checker: bool
    ngram_lm: bool
    diarization: bool
    vad: bool
    streaming: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "model": settings.canary_model,
        "boosted_words": len(_pipeline.boosted_words) if _pipeline else 0,
        "spell_checker": _pipeline.spell_checker_active if _pipeline else False,
        "ngram_lm": settings.ngram_lm_path.exists(),
        "diarization": (_pipeline._diarizer._model is not None) if _pipeline else False,
        "vad": (_pipeline._vad._model is not None) if _pipeline else False,
        "streaming": _live is not None,
    }


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(..., description="WAV / FLAC / MP3 audio file"),
    mode: str = Query("fast", description="'fast' (Canary only) or 'rich' (+ SALM)"),
    denoise: bool = Query(False, description="Apply spectral subtraction before ASR"),
):
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not ready")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio file")

    result = _pipeline.run(audio_bytes)

    return TranscribeResponse(
        segments=[
            SegmentOut(
                speaker=s.speaker,
                start=s.start,
                end=s.end,
                transcript=s.transcript,
                transcript_corrected=s.transcript_corrected,
                transcript_final=s.transcript_final,
                translation=s.translation,
                confidence=s.confidence,
                words=[WordOut(word=w.word, start=w.start, end=w.end, confidence=w.confidence)
                       for w in s.words],
                boosted_words_hit=s.boosted_words_hit,
            )
            for s in result.segments
        ],
        full_transcript=result.full_transcript,
        full_translation=result.full_translation,
        num_speakers=result.num_speakers,
        duration=result.duration,
        language=result.language,
        mode=result.mode,
        latency_ms=result.latency_ms,
    )


@app.get("/word-boost/list")
def list_boosted_words():
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not ready")
    words = _pipeline.boosted_words
    return {"words": words, "count": len(words)}


@app.post("/word-boost/add")
def add_boosted_word(word: str = Query(...)):
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not ready")
    _pipeline._spell.add_to_dictionary(word, count=100)
    return {"added": word}


# ── Live streaming (WebSocket) ─────────────────────────────────────────────────

def _pcm_bytes_to_float32(data: bytes) -> np.ndarray:
    """Decode little-endian int16 mono PCM to float32 in [-1, 1]."""
    if not data:
        return np.zeros(0, dtype=np.float32)
    ints = np.frombuffer(data, dtype="<i2")
    return (ints.astype(np.float32)) / 32768.0


def _event_to_dict(ev) -> dict:
    if isinstance(ev, TurnEvent):
        return {
            "type": "turn",
            "text": ev.text,
            "raw": ev.raw,
            "speaker": ev.speaker,
            "reason": ev.reason,
            "backchannel": ev.backchannel,
        }
    if isinstance(ev, InterimEvent):
        return {"type": "interim", "text": ev.text}
    return {"type": "unknown"}


@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    """Live streaming ASR + end-of-turn detection + speaker attribution.

    Protocol: client streams binary frames of 16 kHz mono little-endian int16
    PCM. The server replies with JSON events:
      {"type":"interim","text":...}                     (partial, per chunk)
      {"type":"turn","text":...,"speaker":...,          (finalized turn)
       "reason":"eou|eob|silence|max_words","backchannel":bool}
    A text message "eos" (or disconnect) flushes the final turn.
    """
    await ws.accept()
    try:
        live = await get_live()
    except Exception as e:  # noqa: BLE001
        await ws.send_json({"type": "error", "detail": f"model load failed: {e}"})
        await ws.close()
        return

    session = live.session()
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if msg.get("bytes") is not None:
                pcm = _pcm_bytes_to_float32(msg["bytes"])
                if pcm.size == 0:
                    continue
                events = await asyncio.to_thread(session.push, pcm)
                for ev in events:
                    await ws.send_json(_event_to_dict(ev))
            elif msg.get("text") is not None:
                if msg["text"].strip().lower() in ("eos", "close", "done"):
                    for ev in await asyncio.to_thread(session.finalize):
                        await ws.send_json(_event_to_dict(ev))
                    break
    except WebSocketDisconnect:
        pass
    finally:
        try:
            for ev in await asyncio.to_thread(session.finalize):
                await ws.send_json(_event_to_dict(ev))
        except Exception:  # noqa: BLE001 - best-effort flush on close
            pass
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass
