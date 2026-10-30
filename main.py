import io
import json
import os
import re

import ollama
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "base")
MAX_HISTORY = 20

SYSTEM_PROMPT = (
    "You are a friendly AI on a live video call. Your replies are spoken aloud, "
    "so use 1-3 short, simple sentences, conversational, with no markdown, lists, "
    "or emojis. If an image is attached, it is the person's current camera view; "
    "describe or use it naturally, as if you can see them on a video call."
)

whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
app = FastAPI()
sessions: dict[str, list[dict]] = {}

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# In "auto" vision mode the camera frame is only given to the model when the question sounds visual.
VISION_HINT = re.compile(
    r"\b(see|saw|look|looks|looking|show|showing|holding|hold|wearing|wear|this|these|"
    r"camera|picture|image|colou?r|read|written|says?|screen|my (shirt|hand|hands|face|room|desk|"
    r"background|hair|outfit)|how do i look|what('s| is) (in|on))\b",
    re.I,
)


def line(obj: dict) -> str:
    return json.dumps(obj) + "\n"


def clean(s: str) -> str:
    return re.sub(r"[*_#`>~]", "", s).strip()  # remove markdown the TTS would read aloud


def transcribe(audio_bytes: bytes) -> str:
    segments, _ = whisper.transcribe(io.BytesIO(audio_bytes), vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


def llm_stream(messages: list[dict], fallback: list[dict] | None):
    opts = {"num_predict": 200}
    try:
        for chunk in ollama.chat(model=OLLAMA_MODEL, messages=messages, stream=True,
                                 options=opts, keep_alive="30m"):
            yield chunk["message"]["content"]
    except ollama.ResponseError:
        if fallback is None:  # no image was sent, so this is a real error
            raise
        # model can't take images: retry text-only
        for chunk in ollama.chat(model=OLLAMA_MODEL, messages=fallback, stream=True,
                                 options=opts, keep_alive="30m"):
            yield chunk["message"]["content"]


def turn_stream(session_id: str, audio_bytes: bytes, frame: bytes | None, vision: str):
    try:
        text = transcribe(audio_bytes)
    except Exception as e:
        yield line({"type": "error", "text": f"Transcription failed: {e}"})
        return
    if len(text) < 2:
        yield line({"type": "empty"})
        return
    yield line({"type": "transcript", "text": text})

    use_frame = bool(frame) and (
        vision == "always" or (vision == "auto" and bool(VISION_HINT.search(text)))
    )
    frame = frame if use_frame else None
    yield line({"type": "vision", "used": use_frame})

    history = sessions.setdefault(session_id, [])
    base = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    plain = {"role": "user", "content": text}
    with_img = {**plain, "images": [frame]} if frame else plain
    fallback = base + [plain] if frame else None

    spoken, buf = "", ""
    try:
        for piece in llm_stream(base + [with_img], fallback):
            buf += piece
            *done, buf = SENTENCE_SPLIT.split(buf)
            for s in done:
                s = clean(s)
                if s:
                    spoken += s + " "
                    yield line({"type": "sentence", "text": s})
        tail = clean(buf)
        if tail:
            spoken += tail
            yield line({"type": "sentence", "text": tail})
    except Exception as e:
        yield line({"type": "error", "text": str(e)})
    finally:
        # Runs even if the browser aborts (user interrupted): keep what was actually said.
        history.append(plain)
        if spoken.strip():
            history.append({"role": "assistant", "content": spoken.strip()})
        del history[:-MAX_HISTORY]


@app.post("/api/ask")
def ask(
    session_id: str = Form(...),
    audio: UploadFile = File(...),
    frame: UploadFile | None = File(None),
    vision: str = Form("auto"),  # auto | always | off
):
    audio_bytes = audio.file.read()
    frame_bytes = frame.file.read() if frame else None
    return StreamingResponse(turn_stream(session_id, audio_bytes, frame_bytes, vision),
                             media_type="application/x-ndjson")


@app.post("/api/reset")
def reset(session_id: str = Form(...)):
    sessions.pop(session_id, None)
    return {"ok": True}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))