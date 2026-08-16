from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .client import complete, stream
from .languages import SUPPORTED_LANGUAGES, language_name, validate_language

app = FastAPI(title="Sunflower Multilingual API", description="Translation and chat powered by Sunbird Sunflower-14B")


class TranslateRequest(BaseModel):
    text: str
    source_language: str
    target_language: str


class ChatRequest(BaseModel):
    message: str
    language: str = "eng"
    stream_response: bool = False


class BatchTranslateRequest(BaseModel):
    texts: list[str]
    source_language: str
    target_language: str


@app.post("/translate")
def translate(body: TranslateRequest):
    for code in [body.source_language, body.target_language]:
        if not validate_language(code):
            raise HTTPException(400, f"Unsupported language: {code}. See /languages for supported codes.")

    src = language_name(body.source_language)
    tgt = language_name(body.target_language)

    system = f"You are a professional translator specialising in African languages. Translate the text from {src} to {tgt}. Output only the translation, nothing else."
    result = complete(system, body.text, max_tokens=512, temperature=0.1)

    return {
        "original": body.text,
        "translation": result,
        "source_language": src,
        "target_language": tgt,
    }


@app.post("/translate/batch")
def batch_translate(body: BatchTranslateRequest):
    for code in [body.source_language, body.target_language]:
        if not validate_language(code):
            raise HTTPException(400, f"Unsupported language: {code}")

    src = language_name(body.source_language)
    tgt = language_name(body.target_language)
    system = f"Translate from {src} to {tgt}. Output only the translation."

    translations = [complete(system, text, max_tokens=256, temperature=0.1) for text in body.texts]
    return {"translations": translations, "source_language": src, "target_language": tgt}


@app.post("/chat")
def chat(body: ChatRequest):
    if not validate_language(body.language):
        raise HTTPException(400, f"Unsupported language: {body.language}")

    lang = language_name(body.language)
    system = (
        f"You are a helpful assistant. Respond in {lang}. "
        f"Be friendly, clear, and culturally appropriate for {lang} speakers."
    )

    if body.stream_response:
        return StreamingResponse(stream(system, body.message), media_type="text/event-stream")

    response = complete(system, body.message, max_tokens=1024, temperature=0.7)
    return {"response": response, "language": lang}


@app.post("/detect-and-translate")
def detect_and_translate(body: dict):
    text = body.get("text", "")
    target_language = body.get("target_language", "eng")

    if not validate_language(target_language):
        raise HTTPException(400, f"Unsupported target language: {target_language}")

    detect_system = (
        "Identify the language of the following text. "
        "Respond with only the ISO 639-3 language code from this list: "
        + ", ".join(SUPPORTED_LANGUAGES.keys())
    )
    detected_code = complete(detect_system, text, max_tokens=10, temperature=0.0).strip().lower()

    if not validate_language(detected_code):
        detected_code = "eng"

    tgt = language_name(target_language)
    src = language_name(detected_code)
    translate_system = f"Translate from {src} to {tgt}. Output only the translation."
    translation = complete(translate_system, text, max_tokens=512, temperature=0.1)

    return {
        "original": text,
        "detected_language": src,
        "detected_code": detected_code,
        "translation": translation,
        "target_language": tgt,
    }


@app.get("/languages")
def languages():
    return {"supported_languages": SUPPORTED_LANGUAGES}


@app.get("/health")
def health():
    return {"status": "ok", "model": "Sunbird/Sunflower-14B-W8A8"}
