import os
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from .health_bot import respond
from data.health_knowledge import UGANDAN_LANGUAGES

app = FastAPI(title="Uganda Community Health Bot")


class HealthRequest(BaseModel):
    question: str
    language: str = "eng"


@app.post("/ask")
def ask(body: HealthRequest):
    if body.language not in UGANDAN_LANGUAGES:
        raise HTTPException(400, f"Unsupported language: {body.language}. Supported: {list(UGANDAN_LANGUAGES.keys())}")
    return respond(body.question, body.language)


@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    form = await request.form()
    message = form.get("Body", "")
    from_number = form.get("From", "")

    language = "lug"
    if message.lower().startswith("[eng]"):
        language = "eng"
        message = message[5:].strip()
    elif message.lower().startswith("[ach]"):
        language = "ach"
        message = message[5:].strip()

    result = respond(message, language)
    reply = result["response"]

    from twilio.twiml.messaging_response import MessagingResponse
    resp = MessagingResponse()
    resp.message(reply)
    return PlainTextResponse(str(resp), media_type="application/xml")


@app.get("/languages")
def languages():
    return {"languages": UGANDAN_LANGUAGES, "usage": "Prefix your message with [lug], [ach], [eng], etc."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Uganda Health Bot</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui; background: #f0fdf4; color: #1a2e1a; max-width: 640px; margin: 0 auto; padding: 1rem; }
    h1 { color: #15803d; }
    .subtitle { color: #6b7280; margin-top: -0.5rem; }
    select, textarea, button { width: 100%; padding: 0.75rem; border-radius: 8px; border: 1px solid #d1fae5; font-size: 1rem; margin: 0.5rem 0; }
    button { background: #16a34a; color: white; border: none; cursor: pointer; font-weight: 600; }
    button:hover { background: #15803d; }
    #response { background: white; border: 1px solid #d1fae5; border-radius: 8px; padding: 1rem; min-height: 80px; white-space: pre-wrap; }
    .emergency { background: #fef2f2; border-color: #fca5a5; color: #991b1b; }
    .lang-badge { display: inline-block; background: #dcfce7; color: #166534; padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.8rem; margin-bottom: 0.5rem; }
  </style>
</head>
<body>
  <h1>🌻 Uganda Health Bot</h1>
  <p class="subtitle">Powered by Sunbird Sunflower-14B · Ask health questions in your language</p>

  <label>Language</label>
  <select id="lang">
    <option value="lug">Luganda</option>
    <option value="ach">Acholi</option>
    <option value="nyn">Runyankole</option>
    <option value="teo">Ateso</option>
    <option value="eng">English</option>
  </select>

  <label>Your Question</label>
  <textarea id="q" rows="3" placeholder="Oli mutamu ki? / What health question do you have?"></textarea>

  <button onclick="ask()">Ask</button>

  <div id="response-area" style="display:none">
    <span class="lang-badge" id="lang-badge"></span>
    <div id="response"></div>
  </div>

  <script>
    async function ask() {
      const q = document.getElementById('q').value;
      const lang = document.getElementById('lang').value;
      if (!q.trim()) return;
      document.getElementById('response').textContent = 'Thinking...';
      document.getElementById('response-area').style.display = 'block';

      const r = await fetch('/ask', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question: q, language: lang})
      });
      const data = await r.json();
      const el = document.getElementById('response');
      el.textContent = data.response;
      el.className = data.is_emergency ? 'emergency' : '';
      document.getElementById('lang-badge').textContent = data.language;
    }
    document.getElementById('q').addEventListener('keydown', e => {
      if (e.key === 'Enter' && e.ctrlKey) ask();
    });
  </script>
</body>
</html>"""
