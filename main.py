import os
from typing import List, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=OPENAI_API_KEY)


class ChatPayload(BaseModel):
    message: str
    history: List[Dict[str, str]] | None = None


def normalize_history(history: List[Dict[str, str]] | None) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    if not history:
        return normalized
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(role, str) and isinstance(content, str) and role in {"user", "assistant"}:
            normalized.append({"role": role, "content": content})
    return normalized


app = FastAPI(title="Career Conversations API")

# TODO: replace with your real WordPress domain(s)
allowed_origins = [
    "https://chat.primeskills.pk",
    "https://primeskills.pk",
    "https://www.primeskills.pk",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat_endpoint(payload: ChatPayload) -> Dict[str, Any]:
    history = normalize_history(payload.history)
    user_message = (payload.message or "").strip()
    if not user_message:
        return {
            "reply": "Please share a question about my experience, projects, or availability.",
            "history": history,
        }

    system_prompt = (
        "You are Nadeem's AI resume. Be concise, friendly, and answer from Nadeem's background, "
        "skills, and experience. If you don't know something, say so briefly."
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}] + history
    messages.append({"role": "user", "content": user_message})

    try:
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        reply = resp.choices[0].message.content
    except Exception as exc:  # noqa: BLE001
        print(f"[chat] error: {exc}", flush=True)
        return {
            "reply": "I hit a snag reaching the server. Please try again shortly.",
            "history": history,
        }

    history += [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply},
    ]
    return {"reply": reply, "history": history}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))
