"""LLM provider runtime: local Ollama (default) or OpenAI-compatible cloud API.

When the user has no Ollama installed/running, they can paste an API key
(in the Streamlit sidebar, via POST /llm/config, or through .env) and all
chat/VLM and embedding calls are routed to an OpenAI-compatible endpoint
instead. Adapters translate the Ollama-style kwargs/responses used by
vlm.py and retriever.py so the rest of the app is unchanged.

Only OpenAI-compatible endpoints are supported: OpenAI, OpenRouter, Groq,
Gemini (its /openai compat surface), LM Studio, OpenLLM gateways, etc.
The key never leaves the process; it is never logged and never echoed back.
"""

import base64
import json
import os
import threading
import time
from pathlib import Path

import requests
import structlog

from config import get_settings

logger = structlog.get_logger()
settings = get_settings()

# Defaults for the cloud path (plain OpenAI /v1 unless overridden).
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


class ProviderError(ConnectionError):
    """HTTP/protocol failure of the cloud provider (retryable upstream)."""


_lock = threading.Lock()

# Runtime state, seeded from env/.env so a headless backend can run in
# cloud mode without any UI interaction.
_state: dict = {
    "provider": (settings.llm_provider or "local").strip().lower(),
    "api_key": (settings.api_key or "").strip(),
    "base_url": (settings.api_base_url or "").strip().rstrip("/"),
    "model": (settings.api_model or "").strip(),
    "embed_model": (settings.api_embed_model or "").strip(),
}
if _state["provider"] not in ("local", "api"):
    _state["provider"] = "local"
if not _state["base_url"]:
    _state["base_url"] = DEFAULT_BASE_URL


def _api_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _effective(key: str, default: str) -> str:
    return _state[key] or default


def chat_model() -> str:
    """Model name to use for chat(`vlm_cache` keys)."""
    if _state["provider"] == "api":
        return _effective("model", DEFAULT_CHAT_MODEL)
    return settings.vlm_model


def embed_model() -> str:
    """Model name to use for embeddings (embedding_cache keys)."""
    if _state["provider"] == "api":
        return _effective("embed_model", DEFAULT_EMBED_MODEL)
    return settings.embed_model


def get_api_key() -> str:
    return _state["api_key"]


def has_api_key() -> bool:
    return bool(_state["api_key"])


def status() -> dict:
    """Current provider info. Never includes the API key itself."""
    base = _state["base_url"] or DEFAULT_BASE_URL
    info = {
        "provider": _state["provider"],
        "chat_model": chat_model(),
        "embed_model": embed_model(),
        "base_url": base if _state["provider"] == "api" else settings.ollama_host,
        "api_key_set": bool(_state["api_key"]),
        "ollama_reachable": ollama_reachable(),
    }
    return info


def configure(provider: str, api_key: str | None = None, base_url: str | None = None,
              model: str | None = None, embed_model: str | None = None) -> dict:
    """Switch provider at runtime.

    provider: "local" (Ollama) or "api" (any OpenAI-compatible endpoint).
    Omitted fields keep their current/default values.
    """
    if provider not in ("local", "api"):
        raise ValueError("provider must be 'local' or 'api'")
    with _lock:
        old_key = _state["api_key"]
        new_key = (api_key or "").strip()
        if provider == "api" and not new_key and not old_key:
            # A switch *to* api mode demands a key the first time.
            raise ValueError("api provider requires a non-empty API key")
        if new_key:
            _state["api_key"] = new_key
        if base_url and base_url.strip():
            _state["base_url"] = base_url.strip().rstrip("/")
        if model and model.strip():
            _state["model"] = model.strip()
        if embed_model and embed_model.strip():
            _state["embed_model"] = embed_model.strip()
        _state["provider"] = provider
    status_after = status()
    logger.info("llm_provider.configured", provider=provider,
                chat_model=status_after["chat_model"],
                embed_model=status_after["embed_model"],
                base_url=status_after["base_url"])
    return status_after


_probe_cache = {"ts": 0.0, "ok": False}
_probe_lock = threading.Lock()


def ollama_reachable(timeout: float = 1.0, ttl_seconds: float = 10.0) -> bool:
    """Cheap health probe so the UI can warn before calls start failing.

    Uses a short connect timeout and caches the result briefly: on machines
    where a dead/firewalled local port *times out* instead of refusing (common
    on Windows), every status call would otherwise stall for the full timeout.
    """
    now = time.time()
    with _probe_lock:
        if now - _probe_cache["ts"] < ttl_seconds:
            return _probe_cache["ok"]
    try:
        r = requests.get(
            f"{settings.ollama_host.rstrip('/')}/api/version",
            timeout=(timeout, timeout),  # short connect timeout, not just read
        )
        ok = r.status_code == 200
    except Exception:
        ok = False
    with _probe_lock:
        _probe_cache["ts"] = time.time()
        _probe_cache["ok"] = ok
    return ok


# --- OpenAI-compatible adapters (Ollama-style I/O) ---

def _to_openai_messages(messages: list[dict]) -> list[dict]:
    """Ollama messages (content str + images as raw PNG bytes) ->
    OpenAI chat.completions messages (text/eight image_url parts with data URIs)."""
    out: list[dict] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        images = m.get("images") or []
        if not images:
            out.append({"role": role, "content": content})
            continue
        parts: list[dict] = [{"type": "text", "text": content}]
        for img in images:
            if isinstance(img, (bytes, bytearray)):
                b64 = base64.b64encode(bytes(img)).decode()
                data_uri = f"data:image/png;base64,{b64}"
            else:
                p = Path(str(img))
                ext = p.suffix.lstrip(".").lower() or "png"
                data_uri = f"data:image/{ext};base64,{base64.b64encode(p.read_bytes()).decode()}"
            parts.append({"type": "image_url", "image_url": {"url": data_uri}})
        out.append({"role": role, "content": parts})
    return out


def _post(path: str, payload: dict, timeout: float) -> dict:
    api_key = _state["api_key"]
    if not api_key:
        raise ProviderError("no API key configured for the cloud provider")
    t0 = time.perf_counter_ns()
    url = f"{_effective('base_url', DEFAULT_BASE_URL)}{path}"
    try:
        resp = requests.post(url, json=payload, headers=_api_headers(api_key), timeout=timeout)
    except requests.exceptions.Timeout as e:
        raise TimeoutError(f"cloud provider timed out: {url}") from e
    except requests.exceptions.RequestException as e:
        raise ProviderError(f"cloud provider unreachable: {e}") from e
    took_ns = time.perf_counter_ns() - t0
    if resp.status_code != 200:
        raise ProviderError(f"cloud provider HTTP {resp.status_code}: {resp.text[:300]}")
    try:
        data = resp.json()
    except ValueError as e:
        raise ProviderError(f"cloud provider returned invalid JSON: {e}") from e
    data["_elapsed_ns"] = took_ns
    return data


def chat_api(**kwargs) -> dict:
    """OpenAI-compatible /chat/completions, accepting ollama.chat-style kwargs.

    Returns an ollama-shaped response: {"message": {"content": ...}, ...timings}
    so existing response handling in vlm.py works unchanged.
    """
    model = _effective("model", DEFAULT_CHAT_MODEL)
    # Ollama messages carry raw image bytes; _to_openai_messages handles that.
    msgs = kwargs.get("messages") or []
    options = kwargs.get("options") or {}
    payload: dict = {
        "model": model,
        "messages": _to_openai_messages(msgs),
        "temperature": options.get("temperature", 0),
    }
    fmt = kwargs.get("format")
    if fmt is not None:
        # Ollama 'json' or a JSON schema -> json_object mode. With a schema we
        # also pin the schema into the conversation so the model follows it.
        payload["response_format"] = {"type": "json_object"}
        schema = fmt if isinstance(fmt, dict) else None
        if schema is not None:
            payload["messages"] = list(payload["messages"]) + [{
                "role": "user",
                "content": "Return ONLY a JSON object matching exactly this JSON schema:\n"
                           + json.dumps(schema),
            }]
    data = _post("/chat/completions", payload, timeout=120.0)

    try:
        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
    except (KeyError, IndexError, TypeError) as e:
        raise ProviderError(f"unexpected chat response shape: {e}") from e
    usage = data.get("usage") or {}
    elapsed_ns = data.get("_elapsed_ns", 0)
    return {
        "message": {"content": content},
        # Ollama-style timing fields (ns) so vlm.py logging stays key-safe.
        "total_duration": elapsed_ns,
        "load_duration": 0,
        "prompt_eval_duration": 0,
        "eval_duration": elapsed_ns,
        "eval_count": usage.get("completion_tokens", 0) or 0,
    }


def embed_api(**kwargs) -> dict:
    """OpenAI-compatible /embeddings. Returns ollama-shape {"embeddings": [...]}."""
    texts_in = kwargs.get("input") or []
    payload = {
        "model": _effective("embed_model", DEFAULT_EMBED_MODEL),
        "input": texts_in,
    }
    data = _post("/embeddings", payload, timeout=60.0)
    try:
        rows = sorted(data["data"], key=lambda d: d.get("index", 0))
        embeddings = [row["embedding"] for row in rows]
    except (KeyError, TypeError) as e:
        raise ProviderError(f"unexpected embeddings response shape: {e}") from e
    if len(embeddings) != len(texts_in):
        raise ProviderError(
            f"embeddings count mismatch: sent {len(texts_in)}, got {len(embeddings)}")
    return {"embeddings": embeddings}


# --- Unified entry points used by vlm.py / retriever.py ---

def chat(**kwargs) -> dict:
    """Route one chat call to the active provider."""
    if _state["provider"] == "api":
        return chat_api(**kwargs)
    import ollama  # deferred so api-only machines can still import this module
    return ollama.chat(**kwargs)


def embed(**kwargs) -> dict:
    """Route one embedding call to the active provider."""
    if _state["provider"] == "api":
        return embed_api(**kwargs)
    import ollama
    return ollama.embed(**kwargs)
