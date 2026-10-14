#!/usr/bin/env python3
"""Offline checks for the cloud-API fallback (no Ollama, no real API needed).

Fakes the OpenAI-compatible HTTP layer so we can verify:
  1. Switching provider to "api" via /llm/config requires a key and persists.
  2. /llm/status reports the provider without ever leaking the key.
  3. vlm.extract() works end-to-end through the cloud adapter
     (Ollama-style images get converted to base64 data URIs, JSON schema is
     appended, and the reply is parsed into an Extraction).
  4. explain() fallback works through the cloud adapter (JSON map reply).
  5. Embeddings route through the cloud adapter (ollama-style {"embeddings"}).
  6. Switching back to "local" restores Ollama as the active backend.
Run from the project root: python test_llm_api_fallback_check.py
"""
import base64
import json

import llm_provider as lp
import retriever
import vlm


FAKE_BASE = "https://fake-provider.test/v1"


def make_client(fake_store: dict):
    """Build a requests-like object standing in for llm_provider.requests.post."""
    import requests

    def fake_post(url, json=None, headers=None, timeout=None):
        fake_post.calls.append({"url": url, "json": json, "headers": headers})
        if "no-auth" in url:
            return _resp(200, {"choices": [{"message": {"content": "hi"}}]}) if False else _resp(401, {"error": "x"})
        if url.endswith("/chat/completions"):
            return _resp(200, fake_store.get("chat_response"))
        if url.endswith("/embeddings"):
            return _resp(200, fake_store.get("embed_response"))
        return _resp(404, {"error": "unknown endpoint"})

    fake_post.calls = []
    return fake_post


def _resp(status: int, payload: dict):
    import requests
    r = requests.Response()
    r.status_code = status
    r._content = json.dumps(payload).encode()
    r.headers["Content-Type"] = "application/json"
    return r


def fake_chat_payload(content: str) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"completion_tokens": 42},
    }


def fake_extract_content() -> str:
    return json.dumps({
        "answer": "The hemoglobin result is 9.8 g/dL, below the reference range.",
        "findings": [{
            "name": "Hemoglobin", "value": "9.8", "unit": "g/dL",
            "reference_range": "12-16", "flag": "LOW", "page": 1, "bbox": [],
        }],
    })


def main():
    failures = []

    def check(name, cond):
        status = "PASS" if cond else "FAIL"
        print(f"{name}: {status}")
        if not cond:
            failures.append(name)

    fake_store = {"chat_response": {}, "embed_response": {}}
    fake_post = make_client(fake_store)

    # --- 0. Force api mode with a fake key and fake endpoint ---
    lp.configure("api", api_key="sk-test-123", base_url=FAKE_BASE,
                 model="test-vision-model", embed_model="test-embed-model")
    check("configure: provider is api", lp.status()["provider"] == "api")
    check("configure: chat model stored", lp.status()["chat_model"] == "test-vision-model")
    check("configure: base url stored", lp.status()["base_url"] == FAKE_BASE)

    import requests
    orig_post = requests.post
    requests.post = fake_post

    # --- 1. Key is never leaked in status and always sent as a Bearer token ---
    s = lp.status()
    check("status: key not leaked", "sk-test-123" not in json.dumps(s))
    check("status: api_key_set flag", s["api_key_set"] is True)

    fake_store["embed_response"] = {
        "data": [{"index": 0, "embedding": [0.1, 0.2]}]
    }
    lp.embed(input=["probe"])
    hdr = fake_post.calls[-1]["headers"]
    check("headers: bearer key sent", hdr.get("Authorization") == "Bearer sk-test-123")

    # --- 2. Reject switching to api when there is genuinely no key anywhere ---
    saved_key = lp._state["api_key"]
    lp._state["api_key"] = ""
    try:
        lp.configure("api", api_key="  ")
        check("no-key switch rejected", False)
    except ValueError:
        check("no-key switch rejected", True)
    finally:
        lp._state["api_key"] = saved_key
    check("state: still api (reconfigure kept old key)", lp.status()["provider"] == "api")

    # --- 3. extract() end-to-end through the cloud adapter ---
    vlm_path = "test_fallback_min.png"
    from PIL import Image
    Image.new("RGB", (16, 16), "white").save(vlm_path)
    question = "what does the fallback test value mean?"
    # Drop any stale cache entry so the fake chat is actually exercised.
    vlm.vlm_cache._backend.delete(
        vlm.vlm_cache.key(lp.chat_model(), question, [vlm._page_hash(vlm_path)]))
    fake_store["chat_response"] = fake_chat_payload(fake_extract_content())
    try:
        result = vlm.extract(question, [{"page": 1, "image_path": vlm_path}])
        check("extract: parsed through api", result.findings[0].name == "Hemoglobin")
        chat_call = next(c for c in fake_post.calls if c["url"].endswith("/chat/completions"))
        sent_msgs = chat_call["json"]["messages"][0]["content"]
        parts = sent_msgs if isinstance(sent_msgs, list) else []
        img_parts = [p for p in parts if p.get("type") == "image_url"]
        data_uri = img_parts[0]["image_url"]["url"] if img_parts else ""
        check("extract: image sent as data URI", data_uri.startswith("data:image/png;base64,"))
        decoded = base64.b64decode(data_uri.split(",", 1)[1]) if img_parts else b""
        check("extract: data URI is a PNG", decoded[:8] == b"\x89PNG\r\n\x1a\n")
        check("extract: model used from config",
              chat_call["json"]["model"] == "test-vision-model")
        schema_part = chat_call["json"]["messages"][-1]["content"]
        check("extract: schema pinned into prompt", '"name"' in schema_part and "findings" in schema_part)
        check("extract: response_format json", chat_call["json"]["response_format"] == {"type": "json_object"})
    finally:
        import os
        try:
            os.remove(vlm_path)
        except OSError:
            pass

    # --- 4. explain() fallback through the cloud adapter ---
    fake_store["chat_response"] = fake_chat_payload(json.dumps(
        {"zzFallbackProbeTerm": "It measures how well the fallback adapter works."}))
    # Fresh term name -> no dictionary hit -> forced VLM fallback.
    class FakeFinding:
        name = "zzFallbackProbeTerm"
    vlm.explain_cache._backend.delete(vlm.explain_cache.key(["zzFallbackProbeTerm"]))
    sentences = vlm.explain([FakeFinding()])
    check("explain: vlm fallback text via api",
          sentences.get("zzFallbackProbeTerm") == "It measures how well the fallback adapter works.")

    # --- 5. Embeddings through the cloud adapter (2 texts, out-of-order data) ---
    fake_store["embed_response"] = {
        "data": [
            {"index": 1, "embedding": [3.0, 4.0]},
            {"index": 0, "embedding": [1.0, 2.0]},
        ]
    }
    out = retriever._embed_with_retry(["q one", "q two"], "search_document")
    check("embed: two texts -> two vectors", len(out) == 2 and all(len(v) == 2 for v in out))
    check("embed: vectors routed by index order", out[0] == [1.0, 2.0] and out[1] == [3.0, 4.0])

    # --- 6. Upstream errors surface as retryable provider errors ---
    lp.requests.post = orig_post  # file system only; keep proxy clean
    try:
        lp.embed(input=["fail"])
        check("embed failure raised", False)
    except Exception as e:
        check("embed failure raised",
              isinstance(e, (lp.ProviderError, requests.exceptions.RequestException,
                             ConnectionError, TimeoutError)))

    # --- 7. /llm/config + /llm/status through the real FastAPI app ---
    import main as main_mod
    from fastapi.testclient import TestClient
    client = TestClient(main_mod.app)
    r = client.post("/llm/config", json={"provider": "nonsense"})
    check("config: 422 on bad provider", r.status_code == 422)
    r = client.get("/llm/status")
    check("status endpoint: 200 + masked key", r.status_code == 200 and "sk-test-123" not in r.text)
    # Restore local mode via the endpoint itself, then confirm.
    r = client.post("/llm/config", json={"provider": "local"})
    check("config: back to local", r.status_code == 200 and r.json()["provider"] == "local")
    check("status: ollama flag present", "ollama_reachable" in client.get("/llm/status").json())
    requests.post = orig_post

    print(f"\n{len(failures)} failure(s)" if failures else "\nAll checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
