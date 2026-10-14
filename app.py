import requests
import streamlit as st
import time

st.set_page_config(page_title="Document Intelligence System", layout="wide")
st.title("Document Intelligence System")
st.caption("Upload a PDF, then ask questions about it. Local Ollama, or a cloud API key — no Ollama needed.")

UPLOAD_URL = "http://127.0.0.1:8000/upload"
QUERY_URL = "http://127.0.0.1:8000/query"
TERMS_URL = "http://127.0.0.1:8000/terms"
LLM_CONFIG_URL = "http://127.0.0.1:8000/llm/config"
LLM_STATUS_URL = "http://127.0.0.1:8000/llm/status"


def fetch_llm_status():
    try:
        r = requests.get(LLM_STATUS_URL, timeout=10)
        return r.json() if r.status_code == 200 else None
    except requests.exceptions.ConnectionError:
        return None

with st.sidebar:
    st.header("Pipeline Settings")
    dpi_setting = st.slider("Render DPI", min_value=150, max_value=600, value=300, step=50)
    top_k = st.slider("Pages to read per question", min_value=1, max_value=3, value=2)

    st.divider()
    st.header("AI Backend")
    backend_mode = st.radio(
        "Choose how the VLM/answers run",
        ["Local Ollama", "Cloud API key (no Ollama needed)"],
        key="ai_backend_mode",
    )

    if backend_mode.startswith("Local"):
        status = fetch_llm_status()
        if status:
            if status.get("provider") == "api":
                st.info("Right now the cloud API is active. Ollama is not being used.")
            elif not status.get("ollama_reachable"):
                st.warning("Ollama is not reachable. Start Ollama, or switch to \"Cloud API key\" above.")
            else:
                st.success(f"Ollama reachable — model: {status.get('chat_model', '?')}")
    else:
        st.caption("Paste an API key from any OpenAI-compatible provider "
                   "(OpenAI, OpenRouter, Groq, Gemini, ...). The backend then does "
                   "all VLM + embedding work through that API instead of Ollama.")
        api_key = st.text_input("API key", type="password", key="cloud_api_key")
        base_url = st.text_input("Endpoint base URL", value="https://api.openai.com/v1",
                                 key="cloud_base_url")
        chat_model = st.text_input("Chat (VLM) model", value="gpt-4o-mini",
                                   key="cloud_chat_model")
        embed_model = st.text_input("Embedding model", value="text-embedding-3-small",
                                    key="cloud_embed_model")
        if st.button("Apply API key", type="primary"):
            if not api_key.strip():
                st.error("The API key field is empty.")
            else:
                try:
                    r = requests.post(
                        LLM_CONFIG_URL,
                        json={
                            "provider": "api",
                            "api_key": api_key.strip(),
                            "base_url": base_url.strip(),
                            "model": chat_model.strip(),
                            "embed_model": embed_model.strip(),
                        },
                        timeout=15,
                    )
                    if r.status_code == 200:
                        info = r.json()
                        st.success(f"Cloud API configured. Chat model: {info.get('chat_model')}")
                    else:
                        try:
                            detail = r.json().get("detail", r.text)
                        except Exception:
                            detail = r.text
                        st.error(f"Could not configure cloud API ({r.status_code}): {detail}")
                except requests.exceptions.ConnectionError:
                    st.error("Connection refused: start the backend with `python main.py` first.")

def fetch_term_index(doc_id, max_wait=120, poll_interval=3):
    """Poll for term index until ready or timeout."""
    start = time.time()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while time.time() - start < max_wait:
        elapsed = int(time.time() - start)
        progress = min(elapsed / max_wait, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Building medical term index... ({elapsed}s)")
        
        try:
            r = requests.get(f"{TERMS_URL}/{doc_id}", timeout=10)
            if r.status_code == 200:
                data = r.json()
                terms = data.get("terms", {})
                # Filter only terms with definitions
                terms_with_def = {k: v for k, v in terms.items() if v.get("definition")}
                progress_bar.progress(1.0)
                status_text.text(f"Term index ready: {len(terms_with_def)} terms found")
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                return terms_with_def
            elif r.status_code == 404:
                # Not ready yet, continue polling
                pass
            else:
                st.warning(f"Term index error: {r.status_code}")
                break
        except requests.exceptions.ConnectionError:
            pass
        
        time.sleep(poll_interval)
    
    progress_bar.empty()
    status_text.empty()
    st.warning("Term index generation timed out. Check backend logs.")
    return {}

uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

if uploaded_file is not None and st.button("Process Document", type="primary"):
    with st.spinner("Rendering and indexing pages..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        try:
            response = requests.post(
                UPLOAD_URL, files=files, params={"dpi": dpi_setting}, timeout=600
            )
            if response.status_code == 200:
                st.session_state["ingest"] = response.json()
                st.session_state.pop("answer", None)
            else:
                st.error(f"Backend API Error ({response.status_code}): {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Connection refused: start the backend with `python main.py` first.")

if "ingest" in st.session_state:
    data = st.session_state["ingest"]
    st.success(f"Indexed {data['total_pages']} page(s) of '{data['doc_id']}' at {data['dpi']} DPI.")

    cols = st.columns(min(data["total_pages"], 3))
    for idx, page_info in enumerate(data["pages"]):
        with cols[idx % 3]:
            st.subheader(f"Page {page_info['page']}")
            st.image(page_info["path"])
            st.caption(page_info["resolution"])

    st.divider()
    
    # Medical Term Index section
    st.subheader("📋 Medical Term Index")
    if "term_index" not in st.session_state or st.session_state.get("term_index_doc_id") != data["doc_id"]:
        with st.spinner("Extracting medical terms..."):
            terms = fetch_term_index(data["doc_id"])
            st.session_state["term_index"] = terms
            st.session_state["term_index_doc_id"] = data["doc_id"]
    else:
        terms = st.session_state["term_index"]
    
    if terms:
        # Build table rows
        term_rows = []
        for term_name, term_data in terms.items():
            term_rows.append({
                "Term": term_name,
                "Definition": term_data.get("definition", ""),
                "Pages": ", ".join(map(str, term_data.get("pages", []))),
                "Example Value": term_data.get("example_value", ""),
                "Unit": term_data.get("unit", ""),
                "Reference Range": term_data.get("reference_range", ""),
            })
        st.dataframe(term_rows, use_container_width=True, hide_index=True)
        st.caption(f"Total terms: {len(term_rows)}")
    else:
        st.info("No medical terms with definitions found yet. The index may still be generating.")
    
    st.divider()
    question = st.text_input("Ask a question about this document")
    if st.button("Ask") and question:
        with st.spinner("Searching pages and reading them with the VLM..."):
            try:
                r = requests.post(
                    QUERY_URL,
                    json={"doc_id": data["doc_id"], "question": question, "k": top_k},
                    timeout=300,
                )
                if r.status_code == 200:
                    st.session_state["answer"] = r.json()
                else:
                    st.error(f"Backend API Error ({r.status_code}): {r.text}")
            except requests.exceptions.ConnectionError:
                st.error("Connection refused: is the backend running?")

    if "answer" in st.session_state:
        ans = st.session_state["answer"]
        st.caption(f"Pages read: {ans['pages_used']}")
        st.write(ans["result"]["answer"])

        findings = ans["result"]["findings"]
        if findings:
            cols_order = ["name", "value", "unit", "reference_range", "status",
                          "what_it_measures", "page"]
            st.dataframe([{c: f.get(c) for c in cols_order} for f in findings])