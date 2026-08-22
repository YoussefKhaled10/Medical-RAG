import base64
import inspect
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.api_client import APIClient
from frontend.components.animated_assistant import render_floating_assistant
from frontend.components.chat import add_message, render_chat_interface
from frontend.components.ingestion import render_sidebar_ingestion

st.set_page_config(
    page_title="RecoveryPath AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

HISTORY_FILE = Path(__file__).parent / ".chat_history.json"
MAX_HISTORY = 30
SUGGESTIONS = [
    "ما أعراض الانسحاب من الكحول؟",
    "ما الأدوية التي يمكن استخدامها بعد الانسحاب الناجح؟",
    "What support may help prevent relapse?",
]


def load_css() -> None:
    path = Path(__file__).parent / "styles" / "custom.css"
    if path.exists():
        st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def logo_uri() -> str:
    raw = (Path(__file__).parent / "assets" / "recoverypath_logo.svg").read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(raw).decode("ascii")


def load_history() -> dict[str, Any]:
    if not HISTORY_FILE.exists():
        return {}
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return data.get("conversations", {}) if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_history() -> None:
    try:
        HISTORY_FILE.write_text(
            json.dumps(
                {"conversations": st.session_state.conversations},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def initialize_state() -> None:
    defaults = {
        "messages": [],
        "latest_response": None,
        "project_id": 2,
        "asset_id": 1,
        "search_scope": "project",
        "generation_provider": "groq",
        "developer_mode": False,
        "current_conv_id": None,
        "conversations": load_history(),
        "pending_question": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def conversation_title(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            text = " ".join(str(message.get("content", "")).split())
            if text:
                return text[:60] + ("…" if len(text) > 60 else "")
    return "New conversation"


def save_current_conversation() -> None:
    if not st.session_state.messages:
        return
    conv_id = st.session_state.current_conv_id or str(uuid.uuid4())
    st.session_state.current_conv_id = conv_id
    st.session_state.conversations[conv_id] = {
        "title": conversation_title(st.session_state.messages),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "messages": st.session_state.messages,
        "latest_response": st.session_state.latest_response,
    }
    ordered = sorted(
        st.session_state.conversations.items(),
        key=lambda item: item[1].get("updated_at", ""),
        reverse=True,
    )[:MAX_HISTORY]
    st.session_state.conversations = dict(ordered)
    save_history()


def new_conversation() -> None:
    save_current_conversation()
    st.session_state.messages = []
    st.session_state.latest_response = None
    st.session_state.current_conv_id = None
    st.session_state.pending_question = None


def load_conversation(conv_id: str) -> None:
    save_current_conversation()
    item = st.session_state.conversations.get(conv_id)
    if not item:
        return
    st.session_state.messages = list(item.get("messages") or [])
    st.session_state.latest_response = item.get("latest_response")
    st.session_state.current_conv_id = conv_id
    st.session_state.pending_question = None


def delete_conversation(conv_id: str) -> None:
    st.session_state.conversations.pop(conv_id, None)
    if st.session_state.current_conv_id == conv_id:
        new_conversation()
    save_history()


def relative_date(value: str) -> str:
    try:
        then = datetime.fromisoformat(value)
        now = datetime.now(then.tzinfo or timezone.utc)
        seconds = max(0, int((now - then).total_seconds()))
        if seconds < 60:
            return "Just now"
        if seconds < 3600:
            return f"{seconds // 60} min ago"
        if seconds < 86400:
            return f"{seconds // 3600} hr ago"
        if seconds < 172800:
            return "Yesterday"
        return f"{seconds // 86400} days ago"
    except Exception:
        return "Recent"


def queue_question(question: str) -> None:
    clean = " ".join(question.split()).strip()
    if clean:
        add_message(st.session_state.messages, "user", clean)
        st.session_state.pending_question = clean


def call_rag_api(client: APIClient, question: str) -> dict[str, Any]:
    kwargs = {
        "question": question,
        "project_id": int(st.session_state.project_id),
        "asset_id": int(st.session_state.asset_id) if st.session_state.search_scope == "document" else None,
        "retrieval_limit": 5,
        "generation_provider": st.session_state.generation_provider,
        "temperature": 0.0,
        "max_output_tokens": 1200,
        "timeout_seconds": 300.0,
    }
    for name in ("ask_rag", "ask_question", "ask"):
        method = getattr(client, name, None)
        if not callable(method):
            continue
        signature = inspect.signature(method)
        accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
        return method(**accepted)
    raise AttributeError("APIClient must expose ask_rag(), ask_question(), or ask().")


def render_developer_settings() -> None:
    if not st.session_state.developer_mode:
        return
    with st.expander("Developer settings", expanded=False):
        st.session_state.project_id = int(st.number_input(
            "Project ID", min_value=1, value=int(st.session_state.project_id), step=1, key="dev_project_id"
        ))
        st.session_state.asset_id = int(st.number_input(
            "Asset ID", min_value=1, value=int(st.session_state.asset_id), step=1, key="dev_asset_id"
        ))
        selected_scope = st.radio(
            "Search scope", ["Entire project", "Single document"],
            index=0 if st.session_state.search_scope == "project" else 1,
            key="dev_search_scope",
        )
        st.session_state.search_scope = "project" if selected_scope == "Entire project" else "document"
        providers = ["groq", "glm", "gemini", "manus"]
        current = st.session_state.generation_provider
        st.session_state.generation_provider = st.selectbox(
            "Generation provider", providers,
            index=providers.index(current) if current in providers else 0,
            key="dev_generation_provider",
        )


def render_sidebar(client: APIClient) -> None:
    with st.sidebar:
        st.markdown(
            f'''<div class="sidebar-brand-v2">
            <img class="sidebar-brand-logo" src="{logo_uri()}" alt="RecoveryPath AI logo">
            <div class="sidebar-brand-copy"><strong>RecoveryPath AI</strong><span>Evidence-based recovery support</span></div>
            </div>''',
            unsafe_allow_html=True,
        )

        # The developer toggle is created exactly once.
        st.session_state.developer_mode = st.toggle(
            "Developer mode",
            value=bool(st.session_state.developer_mode),
            key="developer_mode_toggle",
            help="Show project IDs, citations, retrieval scores, claims, and raw diagnostics.",
        )
        render_developer_settings()

        if st.button("＋  New conversation", type="primary", use_container_width=True, key="new_conversation_button"):
            new_conversation()
            st.rerun()

        st.markdown('<div class="sidebar-divider"></div><div class="sidebar-section-label">CHAT HISTORY</div>', unsafe_allow_html=True)
        search = st.text_input(
            "Search conversations", placeholder="Search conversations…",
            label_visibility="collapsed", key="history_search",
        )
        conversations = sorted(
            st.session_state.conversations.items(),
            key=lambda item: item[1].get("updated_at", ""), reverse=True,
        )
        for conv_id, item in conversations:
            if search.casefold() not in str(item.get("title", "")).casefold():
                continue
            open_col, delete_col = st.columns([5.2, 1])
            with open_col:
                label = f"💬  {item.get('title', 'Conversation')}\n\n{relative_date(item.get('updated_at', ''))}"
                if st.button(label, key=f"open_{conv_id}", use_container_width=True):
                    load_conversation(conv_id)
                    st.rerun()
            with delete_col:
                if st.button("×", key=f"delete_{conv_id}", help="Delete conversation"):
                    delete_conversation(conv_id)
                    st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        render_sidebar_ingestion(client)
        st.markdown(
            '<div class="sidebar-notice"><strong>Informational support only</strong><p>RecoveryPath AI does not replace a qualified doctor, pharmacist, or emergency service.</p></div>',
            unsafe_allow_html=True,
        )


def render_header() -> None:
    st.markdown(
        '''<header class="main-header">
        <div class="header-kicker">RECOVERY SUPPORT</div>
        <h1>Recovery guidance, grounded in evidence.</h1>
        <p>Ask naturally. RecoveryPath AI searches the available guidance, verifies the supporting evidence, and responds in your language.</p>
        </header>''',
        unsafe_allow_html=True,
    )


def main() -> None:
    load_css()
    initialize_state()
    api_url = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")
    try:
        if not os.getenv("BACKEND_API_URL") and "BACKEND_API_URL" in st.secrets:
            api_url = str(st.secrets["BACKEND_API_URL"])
    except Exception:
        pass
    client = APIClient(base_url=api_url)
    render_sidebar(client)
    render_header()
    render_floating_assistant()

    if not st.session_state.messages:
        st.markdown('<div class="suggestions-label">Try a question</div>', unsafe_allow_html=True)
        columns = st.columns(2)
        for index, suggestion in enumerate(SUGGESTIONS):
            with columns[index % 2]:
                if st.button(suggestion, key=f"suggestion_{index}", use_container_width=True):
                    queue_question(suggestion)
                    st.rerun()
    else:
        render_chat_interface(st.session_state.messages)

    pending = st.session_state.pending_question
    if pending:
        try:
            st.markdown("""<div class="rp-processing-card"><div class="rp-processing-bot"><span></span><i></i><b></b></div><div><strong>Checking the available evidence...</strong><small>Searching trusted sources and preparing a safe answer.</small><div class="rp-processing-dots"><em></em><em></em><em></em></div></div></div>""", unsafe_allow_html=True)
            response = call_rag_api(client, pending)
            answer = str(response.get("answer") or response.get("recommendation") or "")
            add_message(st.session_state.messages, "assistant", answer, response)
            st.session_state.latest_response = response
            st.session_state.pending_question = None
            save_current_conversation()
            st.rerun()
        except Exception as exc:
            st.session_state.pending_question = None
            message = "I couldn't complete the request right now. Please try again."
            if st.session_state.developer_mode:
                message += f"\n\nDeveloper detail: {exc}"
            add_message(st.session_state.messages, "assistant", message)
            st.error(message)

    prompt = st.chat_input("Ask anything about alcohol recovery…")
    if prompt:
        queue_question(prompt)
        st.rerun()

    st.markdown(
        '<div class="page-footer">RecoveryPath AI provides evidence-based information and does not replace professional care.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
