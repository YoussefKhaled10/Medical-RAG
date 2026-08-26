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
from frontend.components.chat import (
    add_message,
    render_chat_bottom_anchor,
    render_chat_interface,
)
from frontend.components.ingestion import render_sidebar_ingestion

st.set_page_config(
    page_title="RecoveryPath AI",
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
        "project_id": None,
        "asset_id": None,
        "search_scope": "all_projects",
        "active_query_mode": "global_kb",
        "ephemeral_uploaded_doc": None,
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


def build_conversation_history() -> list[dict[str, str]]:
    """Return recent turns before the latest queued user message."""
    messages = st.session_state.get("messages", [])
    if messages and messages[-1].get("role") == "user":
        candidates = messages[:-1]
    else:
        candidates = messages

    history: list[dict[str, str]] = []
    for message in candidates[-6:]:
        role = str(message.get("role") or "").strip().lower()
        content = " ".join(str(message.get("content") or "").split()).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        history.append({"role": role, "content": content[:1500]})
    return history


def call_rag_api(client: APIClient, question: str) -> dict[str, Any]:
    active_mode = st.session_state.get("active_query_mode", "global_kb")
    ephemeral_doc = st.session_state.get("ephemeral_uploaded_doc")

    # User mode automatically searches the trusted global knowledge base
    # together with the temporary uploaded document. The document remains
    # session-only and is never written to the database.
    if ephemeral_doc is not None and not st.session_state.get("developer_mode", False):
        return client.ask_document(
            question=question,
            file_bytes=ephemeral_doc["bytes"],
            file_name=ephemeral_doc["name"],
            generation_provider=st.session_state.generation_provider,
            temperature=0.0,
            max_output_tokens=1200,
            conversation_history=build_conversation_history(),
            include_global_knowledge=True,
        )

    # Developer mode keeps the previous explicit uploaded-document-only mode.
    if active_mode == "uploaded_doc" and ephemeral_doc is not None:
        return client.ask_document(
            question=question,
            file_bytes=ephemeral_doc["bytes"],
            file_name=ephemeral_doc["name"],
            generation_provider=st.session_state.generation_provider,
            temperature=0.0,
            max_output_tokens=1200,
            conversation_history=build_conversation_history(),
            include_global_knowledge=False,
        )

    search_scope = st.session_state.get("search_scope", "all_projects")
    project_id = st.session_state.get("project_id")
    asset_id = st.session_state.get("asset_id")

    if search_scope == "all_projects":
        selected_project_id = None
        selected_asset_id = None
    elif search_scope == "project":
        selected_project_id = int(project_id) if project_id is not None else None
        selected_asset_id = None
    else:
        selected_project_id = int(project_id) if project_id is not None else None
        selected_asset_id = int(asset_id) if asset_id is not None else None

    kwargs = {
        "question": question,
        "conversation_history": build_conversation_history(),
        "project_id": selected_project_id,
        "asset_id": selected_asset_id,
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
        accepted = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
        }
        return method(**accepted)
    raise AttributeError(
        "APIClient must expose ask_rag(), ask_question(), or ask()."
    )


def render_scope_selector() -> None:
    if not st.session_state.get("developer_mode", False):
        st.session_state.active_query_mode = "combined" if (
            st.session_state.get("ephemeral_uploaded_doc") is not None
        ) else "global_kb"
        return

    ephemeral_doc = st.session_state.get("ephemeral_uploaded_doc")
    if ephemeral_doc is not None:
        doc_name = ephemeral_doc.get("name", "Document")
        options = [
            "🌐 الموسوعة الطبية الشاملة (جميع المراجع المعتمدة)",
            f"📄 الملف المرفوع فقط ({doc_name})",
        ]
        current = st.session_state.get("active_query_mode", "uploaded_doc")
        idx = 1 if current == "uploaded_doc" else 0
        selected = st.radio(
            "نطاق البحث والإجابة:",
            options,
            index=idx,
            horizontal=True,
            key="scope_selector_radio",
        )
        st.session_state.active_query_mode = "uploaded_doc" if selected == options[1] else "global_kb"
    else:
        st.session_state.active_query_mode = "global_kb"


def render_developer_settings() -> None:
    if not st.session_state.developer_mode:
        return

    with st.expander("Developer settings", expanded=False):
        scope_names = [
            "All projects",
            "Single project",
            "Single document",
        ]
        scope_indexes = {
            "all_projects": 0,
            "project": 1,
            "document": 2,
        }
        current_scope = st.session_state.get(
            "search_scope",
            "all_projects",
        )
        selected_scope = st.radio(
            "Search scope",
            scope_names,
            index=scope_indexes.get(current_scope, 0),
            key="dev_search_scope",
        )

        if selected_scope == "All projects":
            st.session_state.search_scope = "all_projects"
            st.session_state.project_id = None
            st.session_state.asset_id = None
            st.info("Searching every indexed project and document.")

        elif selected_scope == "Single project":
            st.session_state.search_scope = "project"
            default_project_id = st.session_state.project_id or 2
            st.session_state.project_id = int(
                st.number_input(
                    "Project ID",
                    min_value=1,
                    value=int(default_project_id),
                    step=1,
                    key="dev_project_id",
                )
            )
            st.session_state.asset_id = None

        else:
            st.session_state.search_scope = "document"
            default_project_id = st.session_state.project_id or 2
            default_asset_id = st.session_state.asset_id or 1
            st.session_state.project_id = int(
                st.number_input(
                    "Project ID",
                    min_value=1,
                    value=int(default_project_id),
                    step=1,
                    key="dev_project_id",
                )
            )
            st.session_state.asset_id = int(
                st.number_input(
                    "Asset ID",
                    min_value=1,
                    value=int(default_asset_id),
                    step=1,
                    key="dev_asset_id",
                )
            )

        providers = ["groq", "glm", "gemini", "manus"]
        current = st.session_state.generation_provider
        st.session_state.generation_provider = st.selectbox(
            "Generation provider",
            providers,
            index=providers.index(current) if current in providers else 0,
            key="dev_generation_provider",
        )

def render_sidebar(client: APIClient) -> None:
    with st.sidebar:
        st.markdown(
            f'''
            <div class="sidebar-brand-v2">
                <img
                    class="sidebar-brand-logo"
                    src="{logo_uri()}"
                    alt="RecoveryPath AI logo"
                >
                <div class="sidebar-brand-copy">
                    <strong>RecoveryPath AI</strong>
                    <span>Evidence-based recovery support</span>
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        # Developer Mode is hidden by default.
        # Set SHOW_DEVELOPER_MODE=true locally to display it.
        show_developer_mode = (
            os.getenv(
                "SHOW_DEVELOPER_MODE",
                "true",
            )
            .strip()
            .lower()
            == "true"
        )

        if show_developer_mode:
            st.session_state.developer_mode = st.toggle(
                "Developer mode",
                value=bool(
                    st.session_state.get(
                        "developer_mode",
                        False,
                    )
                ),
                key="developer_mode_toggle",
                help=(
                    "Show project IDs, citations, retrieval scores, "
                    "claims, and raw diagnostics."
                ),
            )

            render_developer_settings()

        else:
            # User Mode searches all indexed projects and documents.
            st.session_state.developer_mode = False
            st.session_state.search_scope = "all_projects"
            st.session_state.project_id = None
            st.session_state.asset_id = None

        if st.button(
            "＋  New conversation",
            type="primary",
            use_container_width=True,
            key="new_conversation_button",
        ):
            new_conversation()
            st.rerun()

        st.markdown(
            '''
            <div class="sidebar-divider"></div>
            <div class="sidebar-section-label">
                CHAT HISTORY
            </div>
            ''',
            unsafe_allow_html=True,
        )

        search = st.text_input(
            "Search conversations",
            placeholder="Search conversations…",
            label_visibility="collapsed",
            key="history_search",
        )

        conversations = sorted(
            st.session_state.conversations.items(),
            key=lambda item: item[1].get(
                "updated_at",
                "",
            ),
            reverse=True,
        )

        for conv_id, item in conversations:
            title = str(
                item.get(
                    "title",
                    "Conversation",
                )
            )

            if search.casefold() not in title.casefold():
                continue

            open_col, delete_col = st.columns(
                [5.2, 1]
            )

            with open_col:
                updated_at = item.get(
                    "updated_at",
                    "",
                )

                label = (
                    f" {title}\n\n"
                    f"{relative_date(updated_at)}"
                )

                if st.button(
                    label,
                    key=f"open_{conv_id}",
                    use_container_width=True,
                ):
                    load_conversation(conv_id)
                    st.rerun()

            with delete_col:
                if st.button(
                    "×",
                    key=f"delete_{conv_id}",
                    help="Delete conversation",
                ):
                    delete_conversation(conv_id)
                    st.rerun()

        st.markdown(
            '<div class="sidebar-divider"></div>',
            unsafe_allow_html=True,
        )

        render_sidebar_ingestion(client)

        st.markdown(
            '''
            <div class="sidebar-notice">
                <strong>
                    Informational support only
                </strong>
                <p>
                    RecoveryPath AI does not replace a qualified
                    doctor, pharmacist, or emergency service.
                </p>
            </div>
            ''',
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
    render_scope_selector()
    render_floating_assistant()
    if st.session_state.get("messages"):
        st.markdown(
            '<a class="rp-chat-top-button" href="#rp-chat-top" '
            'title="Back to the first message" '
            'aria-label="Back to the first message">↑</a>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<a class="rp-chat-bottom-button" href="#rp-chat-bottom" '
            'title="Go to the latest message" '
            'aria-label="Go to the latest message">&#8595;</a>',
            unsafe_allow_html=True,
        )


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

            message = (
                "Request failed: "
                f"{type(exc).__name__}: {exc}"
            )

            add_message(
                st.session_state.messages,
                "assistant",
                message,
            )

            st.error(message)

    render_chat_bottom_anchor()
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
