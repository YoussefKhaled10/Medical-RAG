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
from frontend.components.auth import (
    render_login_page,
    render_otp_page,
    render_register_page,
    render_user_profile_sidebar,
)
from frontend.components.chat import (
    add_message,
    render_chat_bottom_anchor,
    render_chat_interface,
    render_streaming_assistant,
)
from frontend.components.ingestion import render_sidebar_ingestion



st.set_page_config(
    page_title="RecoveryPath AI",
    layout="wide",
    initial_sidebar_state="expanded",
)
MAX_HISTORY = 30
SUGGESTIONS = [
    "ما أعراض الانسحاب من الكحول؟",
    "ما الأدوية التي يمكن استخدامها بعد الانسحاب الناجح؟",
    "What support may help prevent relapse?",
]


def get_boolean_setting(
    name: str,
    default: bool = False,
) -> bool:
    """Read a boolean from Streamlit Secrets first, then the environment."""
    raw_value: Any = None

    try:
        raw_value = st.secrets.get(name, None)
    except Exception:
        raw_value = None

    if raw_value is None:
        raw_value = os.getenv(name, str(default))

    return str(raw_value).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def load_css() -> None:
    path = Path(__file__).parent / "styles" / "custom.css"
    if path.exists():
        st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        [data-testid="stChatInput"] {
            background: linear-gradient(135deg, #dffbf7 0%, #eef9ff 100%) !important;
            border: 2px solid #16a6a1 !important;
            border-radius: 22px !important;
            box-shadow: 0 12px 30px rgba(22, 166, 161, 0.18) !important;
            transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease !important;
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: #087f7a !important;
            box-shadow: 0 0 0 5px rgba(22, 166, 161, 0.18),
                        0 16px 36px rgba(8, 127, 122, 0.22) !important;
            transform: translateY(-1px) !important;
        }
        [data-testid="stChatInput"] textarea {
            color: #123b5d !important;
            -webkit-text-fill-color: #123b5d !important;
            caret-color: #087f7a !important;
            font-weight: 650 !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: #3f6f84 !important;
            -webkit-text-fill-color: #3f6f84 !important;
            opacity: 1 !important;
        }
        [data-testid="stChatInput"] button {
            background: #0f9f99 !important;
            color: #ffffff !important;
            border-radius: 13px !important;
        }
        [data-testid="stChatInput"] button:hover {
            background: #087f7a !important;
        }
        .rp-streaming-answer {
            border-left: 3px solid #23c7bd !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def logo_uri() -> str:
    raw = (Path(__file__).parent / "assets" / "recoverypath_logo.svg").read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(raw).decode("ascii")


def load_history() -> dict[str, Any]:
    """Never load shared conversations from disk."""
    return {}



def save_history() -> None:
    """Chat history is session-only and is never written to disk."""
    return None



def initialize_state() -> None:
    defaults = {
        "messages": [],
        "latest_response": None,
        "ephemeral_uploaded_doc": None,
        "user": None,
        "auth_token": None,
        "refresh_token": None,
        "current_page": "app",
        "pending_registration": None,
        "generation_provider": "groq",
        "developer_mode": False,
        "current_conv_id": None,
        "conversations": {},
        "conversations_synced": False,
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
    
    # If signed in, we need to fetch messages from DB
    if st.session_state.get("auth_token") and str(conv_id).isdigit():
        import os
        api_url = str(os.getenv("BACKEND_API_URL")).rstrip("/") if os.getenv("BACKEND_API_URL") else "http://127.0.0.1:8000"
        temp_client = APIClient(base_url=api_url, auth_token=st.session_state.get("auth_token"))
        try:
            msg_data = temp_client.get_messages(int(conv_id), limit=100)
            # Backend returns items in descending order (newest first). Let's reverse them for UI.
            # Wait, backend list_messages might return newest first, we'll sort by created_at.
            db_msgs = msg_data.get("items", [])
            db_msgs = sorted(db_msgs, key=lambda x: x.get("created_at", ""))
            
            # Format to match frontend structure
            formatted_msgs = []
            for m in db_msgs:
                formatted_msgs.append({
                    "role": m["role"],
                    "content": m["content"],
                    "sources": m.get("sources", []),
                    "metadata": m.get("metadata", {})
                })
            st.session_state.messages = formatted_msgs
        except Exception:
            st.session_state.messages = list(item.get("messages") or [])
    else:
        st.session_state.messages = list(item.get("messages") or [])
        
    st.session_state.latest_response = item.get("latest_response")
    st.session_state.current_conv_id = conv_id
    st.session_state.pending_question = None


def delete_conversation(conv_id: str) -> None:
    if st.session_state.get("auth_token") and str(conv_id).isdigit():
        import os
        api_url = str(os.getenv("BACKEND_API_URL")).rstrip("/") if os.getenv("BACKEND_API_URL") else "http://127.0.0.1:8000"
        temp_client = APIClient(base_url=api_url, auth_token=st.session_state.get("auth_token"))
        try:
            temp_client.delete_conversation(int(conv_id))
        except Exception:
            pass
    st.session_state.conversations.pop(conv_id, None)
    if st.session_state.current_conv_id == conv_id:
        st.session_state.current_conv_id = None
        st.session_state.messages = []
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
    """Return up to the latest 4 user and 4 assistant messages."""
    messages = st.session_state.get("messages", [])
    if messages and str(messages[-1].get("role") or "").lower() == "user":
        candidates = messages[:-1]
    else:
        candidates = messages

    selected_reversed: list[dict[str, str]] = []
    role_counts = {"user": 0, "assistant": 0}
    for message in reversed(candidates):
        role = str(message.get("role") or "").strip().lower()
        if role not in role_counts or role_counts[role] >= 4:
            continue
        content = " ".join(str(message.get("content") or "").split()).strip()
        if not content or content.lower().startswith("request failed:"):
            continue
        selected_reversed.append({"role": role, "content": content[:1500]})
        role_counts[role] += 1
        if role_counts["user"] == 4 and role_counts["assistant"] == 4:
            break
    return list(reversed(selected_reversed))


def call_rag_api(client: APIClient, question: str) -> dict[str, Any]:
    """Answer from System Knowledge only, unless a temporary file is uploaded."""
    ephemeral_doc = st.session_state.get("ephemeral_uploaded_doc")

    # A temporary upload is queried directly and remains session-only.
    if ephemeral_doc is not None:
        return client.ask_document(
            question=question,
            file_bytes=ephemeral_doc["bytes"],
            file_name=ephemeral_doc["name"],
            generation_provider=st.session_state.generation_provider,
            temperature=0.0,
            max_output_tokens=1200,
            conversation_history=build_conversation_history(),
        )

    # Every normal question uses System Knowledge automatically.
    if st.session_state.get("auth_token"):
        conv_id_str = st.session_state.get("current_conv_id")
        if not conv_id_str or not str(conv_id_str).isdigit():
            try:
                new_conv = client.create_conversation(
                    title=question[:50] or "New Conversation"
                )
                conv_id = int(new_conv["id"])
                st.session_state.current_conv_id = str(conv_id)
                st.session_state.conversations[str(conv_id)] = new_conv
            except Exception as exc:
                st.error(f"Failed to create conversation: {exc}")
                raise
        else:
            conv_id = int(conv_id_str)

        request_id = str(uuid.uuid4())
        try:
            result = client.send_message_to_conversation(
                conversation_id=conv_id,
                question=question,
                client_request_id=request_id,
                generation_provider=st.session_state.generation_provider,
                retrieval_limit=10,
                temperature=0.0,
                max_output_tokens=1200,
            )
            st.session_state.conversations[str(conv_id)] = result["conversation"]
            return result["rag"]
        except Exception as exc:
            st.error(f"Failed to send message: {exc}")
            raise

    return client.ask_rag(
        question=question,
        project_id=2,
        search_scope="system",
        conversation_history=build_conversation_history(),
        retrieval_limit=10,
        retrieval_mode="hybrid",
        generation_provider=st.session_state.generation_provider,
        temperature=0.0,
        max_output_tokens=1200,
        timeout_seconds=300.0,
    )


def render_developer_settings() -> None:
    if not st.session_state.developer_mode:
        return
    with st.expander("Developer settings", expanded=False):
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

        render_user_profile_sidebar(client)

        # Hidden by default in production. It can be enabled locally or in

        # Streamlit Secrets with SHOW_DEVELOPER_MODE=true.
        show_developer_mode = get_boolean_setting(
            "SHOW_DEVELOPER_MODE",
            False,
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
            st.session_state.developer_mode = False

        if st.button(
            "+ New conversation",
            type="primary",
            use_container_width=True,
            key="new_conversation_button",
        ):
            new_conversation()
            st.rerun()

        st.markdown(
            '''
            <div class="sidebar-divider"></div>
            <div class="rp-sidebar-section-title">
                CHAT HISTORY
            </div>
            ''',
            unsafe_allow_html=True,
        )

        search = st.text_input(
            "Search conversations",
            placeholder="Search conversations...",
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
                    f"{title}\n\n"
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
                    "X",
                    key=f"delete_{conv_id}",
                    help="Delete conversation",
                ):
                    delete_conversation(conv_id)
                    st.rerun()

        st.markdown(
            '''
            <div class="sidebar-divider"></div>
            <div class="rp-sidebar-section-title">
                DOCUMENT ANALYSIS & VAULT
            </div>
            <p style="font-size: 0.76rem; color: #94a3b8; line-height: 1.45; margin: 0 0 10px 0;">
                Upload a medical report or guideline (PDF or TXT) to query it directly.
            </p>
            ''',
            unsafe_allow_html=True,
        )

        render_sidebar_ingestion(client)

        st.markdown(
            '''
            <div class="rp-sidebar-footer">
                &copy; 2025 RecoveryPath AI
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
    initialize_state()
    api_url = "http://127.0.0.1:8000"
    try:
        if "BACKEND_API_URL" in st.secrets:
            api_url = str(st.secrets["BACKEND_API_URL"]).rstrip("/")
        elif os.getenv("BACKEND_API_URL"):
            api_url = str(os.getenv("BACKEND_API_URL")).rstrip("/")
    except Exception:
        if os.getenv("BACKEND_API_URL"):
            api_url = str(os.getenv("BACKEND_API_URL")).rstrip("/")
    token = st.session_state.get("auth_token")
    client = APIClient(base_url=api_url, auth_token=token)

    # Sync conversations if logged in
    if token and not st.session_state.get("conversations_synced"):
        try:
            convs = client.list_conversations(limit=50)
            st.session_state.conversations = {str(c["id"]): c for c in convs.get("items", [])}
            st.session_state.conversations_synced = True
        except Exception:
            pass

    # Route to standalone login / register / otp pages
    current_page = st.session_state.get("current_page", "app")
    if current_page == "login":
        render_login_page(client)
        return
    elif current_page == "register":
        render_register_page(client)
        return
    elif current_page == "otp":
        render_otp_page(client)
        return

    # Load the chat theme only after standalone auth routing.
    load_css()
    render_sidebar(client)

    render_header()
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
            render_streaming_assistant(
                answer,
                language=str(response.get("answer_language") or "") or None,
                delay_seconds=0.025,
            )
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
