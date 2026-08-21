import json
import os
import re
import sys
import time
import random
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.api_client import APIClient
from frontend.components.chat import (
    _render_details,
    _render_metadata,
    _safe_html,
    add_message,
    render_chat_interface,
    stream_new_message,
)
from frontend.components.ingestion import render_sidebar_ingestion


st.set_page_config(
    page_title="RecoveryPath AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Constants ────────────────────────────────────────────────────────────────

HISTORY_FILE = Path(__file__).parent / ".chat_history.json"
MAX_SAVED_CONVERSATIONS = 30
MAX_TITLE_LENGTH = 55


# ─── Small-talk detection ─────────────────────────────────────────────────────

_ARABIC_CHARS = re.compile(r"[\u0600-\u06FF]")

_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"^(hi|hello|hey|howdy|greetings?|good\s*(morning|afternoon|evening|night|day))[\s!.؟?]*$",
            re.I | re.U,
        ),
        "greeting",
    ),
    (
        re.compile(
            r"^(مرحبا?|اهلا?|أهلا?|هلا|هاي|هالو|السلام\s*عليكم|وعليكم\s*السلام"
            r"|صباح\s*الخير|مساء\s*الخير|صباح\s*النور|مساء\s*النور)[\s!.؟?]*$",
            re.I | re.U,
        ),
        "greeting",
    ),
    (
        re.compile(
            r"^(how\s+are\s+you|how\s+is\s+it\s+going|how\s+do\s+you\s+do"
            r"|كيف\s*(حالك|الحال|أحوالك)|عامل\s*ايه|كيفك|شو\s*حالك)[\s?!.؟]*$",
            re.I | re.U,
        ),
        "how_are_you",
    ),
    (
        re.compile(
            r"^(thanks?|thank\s+you|thx|ty|شكرا?ً?|مشكور|يسلمو|ممنون)[\s!.؟?]*$",
            re.I | re.U,
        ),
        "thanks",
    ),
    (
        re.compile(
            r"^(bye|goodbye|see\s+you|cya|take\s+care|باي|مع\s*السلامة|تصبح\s*على\s*خير|وداعا?ً?)[\s!.؟?]*$",
            re.I | re.U,
        ),
        "farewell",
    ),
    (
        re.compile(
            r"^(who\s+are\s+you|what\s+are\s+you|ما\s*اسمك|من\s*(انت|أنت)|ايه\s*(ده|انت\s*ده))[\s?!.؟?]*$",
            re.I | re.U,
        ),
        "who_are_you",
    ),
    (
        re.compile(
            r"^(ok|okay|alright|sure|cool|great|nice|awesome|perfect"
            r"|اوكي|تمام|حسنا?ً?|عظيم|رائع|صح|زين)[\s!.؟?]*$",
            re.I | re.U,
        ),
        "ok",
    ),
]

_REPLIES: dict[str, dict[str, list[str]]] = {
    "en": {
        "greeting": [
            "Hello! 👋 I'm **RecoveryPath AI**, your evidence-based alcohol recovery assistant. Feel free to ask about withdrawal support, relapse prevention, assessments, or recovery care.",
            "Hi there! 😊 I'm RecoveryPath AI — here to provide evidence-based guidance on alcohol recovery. What would you like to know?",
        ],
        "how_are_you": [
            "Doing well, thanks for asking! 😊 Ready to help with any questions about alcohol recovery. What's on your mind?",
        ],
        "thanks": [
            "You're welcome! 😊 Feel free to ask anything else about recovery.",
            "Happy to help! Let me know if you have more questions.",
        ],
        "farewell": [
            "Take care! 👋 I'm here whenever you need evidence-based guidance on alcohol recovery.",
        ],
        "who_are_you": [
            "I'm **RecoveryPath AI** — an evidence-based assistant specialized in alcohol recovery guidance. I draw answers directly from indexed clinical guidelines and research. How can I help? 🤖",
        ],
        "ok": [
            "Great! 😊 Let me know if you have any questions about alcohol recovery.",
            "Sure! Feel free to ask anything about recovery guidance.",
        ],
        "default": [
            "I'm RecoveryPath AI! 😊 I'm best suited for questions about alcohol recovery, withdrawal, relapse prevention, and clinical guidance. Feel free to ask!",
        ],
    },
    "ar": {
        "greeting": [
            "أهلاً وسهلاً! 👋 أنا **RecoveryPath AI**، مساعدك المتخصص في توجيهات التعافي من الكحول المبنية على الأدلة. يمكنك السؤال عن دعم الانسحاب، الوقاية من الانتكاس، التقييم، أو رعاية التعافي.",
            "مرحباً! 😊 أنا RecoveryPath AI — هنا لمساعدتك في أسئلة التعافي المبنية على الأدلة. كيف يمكنني مساعدتك؟",
        ],
        "how_are_you": [
            "بخير، شكراً على السؤال! 😊 أنا جاهز لمساعدتك في أي استفسارات حول التعافي. بماذا يمكنني خدمتك؟",
        ],
        "thanks": [
            "على الرحب والسعة! 😊 لا تتردد في السؤال عن أي شيء آخر.",
            "سعيد بمساعدتك! إذا كان لديك أي سؤال آخر، أنا هنا.",
        ],
        "farewell": [
            "مع السلامة! 👋 أنا دائماً هنا إذا احتجت توجيهاً حول التعافي.",
        ],
        "who_are_you": [
            "أنا **RecoveryPath AI** — مساعد متخصص في توجيهات التعافي من الكحول المبنية على الأدلة. أجيب من الإرشادات السريرية والأبحاث المفهرسة. كيف يمكنني مساعدتك؟ 🤖",
        ],
        "ok": [
            "رائع! 😊 أخبرني إذا كان لديك أي سؤال حول التعافي.",
        ],
        "default": [
            "أنا RecoveryPath AI! 😊 متخصص في أسئلة التعافي من الكحول والانسحاب والوقاية من الانتكاس. اسأل بحرية!",
        ],
    },
}

_SUGGESTED_QUESTIONS: list[tuple[str, str]] = [
    ("What are the symptoms of alcohol withdrawal?", "en"),
    ("What medications are used for alcohol withdrawal?", "en"),
    ("How do I screen for alcohol use disorder?", "en"),
    ("ما هي أعراض الانسحاب من الكحول؟", "ar"),
    ("ما الأدوية المستخدمة في علاج الانسحاب؟", "ar"),
    ("كيف يتم تقييم اضطراب استخدام الكحول؟", "ar"),
]


def _classify_small_talk(text: str) -> tuple[bool, str, str]:
    lang = "ar" if _ARABIC_CHARS.search(text) else "en"
    for pattern, category in _PATTERNS:
        if pattern.search(text.strip()):
            return True, category, lang
    return False, "default", lang


def _small_talk_reply(category: str, lang: str) -> tuple[str, str]:
    options = _REPLIES.get(lang, _REPLIES["en"]).get(
        category, _REPLIES["en"]["default"]
    )
    return random.choice(options), lang


# ─── CSS ──────────────────────────────────────────────────────────────────────

def _load_css() -> None:
    css_path = Path(__file__).parent / "styles" / "custom.css"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


# ─── Conversation history persistence ─────────────────────────────────────────

def _load_history_file() -> dict:
    """Load saved conversations from disk."""
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"conversations": {}}


def _write_history_file(history: dict) -> None:
    """Persist conversations to disk."""
    try:
        HISTORY_FILE.write_text(
            json.dumps(history, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    except Exception:
        pass  # Non-fatal — history only affects UX


def _serialize_message(msg: dict) -> dict:
    """Strip heavy fields before saving to disk."""
    out: dict = {
        "role": msg.get("role"),
        "content": msg.get("content"),
        "language": msg.get("language", "en"),
        "grounded": msg.get("grounded", False),
        "refused": msg.get("refused", False),
        "timings_ms": msg.get("timings_ms") or {},
        "relevance": msg.get("relevance") or {},
        "sources": msg.get("sources") or [],
    }
    payload = msg.get("payload") or {}
    if payload:
        # Drop large retrieval blob; keep everything else for evidence panel
        out["payload"] = {k: v for k, v in payload.items() if k != "retrieval"}
    return out


def _conversation_title(messages: list[dict]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            content = str(msg.get("content", "")).strip()
            if content:
                return content[:MAX_TITLE_LENGTH] + ("…" if len(content) > MAX_TITLE_LENGTH else "")
    return "New conversation"


def _save_current_conversation() -> None:
    """Persist the active conversation to session state + disk."""
    messages = st.session_state.get("messages", [])
    if not messages:
        return

    conv_id = st.session_state.get("current_conv_id")
    if conv_id is None:
        conv_id = uuid.uuid4().hex[:10]
        st.session_state["current_conv_id"] = conv_id

    now = datetime.now().isoformat(timespec="seconds")
    conversations: dict = st.session_state.setdefault("conversations", {})

    created_at = conversations.get(conv_id, {}).get("created_at", now)

    conversations[conv_id] = {
        "id": conv_id,
        "title": _conversation_title(messages),
        "created_at": created_at,
        "updated_at": now,
        "messages": [_serialize_message(m) for m in messages],
        "latest_response": st.session_state.get("latest_response"),
    }

    # Keep only the most recent N conversations on disk
    trimmed = dict(
        sorted(conversations.items(), key=lambda kv: kv[1].get("updated_at", ""), reverse=True)[
            :MAX_SAVED_CONVERSATIONS
        ]
    )
    _write_history_file({"conversations": trimmed})


def _load_conversation(conv_id: str) -> None:
    """Switch to a saved conversation."""
    _save_current_conversation()  # persist the one being left
    conv = st.session_state.get("conversations", {}).get(conv_id)
    if not conv:
        return
    st.session_state["messages"] = list(conv.get("messages") or [])
    st.session_state["latest_response"] = conv.get("latest_response")
    st.session_state["current_conv_id"] = conv_id
    st.session_state["pending_question"] = None


def _delete_conversation(conv_id: str) -> None:
    """Remove a conversation from history."""
    conversations: dict = st.session_state.get("conversations", {})
    conversations.pop(conv_id, None)
    if st.session_state.get("current_conv_id") == conv_id:
        st.session_state["messages"] = []
        st.session_state["latest_response"] = None
        st.session_state["current_conv_id"] = None
        st.session_state["pending_question"] = None
    _write_history_file({"conversations": conversations})


# ─── Session state ────────────────────────────────────────────────────────────


def _scroll_to_latest_message() -> None:
    """Scroll the parent Streamlit page to the newest chat message."""
    components.html(
        """
        <script>
        const parentDocument = window.parent.document;
        const scrollToLatest = () => {
            const messages = parentDocument.querySelectorAll(
                '[data-testid="stChatMessage"]'
            );
            if (messages.length > 0) {
                messages[messages.length - 1].scrollIntoView({
                    behavior: 'smooth',
                    block: 'center'
                });
            }
        };
        window.setTimeout(scrollToLatest, 80);
        window.setTimeout(scrollToLatest, 350);
        </script>
        """,
        height=0,
        width=0,
    )

def _initialize_state() -> None:
    defaults: dict = {
        "messages": [],
        "latest_response": None,
        "project_id": 2,
        "asset_id": 1,
        "search_scope": "project",
        "last_ingestion": None,
        "pending_question": None,
        "scroll_to_latest": False,
        "generation_provider": "groq",
        "current_conv_id": None,
        "conversations": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Load history from disk once per session
    if not st.session_state.get("_history_loaded"):
        history = _load_history_file()
        st.session_state["conversations"] = history.get("conversations", {})
        st.session_state["_history_loaded"] = True


def _new_conversation() -> None:
    _save_current_conversation()
    st.session_state["messages"] = []
    st.session_state["latest_response"] = None
    st.session_state["pending_question"] = None
    st.session_state["current_conv_id"] = None


def _queue_user_question(question: str) -> None:
    clean = " ".join(question.split()).strip()
    if not clean:
        return
    add_message(st.session_state["messages"], "user", clean)
    st.session_state["pending_question"] = clean


# ─── Progress indicator ───────────────────────────────────────────────────────

def _show_progress(step: str) -> str:
    steps = {
        "retrieve": (
            '<div class="progress-steps">'
            '<span class="progress-step step-active">🔍 Searching evidence</span>'
            '<span class="progress-step step-pending">🤖 Generating</span>'
            '<span class="progress-step step-pending">Done</span>'
            "</div>"
        ),
        "generate": (
            '<div class="progress-steps">'
            '<span class="progress-step step-done">Evidence retrieved</span>'
            '<span class="progress-step step-active">🤖 Generating answer</span>'
            '<span class="progress-step step-pending">Done</span>'
            "</div>"
        ),
    }
    return steps.get(step, "")


# ─── Answer generation ────────────────────────────────────────────────────────

def _generate_pending_answer(api_client: APIClient) -> None:
    pending = st.session_state.get("pending_question")
    if not pending:
        return

    st.session_state["pending_question"] = None

    # ── Small talk ────────────────────────────────────────────────────────────
    is_small, category, lang = _classify_small_talk(pending)
    if is_small:
        answer, answer_lang = _small_talk_reply(category, lang)
        stream_new_message(answer, response_payload={"answer_language": answer_lang})
        add_message(
            st.session_state["messages"],
            "assistant",
            answer,
            {
                "answer_language": answer_lang,
                "grounded": False,
                "refused": False,
                "timings_ms": {},
                "sources": [],
                "relevance": {},
                "evidence": [],
            },
        )
        _save_current_conversation()
        return

    # ── RAG call ──────────────────────────────────────────────────────────────
    try:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(
                '<div class="assistant-title">RecoveryPath AI</div>',
                unsafe_allow_html=True,
            )
            prog = st.empty()
            prog.markdown(_show_progress("retrieve"), unsafe_allow_html=True)

            selected_asset_id = (
                int(st.session_state["asset_id"])
                if st.session_state.get("search_scope") == "asset"
                else None
            )
            response = api_client.ask_rag(
                question=pending,
                project_id=int(st.session_state["project_id"]),
                asset_id=selected_asset_id,
                retrieval_limit=5,
                generation_provider=st.session_state.get("generation_provider", "groq"),
                temperature=0.0,
                max_output_tokens=1200,
                timeout_seconds=300.0,
            )

            prog.markdown(_show_progress("generate"), unsafe_allow_html=True)

            response["search_scope"] = {
                "mode": st.session_state.get("search_scope", "project"),
                "project_id": int(st.session_state["project_id"]),
                "asset_id": selected_asset_id,
                "label": (
                    "Entire project"
                    if selected_asset_id is None
                    else f"Single document (Asset {selected_asset_id})"
                ),
            }
            answer = response.get("recommendation") or response.get("answer", "")
            lang = response.get("answer_language", "en")
            direction = "rtl" if lang == "ar" else "ltr"
            lang_class = "arabic-message" if lang == "ar" else "english-message"

            prog.empty()

            # Stream word by word
            placeholder = st.empty()
            if answer:
                words = answer.split(" ")
                accumulated = ""
                for i, word in enumerate(words):
                    accumulated += word + (" " if i < len(words) - 1 else "")
                    placeholder.markdown(
                        f'<div dir="{direction}" class="assistant-message {lang_class} streaming">'
                        f"{_safe_html(accumulated)}"
                        f'<span class="streaming-cursor">▋</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.022)
                placeholder.markdown(
                    f'<div dir="{direction}" class="assistant-message {lang_class}">'
                    f"{_safe_html(accumulated)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            temp_msg = {
                "grounded": response.get("grounded", False),
                "refused": response.get("refused", False),
                "relevance": response.get("relevance", {}),
                "sources": response.get("sources", []),
                "timings_ms": response.get("timings_ms", {}),
                "language": lang,
                "evidence_strength": response.get("evidence_strength", {}),
                "payload": response,
            }
            _render_metadata(temp_msg)
            _render_details(temp_msg)

        st.session_state["latest_response"] = response
        add_message(st.session_state["messages"], "assistant", answer, response)
        _save_current_conversation()

    except Exception as exc:
        err_msg = f"تعذر إكمال الطلب حاليًا. {exc}"
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(
                '<div class="assistant-title">RecoveryPath AI</div>',
                unsafe_allow_html=True,
            )
            st.error(err_msg)
        add_message(
            st.session_state["messages"],
            "assistant",
            err_msg,
            {"answer_language": "ar", "error": True},
        )


# ─── App layout ───────────────────────────────────────────────────────────────

_load_css()
_initialize_state()

api_url = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")
api_client = APIClient(base_url=api_url)
backend_alive = api_client.health_check()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-symbol">RP</div>
            <div>
                <div class="brand-name">RecoveryPath AI</div>
                <div class="brand-subtitle">Alcohol recovery evidence assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Backend status
    status_class = "status-online" if backend_alive else "status-offline"
    status_text = "Backend connected" if backend_alive else "Backend offline"
    st.markdown(
        f'<div class="backend-status {status_class}"><span></span>{status_text}</div>',
        unsafe_allow_html=True,
    )

    # New conversation
    if st.button("New conversation", icon="➕", type="primary", use_container_width=True):
        _new_conversation()
        st.rerun()

    # ── Conversation history ───────────────────────────────────────────────────
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-section-label">CONVERSATION HISTORY</div>',
        unsafe_allow_html=True,
    )

    conversations: dict = st.session_state.get("conversations", {})
    sorted_convs = sorted(
        conversations.values(),
        key=lambda c: c.get("updated_at", ""),
        reverse=True,
    )

    if not sorted_convs:
        st.markdown(
            '<p class="history-empty">No saved conversations yet.<br>Start asking to save history.</p>',
            unsafe_allow_html=True,
        )
    else:
        current_id = st.session_state.get("current_conv_id")
        for conv in sorted_convs:
            cid = conv["id"]
            title = conv.get("title", "Untitled")
            date_str = conv.get("updated_at", "")[:10]
            is_active = cid == current_id

            active_cls = "history-item-active" if is_active else "history-item"
            dot = "◉ " if is_active else ""

            col_title, col_del = st.columns([9, 1])
            with col_title:
                if st.button(
                    f"{dot}{title}",
                    key=f"hist_{cid}",
                    use_container_width=True,
                    help=f"Last updated: {date_str}",
                ):
                    if cid != current_id:
                        _load_conversation(cid)
                        st.rerun()
            with col_del:
                if st.button("✕", key=f"del_{cid}", help="Delete conversation"):
                    _delete_conversation(cid)
                    st.rerun()

    # ── Document upload ────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    render_sidebar_ingestion(api_client)

    # ── Search scope ───────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-section-label">SEARCH SCOPE</div>',
        unsafe_allow_html=True,
    )
    scope_choice = st.radio(
        "Choose where RecoveryPath AI searches",
        options=["Entire project", "Single document"],
        index=0 if st.session_state.get("search_scope", "project") == "project" else 1,
        help="Entire project searches every indexed document in the project. Single document restricts retrieval to one asset.",
        key="search_scope_choice",
    )
    st.session_state["search_scope"] = "project" if scope_choice == "Entire project" else "asset"

    project_id = st.number_input(
        "Project ID", min_value=1,
        value=int(st.session_state["project_id"]), step=1,
        help="Project containing the indexed recovery documents.",
        key="active_project_id",
    )
    st.session_state["project_id"] = int(project_id)

    if st.session_state["search_scope"] == "asset":
        asset_id = st.number_input(
            "Asset ID", min_value=1,
            value=int(st.session_state.get("asset_id") or 1), step=1,
            help="Restrict retrieval to this indexed document only.",
            key="active_asset_id",
        )
        st.session_state["asset_id"] = int(asset_id)
        st.markdown(
            f'<div class="scope-summary scope-document"><strong>Single document</strong><span>Project {int(project_id)} • Asset {int(asset_id)}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="scope-summary scope-project"><strong>Entire project</strong><span>Searching all indexed files in Project {int(project_id)}</span></div>',
            unsafe_allow_html=True,
        )

    # ── Generation model ───────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-section-label">GENERATION MODEL</div>',
        unsafe_allow_html=True,
    )
    provider_options = ["groq", "glm", "gemini"]
    current_provider = st.session_state.get(
        "generation_provider",
        "groq",
    )
    if current_provider not in provider_options:
        current_provider = "groq"

    provider_choice = st.selectbox(
        "Provider",
        options=provider_options,
        index=provider_options.index(current_provider),
        format_func=lambda value: {
            "groq": "Groq - GPT OSS 120B",
            "glm": "GLM 4.7 Flash - Z.AI",
            "gemini": "Gemini 2.5 Flash",
        }.get(value, value.upper()),
        label_visibility="collapsed",
    )
    st.session_state["generation_provider"] = provider_choice

    # ── Notice ─────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="sidebar-notice">
            <strong>Private, evidence-focused support</strong>
            <p>RecoveryPath AI explains indexed guidance and does not replace care from a qualified professional.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Main header ───────────────────────────────────────────────────────────────
header_left, header_right = st.columns([5, 1])
with header_left:
    st.markdown(
        """
        <header class="main-header">
            <div class="header-kicker">ALCOHOL RECOVERY EVIDENCE ASSISTANT</div>
            <h1>Recovery guidance, grounded in evidence.</h1>
            <p>Ask about withdrawal support, relapse prevention, assessments, and recovery care.</p>
        </header>
        """,
        unsafe_allow_html=True,
    )
with header_right:
    st.markdown(
        """
        <div class="trust-stack">
            <span>Evidence-based</span>
            <span>Source cited</span>
            <span>Arabic + English</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Chat area ─────────────────────────────────────────────────────────────────
chat_shell = st.container()

with chat_shell:
    if not st.session_state["messages"]:
        st.markdown(
            """
            <section class="empty-chat">
                <div class="empty-chat-symbol">🤖</div>
                <h2>كيف يمكنني مساعدتك في رحلة التعافي؟</h2>
                <p class="empty-chat-ar">
                    اسأل عن دعم الانسحاب، منع الانتكاس، أدوات التقييم، أو الرعاية
                    المذكورة في المستندات المفهرسة.
                </p>
                <p class="empty-chat-en">Ask a recovery question in Arabic or English, or choose a suggestion below.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="suggestions-label">Suggested questions</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for idx, (suggestion, _lang) in enumerate(_SUGGESTED_QUESTIONS):
            with cols[idx % 2]:
                if st.button(suggestion, key=f"sugg_{idx}", use_container_width=True):
                    _queue_user_question(suggestion)
                    st.rerun()
    else:
        render_chat_interface(st.session_state["messages"])
        if st.session_state.get("scroll_to_latest"):
            _scroll_to_latest_message()
            st.session_state["scroll_to_latest"] = False

    # Stream pending answer inside same container (appears after history)
    if st.session_state.get("pending_question"):
        _generate_pending_answer(api_client)
        st.rerun()


# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("اكتب سؤالك هنا...  |  Ask about alcohol recovery")
if prompt:
    _queue_user_question(prompt)
    st.rerun()


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="page-footer">RecoveryPath AI provides evidence-based information and does not replace professional care.</div>',
    unsafe_allow_html=True,
)
