import html
import re
import time
from typing import Any
import streamlit as st
from markdown_it import MarkdownIt

ARABIC = re.compile(r"[\u0600-\u06FF]")
CITATIONS = re.compile(r"(?<!\w)\s*\[S\d+\](?!\w)")
MARKDOWN = MarkdownIt(
    "commonmark",
    {"html": False, "breaks": True, "linkify": False},
)


def _direction(text: str, language: str | None = None) -> str:
    """Resolve direction without modifying answer content."""
    if str(language or "").lower().startswith("ar"):
        return "rtl"
    return "rtl" if ARABIC.search(text) else "ltr"


def _dev() -> bool:
    return bool(st.session_state.get("developer_mode", False))


def _answer(text: str) -> str:
    if _dev():
        return text
    return re.sub(r"\s+([.,;:!?،؛؟])", r"\1", CITATIONS.sub("", text)).strip()


def _html(text: str) -> str:
    value = html.escape(_answer(text)).replace("\n", "<br>")
    if _dev():
        value = re.sub(r"(\[S\d+\])", r'<span class="inline-citation">\1</span>', value)
    return value


def _markdown_html(text: str) -> str:
    """Render model Markdown safely while preserving developer citations."""
    clean = _answer(str(text or ""))
    rendered = MARKDOWN.render(clean)
    if _dev():
        rendered = re.sub(
            r"(\[S\d+\])",
            r'<span class="inline-citation">\1</span>',
            rendered,
        )
    return rendered


def _metadata(message: dict[str, Any]) -> None:
    # Status, evidence strength, source IDs, and scores are developer-only.
    if not _dev():
        return

    payload = message.get("payload") or {}
    chips: list[str] = []

    if message.get("grounded"):
        chips.append(
            '<span class="message-chip chip-grounded">'
            '<span class="chip-dot"></span>Verified answer</span>'
        )
    elif message.get("refused"):
        chips.append(
            '<span class="message-chip chip-refused">'
            '<span class="chip-dot"></span>Safe guidance</span>'
        )

    level = (
        message.get("evidence_strength")
        or payload.get("evidence_strength")
        or {}
    ).get("level")
    if level:
        label = {
            "strong": "Strong evidence",
            "moderate": "Moderate evidence",
            "insufficient": "Limited evidence",
        }.get(level, level)
        css_level = level if level != "insufficient" else "low"
        chips.append(
            f'<span class="message-chip chip-strength-{css_level}">'
            f'{label}</span>'
        )

    relation = (message.get("relevance") or {}).get("top_score")
    if relation is not None:
        chips.append(
            '<span class="message-chip chip-muted">'
            f'Relevance {float(relation):.3f}</span>'
        )

    for source in (message.get("sources") or [])[:3]:
        chips.append(
            '<span class="message-chip chip-source">'
            f'[{source.get("source_id")}] '
            f'p.{source.get("page_number")}</span>'
        )

    if chips:
        st.markdown(
            f'<div class="message-chips">{"".join(chips)}</div>',
            unsafe_allow_html=True,
        )


def _details(message: dict[str, Any]) -> None:
    # Raw citations and supporting evidence are developer-only.
    if not _dev():
        return
    payload = message.get("payload") or {}
    if payload:
        with st.expander("Developer diagnostics", expanded=False):
            st.json(payload)


def render_chat_interface(messages: list[dict[str, Any]]) -> None:
    # Stable target used by the fixed "back to first message" control.
    st.markdown(
        '<div id="rp-chat-top" class="rp-chat-top-anchor" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )
    for message in messages:
        content = str(message.get("content", ""))
        language = message.get("language")
        direction = _direction(content, language)
        if message.get("role") == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(f'<div class="user-message-row"><div class="user-message" dir="{direction}"><bdi>{_html(content)}</bdi></div></div>', unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(
                    f'<div class="assistant-title">RecoveryPath AI</div>'
                    f'<div class="assistant-message rp-markdown-message" '
                    f'dir="{direction}">{_markdown_html(content)}</div>',
                    unsafe_allow_html=True,
                )
                _metadata(message)
                _details(message)




def render_streaming_assistant(
    text: str,
    *,
    language: str | None = None,
    delay_seconds: float = 0.018,
) -> None:
    """Stream a safe preview, then commit fully rendered Markdown."""
    clean = str(text or "").strip()
    if not clean:
        return

    direction = _direction(clean, language)
    words = clean.split()
    group_size = 4 if direction == "rtl" else 3
    groups = [words[i : i + group_size] for i in range(0, len(words), group_size)]

    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(
            '<div class="assistant-title">RecoveryPath AI</div>',
            unsafe_allow_html=True,
        )
        placeholder = st.empty()
        rendered: list[str] = []

        # During animation use escaped text so incomplete Markdown cannot break.
        for index, group in enumerate(groups):
            rendered.extend(group)
            cursor = " ▌" if index < len(groups) - 1 else ""
            preview = _html(" ".join(rendered))
            placeholder.markdown(
                f'<div class="assistant-message rp-streaming-answer" '
                f'dir="{direction}"><bdi>{preview}</bdi>{cursor}</div>',
                unsafe_allow_html=True,
            )
            if delay_seconds > 0 and index < len(groups) - 1:
                time.sleep(delay_seconds)

        # Final frame parses headings, lists, emphasis, and numbered steps.
        placeholder.markdown(
            f'<div class="assistant-message rp-markdown-message" '
            f'dir="{direction}">{_markdown_html(clean)}</div>',
            unsafe_allow_html=True,
        )

def render_chat_bottom_anchor() -> None:
    st.markdown(
        '<div id="rp-chat-bottom" class="rp-chat-bottom-anchor" '
        'aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )

def add_message(messages_state: list[dict[str, Any]], role: str, content: str, response_payload: dict[str, Any] | None = None) -> None:
    message: dict[str, Any] = {"role": role, "content": content}
    if response_payload:
        message.update({"sources": response_payload.get("sources", []), "grounded": response_payload.get("grounded", False), "refused": response_payload.get("refused", False), "language": response_payload.get("answer_language", "en"), "relevance": response_payload.get("relevance", {}), "evidence_strength": response_payload.get("evidence_strength", {}), "payload": response_payload})
    messages_state.append(message)
