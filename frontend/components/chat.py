import html
import re
from typing import Any
import streamlit as st

ARABIC = re.compile(r"[\u0600-\u06FF]")
CITATIONS = re.compile(r"\s*\[S\d+\]")


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
    for message in messages:
        content = str(message.get("content", ""))
        language = message.get("language") or ("ar" if ARABIC.search(content) else "en")
        direction = "rtl" if language == "ar" else "ltr"
        if message.get("role") == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(f'<div class="user-message-row"><div class="user-message" dir="{direction}">{_html(content)}</div></div>', unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f'<div class="assistant-title">RecoveryPath AI</div><div class="assistant-message" dir="{direction}">{_html(content)}</div>', unsafe_allow_html=True)
                _metadata(message)
                _details(message)


def add_message(messages_state: list[dict[str, Any]], role: str, content: str, response_payload: dict[str, Any] | None = None) -> None:
    message: dict[str, Any] = {"role": role, "content": content}
    if response_payload:
        message.update({"sources": response_payload.get("sources", []), "grounded": response_payload.get("grounded", False), "refused": response_payload.get("refused", False), "language": response_payload.get("answer_language", "en"), "relevance": response_payload.get("relevance", {}), "evidence_strength": response_payload.get("evidence_strength", {}), "payload": response_payload})
    messages_state.append(message)
