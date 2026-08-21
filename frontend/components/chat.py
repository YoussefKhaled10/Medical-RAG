import html
import re
import time
from typing import Any

import streamlit as st


ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
CITATION_PATTERN = re.compile(r"(\[S\d+\])")


def is_arabic(text: str) -> bool:
    return bool(ARABIC_PATTERN.search(text))


def _safe_html(text: str) -> str:
    escaped = html.escape(text)
    escaped = CITATION_PATTERN.sub(
        r'<span class="inline-citation">\1</span>',
        escaped,
    )
    return escaped.replace("\n", "<br>")


def _seconds(value: Any) -> str:
    try:
        return f"{float(value) / 1000:.2f}s"
    except (TypeError, ValueError):
        return ""


def _evidence_strength_chip(evidence_strength: dict) -> str:
    level = evidence_strength.get("level", "")
    if level == "strong":
        return '<span class="message-chip chip-strength-strong">💪 Strong evidence</span>'
    if level == "moderate":
        return '<span class="message-chip chip-strength-moderate">⚡ Moderate evidence</span>'
    if level == "insufficient":
        return '<span class="message-chip chip-strength-low">⚠ Insufficient</span>'
    return ""


def _render_metadata(message: dict[str, Any]) -> None:
    grounded = bool(message.get("grounded", False))
    refused = bool(message.get("refused", False))
    relevance = message.get("relevance") or {}
    sources = message.get("sources") or []
    timings = message.get("timings_ms") or {}
    payload = message.get("payload") or {}
    evidence_strength = (
        message.get("evidence_strength")
        or payload.get("evidence_strength")
        or {}
    )
    claim_validation = payload.get("claim_validation") or {}

    chips: list[str] = []

    if grounded:
        chips.append(
            '<span class="message-chip chip-grounded">'
            '<span class="chip-dot"></span>Grounded'
            "</span>"
        )
    elif refused:
        chips.append('<span class="message-chip chip-refused">Safe refusal</span>')

    if evidence_strength:
        badge = _evidence_strength_chip(evidence_strength)
        if badge:
            chips.append(badge)

    top_score = relevance.get("top_score")
    if top_score is not None:
        chips.append(
            f'<span class="message-chip chip-muted">Relevance {float(top_score):.3f}</span>'
        )

    for source in sources[:3]:
        source_id = html.escape(str(source.get("source_id", "S")))
        page = html.escape(str(source.get("page_number", "?")))
        chips.append(
            f'<span class="message-chip chip-source">[{source_id}] p.{page}</span>'
        )

    if claim_validation and grounded:
        total = claim_validation.get("total_claims", 0)
        supported = claim_validation.get("supported_claims", 0)
        if total > 0:
            chips.append(
                f'<span class="message-chip chip-muted">✓ {supported}/{total} claims</span>'
            )

    total_time = _seconds(timings.get("total"))
    if total_time:
        chips.append(f'<span class="message-chip chip-muted">{total_time}</span>')

    if chips:
        st.markdown(
            f'<div class="message-chips">{"".join(chips)}</div>',
            unsafe_allow_html=True,
        )


def _render_details(message: dict[str, Any]) -> None:
    payload = message.get("payload") or {}
    evidence = payload.get("evidence") or []
    refusal = payload.get("refusal") or {}
    retrieval = payload.get("retrieval_summary") or {}
    timings = payload.get("timings_ms") or {}

    if not evidence and not refusal and not retrieval:
        return

    label = (
        "عرض المصادر والتفاصيل"
        if message.get("language") == "ar"
        else "View sources and details"
    )
    with st.expander(label, expanded=False):
        if refusal:
            st.markdown(
                '<div class="refusal-details">'
                f'<strong>{html.escape(str(refusal.get("reason", "insufficient_evidence")).replace("_", " ").title())}</strong>'
                f'<span>Stage: {html.escape(str(refusal.get("stage", "unknown")).replace("_", " "))}</span>'
                f'<span>Generation skipped: {str(refusal.get("generation_skipped", False)).lower()}</span>'
                "</div>",
                unsafe_allow_html=True,
            )

        for item in evidence:
            source_id = html.escape(str(item.get("source_id", "S")))
            document = html.escape(str(item.get("document_name") or "Unknown document"))
            section = html.escape(str(item.get("section_title") or "Unknown section"))
            page = html.escape(str(item.get("page_number") or "?"))
            chunk_id = html.escape(str(item.get("chunk_id") or "?"))
            excerpt = html.escape(str(item.get("excerpt") or ""))
            citation = html.escape(str(item.get("citation") or ""))
            score = item.get("rerank_score")
            score_text = f"{float(score):.4f}" if score is not None else "N/A"

            st.markdown(
                f"""
                <article class="evidence-card">
                    <div class="evidence-head">
                        <span class="evidence-source">{source_id}</span>
                        <div>
                            <strong>{document}</strong>
                            <small>{section}</small>
                        </div>
                        <span class="evidence-score">{score_text}</span>
                    </div>
                    <div class="evidence-meta">Page {page} • {chunk_id}</div>
                    <div class="evidence-excerpt">{excerpt}</div>
                    <div class="evidence-citation">{citation}</div>
                </article>
                """,
                unsafe_allow_html=True,
            )

        if retrieval:
            st.markdown("**Retrieval summary**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Candidates", retrieval.get("pre_dedup_count", 0))
            col2.metric("After dedup", retrieval.get("post_dedup_count", 0))
            col3.metric("Final", retrieval.get("final_result_count", 0))

        if timings:
            st.caption(
                f"Retrieval {_seconds(timings.get('retrieval'))} • "
                f"Generation {_seconds(timings.get('generation'))} • "
                f"Total {_seconds(timings.get('total'))}"
            )


def render_chat_interface(messages: list[dict[str, Any]]) -> None:
    """Render conversation history (static, no streaming)."""
    for message in messages:
        role = message.get("role", "assistant")
        content = str(message.get("content", ""))
        language = message.get("language") or (
            "ar" if is_arabic(content) else "en"
        )
        direction = "rtl" if language == "ar" else "ltr"
        language_class = "arabic-message" if language == "ar" else "english-message"

        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(
                    '<div class="user-message-row">'
                    f'<div dir="{direction}" class="user-message {language_class}">'
                    f"{_safe_html(content)}"
                    "</div></div>",
                    unsafe_allow_html=True,
                )
            continue

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(
                '<div class="assistant-title">RecoveryPath AI</div>'
                f'<div dir="{direction}" class="assistant-message {language_class}">'
                f"{_safe_html(content)}"
                "</div>",
                unsafe_allow_html=True,
            )
            _render_metadata(message)
            _render_details(message)


def stream_new_message(
    answer: str,
    *,
    response_payload: dict[str, Any] | None = None,
    delay: float = 0.022,
) -> None:
    """Render a NEW assistant message with word-by-word streaming typing effect."""
    language = (
        response_payload.get("answer_language", "en")
        if response_payload
        else ("ar" if is_arabic(answer) else "en")
    )
    direction = "rtl" if language == "ar" else "ltr"
    lang_class = "arabic-message" if language == "ar" else "english-message"

    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(
            '<div class="assistant-title">RecoveryPath AI</div>',
            unsafe_allow_html=True,
        )
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
                time.sleep(delay)

            # Final render — remove blinking cursor
            placeholder.markdown(
                f'<div dir="{direction}" class="assistant-message {lang_class}">'
                f"{_safe_html(accumulated)}"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            placeholder.markdown(
                f'<div dir="{direction}" class="assistant-message {lang_class}"> </div>',
                unsafe_allow_html=True,
            )

        # Show metadata chips + details after streaming finishes
        if response_payload:
            temp_msg: dict[str, Any] = {
                "grounded": response_payload.get("grounded", False),
                "refused": response_payload.get("refused", False),
                "relevance": response_payload.get("relevance", {}),
                "sources": response_payload.get("sources", []),
                "timings_ms": response_payload.get("timings_ms", {}),
                "language": language,
                "evidence_strength": response_payload.get("evidence_strength", {}),
                "payload": response_payload,
            }
            _render_metadata(temp_msg)
            _render_details(temp_msg)


def add_message(
    messages_state: list[dict[str, Any]],
    role: str,
    content: str,
    response_payload: dict[str, Any] | None = None,
) -> None:
    message: dict[str, Any] = {"role": role, "content": content}
    if response_payload:
        message.update(
            {
                "sources": response_payload.get("sources", []),
                "grounded": response_payload.get("grounded", False),
                "refused": response_payload.get("refused", False),
                "timings_ms": response_payload.get("timings_ms", {}),
                "language": response_payload.get("answer_language", "en"),
                "relevance": response_payload.get("relevance", {}),
                "payload": response_payload,
                "error": response_payload.get("error", False),
            }
        )
    messages_state.append(message)
