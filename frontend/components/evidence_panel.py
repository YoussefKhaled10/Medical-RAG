import html
from typing import Any

import streamlit as st


def _fmt_score(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def _seconds(ms: Any) -> str:
    try:
        return f"{float(ms) / 1000:.2f}s"
    except (TypeError, ValueError):
        return "—"


def render_evidence_panel(response: dict[str, Any] | None) -> None:
    st.markdown(
        '<div class="section-heading"><div><h3>Evidence inspector</h3>'
        "<p>Verify relevance, citations, and source excerpts.</p></div></div>",
        unsafe_allow_html=True,
    )

    if not response:
        st.markdown(
            """
            <div class="empty-evidence">
                <div class="empty-evidence-icon">⌁</div>
                <strong>No evidence selected</strong>
                <p>Submit a clinical question to inspect relevance decisions,
                source excerpts, and latency.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    grounded = bool(response.get("grounded", False))
    refused  = bool(response.get("refused", False))
    relevance = response.get("relevance") or {}
    refusal   = response.get("refusal")   or {}
    timings   = response.get("timings_ms") or {}
    provider  = response.get("provider") or "Skipped"
    model     = response.get("model")    or "No generation"
    evidence_strength = response.get("evidence_strength") or {}
    search_scope = response.get("search_scope") or {}

    # ── Decision card ────────────────────────────────────────────────────────
    if grounded:
        state_class, state_icon, state_title = (
            "decision-pass", "✓", "Grounded response verified",
        )
    elif refused:
        state_class, state_icon, state_title = (
            "decision-refused", "!", "Safe refusal activated",
        )
    else:
        state_class, state_icon, state_title = (
            "decision-warning", "!", "Response requires review",
        )

    top_score = relevance.get("top_score")
    threshold = relevance.get("threshold")
    st.markdown(
        f"""
        <div class="decision-card {state_class}">
            <div class="decision-icon">{state_icon}</div>
            <div class="decision-content">
                <div class="decision-title">{state_title}</div>
                <div class="decision-meta">
                    Top relevance <strong>{_fmt_score(top_score)}</strong>
                    <span>•</span>
                    Gate <strong>{_fmt_score(threshold)}</strong>
                    <span>•</span>
                    Provider <strong>{html.escape(provider)}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if search_scope:
        scope_label = html.escape(str(search_scope.get("label") or "Unknown scope"))
        project_id = html.escape(str(search_scope.get("project_id") or "?"))
        st.markdown(
            f'<div class="evidence-scope-banner"><strong>Search scope: {scope_label}</strong><span>Project {project_id}</span></div>',
            unsafe_allow_html=True,
        )

    # ── Evidence strength card ───────────────────────────────────────────────
    if evidence_strength:
        level = evidence_strength.get("level", "")
        rationale = html.escape(evidence_strength.get("rationale", ""))
        policy    = html.escape(evidence_strength.get("language_policy", ""))
        badge_map = {
            "strong":       ("chip-strength-strong",   "💪 Strong evidence"),
            "moderate":     ("chip-strength-moderate", "⚡ Moderate evidence"),
            "insufficient": ("chip-strength-low",      "⚠ Insufficient evidence"),
        }
        badge_cls, badge_label = badge_map.get(level, ("chip-muted", level))
        st.markdown(
            f'<div style="margin-bottom:.8rem">'
            f'<span class="message-chip {badge_cls}" style="font-size:.76rem;padding:.3rem .65rem">'
            f"{badge_label}</span>"
            f'<span style="color:var(--muted);font-size:.72rem;margin-left:.6rem">'
            f"{rationale}</span>"
            f'<br><span style="color:#8a9eb0;font-size:.66rem">Policy: {policy}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Refusal card ─────────────────────────────────────────────────────────
    if refused and refusal:
        gen_skipped = bool(refusal.get("generation_skipped", False))
        st.markdown(
            f"""
            <div class="refusal-card">
                <div class="refusal-label">REFUSAL DETAILS</div>
                <strong>{html.escape(str(refusal.get("reason","insufficient_evidence")).replace("_"," ").title())}</strong>
                <p style="margin:.3rem 0 0;font-size:.78rem;color:#5e3f4a">
                    Stopped at <em>{html.escape(str(refusal.get("stage","unknown")).replace("_"," "))}</em> stage.
                    Generation skipped: <strong>{str(gen_skipped).lower()}</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Tabs ─────────────────────────────────────────────────────────────────
    source_tab, claims_tab, retrieval_tab, latency_tab, raw_tab = st.tabs(
        ["Sources", "Claims", "Retrieval", "Latency", "Raw JSON"]
    )

    # Sources
    with source_tab:
        evidence = response.get("evidence", [])
        if not evidence:
            st.caption("No source excerpts are attached to this response.")
        for index, item in enumerate(evidence, start=1):
            source_id = html.escape(str(item.get("source_id", f"S{index}")))
            citation  = html.escape(str(item.get("citation", "Citation unavailable")))
            excerpt   = html.escape(str(item.get("excerpt", "")))
            section   = html.escape(str(item.get("section_title") or "Unknown section"))
            document  = html.escape(str(item.get("document_name") or "Unknown document"))
            page      = html.escape(str(item.get("page_number") or "?"))
            chunk_id  = html.escape(str(item.get("chunk_id") or "?"))
            score     = _fmt_score(item.get("rerank_score"))

            st.markdown(
                f"""
                <article class="source-card">
                    <div class="source-card-head">
                        <span class="source-index">{source_id}</span>
                        <div><strong>{document}</strong><small>{section}</small></div>
                        <span class="source-score">{score}</span>
                    </div>
                    <div class="source-meta">Page {page} • {chunk_id}</div>
                    <blockquote>{excerpt}</blockquote>
                    <div class="source-citation">{citation}</div>
                </article>
                """,
                unsafe_allow_html=True,
            )

    # Claims
    with claims_tab:
        claim_results = response.get("claim_results", [])
        claim_validation = response.get("claim_validation") or {}

        if claim_validation:
            total     = claim_validation.get("total_claims", 0)
            supported = claim_validation.get("supported_claims", 0)
            faith     = claim_validation.get("faithfulness", 1.0)
            passed    = claim_validation.get("passed", True)
            cv1, cv2, cv3 = st.columns(3)
            cv1.metric("Total claims", total)
            cv2.metric("Supported", supported)
            cv3.metric("Faithfulness", f"{float(faith):.0%}")
            status_txt = "✅ Passed" if passed else "❌ Failed"
            st.caption(f"Validation: **{status_txt}** — {claim_validation.get('reason','')}")

        if not claim_results:
            st.caption("No claim-level results available.")
        for cr in claim_results:
            support_icon = "✅" if cr.get("supported") else "❌"
            score_val = cr.get("support_score")
            score_str = f"{float(score_val):.2f}" if score_val is not None else "N/A"
            st.markdown(
                f"**{support_icon} {html.escape(cr.get('claim_id',''))}** "
                f"(score: `{score_str}`)"
                f"\n\n{html.escape(str(cr.get('claim','')))}",
            )
            if cr.get("reason"):
                st.caption(f"Reason: {cr['reason']}")
            st.markdown("---")

    # Retrieval
    with retrieval_tab:
        summary = response.get("retrieval_summary") or {}
        if not summary:
            st.caption("Retrieval summary is unavailable.")
        else:
            r1, r2, r3 = st.columns(3)
            r1.metric("Candidates", summary.get("pre_dedup_count", 0))
            r2.metric("After dedup", summary.get("post_dedup_count", 0))
            r3.metric("Final top K", summary.get("final_result_count", 0))

            translated = bool(summary.get("cross_language_keyword_used", False))
            st.markdown(
                f"**Cross-language keyword search:** {'✅ Enabled' if translated else 'Not required'}"
            )
            kw = summary.get("effective_keyword_query")
            if kw:
                st.code(str(kw), language=None)

            chunk_ids = summary.get("chunk_ids", [])
            if chunk_ids:
                st.markdown("**Retrieved chunk IDs**")
                st.write("  ".join(f"`{cid}`" for cid in chunk_ids))

    # Latency
    with latency_tab:
        if not timings:
            st.caption("Timing data is unavailable.")
        else:
            t1, t2 = st.columns(2)
            t1.metric("Retrieval",  _seconds(timings.get("retrieval", 0)))
            t2.metric("Generation", _seconds(timings.get("generation", 0)))
            t3, t4 = st.columns(2)
            t3.metric("Citation repair",    f"{float(timings.get('citation_repair',0)):.1f} ms")
            t4.metric("Claim validation",   f"{float(timings.get('claim_validation',0)):.1f} ms")
            st.metric("⏱ Total response time", _seconds(timings.get("total", 0)))
            st.caption(f"Model: **{provider}** / {model}")

    # Raw JSON
    with raw_tab:
        st.json(response)
