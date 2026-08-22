from typing import Any
import streamlit as st


def render_evidence_panel(response: dict[str, Any] | None) -> None:
    if not response:
        st.info("Ask a recovery question to review supporting evidence.")
        return
    if st.session_state.get("developer_mode", False):
        st.subheader("Developer evidence inspector")
        st.json(response)
        return
    st.subheader("Supporting evidence")
    evidence = response.get("evidence") or []
    if not evidence:
        st.caption("No supporting excerpts are attached to this response.")
    for item in evidence:
        st.markdown(f"**{item.get('document_name','Source')}**  ·  Page {item.get('page_number','?')}")
        st.caption(str(item.get("section_title") or "Relevant section"))
        st.info(str(item.get("excerpt") or ""))
