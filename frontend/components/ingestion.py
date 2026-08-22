import streamlit as st
from frontend.api_client import APIClient


def render_sidebar_ingestion(api_client: APIClient) -> None:
    st.markdown('<div class="sidebar-section-label">DOCUMENTS</div>', unsafe_allow_html=True)
    st.caption("Add an alcohol recovery or treatment guideline PDF.")
    project_id = int(st.session_state.get("project_id", 2))
    if st.session_state.get("developer_mode", False):
        with st.expander("Developer upload settings", expanded=False):
            project_id = int(st.number_input("Upload project ID", min_value=1, value=project_id, step=1, key="upload_project_id"))
    else:
        st.markdown('<div class="upload-destination"><span>Upload destination</span><strong>Recovery knowledge base</strong></div>', unsafe_allow_html=True)
    file = st.file_uploader("Choose PDF", type=["pdf"], label_visibility="collapsed")
    if file is None:
        return
    st.markdown(f'<div class="upload-summary"><strong>{file.name}</strong><span>{file.size/1024:.1f} KB</span></div>', unsafe_allow_html=True)
    if st.button("Upload document", icon="⬆️", use_container_width=True, key="upload_index_button"):
        try:
            with st.spinner("Preparing the document for search..."):
                result = api_client.upload_pdf(project_id=project_id, file_bytes=file.getvalue(), file_name=file.name, timeout_seconds=300.0)
            if result.get("asset_id") is not None:
                st.session_state.asset_id = int(result["asset_id"])
            st.success("Document is ready")
            if st.session_state.get("developer_mode", False):
                st.json(result)
        except Exception as exc:
            st.error(f"Upload failed: {exc}")
