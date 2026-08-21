import streamlit as st

from frontend.api_client import APIClient


def render_sidebar_ingestion(api_client: APIClient) -> None:
    st.markdown(
        '<div class="sidebar-section-label">RECOVERY DOCUMENT UPLOAD</div>',
        unsafe_allow_html=True,
    )
    st.caption("Upload and index an alcohol recovery or treatment guideline PDF.")

    upload_project_id = st.number_input(
        "Upload to project",
        min_value=1,
        value=int(st.session_state.get("project_id", 2)),
        step=1,
        key="upload_project_id",
    )

    uploaded_file = st.file_uploader(
        "Choose PDF",
        type=["pdf"],
        help="Select an alcohol recovery or treatment guideline PDF.",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        st.markdown(
            f"<div class='upload-summary'><strong>{uploaded_file.name}</strong>"
            f"<span>{uploaded_file.size / 1024:.1f} KB</span></div>",
            unsafe_allow_html=True,
        )

        if st.button(
            "Upload and index",
            icon="⬆️",
            use_container_width=True,
            key="upload_index_button",
        ):
            try:
                with st.spinner("Indexing document..."):
                    result = api_client.upload_pdf(
                        project_id=int(upload_project_id),
                        file_bytes=uploaded_file.getvalue(),
                        file_name=uploaded_file.name,
                        timeout_seconds=300.0,
                    )

                asset_id = result.get("asset_id")
                if asset_id is not None:
                    st.session_state["asset_id"] = int(asset_id)
                st.session_state["project_id"] = int(upload_project_id)
                st.session_state["last_ingestion"] = result

                st.success("Document indexed")
                st.markdown(
                    f"""
                    <div class="ingestion-result">
                        <div><span>Asset</span><strong>{asset_id or '-'}</strong></div>
                        <div><span>Pages</span><strong>{result.get('total_pages', 0)}</strong></div>
                        <div><span>Chunks</span><strong>{result.get('total_chunks', 0)}</strong></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.error(f"Upload failed: {exc}")
