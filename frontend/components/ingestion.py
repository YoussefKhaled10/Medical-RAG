import streamlit as st
from frontend.api_client import APIClient


DEFAULT_UPLOAD_PROJECT_ID = 2


def render_sidebar_ingestion(
    api_client: APIClient,
) -> None:
    st.markdown(
        (
            '<div class="sidebar-section-label">'
            "DOCUMENT ANALYSIS"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    st.caption(
        "Upload a medical report or guideline (PDF or TXT) to query it directly."
    )

    uploaded_file = st.file_uploader(
        "Choose document (PDF or TXT)",
        type=["pdf", "txt"],
        label_visibility="collapsed",
        key="recovery_document_uploader",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
        file_size = len(file_bytes)

        st.session_state["ephemeral_uploaded_doc"] = {
            "name": file_name,
            "bytes": file_bytes,
            "size": file_size,
        }
        st.session_state["active_query_mode"] = "uploaded_doc"

        st.markdown(
            f"""
            <div class="upload-summary" style="border-left: 3px solid #10b981; padding: 6px 10px; background: rgba(16,185,129,0.08); border-radius: 4px; margin-bottom: 8px;">
                <strong style="color: #10b981;">📄 {file_name}</strong><br>
                <small style="color: #6b7280;">{file_size / 1024:.1f} KB (In-Memory Session Only - No DB Storage)</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("🗑️ Remove uploaded document", use_container_width=True, key="clear_ephemeral_doc_btn"):
            st.session_state["ephemeral_uploaded_doc"] = None
            st.session_state["active_query_mode"] = "global_kb"
            st.rerun()

    if st.session_state.get("developer_mode", False):
        with st.expander("Permanent database indexing (Dev only)", expanded=False):
            upload_project_id = int(
                st.number_input(
                    "Permanent Project ID",
                    min_value=1,
                    value=DEFAULT_UPLOAD_PROJECT_ID,
                    step=1,
                    key="upload_project_id",
                )
            )
            if uploaded_file is not None:
                if st.button(
                    "Save permanently to Database",
                    icon="💾",
                    use_container_width=True,
                    key="permanent_index_button",
                ):
                    try:
                        with st.spinner("Indexing into PostgreSQL / pgvector..."):
                            result = api_client.upload_document(
                                project_id=upload_project_id,
                                file_bytes=uploaded_file.getvalue(),
                                file_name=uploaded_file.name,
                                timeout_seconds=300.0,
                            )
                        st.success("Successfully saved to database!")
                        st.json(result)
                    except Exception as exc:
                        st.error(f"Permanent index failed: {exc}")
