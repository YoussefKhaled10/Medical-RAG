import streamlit as st
from frontend.api_client import APIClient


def render_sidebar_ingestion(
    api_client: APIClient,
) -> None:
    st.markdown(
        (
            '<div class="sidebar-section-label">'
            "DOCUMENT ANALYSIS & VAULT"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    user = st.session_state.get("user")
    if user:
        vault_id = user.get("private_project_id")
        st.caption(f"Upload documents directly to your Private Vault (#{vault_id}) or test in-memory.")
    else:
        st.caption("Upload a medical report or guideline (PDF or TXT) to query it directly.")

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

        st.markdown(
            f"""
            <div class="upload-summary" style="border-left: 3px solid #0d9488; padding: 8px 10px; background: rgba(13,148,136,0.12); border-radius: 8px; margin-bottom: 8px;">
                <strong style="color: #0d9488;">{file_name}</strong><br>
                <small style="color: #94a3b8;">{file_size / 1024:.1f} KB (Ready for query)</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_action1, col_action2 = st.columns(2)
        with col_action1:
            if st.button("Delete", use_container_width=True, key="clear_ephemeral_doc_btn"):
                st.session_state["ephemeral_uploaded_doc"] = None
                st.session_state["active_query_mode"] = "global_kb"
                st.rerun()

        with col_action2:
            if user is not None:
                if st.button("Save to Vault", use_container_width=True, key="save_to_private_vault_btn"):
                    try:
                        with st.spinner("Saving document into your Private Vault..."):
                            vault_id = user.get("private_project_id")
                            result = api_client.upload_document(
                                project_id=vault_id,
                                file_bytes=file_bytes,
                                file_name=file_name,
                                timeout_seconds=300.0,
                            )
                        st.success("Saved to your Private Vault successfully.")
                        user["document_count"] = user.get("document_count", 0) + 1
                        st.session_state["user"] = user
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Save failed: {exc}")
            else:
                st.caption("Sign in to save files to your vault.")


    if st.session_state.get("developer_mode", False):
        with st.expander("Developer upload settings (Dev only)", expanded=False):
            upload_project_id = int(
                st.number_input(
                    "Manual Project ID",
                    min_value=1,
                    value=1,
                    step=1,
                    key="dev_upload_project_id",
                )
            )
            if uploaded_file is not None:
                if st.button(
                    "Save to specified Project ID",
                    use_container_width=True,
                    key="dev_permanent_index_button",
                ):
                    try:
                        with st.spinner("Indexing into specified project..."):
                            result = api_client.upload_document(
                                project_id=upload_project_id,
                                file_bytes=uploaded_file.getvalue(),
                                file_name=uploaded_file.name,
                                timeout_seconds=300.0,
                            )
                        st.success(f"Successfully saved to Project {upload_project_id}!")
                        st.json(result)
                    except Exception as exc:
                        st.error(f"Dev upload failed: {exc}")