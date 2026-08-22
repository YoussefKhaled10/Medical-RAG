from __future__ import annotations

from typing import Any
import httpx


class APIClient:
    """Small HTTP client used by the Streamlit application."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = str(base_url).rstrip("/")

    def ask_rag(
        self,
        *,
        question: str,
        project_id: int | None = None,
        asset_id: int | None = None,
        retrieval_limit: int = 5,
        generation_provider: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1200,
        timeout_seconds: float = 300.0,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": question,
            "retrieval_limit": retrieval_limit,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if project_id is not None:
            payload["project_id"] = project_id
        if asset_id is not None:
            payload["asset_id"] = asset_id
        if generation_provider:
            payload["generation_provider"] = generation_provider
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(f"{self.base_url}/api/v1/rag/ask", json=payload)
            response.raise_for_status()
            return response.json()

    ask_question = ask_rag
    ask = ask_rag

    def upload_pdf(
        self,
        *,
        project_id: int,
        file_bytes: bytes,
        file_name: str,
        timeout_seconds: float = 300.0,
    ) -> dict[str, Any]:
        endpoint = (
            f"{self.base_url}/api/v1/ingestion/upload-index"
        )

        files = {
            "file": (
                file_name,
                file_bytes,
                "application/pdf",
            ),
        }
        data = {
            "project_id": str(project_id),
        }

        try:
            with httpx.Client(
                timeout=timeout_seconds,
            ) as client:
                response = client.post(
                    endpoint,
                    files=files,
                    data=data,
                )

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json()
            except ValueError:
                detail = exc.response.text

            raise RuntimeError(
                "Upload API returned HTTP "
                f"{exc.response.status_code}: {detail}"
            ) from exc

        except httpx.RequestError as exc:
            raise RuntimeError(
                "Could not connect to the upload API at "
                f"{endpoint}: {exc}"
            ) from exc
