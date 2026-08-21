from typing import Any

import httpx


class APIClientError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class APIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def _timeout(seconds: float) -> httpx.Timeout:
        return httpx.Timeout(
            connect=min(10.0, seconds),
            read=seconds,
            write=min(30.0, seconds),
            pool=min(10.0, seconds),
        )

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=self._timeout(3.0)) as client:
                response = client.get(f"{self.base_url}/api/v1/")
                return response.status_code == 200
        except httpx.HTTPError:
            return False

    @staticmethod
    def _raise_error(response: httpx.Response, fallback: str) -> None:
        try:
            payload = response.json()
        except Exception:
            payload = {}

        detail = payload.get("detail", payload)
        if isinstance(detail, dict):
            message = str(
                detail.get("message")
                or detail.get("error")
                or fallback
            )
            retry_after = detail.get("retry_after_seconds")
            if retry_after is not None:
                message = f"{message} Retry after {retry_after} seconds."
            retryable = bool(detail.get("retryable", False))
        else:
            message = str(detail or response.text or fallback)
            retryable = response.status_code in {429, 502, 503, 504}

        raise APIClientError(
            message,
            status_code=response.status_code,
            retryable=retryable,
        )

    def ask_rag(
        self,
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
        if generation_provider is not None:
            payload["generation_provider"] = generation_provider

        try:
            with httpx.Client(timeout=self._timeout(timeout_seconds)) as client:
                response = client.post(
                    f"{self.base_url}/api/v1/rag/ask",
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise APIClientError(
                "The request timed out while waiting for retrieval or generation.",
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise APIClientError(
                "Could not connect to the Medical RAG backend.",
                retryable=True,
            ) from exc

        if response.status_code != 200:
            self._raise_error(response, "RAG request failed")
        return response.json()

    def upload_pdf(
        self,
        project_id: int,
        file_bytes: bytes,
        file_name: str,
        timeout_seconds: float = 300.0,
    ) -> dict[str, Any]:
        files = {"file": (file_name, file_bytes, "application/pdf")}
        data = {"project_id": project_id}

        try:
            with httpx.Client(timeout=self._timeout(timeout_seconds)) as client:
                response = client.post(
                    f"{self.base_url}/api/v1/ingestion/upload-index",
                    data=data,
                    files=files,
                )
        except httpx.TimeoutException as exc:
            raise APIClientError(
                "Document indexing timed out.",
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise APIClientError(
                "Could not connect to the ingestion API.",
                retryable=True,
            ) from exc

        if response.status_code not in (200, 201):
            self._raise_error(response, "Document upload failed")
        return response.json()
