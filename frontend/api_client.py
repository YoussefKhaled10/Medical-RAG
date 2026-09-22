import json
from typing import Any
import httpx


class APIClient:
    """HTTP client used by the Streamlit application with Authentication & Multi-Tenant Support."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", auth_token: str | None = None) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.auth_token = auth_token

    def set_token(self, token: str | None) -> None:
        self.auth_token = token

    def _headers(self, custom: dict[str, str] | None = None) -> dict[str, str]:
        headers = dict(custom or {})
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    # ================= Authentication Methods =================

    def send_otp(self, email: str, timeout_seconds: float = 30.0) -> dict[str, Any]:
        """Request 6-digit OTP sent to user's Gmail."""
        endpoint = f"{self.base_url}/api/v1/auth/send-otp"
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(endpoint, json={"email": email.strip()})
        try:
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            detail = self._parse_error_detail(exc)
            raise RuntimeError(f"Failed to send OTP: {detail}") from exc

    def register(
        self,
        email: str,
        username: str,
        password: str,
        otp_code: str,
        full_name: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        """Register a new user with verified OTP."""
        endpoint = f"{self.base_url}/api/v1/auth/register"
        payload = {
            "email": email.strip(),
            "username": username.strip(),
            "password": password,
            "otp_code": otp_code.strip(),
        }
        if full_name:
            payload["full_name"] = full_name.strip()

        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(endpoint, json=payload)
        try:
            response.raise_for_status()
            data = response.json()
            if "access_token" in data:
                self.set_token(data["access_token"])
            return data
        except httpx.HTTPStatusError as exc:
            detail = self._parse_error_detail(exc)
            raise RuntimeError(f"Registration failed: {detail}") from exc

    def login(
        self,
        email_or_username: str,
        password: str,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        """Authenticate user and retrieve tokens."""
        endpoint = f"{self.base_url}/api/v1/auth/login"
        payload = {
            "email_or_username": email_or_username.strip(),
            "password": password,
        }
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(endpoint, json=payload)
        try:
            response.raise_for_status()
            data = response.json()
            if "access_token" in data:
                self.set_token(data["access_token"])
            return data
        except httpx.HTTPStatusError as exc:
            detail = self._parse_error_detail(exc)
            raise RuntimeError(f"Login failed: {detail}") from exc

    def get_me(self, timeout_seconds: float = 20.0) -> dict[str, Any]:
        """Get profile of the currently logged in user."""
        endpoint = f"{self.base_url}/api/v1/auth/me"
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.get(endpoint, headers=self._headers())
        try:
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            detail = self._parse_error_detail(exc)
            raise RuntimeError(f"Failed to fetch profile: {detail}") from exc

    def logout(self, refresh_token: str | None = None, timeout_seconds: float = 15.0) -> None:
        """Logout user."""
        endpoint = f"{self.base_url}/api/v1/auth/logout"
        try:
            with httpx.Client(timeout=timeout_seconds) as client:
                client.post(endpoint, json={"refresh_token": refresh_token}, headers=self._headers())
        except Exception:
            pass
        finally:
            self.set_token(None)

    
    # ================= Conversations Methods =================

    def list_conversations(self, limit: int = 30, offset: int = 0, timeout_seconds: float = 30.0) -> dict[str, Any]:
        endpoint = f"{self.base_url}/api/v1/conversations"
        params = {"limit": limit, "offset": offset}
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.get(endpoint, params=params, headers=self._headers())
        try:
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            detail = self._parse_error_detail(exc)
            raise RuntimeError(f"Failed to fetch conversations: {detail}") from exc

    def create_conversation(self, title: str, timeout_seconds: float = 30.0) -> dict[str, Any]:
        endpoint = f"{self.base_url}/api/v1/conversations"
        payload = {"title": title}
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(endpoint, json=payload, headers=self._headers())
        try:
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            detail = self._parse_error_detail(exc)
            raise RuntimeError(f"Failed to create conversation: {detail}") from exc

    def get_messages(self, conversation_id: int, limit: int = 50, offset: int = 0, timeout_seconds: float = 30.0) -> dict[str, Any]:
        endpoint = f"{self.base_url}/api/v1/conversations/{conversation_id}/messages"
        params = {"limit": limit, "offset": offset}
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.get(endpoint, params=params, headers=self._headers())
        try:
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            detail = self._parse_error_detail(exc)
            raise RuntimeError(f"Failed to fetch messages: {detail}") from exc

    def send_message_to_conversation(
        self,
        conversation_id: int,
        question: str,
        client_request_id: str,
        generation_provider: str | None = None,
        retrieval_limit: int = 5,
        temperature: float = 0.0,
        max_output_tokens: int = 1200,
        timeout_seconds: float = 300.0,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        endpoint = f"{self.base_url}/api/v1/conversations/{conversation_id}/messages"
        payload = {
            "question": question,
            "client_request_id": client_request_id,
            "retrieval_limit": retrieval_limit,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if generation_provider:
            payload["generation_provider"] = generation_provider

        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(endpoint, json=payload, headers=self._headers())
        try:
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            detail = self._parse_error_detail(exc)
            raise RuntimeError(f"Failed to send message: {detail}") from exc

    def delete_conversation(self, conversation_id: int, timeout_seconds: float = 30.0) -> None:
        endpoint = f"{self.base_url}/api/v1/conversations/{conversation_id}"
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.delete(endpoint, headers=self._headers())
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = self._parse_error_detail(exc)
            raise RuntimeError(f"Failed to delete conversation: {detail}") from exc

    # ================= RAG & Ingestion Methods =================

    def ask_rag(
        self,
        *,
        question: str,
        project_id: int | None = None,
        asset_id: int | None = None,
        search_scope: str = "system",
        retrieval_limit: int = 10,
        retrieval_mode: str = "hybrid",
        generation_provider: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1200,
        timeout_seconds: float = 300.0,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": question,
            "conversation_history": conversation_history or [],
            "search_scope": search_scope,
            "retrieval_limit": retrieval_limit,
            "retrieval_mode": retrieval_mode,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if project_id is not None:
            payload["project_id"] = project_id
        if asset_id is not None:
            payload["asset_id"] = asset_id
        if generation_provider:
            payload["generation_provider"] = generation_provider
        endpoint = f"{self.base_url}/api/v1/rag/ask"
        try:
            with httpx.Client(timeout=timeout_seconds) as client:
                response = client.post(endpoint, json=payload, headers=self._headers())
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            detail = self._parse_error_detail(exc)
            raise RuntimeError(f"RAG API returned HTTP {exc.response.status_code}: {detail}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Could not connect to the RAG API at {endpoint}: {exc}") from exc

    ask_question = ask_rag
    ask = ask_rag

    def ask_document(
        self,
        *,
        question: str,
        file_bytes: bytes,
        file_name: str,
        generation_provider: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1200,
        timeout_seconds: float = 300.0,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        endpoint = f"{self.base_url}/api/v1/rag/ask-document"
        content_type = "text/plain" if file_name.lower().endswith(".txt") else "application/pdf"
        files = {
            "file": (file_name, file_bytes, content_type),
        }
        data = {
            "question": question,
            "conversation_history": json.dumps(conversation_history or []),
            "temperature": str(temperature),
            "max_output_tokens": str(max_output_tokens),
        }
        if generation_provider:
            data["generation_provider"] = generation_provider

        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(endpoint, files=files, data=data, headers=self._headers())
        try:
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            detail = self._parse_error_detail(exc)
            raise RuntimeError(f"Document RAG API returned HTTP {exc.response.status_code}: {detail}") from exc

    def list_my_assets(self, timeout_seconds: float = 30.0) -> list[dict[str, Any]]:
        endpoint = f"{self.base_url}/api/v1/ingestion/my-assets"
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.get(endpoint, headers=self._headers())
        try:
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else []
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Could not load private files: {self._parse_error_detail(exc)}") from exc

    def upload_document(
        self,
        *,
        project_id: int | None = None,
        file_bytes: bytes,
        file_name: str,
        timeout_seconds: float = 300.0,
    ) -> dict[str, Any]:
        endpoint = f"{self.base_url}/api/v1/ingestion/upload-index"
        content_type = "text/plain" if file_name.lower().endswith(".txt") else "application/pdf"
        files = {
            "file": (file_name, file_bytes, content_type),
        }
        data: dict[str, str] = {}
        if project_id is not None:
            data["project_id"] = str(project_id)

        try:
            with httpx.Client(timeout=timeout_seconds) as client:
                response = client.post(endpoint, files=files, data=data, headers=self._headers())
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            detail = self._parse_error_detail(exc)
            raise RuntimeError(f"Upload API returned HTTP {exc.response.status_code}: {detail}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Could not connect to the upload API at {endpoint}: {exc}") from exc

    upload_pdf = upload_document

    @staticmethod
    def _parse_error_detail(exc: httpx.HTTPStatusError) -> Any:
        try:
            body = exc.response.json()
            if isinstance(body, dict) and "detail" in body:
                return body["detail"]
            return body
        except Exception:
            return exc.response.text
