import asyncio
import json
import urllib.parse
import urllib.request
from typing import Any

from src.stores.llm.GenerationInterface import (
    GenerationInterface,
    GenerationResult,
)


class ManusProvider(GenerationInterface):
    """Optional Manus v2 asynchronous task provider."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "manus-1.6",
        base_url: str = "https://api.manus.ai",
        poll_interval_seconds: float = 2.0,
        timeout_seconds: float = 120.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Manus API key must not be empty")
        if not model_name.strip():
            raise ValueError("Manus agent profile must not be empty")

        self._api_key = api_key
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._poll_interval = poll_interval_seconds
        self._timeout = timeout_seconds

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = (
            json.dumps(payload).encode("utf-8")
            if payload is not None
            else None
        )
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            data=data,
            method=method,
            headers={
                "x-manus-api-key": self._api_key,
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _extract_task_id(response: dict[str, Any]) -> str:
        for source in (response, response.get("data") or {}):
            if isinstance(source, dict):
                for key in ("task_id", "id"):
                    if source.get(key):
                        return str(source[key])
        raise RuntimeError(
            f"Manus task.create returned no task id: {response}"
        )

    @staticmethod
    def _extract_status_and_answer(
        response: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        messages = response.get("messages") or response.get("data") or []
        if isinstance(messages, dict):
            messages = (
                messages.get("messages")
                or messages.get("items")
                or []
            )

        status: str | None = None
        answer: str | None = None

        for event in messages:
            if not isinstance(event, dict):
                continue

            if event.get("type") == "status_update":
                status = (
                    event.get("status_update") or {}
                ).get("agent_status")

            if event.get("type") == "assistant_message":
                message = event.get("assistant_message") or event
                content = message.get("content")
                if isinstance(content, str):
                    answer = content.strip() or answer
                elif isinstance(content, list):
                    parts = [
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict)
                    ]
                    combined = "\n".join(
                        part for part in parts if part
                    ).strip()
                    answer = combined or answer

        return status, answer

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_output_tokens: int = 1200,
    ) -> GenerationResult:
        del temperature, max_output_tokens

        instruction = (
            f"{system_prompt}\n\n"
            f"USER REQUEST:\n{user_prompt}"
        )

        created = await asyncio.to_thread(
            self._request,
            "POST",
            "/v2/task.create",
            {
                "message": {
                    "content": [
                        {"type": "text", "text": instruction}
                    ]
                },
                "interactive_mode": False,
                "hide_in_task_list": True,
                "share_visibility": "private",
                "agent_profile": self._model_name,
                "title": "Medical RAG grounded answer",
            },
        )
        task_id = self._extract_task_id(created)
        deadline = (
            asyncio.get_running_loop().time() + self._timeout
        )

        while asyncio.get_running_loop().time() < deadline:
            query_string = urllib.parse.urlencode(
                {
                    "task_id": task_id,
                    "order": "desc",
                    "limit": 20,
                }
            )
            response = await asyncio.to_thread(
                self._request,
                "GET",
                f"/v2/task.listMessages?{query_string}",
            )
            status, answer = self._extract_status_and_answer(response)

            if status == "stopped" and answer:
                return GenerationResult(
                    text=answer,
                    provider="manus",
                    model=self._model_name,
                    request_id=task_id,
                )
            if status == "error":
                raise RuntimeError(f"Manus task failed: {response}")
            if status == "waiting":
                raise RuntimeError(
                    "Manus task requires interactive confirmation"
                )

            await asyncio.sleep(self._poll_interval)

        raise TimeoutError(
            f"Manus task {task_id} exceeded {self._timeout}s"
        )

    async def close(self) -> None:
        return None
