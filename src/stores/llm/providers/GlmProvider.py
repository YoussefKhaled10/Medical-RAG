import asyncio
import logging
from typing import Any

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
)

from src.stores.llm.GenerationExceptions import (
    GenerationAuthenticationError,
    GenerationEmptyResponseError,
    GenerationModelNotFoundError,
    GenerationProviderError,
    GenerationRateLimitError,
    GenerationTimeoutError,
)
from src.stores.llm.GenerationInterface import GenerationInterface, GenerationResult

logger = logging.getLogger(__name__)


class GlmProvider(GenerationInterface):
    """Z.AI GLM provider using the OpenAI-compatible chat API."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "glm-4.7-flash",
        base_url: str = "https://api.z.ai/api/paas/v4/",
        timeout_seconds: float = 120.0,
        empty_response_retries: int = 2,
        empty_response_base_delay_seconds: float = 2.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Z.AI API key must not be empty")
        if not model_name.strip():
            raise ValueError("GLM model name must not be empty")
        if not base_url.strip():
            raise ValueError("Z.AI base URL must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/") + "/",
            timeout=httpx.Timeout(timeout_seconds),
            max_retries=0,
        )
        self._model_name = model_name
        self._empty_response_retries = empty_response_retries
        self._empty_response_base_delay_seconds = empty_response_base_delay_seconds
        self._closed = False

    @staticmethod
    def _extract_text(response: Any) -> tuple[str, str | None]:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return "", None
        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) if message else None
        if isinstance(content, str):
            return content.strip(), finish_reason
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                text = part if isinstance(part, str) else getattr(part, "text", None)
                if isinstance(part, dict):
                    text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            return "\n".join(parts).strip(), finish_reason
        return "", finish_reason

    async def _create_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> Any:
        return await self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=temperature,
            max_tokens=max_output_tokens,
            stream=False,
        )

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_output_tokens: int = 1200,
    ) -> GenerationResult:
        if self._closed:
            raise RuntimeError("GLM provider is already closed")
        if not system_prompt.strip() or not user_prompt.strip():
            raise ValueError("GLM prompts must not be empty")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")

        total_attempts = self._empty_response_retries + 1
        last_failure = "empty content"
        for attempt_index in range(total_attempts):
            response = await self._request_with_error_mapping(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            request_id = getattr(response, "request_id", None) or getattr(response, "id", None)
            choices = getattr(response, "choices", None) or []
            text, finish_reason = self._extract_text(response)
            truncated = finish_reason == "length"
            logger.info(
                "GLM completion diagnostics. model=%s attempt=%s/%s request_id=%s "
                "finish_reason=%s content_length=%s usage=%s",
                self._model_name, attempt_index + 1, total_attempts, request_id,
                finish_reason, len(text), getattr(response, "usage", None),
            )
            if text and not truncated:
                return GenerationResult(
                    text=text,
                    provider="glm",
                    model=self._model_name,
                    request_id=request_id,
                )
            if truncated:
                last_failure = "truncated response"
            elif not choices:
                last_failure = "no completion choices"
            else:
                last_failure = "empty content"
            if attempt_index < total_attempts - 1:
                delay = self._empty_response_base_delay_seconds * (2 ** attempt_index)
                if delay > 0:
                    await asyncio.sleep(delay)

        raise GenerationEmptyResponseError(
            "GLM returned no complete answer after "
            f"{total_attempts} attempts. Last result: {last_failure}.",
            provider="glm",
            status_code=502,
            retryable=True,
        )

    async def _request_with_error_mapping(self, **kwargs: Any) -> Any:
        try:
            return await self._create_completion(**kwargs)
        except AuthenticationError as exc:
            raise GenerationAuthenticationError(
                "Z.AI rejected the configured API key.", provider="glm", status_code=401
            ) from exc
        except NotFoundError as exc:
            raise GenerationModelNotFoundError(
                f"GLM model is unavailable: {self._model_name}",
                provider="glm", status_code=503,
            ) from exc
        except RateLimitError as exc:
            retry_after = self._read_retry_after(exc)
            raise GenerationRateLimitError(
                "Z.AI rate limit reached. Retry the request later.",
                provider="glm", status_code=429, retryable=True,
                retry_after_seconds=retry_after,
            ) from exc
        except APITimeoutError as exc:
            raise GenerationTimeoutError(
                "Z.AI generation request timed out.",
                provider="glm", status_code=504, retryable=True,
            ) from exc
        except APIConnectionError as exc:
            raise GenerationProviderError(
                "Could not connect to Z.AI.",
                provider="glm", status_code=503, retryable=True,
            ) from exc
        except APIStatusError as exc:
            raise GenerationProviderError(
                f"Z.AI returned HTTP {exc.status_code}.",
                provider="glm", status_code=502, retryable=exc.status_code >= 500,
            ) from exc

    @staticmethod
    def _read_retry_after(exc: RateLimitError) -> float | None:
        response = getattr(exc, "response", None)
        value = response.headers.get("retry-after") if response is not None else None
        try:
            return float(value) if value is not None else None
        except ValueError:
            return None

    async def close(self) -> None:
        if self._closed:
            return
        await self._client.close()
        self._closed = True
