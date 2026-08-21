import asyncio
import logging
from typing import Any

import httpx
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncGroq,
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
from src.stores.llm.GenerationInterface import (
    GenerationInterface,
    GenerationResult,
)


logger = logging.getLogger(__name__)


class GroqProvider(GenerationInterface):
    """Generate grounded answers through Groq with resilient retries."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "openai/gpt-oss-120b",
        timeout_seconds: float = 120.0,
        empty_response_retries: int = 2,
        empty_response_base_delay_seconds: float = 2.0,
        minimum_completion_tokens: int = 3000,
        retry_token_increment: int = 1000,
        reasoning_effort: str = "low",
        reasoning_format: str = "hidden",
    ) -> None:
        if not api_key.strip():
            raise ValueError("Groq API key must not be empty")
        if not model_name.strip():
            raise ValueError("Groq model name must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if empty_response_retries < 0:
            raise ValueError("empty_response_retries must be zero or greater")
        if empty_response_base_delay_seconds < 0:
            raise ValueError(
                "empty_response_base_delay_seconds must be zero or greater"
            )
        if minimum_completion_tokens <= 0:
            raise ValueError("minimum_completion_tokens must be greater than zero")
        if retry_token_increment < 0:
            raise ValueError("retry_token_increment must be zero or greater")
        if reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("reasoning_effort must be low, medium, or high")
        if reasoning_format not in {"hidden", "parsed", "raw"}:
            raise ValueError("reasoning_format must be hidden, parsed, or raw")

        self._client = AsyncGroq(
            api_key=api_key,
            timeout=httpx.Timeout(timeout_seconds),
        )
        self._model_name = model_name
        self._empty_response_retries = empty_response_retries
        self._empty_response_base_delay_seconds = (
            empty_response_base_delay_seconds
        )
        self._minimum_completion_tokens = minimum_completion_tokens
        self._retry_token_increment = retry_token_increment
        self._reasoning_effort = reasoning_effort
        self._reasoning_format = reasoning_format
        self._closed = False

    @staticmethod
    def _extract_text(response: Any) -> tuple[str, str | None]:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return "", None

        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        message = getattr(choice, "message", None)
        if message is None:
            return "", finish_reason

        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip(), finish_reason

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                item_text = getattr(item, "text", None)
                if isinstance(item, dict):
                    item_text = item.get("text")
                if isinstance(item_text, str):
                    parts.append(item_text)
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
            max_completion_tokens=max_output_tokens,
            reasoning_effort=self._reasoning_effort,
            reasoning_format=self._reasoning_format,
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
            raise RuntimeError("Groq provider is already closed")
        if not system_prompt.strip():
            raise ValueError("system_prompt must not be empty")
        if not user_prompt.strip():
            raise ValueError("user_prompt must not be empty")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")

        total_attempts = self._empty_response_retries + 1
        last_request_id: str | None = None
        last_finish_reason: str | None = None
        last_failure_kind = "empty content"

        for attempt_index in range(total_attempts):
            attempt_token_budget = max(
                max_output_tokens,
                self._minimum_completion_tokens,
            ) + (self._retry_token_increment * attempt_index)

            response = await self._request_with_error_mapping(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=attempt_token_budget,
            )

            last_request_id = getattr(response, "id", None)
            choices = getattr(response, "choices", None) or []
            text, last_finish_reason = self._extract_text(response)
            usage = getattr(response, "usage", None)
            is_truncated = last_finish_reason == "length"

            logger.info(
                "Groq completion diagnostics. model=%s attempt=%s/%s "
                "token_budget=%s request_id=%s finish_reason=%s "
                "content_length=%s usage=%s",
                self._model_name,
                attempt_index + 1,
                total_attempts,
                attempt_token_budget,
                last_request_id,
                last_finish_reason,
                len(text),
                usage,
            )

            if text and not is_truncated:
                if attempt_index > 0:
                    logger.info(
                        "Groq retry succeeded. model=%s attempt=%s "
                        "request_id=%s",
                        self._model_name,
                        attempt_index + 1,
                        last_request_id,
                    )
                return GenerationResult(
                    text=text,
                    provider="groq",
                    model=self._model_name,
                    request_id=last_request_id,
                )

            if is_truncated:
                last_failure_kind = "truncated response"
            elif not choices:
                last_failure_kind = "no completion choices"
            else:
                last_failure_kind = "empty content"

            logger.warning(
                "Groq returned %s. model=%s attempt=%s/%s "
                "token_budget=%s finish_reason=%s request_id=%s",
                last_failure_kind,
                self._model_name,
                attempt_index + 1,
                total_attempts,
                attempt_token_budget,
                last_finish_reason,
                last_request_id,
            )

            if attempt_index < total_attempts - 1:
                delay = self._empty_response_base_delay_seconds * (
                    2 ** attempt_index
                )
                if delay > 0:
                    await asyncio.sleep(delay)

        raise GenerationEmptyResponseError(
            "Groq returned no complete answer after "
            f"{total_attempts} attempts. Last result: {last_failure_kind}.",
            provider="groq",
            status_code=502,
            retryable=True,
        )

    async def _request_with_error_mapping(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> Any:
        try:
            return await self._create_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        except AuthenticationError as exc:
            raise GenerationAuthenticationError(
                "Groq rejected the configured API key.",
                provider="groq",
                status_code=401,
            ) from exc
        except NotFoundError as exc:
            raise GenerationModelNotFoundError(
                f"Groq model is unavailable: {self._model_name}",
                provider="groq",
                status_code=503,
            ) from exc
        except RateLimitError as exc:
            retry_after = self._read_retry_after(exc)
            raise GenerationRateLimitError(
                "Groq rate limit reached. Retry the request later.",
                provider="groq",
                status_code=429,
                retryable=True,
                retry_after_seconds=retry_after,
            ) from exc
        except APITimeoutError as exc:
            raise GenerationTimeoutError(
                "Groq generation request timed out.",
                provider="groq",
                status_code=504,
                retryable=True,
            ) from exc
        except APIConnectionError as exc:
            raise GenerationProviderError(
                "Could not connect to Groq.",
                provider="groq",
                status_code=503,
                retryable=True,
            ) from exc
        except APIStatusError as exc:
            raise GenerationProviderError(
                f"Groq returned HTTP {exc.status_code}.",
                provider="groq",
                status_code=502,
                retryable=exc.status_code >= 500,
            ) from exc

    @staticmethod
    def _read_retry_after(exc: RateLimitError) -> float | None:
        response = getattr(exc, "response", None)
        if response is None:
            return None
        value = response.headers.get("retry-after")
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    async def close(self) -> None:
        if self._closed:
            return
        await self._client.close()
        self._closed = True
