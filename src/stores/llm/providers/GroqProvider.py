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


class GroqProvider(GenerationInterface):
    """Generate grounded answers through the asynchronous Groq API."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "openai/gpt-oss-120b",
        timeout_seconds: float = 120.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Groq API key must not be empty")
        if not model_name.strip():
            raise ValueError("Groq model name must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        self._client = AsyncGroq(
            api_key=api_key,
            timeout=httpx.Timeout(timeout_seconds),
        )
        self._model_name = model_name
        self._closed = False

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

        try:
            response = await self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_prompt.strip()},
                ],
                temperature=temperature,
                max_completion_tokens=max_output_tokens,
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

        if not response.choices:
            raise GenerationEmptyResponseError(
                "Groq returned no completion choices.",
                provider="groq",
                status_code=502,
                retryable=True,
            )

        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise GenerationEmptyResponseError(
                "Groq returned an empty response.",
                provider="groq",
                status_code=502,
                retryable=True,
            )

        return GenerationResult(
            text=text,
            provider="groq",
            model=self._model_name,
            request_id=getattr(response, "id", None),
        )

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
