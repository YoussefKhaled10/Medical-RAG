from google import genai
from google.genai import types

from src.stores.llm.GenerationInterface import (
    GenerationInterface,
    GenerationResult,
)


class GeminiProvider(GenerationInterface):
    """Generate text with Gemini using the asynchronous Google Gen AI client."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        if not api_key.strip():
            raise ValueError("Gemini API key must not be empty")
        if not model_name.strip():
            raise ValueError("Gemini model name must not be empty")

        self._client = genai.Client(api_key=api_key).aio
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
            raise RuntimeError("Gemini provider is already closed")
        if not system_prompt.strip():
            raise ValueError("system_prompt must not be empty")
        if not user_prompt.strip():
            raise ValueError("user_prompt must not be empty")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")

        response = await self._client.models.generate_content(
            model=self._model_name,
            contents=user_prompt.strip(),
            config=types.GenerateContentConfig(
                system_instruction=system_prompt.strip(),
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                automatic_function_calling=(
                    types.AutomaticFunctionCallingConfig(
                        disable=True,
                    )
                ),
            ),
        )

        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty response")

        return GenerationResult(
            text=text,
            provider="gemini",
            model=self._model_name,
            request_id=getattr(response, "response_id", None),
        )

    async def close(self) -> None:
        if self._closed:
            return
        await self._client.aclose()
        self._closed = True
