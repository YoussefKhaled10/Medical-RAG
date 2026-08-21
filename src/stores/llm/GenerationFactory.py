from enum import StrEnum

from src.stores.llm.GenerationInterface import GenerationInterface
from src.stores.llm.providers.GeminiProvider import GeminiProvider
from src.stores.llm.providers.GroqProvider import GroqProvider
from src.stores.llm.providers.ManusProvider import ManusProvider
from src.stores.llm.providers.GlmProvider import (
    GlmProvider,
)


class GenerationProviderEnums(StrEnum):
    GEMINI = "gemini"
    GROQ = "groq"
    MANUS = "manus"
    GLM = "glm"


class GenerationFactory:
    @staticmethod
    def create(
        provider: str,
        *,
        api_key: str,
        model_name: str,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> GenerationInterface:
        try:
            provider_enum = GenerationProviderEnums(provider.lower())
        except ValueError as exc:
            supported = ", ".join(
                item.value for item in GenerationProviderEnums
            )
            raise ValueError(
                f"Unsupported generation provider: {provider}. "
                f"Supported providers: {supported}"
            ) from exc

        if provider_enum == GenerationProviderEnums.GEMINI:
            return GeminiProvider(
                api_key=api_key,
                model_name=model_name,
            )

        if provider_enum == GenerationProviderEnums.GROQ:
            return GroqProvider(
                api_key=api_key,
                model_name=model_name,
            )

        if provider_enum == GenerationProviderEnums.GLM:
            return GlmProvider(
                api_key=api_key,
                model_name=model_name,
                base_url=(
                    base_url
                    or "https://api.z.ai/api/paas/v4/"
                ),
                timeout_seconds=timeout_seconds or 120.0,
            )

        if provider_enum == GenerationProviderEnums.MANUS:
            return ManusProvider(
                api_key=api_key,
                model_name=model_name,
            )

        raise AssertionError("Unhandled generation provider")
