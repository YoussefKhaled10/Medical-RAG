from enum import StrEnum

from src.stores.llm.GenerationInterface import GenerationInterface
from src.stores.llm.providers.GeminiProvider import GeminiProvider
from src.stores.llm.providers.GroqProvider import GroqProvider
from src.stores.llm.providers.ManusProvider import ManusProvider


class GenerationProviderEnums(StrEnum):
    GEMINI = "gemini"
    GROQ = "groq"
    MANUS = "manus"


class GenerationFactory:
    @staticmethod
    def create(
        provider: str,
        *,
        api_key: str,
        model_name: str,
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

        return ManusProvider(
            api_key=api_key,
            model_name=model_name,
        )
