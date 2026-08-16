from typing import Any

from .LLMEnums import LLMEnums
from .LLMInterface import LLMInterface
from .providers.CohereProvider import CohereProvider


class LLMFactory:
    @staticmethod
    def create(
        provider: LLMEnums | str,
        **provider_options: Any,
    ) -> LLMInterface:
        provider_value = provider.value if isinstance(provider, LLMEnums) else provider
        provider_value = provider_value.strip().upper()

        if provider_value == LLMEnums.COHERE.value:
            return CohereProvider(**provider_options)

        raise ValueError(f"Unsupported LLM provider: {provider_value}")
