from typing import Any

from .VectorDBEnums import VectorDBEnums
from .providers.PGVector_Provider import PGVector_Provider
from .VectorDBInterface import VectorDBInterface


class VectorDBFactory:
    @staticmethod
    def create(
        provider: VectorDBEnums | str,
        database_url: str,
        embedding_dimension: int,
        **provider_options: Any,
    ) -> VectorDBInterface:
        provider_value = (
            provider.value if isinstance(provider, VectorDBEnums) else provider
        )
        provider_value = provider_value.strip().upper()

        if provider_value == VectorDBEnums.PGVECTOR.value:
            return PGVector_Provider(
                database_url=database_url,
                embedding_dimension=embedding_dimension,
                **provider_options,
            )

        raise ValueError(f"Unsupported vector database provider: {provider_value}")
