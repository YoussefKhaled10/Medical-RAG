from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    provider: str
    model: str
    request_id: str | None = None


class GenerationInterface(ABC):
    """Common contract for answer-generation providers."""

    @abstractmethod
    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_output_tokens: int = 1200,
    ) -> GenerationResult:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError
