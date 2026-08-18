class GenerationProviderError(RuntimeError):
    """Base error raised by answer-generation providers."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int = 502,
        retryable: bool = False,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds


class GenerationAuthenticationError(GenerationProviderError):
    """Raised when a provider rejects the API credentials."""


class GenerationModelNotFoundError(GenerationProviderError):
    """Raised when the configured model is unavailable."""


class GenerationRateLimitError(GenerationProviderError):
    """Raised when a provider rate limit is reached."""


class GenerationTimeoutError(GenerationProviderError):
    """Raised when a provider request times out."""


class GenerationEmptyResponseError(GenerationProviderError):
    """Raised when a provider returns no usable answer text."""
