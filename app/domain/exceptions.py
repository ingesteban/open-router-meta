"""Domain errors. The presentation layer maps them to HTTP status codes."""


class DomainError(Exception):
    """Base class for expected, user-facing errors."""


class InvalidRequestError(DomainError):
    """The input to a use case is invalid (maps to 422)."""


class LLMProviderError(DomainError):
    """The LLM provider failed or returned an unusable response (maps to 502)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMAuthError(LLMProviderError):
    """Missing or invalid API key (maps to 401)."""


class LLMRateLimitError(LLMProviderError):
    """Rate limited or out of credits (maps to 429)."""
