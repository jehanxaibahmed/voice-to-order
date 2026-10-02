"""Exception hierarchy. Every error raised by the pipeline derives from VoiceToOrderError."""


class VoiceToOrderError(Exception):
    """Base class for all pipeline errors."""


class AudioError(VoiceToOrderError):
    """Audio could not be probed or converted."""


class TranscriptionError(VoiceToOrderError):
    """A speech-to-text provider failed."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.message = message


class LLMError(VoiceToOrderError):
    """The LLM call failed, was refused, or returned unusable output."""


class ExtractionError(VoiceToOrderError):
    """An order could not be extracted from the transcript."""
