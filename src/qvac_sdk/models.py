"""Response models for QVAC SDK."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CompletionResponse(BaseModel):
    """Response from a completion request."""

    text: str
    model: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)


class CompletionChunk(BaseModel):
    """A single chunk from a streaming completion."""

    text: str
    done: bool = False


class ChatMessage(BaseModel):
    """A chat message."""

    role: str  # "system" | "user" | "assistant"
    content: str


class TranscriptionResponse(BaseModel):
    """Response from a speech-to-text request."""

    text: str
    language: str = ""
    duration_seconds: float = 0.0
    segments: list[dict[str, Any]] = Field(default_factory=list)


class TranslationResponse(BaseModel):
    """Response from a translation request."""

    text: str
    source_language: str = ""
    target_language: str = ""


class ImageResponse(BaseModel):
    """Response from an image generation request."""

    data: bytes = b""
    width: int = 0
    height: int = 0
    format: str = "png"

    model_config = {"arbitrary_types_allowed": True}

    def save(self, path: str) -> None:
        """Save image to file."""
        with open(path, "wb") as f:
            f.write(self.data)


class EmbeddingResponse(BaseModel):
    """Response from an embedding request."""

    vectors: list[list[float]] = Field(default_factory=list)
    model: str = ""
    dimensions: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        """Return (num_vectors, dimensions) shape tuple."""
        if not self.vectors:
            return (0, 0)
        return (len(self.vectors), len(self.vectors[0]))


class TTSResponse(BaseModel):
    """Response from a text-to-speech request."""

    audio: bytes = b""
    sample_rate: int = 22050
    format: str = "wav"

    model_config = {"arbitrary_types_allowed": True}

    def save(self, path: str) -> None:
        """Save audio to file."""
        with open(path, "wb") as f:
            f.write(self.audio)


class OCRResponse(BaseModel):
    """Response from an OCR request."""

    text: str
    confidence: float = 0.0
    regions: list[dict[str, Any]] = Field(default_factory=list)
