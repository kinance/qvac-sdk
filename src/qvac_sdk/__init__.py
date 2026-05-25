"""Python SDK for Tether QVAC — local-first AI inference on any device."""

from qvac_sdk.client import AsyncQVACClient, QVACClient
from qvac_sdk.config import QVACConfig
from qvac_sdk.models import (
    CompletionResponse,
    EmbeddingResponse,
    ImageResponse,
    TranscriptionResponse,
)

__version__ = "0.1.0"

__all__ = [
    "AsyncQVACClient",
    "CompletionResponse",
    "EmbeddingResponse",
    "ImageResponse",
    "QVACClient",
    "QVACConfig",
    "TranscriptionResponse",
]
