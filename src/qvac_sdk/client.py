"""Sync and async clients for QVAC API."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from qvac_sdk.config import QVACConfig
from qvac_sdk.models import (
    CompletionChunk,
    CompletionResponse,
    EmbeddingResponse,
    ImageResponse,
    OCRResponse,
    TranscriptionResponse,
    TranslationResponse,
    TTSResponse,
)


class QVACClient:
    """Synchronous Python client for a local QVAC instance.

    Usage::

        client = QVACClient()
        response = client.complete("Hello, world!")
        print(response.text)
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        model: str | None = None,
    ) -> None:
        cfg = QVACConfig(
            **({"base_url": base_url} if base_url else {}),
            **({"timeout": timeout} if timeout is not None else {}),
            **({"model": model} if model else {}),
        )
        self._config = cfg
        self._http = httpx.Client(base_url=cfg.api_base, timeout=cfg.timeout)

    def close(self) -> None:
        """Close the underlying HTTP connection."""
        self._http.close()

    def __enter__(self) -> QVACClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # -- LLM completions --

    def complete(self, prompt: str, *, model: str | None = None, **kwargs: Any) -> CompletionResponse:
        """Run a single-turn LLM completion."""
        payload = {
            "prompt": prompt,
            "model": model or self._config.model,
            **kwargs,
        }
        resp = self._http.post("/v1/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return CompletionResponse(
            text=data.get("text", data.get("choices", [{}])[0].get("text", "")),
            model=data.get("model", ""),
            usage=data.get("usage", {}),
            raw=data,
        )

    def complete_stream(
        self, prompt: str, *, model: str | None = None, **kwargs: Any
    ) -> Iterator[CompletionChunk]:
        """Stream a completion token by token."""
        payload = {
            "prompt": prompt,
            "model": model or self._config.model,
            "stream": True,
            **kwargs,
        }
        with self._http.stream("POST", "/v1/completions", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                if line.strip() == "[DONE]":
                    break
                import json

                chunk_data = json.loads(line)
                yield CompletionChunk(
                    text=chunk_data.get("text", chunk_data.get("choices", [{}])[0].get("text", "")),
                    done=chunk_data.get("done", False),
                )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Multi-turn chat completion."""
        payload = {
            "messages": messages,
            "model": model or self._config.model,
            **kwargs,
        }
        resp = self._http.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        return CompletionResponse(
            text=choice.get("message", {}).get("content", ""),
            model=data.get("model", ""),
            usage=data.get("usage", {}),
            raw=data,
        )

    # -- Speech-to-text --

    def transcribe(self, audio_path: str | Path, **kwargs: Any) -> TranscriptionResponse:
        """Transcribe an audio file to text."""
        path = Path(audio_path)
        with open(path, "rb") as f:
            files = {"file": (path.name, f, "audio/wav")}
            resp = self._http.post("/v1/audio/transcriptions", files=files, data=kwargs)
        resp.raise_for_status()
        data = resp.json()
        return TranscriptionResponse(
            text=data.get("text", ""),
            language=data.get("language", ""),
            duration_seconds=data.get("duration", 0.0),
            segments=data.get("segments", []),
        )

    # -- Translation --

    def translate(
        self, text: str, *, target_language: str = "en", **kwargs: Any
    ) -> TranslationResponse:
        """Translate text to the target language."""
        payload = {"text": text, "target_language": target_language, **kwargs}
        resp = self._http.post("/v1/translations", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return TranslationResponse(
            text=data.get("text", ""),
            source_language=data.get("source_language", ""),
            target_language=data.get("target_language", target_language),
        )

    # -- Image generation --

    def generate_image(
        self, prompt: str, *, width: int = 512, height: int = 512, **kwargs: Any
    ) -> ImageResponse:
        """Generate an image from a text prompt."""
        payload = {"prompt": prompt, "width": width, "height": height, **kwargs}
        resp = self._http.post("/v1/images/generations", json=payload)
        resp.raise_for_status()
        return ImageResponse(
            data=resp.content,
            width=width,
            height=height,
        )

    # -- Embeddings --

    def embed(self, inputs: list[str], *, model: str | None = None, **kwargs: Any) -> EmbeddingResponse:
        """Generate embeddings for a list of texts."""
        payload = {"input": inputs, "model": model or self._config.model, **kwargs}
        resp = self._http.post("/v1/embeddings", json=payload)
        resp.raise_for_status()
        data = resp.json()
        vectors = [item["embedding"] for item in data.get("data", [])]
        return EmbeddingResponse(
            vectors=vectors,
            model=data.get("model", ""),
            dimensions=len(vectors[0]) if vectors else 0,
        )

    # -- Text-to-speech --

    def speak(self, text: str, **kwargs: Any) -> TTSResponse:
        """Convert text to speech audio."""
        payload = {"input": text, **kwargs}
        resp = self._http.post("/v1/audio/speech", json=payload)
        resp.raise_for_status()
        return TTSResponse(audio=resp.content)

    # -- OCR --

    def ocr(self, image_path: str | Path, **kwargs: Any) -> OCRResponse:
        """Extract text from an image."""
        path = Path(image_path)
        with open(path, "rb") as f:
            files = {"file": (path.name, f, "image/png")}
            resp = self._http.post("/v1/ocr", files=files, data=kwargs)
        resp.raise_for_status()
        data = resp.json()
        return OCRResponse(
            text=data.get("text", ""),
            confidence=data.get("confidence", 0.0),
            regions=data.get("regions", []),
        )


class AsyncQVACClient:
    """Async Python client for a local QVAC instance.

    Usage::

        async with AsyncQVACClient() as client:
            response = await client.complete("Hello, world!")
            print(response.text)
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        model: str | None = None,
    ) -> None:
        cfg = QVACConfig(
            **({"base_url": base_url} if base_url else {}),
            **({"timeout": timeout} if timeout is not None else {}),
            **({"model": model} if model else {}),
        )
        self._config = cfg
        self._http = httpx.AsyncClient(base_url=cfg.api_base, timeout=cfg.timeout)

    async def close(self) -> None:
        """Close the underlying HTTP connection."""
        await self._http.aclose()

    async def __aenter__(self) -> AsyncQVACClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def complete(
        self, prompt: str, *, model: str | None = None, **kwargs: Any
    ) -> CompletionResponse:
        """Run a single-turn LLM completion."""
        payload = {
            "prompt": prompt,
            "model": model or self._config.model,
            **kwargs,
        }
        resp = await self._http.post("/v1/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return CompletionResponse(
            text=data.get("text", data.get("choices", [{}])[0].get("text", "")),
            model=data.get("model", ""),
            usage=data.get("usage", {}),
            raw=data,
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Multi-turn chat completion."""
        payload = {
            "messages": messages,
            "model": model or self._config.model,
            **kwargs,
        }
        resp = await self._http.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        return CompletionResponse(
            text=choice.get("message", {}).get("content", ""),
            model=data.get("model", ""),
            usage=data.get("usage", {}),
            raw=data,
        )

    async def transcribe(
        self, audio_path: str | Path, **kwargs: Any
    ) -> TranscriptionResponse:
        """Transcribe an audio file to text."""
        path = Path(audio_path)
        with open(path, "rb") as f:
            files = {"file": (path.name, f, "audio/wav")}
            resp = await self._http.post("/v1/audio/transcriptions", files=files, data=kwargs)
        resp.raise_for_status()
        data = resp.json()
        return TranscriptionResponse(
            text=data.get("text", ""),
            language=data.get("language", ""),
            duration_seconds=data.get("duration", 0.0),
            segments=data.get("segments", []),
        )

    async def translate(
        self, text: str, *, target_language: str = "en", **kwargs: Any
    ) -> TranslationResponse:
        """Translate text to the target language."""
        payload = {"text": text, "target_language": target_language, **kwargs}
        resp = await self._http.post("/v1/translations", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return TranslationResponse(
            text=data.get("text", ""),
            source_language=data.get("source_language", ""),
            target_language=data.get("target_language", target_language),
        )

    async def generate_image(
        self, prompt: str, *, width: int = 512, height: int = 512, **kwargs: Any
    ) -> ImageResponse:
        """Generate an image from a text prompt."""
        payload = {"prompt": prompt, "width": width, "height": height, **kwargs}
        resp = await self._http.post("/v1/images/generations", json=payload)
        resp.raise_for_status()
        return ImageResponse(data=resp.content, width=width, height=height)

    async def embed(
        self, inputs: list[str], *, model: str | None = None, **kwargs: Any
    ) -> EmbeddingResponse:
        """Generate embeddings for a list of texts."""
        payload = {"input": inputs, "model": model or self._config.model, **kwargs}
        resp = await self._http.post("/v1/embeddings", json=payload)
        resp.raise_for_status()
        data = resp.json()
        vectors = [item["embedding"] for item in data.get("data", [])]
        return EmbeddingResponse(
            vectors=vectors,
            model=data.get("model", ""),
            dimensions=len(vectors[0]) if vectors else 0,
        )

    async def speak(self, text: str, **kwargs: Any) -> TTSResponse:
        """Convert text to speech audio."""
        payload = {"input": text, **kwargs}
        resp = await self._http.post("/v1/audio/speech", json=payload)
        resp.raise_for_status()
        return TTSResponse(audio=resp.content)

    async def ocr(self, image_path: str | Path, **kwargs: Any) -> OCRResponse:
        """Extract text from an image."""
        path = Path(image_path)
        with open(path, "rb") as f:
            files = {"file": (path.name, f, "image/png")}
            resp = await self._http.post("/v1/ocr", files=files, data=kwargs)
        resp.raise_for_status()
        data = resp.json()
        return OCRResponse(
            text=data.get("text", ""),
            confidence=data.get("confidence", 0.0),
            regions=data.get("regions", []),
        )
