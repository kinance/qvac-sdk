"""Tests for QVACClient."""

from __future__ import annotations

import json

import pytest
from pytest_httpx import HTTPXMock

from qvac_sdk import QVACClient
from qvac_sdk.models import CompletionResponse, EmbeddingResponse


class TestComplete:
    def test_basic_completion(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url="http://localhost:8080/v1/completions",
            json={
                "text": "Hello, world!",
                "model": "test-model",
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
        )

        client = QVACClient()
        resp = client.complete("Say hello")

        assert isinstance(resp, CompletionResponse)
        assert resp.text == "Hello, world!"
        assert resp.model == "test-model"
        assert resp.usage["prompt_tokens"] == 5
        client.close()

    def test_openai_format_response(self, httpx_mock: HTTPXMock) -> None:
        """QVAC may return OpenAI-style responses."""
        httpx_mock.add_response(
            url="http://localhost:8080/v1/completions",
            json={
                "choices": [{"text": "Hello from OpenAI format"}],
                "model": "local",
                "usage": {},
            },
        )

        with QVACClient() as client:
            resp = client.complete("test")
            assert resp.text == "Hello from OpenAI format"


class TestChat:
    def test_chat_completion(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url="http://localhost:8080/v1/chat/completions",
            json={
                "choices": [{"message": {"role": "assistant", "content": "Hi there"}}],
                "model": "local",
                "usage": {},
            },
        )

        with QVACClient() as client:
            resp = client.chat([{"role": "user", "content": "Hello"}])
            assert resp.text == "Hi there"


class TestEmbed:
    def test_embeddings(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url="http://localhost:8080/v1/embeddings",
            json={
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]},
                ],
                "model": "embed-model",
            },
        )

        with QVACClient() as client:
            resp = client.embed(["one", "two"])
            assert isinstance(resp, EmbeddingResponse)
            assert resp.shape == (2, 3)
            assert resp.vectors[0] == [0.1, 0.2, 0.3]


class TestConfig:
    def test_custom_base_url(self) -> None:
        client = QVACClient(base_url="http://myhost:9090")
        assert client._config.api_base == "http://myhost:9090"
        client.close()

    def test_trailing_slash_stripped(self) -> None:
        client = QVACClient(base_url="http://myhost:9090/")
        assert client._config.api_base == "http://myhost:9090"
        client.close()
