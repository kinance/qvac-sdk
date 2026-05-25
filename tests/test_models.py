"""Tests for response models."""

from __future__ import annotations

from qvac_sdk.models import CompletionResponse, EmbeddingResponse, ImageResponse


class TestCompletionResponse:
    def test_basic(self) -> None:
        resp = CompletionResponse(text="hello", model="m1")
        assert resp.text == "hello"
        assert resp.model == "m1"
        assert resp.usage == {}


class TestEmbeddingResponse:
    def test_shape_empty(self) -> None:
        resp = EmbeddingResponse()
        assert resp.shape == (0, 0)

    def test_shape(self) -> None:
        resp = EmbeddingResponse(vectors=[[1.0, 2.0], [3.0, 4.0]], dimensions=2)
        assert resp.shape == (2, 2)


class TestImageResponse:
    def test_save(self, tmp_path: str) -> None:
        import pathlib

        out = pathlib.Path(tmp_path) / "test.png"
        resp = ImageResponse(data=b"fake-png-data", width=64, height=64)
        resp.save(str(out))
        assert out.read_bytes() == b"fake-png-data"
