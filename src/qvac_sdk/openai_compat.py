"""OpenAI-compatible wrapper for QVAC.

QVAC exposes an OpenAI-compatible API via ``qvac serve openai``.
This module provides a thin wrapper that pre-configures the OpenAI
client to point at the local QVAC endpoint.

Usage::

    from qvac_sdk.openai_compat import QVACOpenAI

    client = QVACOpenAI()
    response = client.chat.completions.create(
        model="local",
        messages=[{"role": "user", "content": "Hello"}],
    )
"""

from __future__ import annotations

from qvac_sdk.config import QVACConfig

try:
    from openai import AsyncOpenAI, OpenAI
except ImportError as e:
    raise ImportError(
        "The openai package is required for OpenAI-compatible mode. "
        "Install it with: pip install qvac-sdk[openai]"
    ) from e


def QVACOpenAI(  # noqa: N802 — function named like a class for ergonomic API
    base_url: str | None = None,
    timeout: float | None = None,
) -> OpenAI:
    """Create an OpenAI client pointed at the local QVAC instance.

    Args:
        base_url: Override QVAC endpoint. Defaults to ``QVAC_BASE_URL`` env var
            or ``http://localhost:8080``.
        timeout: Request timeout in seconds.

    Returns:
        A configured ``openai.OpenAI`` client.
    """
    cfg = QVACConfig(
        **({"base_url": base_url} if base_url else {}),
        **({"timeout": timeout} if timeout is not None else {}),
    )
    return OpenAI(
        base_url=f"{cfg.api_base}/v1",
        api_key="not-needed",
        timeout=cfg.timeout,
    )


def AsyncQVACOpenAI(  # noqa: N802
    base_url: str | None = None,
    timeout: float | None = None,
) -> AsyncOpenAI:
    """Create an async OpenAI client pointed at the local QVAC instance."""
    cfg = QVACConfig(
        **({"base_url": base_url} if base_url else {}),
        **({"timeout": timeout} if timeout is not None else {}),
    )
    return AsyncOpenAI(
        base_url=f"{cfg.api_base}/v1",
        api_key="not-needed",
        timeout=cfg.timeout,
    )
