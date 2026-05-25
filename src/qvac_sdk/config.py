"""Configuration for QVAC SDK."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class QVACConfig:
    """Configuration for connecting to a QVAC instance.

    Values are resolved in order: explicit argument > environment variable > default.
    """

    base_url: str = field(
        default_factory=lambda: os.environ.get("QVAC_BASE_URL", "http://localhost:8080")
    )
    timeout: float = field(
        default_factory=lambda: float(os.environ.get("QVAC_TIMEOUT", "30.0"))
    )
    model: str = field(
        default_factory=lambda: os.environ.get("QVAC_MODEL", "default")
    )

    @property
    def api_base(self) -> str:
        """Base URL normalized without trailing slash."""
        return self.base_url.rstrip("/")
