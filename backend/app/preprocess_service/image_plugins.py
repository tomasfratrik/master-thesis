from __future__ import annotations

try:
    import pillow_avif  # type: ignore # noqa: F401

    AVIF_PLUGIN_AVAILABLE = True
except ImportError:
    AVIF_PLUGIN_AVAILABLE = False

