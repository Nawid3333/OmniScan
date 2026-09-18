"""Rich logging setup for OmniScan."""

import logging

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging with a rich handler (idempotent: calling twice must not add a second handler)."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,  # replaces any previous handlers, so repeat calls never stack a second one
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # per-request INFO logs are noise
