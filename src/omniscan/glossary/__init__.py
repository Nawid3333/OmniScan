"""Per-series glossary: SQLite working copy, human-editable YAML, particle-aware matcher."""

from omniscan.glossary.match import (
    JAPANESE_PARTICLES,
    KOREAN_PARTICLES,
    PARTICLES_BY_LANG,
    Match,
    find_terms,
    term_present,
)
from omniscan.glossary.store import GlossaryStore
from omniscan.glossary.yaml_io import export_yaml, import_yaml

__all__ = [
    "JAPANESE_PARTICLES",
    "KOREAN_PARTICLES",
    "PARTICLES_BY_LANG",
    "GlossaryStore",
    "Match",
    "export_yaml",
    "find_terms",
    "import_yaml",
    "term_present",
]
