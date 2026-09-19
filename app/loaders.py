"""Document loaders for .md, .txt, and .pdf files.

The org_id is derived from the directory name (data/<org_id>/), which means
a document physically cannot be loaded without an owner — the isolation
boundary starts at the filesystem level, before any indexing happens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawDocument:
    """Immutable container for a loaded document's text and provenance."""

    text: str
    filename: str
    organization_id: str


def _read_text(path: Path) -> str:
    """Read UTF-8 text, replacing encoding errors so we never crash on one bad file."""
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF via pypdf — lightweight and pure-Python."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pypdf is required for PDF loading: pip install pypdf") from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


_LOADERS: dict[str, callable] = {
    ".md": _read_text,
    ".txt": _read_text,
    ".pdf": _read_pdf,
}


def load_documents(org_id: str) -> list[RawDocument]:
    """Walk data/<org_id>/ and return every supported document.

    Raises FileNotFoundError if the org directory doesn't exist — callers
    treat this as "no data" rather than a silent empty search.
    """
    org_dir = (settings.DATA_DIR / org_id).resolve()

    # Verify the resolved path is inside DATA_DIR to block symlink/traversal tricks.
    data_root = settings.DATA_DIR.resolve()
    if not str(org_dir).startswith(str(data_root)):
        raise ValueError(f"Path traversal blocked: {org_id!r}")

    if not org_dir.is_dir():
        raise FileNotFoundError(f"No data directory for organization {org_id!r}")

    documents: list[RawDocument] = []
    for path in sorted(org_dir.iterdir()):
        loader = _LOADERS.get(path.suffix.lower())
        if loader is None:
            logger.debug("Skipping unsupported file: %s", path.name)
            continue

        try:
            text = loader(path)
            if text.strip():
                documents.append(RawDocument(text=text, filename=path.name, organization_id=org_id))
                logger.info("Loaded %s (%d chars)", path.name, len(text))
            else:
                logger.warning("Empty file skipped: %s", path.name)
        except Exception:
            logger.exception("Failed to load %s", path.name)

    return documents
