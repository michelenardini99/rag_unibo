import json
import uuid
from datetime import date
from pathlib import Path

from config import settings

ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "rag-unibo")


def current_academic_year(today: date | None = None, start_month: int = settings.academic_year_start_month) -> str:
    """Computes the Italian academic year (e.g. "2026/2027") a document is
    being processed in, rather than one encoded in its folder path.

    Args:
        today (date | None): Date to compute from. Defaults to `date.today()`.
        start_month (int): Month the academic year starts on (1-12).

    Returns:
        str: "<start_year>/<start_year + 1>".
    """
    today = today or date.today()
    start_year = today.year if today.month >= start_month else today.year - 1
    return f"{start_year}/{start_year + 1}"


def read_checksum(json_path: Path) -> str | None:
    """Reads Docling's own hash of the original source file out of a
    converted document JSON, so change-detection can reuse it instead of
    hashing anything ourselves.

    Docling stores this as `origin.binary_hash` (the low 64 bits of a sha256
    of the raw file, see `DocumentOrigin.parse_hex_string` in docling_core),
    computed once at conversion time via `docling.utils.utils.create_file_hash`.
    Returned as a hex string rather than the raw int: Qdrant's payload
    integers are signed 64-bit, and this unsigned value can exceed that range.

    Args:
        json_path (Path): Path to the converted document JSON.

    Returns:
        str | None: The checksum as a 16-hex-digit string, or None if the
            file can't be read/parsed or doesn't carry one (e.g. a
            non-Docling JSON).
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return f"{data['origin']['binary_hash']:016x}"
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def read_source_url(json_path: Path) -> str | None:
    """Reads a document's original web page URL, if it has one (see
    `ingestion.web_crawler`'s sidecar, carried over by
    `ingestion.converter.backfill_url_sidecars`). PDF/DOCX/PPTX documents
    never have one — they were never a web page to begin with.

    Args:
        json_path (Path): Path to the converted document JSON.

    Returns:
        str | None: The page's URL, or None if there isn't one.
    """
    sidecar = json_path.with_name(json_path.name + ".url")
    try:
        return sidecar.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def extract_metadata(json_path: Path, converted_root: Path, schema: list[str] | None = None) -> dict:
    """Derives document metadata from its folder position under `converted_root`.

    Args:
        json_path (Path): Path to the converted document JSON.
        converted_root (Path): Root of the converted-documents tree.
        schema (list[str] | None): Name for each directory level, in order.
            Defaults to `settings.metadata_schema`. Levels beyond the actual
            folder depth are left `None`.

    Returns:
        dict: Schema fields plus "anno_accademico", "stato", "id",
        "checksum", "source_file", "source_path" and "source_url".
    """
    rel = json_path.relative_to(converted_root)
    dir_parts = rel.parts[:-1]
    schema = schema or settings.metadata_schema

    metadata = {name: (dir_parts[i] if i < len(dir_parts) else None) for i, name in enumerate(schema)}
    return {
        **metadata,
        "anno_accademico": current_academic_year(),
        "stato": "vigente",
        "id": str(uuid.uuid5(ID_NAMESPACE, str(rel))),
        "checksum": read_checksum(json_path),
        "source_file": rel.name,
        "source_path": str(rel),
        "source_url": read_source_url(json_path),
    }
