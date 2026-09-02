from pathlib import Path

from config import settings


def extract_metadata(json_path: Path, converted_root: Path, schema: list[str] | None = None) -> dict:
    """Derives document metadata from its folder position under `converted_root`.

    Args:
        json_path (Path): Path to the converted document JSON.
        converted_root (Path): Root of the converted-documents tree, e.g.
            `<anno>/<corso>/<categoria>/file.json`.
        schema (list[str] | None): Name for each directory level, in order
            (default `settings.metadata_schema`). Levels beyond the actual
            folder depth are left `None`.

    Returns:
        dict: Schema fields plus "stato", "source_file", "source_path".
    """
    schema = schema if schema is not None else settings.metadata_schema
    rel = json_path.relative_to(converted_root)
    dir_parts = rel.parts[:-1]

    metadata = {name: (dir_parts[i] if i < len(dir_parts) else None) for i, name in enumerate(schema)}
    return {
        **metadata,
        "stato": "vigente",
        "source_file": rel.name,
        "source_path": str(rel),
    }
