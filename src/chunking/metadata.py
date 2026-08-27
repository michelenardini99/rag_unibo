from pathlib import Path

from config import settings


def extract_metadata(json_path: Path, converted_root: Path, schema: list[str] | None = None) -> dict:
    """Derives metadata from the document's position in the folder tree under
    `converted_root`, e.g. `<anno_accademico>/<corso>/<categoria>/file.json`.
    `schema` names each directory level in order (default: `settings.metadata_schema`)
    so a different institution/domain can adapt the taxonomy via config instead
    of editing this function. Levels beyond the actual folder depth (e.g. a
    document without a `materia` subfolder) are left as `None`.
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
