from pathlib import Path

def extract_metadata(json_path: Path, converted_root = Path("dataset/converted/")) -> dict:
    rel = json_path.relative_to(converted_root)
    parts = rel.parts
    anno_accademico, corso, categoria = parts[0], parts[1], parts[2]
    materia = parts[3] if categoria == "corsi" and len(parts) > 4 else None
    return {
        "anno_accademico": anno_accademico,
        "corso": corso,
        "categoria": categoria,
        "materia": materia,
        "stato": "vigente",
        "source_file": rel.name,
        "source_path": str(rel),
    }