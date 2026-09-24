import argparse
import io
import json
import multiprocessing as mp
import time
from pathlib import Path
from queue import Empty
from bs4 import BeautifulSoup
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import ConversionStatus
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.utils.utils import create_file_hash
from docling_core.types.doc import ImageRefMode
from ingestion.picture_descriptions import picture_descriptions_options
from ingestion.normalize import normalize_document
from config import settings


def pipeline_options(device: int) -> PdfPipelineOptions:
    """Builds Docling's PDF pipeline options: OCR, table structure and picture
    description all enabled.

    Args:
        device (int): CUDA device index for Docling's local models (OCR, layout).

    Returns:
        PdfPipelineOptions: Options passed to `PdfFormatOption`.
    """
    return PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=f"cuda:{device}"),
        table_structure_options=TableStructureOptions(do_cell_matching=True),
        do_ocr=True,
        do_table_structure=True,
        # Captioning calls settings.generation_model over the API; skip it
        # when that deployment isn't vision-capable (see generation_model_vision
        # in prompt.py) — it would just 400 on every image otherwise.
        do_picture_description=settings.generation_model_vision,
        generate_picture_images=True,
        picture_description_options=picture_descriptions_options,
        enable_remote_services=True
    )

def build_converter(device: int) -> DocumentConverter:
    """Builds the Docling converter for the supported input formats.

    Args:
        device (int): CUDA device index for the PDF pipeline's local models.

    Returns:
        DocumentConverter: Converter for PDF, DOCX, HTML and PPTX.
    """
    return DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.DOCX,
            InputFormat.HTML,
            InputFormat.PPTX
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options(device),
            )
        }
    )


def _raw_file_hash(path: Path) -> int:
    """Hashes a raw source file the same way Docling hashes it internally
    (`DocumentOrigin.parse_hex_string`), so it can be compared directly
    against the `origin.binary_hash` a prior conversion already stored.

    Args:
        path (Path): Raw source file.

    Returns:
        int: Low 64 bits of the file's sha256, as an unsigned integer.
    """
    return int(create_file_hash(path), 16) & 0xFFFFFFFFFFFFFFFF


def file_already_converted(file: Path, input_dir: Path, output_dir: Path) -> bool:
    """Checks whether `file` has already been converted with its current
    content, so unchanged files are skipped and modified ones are redone.

    Args:
        file (Path): Raw source file.
        input_dir (Path): Root of raw documents `file` is relative to.
        output_dir (Path): Root of converted-documents output.

    Returns:
        bool: True if a converted JSON exists and its recorded source hash
            still matches `file`'s current content.
    """
    rel = file.relative_to(input_dir)
    target = (output_dir / rel).with_suffix(".json")
    if not target.exists():
        return False
    try:
        recorded_hash = json.loads(target.read_text(encoding="utf-8"))["origin"]["binary_hash"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return False
    return recorded_hash == _raw_file_hash(file)


def convert_corpus(input_dir: Path, output_dir: Path, device: int) -> dict[str, list[str]]:
    """Converts every unconverted document under `input_dir` to normalized
    Markdown + JSON under `output_dir`.

    Args:
        input_dir (Path): Root of raw documents, or a single file.
        output_dir (Path): Root to write converted Markdown/JSON to, mirroring
            `input_dir`'s folder structure.
        device (int): CUDA device index for Docling's local models.

    Returns:
        dict[str, list[str]]: {"failures": [...]} paths that failed to convert.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    converter = build_converter(device)
    if input_dir.is_file():
        paths = [input_dir]
        base_dir = input_dir.parent
    else:
        paths = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in {".pdf", ".docx", ".pptx", ".html"})
        base_dir = input_dir
    failures: list[str] = []

    paths = [f for f in paths if not file_already_converted(f, input_dir, output_dir)]

    times: list[float] = []
    t_prev = time.perf_counter()
    for result in converter.convert_all(paths, raises_on_error=False):
        now = time.perf_counter()
        times.append(now - t_prev)
        t_prev = now

        stem = result.input.file.stem
        dest = output_dir / result.input.file.relative_to(base_dir).parent
        dest.mkdir(parents=True, exist_ok=True)

        if result.status == ConversionStatus.SUCCESS:
            doc = result.document
            normalize_document(doc)
            doc.save_as_markdown(dest / f"{stem}.md", image_mode=ImageRefMode.REFERENCED)
            doc.save_as_json(dest / f"{stem}.json", image_mode=ImageRefMode.REFERENCED)
        else:
            failures.append(str(result.input.file))

    if times:
        print(f"Converted {len(times)} document(s) in {sum(times):.2f}s "
              f"({sum(times) / len(times):.2f}s/document on average)")

    return {"failures": failures}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type = Path, default=settings.data_raw_dir, help="Cartella sorgente dei documenti da convertire (PDF, DOCX, PPTX, HTML). Se non specificata, usa la cartella predefinita in config.py")
    parser.add_argument("--destination", type = Path, default=settings.data_converted_dir, help="Cartella di destinazione dei documenti convertiti (Markdown + JSON). Se non specificata, usa la cartella predefinita in config.py")
    args = parser.parse_args()

    result = convert_corpus(
        args.source,
        args.destination,
        device=settings.embeddings_device_id,
    )
    if result["failures"]:
        print(f"Conversione completata con {len(result['failures'])} errore/i:")
        for failure in result["failures"]:
            print(f"  - {failure}")
    else:
        print(f"Conversione completata senza errori. Output in {settings.data_converted_dir}")