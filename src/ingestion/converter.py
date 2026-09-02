import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionStatus
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode
from ingestion.picture_descriptions import picture_descriptions_options
from ingestion.normalize import normalize_document


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
        do_picture_description=True,
        generate_picture_images=True,
        picture_description_options=picture_descriptions_options,
        enable_remote_services=True  # required: picture captioning calls the vLLM HTTP endpoint
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

    def file_already_converted(file: Path) -> bool:
        rel = file.relative_to(input_dir)
        target = (output_dir / rel).with_suffix(".json")
        return target.exists()

    # Reruns only convert new files, not the whole corpus — conversion (OCR,
    # picture captioning) is the slowest stage in the offline pipeline.
    paths = [f for f in paths if not file_already_converted(f)]
    
    for result in converter.convert_all(paths, raises_on_error=False):
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

    return {"failures": failures}


if __name__ == "__main__":
    from config import settings

    result = convert_corpus(
        settings.data_raw_dir,
        settings.data_converted_dir,
        device=settings.embeddings_device_id,
    )
    if result["failures"]:
        print(f"Conversione completata con {len(result['failures'])} errore/i:")
        for failure in result["failures"]:
            print(f"  - {failure}")
    else:
        print(f"Conversione completata senza errori. Output in {settings.data_converted_dir}")