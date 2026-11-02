import argparse
import io
import json
import multiprocessing as mp
import re
import time
from html import escape
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
from ingestion.picture_descriptions import check_vision_server, picture_descriptions_options
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
        # Described by the vision model's own server, see picture_descriptions.py.
        do_picture_description=settings.picture_description,
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


def backfill_url_sidecars(input_dir: Path, output_dir: Path) -> int:
    """Copies each raw file's URL sidecar (see `ingestion.web_crawler`) to its
    converted JSON's matching sidecar, for documents whose conversion was
    skipped by `file_already_converted` — they never got the chance to carry
    it over themselves. Kept as a sidecar rather than injected into Docling's
    own JSON, so it can't collide with or be dropped by Docling's own
    (de)serialization. A no-op for PDF/DOCX/PPTX sources: they were never a
    web page, so they never have a `.url` sidecar to begin with.

    Args:
        input_dir (Path): Root of raw documents.
        output_dir (Path): Root of converted-documents output.

    Returns:
        int: Number of sidecars backfilled.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    count = 0
    for raw_sidecar in input_dir.rglob("*.url"):
        raw_file = raw_sidecar.with_name(raw_sidecar.name[: -len(".url")])
        converted_json = output_dir / raw_file.relative_to(input_dir).parent / f"{raw_file.stem}.json"
        if not converted_json.exists():
            continue
        converted_sidecar = converted_json.with_name(converted_json.name + ".url")
        if not converted_sidecar.exists():
            converted_sidecar.write_text(raw_sidecar.read_text(encoding="utf-8"), encoding="utf-8")
            count += 1
    return count


# Interface controls inside <main> that carry no content of their own: tabs
# switching between course codes ("Codice 6673 Codice 8615"), filter forms
# with their dropdowns ("A.A. 2016/2017 ... A.A. 2010/2011 Vai").
_HTML_UI_SELECTORS = ("form", "select", "ul.tabs")


def _move_header_legends_below_table(soup: BeautifulSoup, main) -> None:
    """Moves "?" tooltips out of table header cells into a legend paragraph
    right after the table, e.g. the study plans' `<th>periodo<span
    class="legend">? L'arco di tempo... 1: 21/09/2026 - 23/12/2026 ...</span>`.
    Left in place, the tooltip becomes part of the column header, which the
    chunker repeats on every row of the table.
    """
    for table in main.find_all("table"):
        legends = []
        for legend in table.select("th span.legend"):
            header = legend.find_parent("th")
            legend.extract()
            text = legend.get_text("\n", strip=True).lstrip("?").strip()
            if text:
                legends.append((header.get_text(" ", strip=True), text))
        for header, text in reversed(legends):
            p = soup.new_tag("p")
            p.string = f"Legenda — {header}: {text}"
            table.insert_after(p)


def _flatten_accordion_lists(soup: BeautifulSoup, main) -> None:
    """Turns www.unibo.it accordion lists (`ul.accordion-box > li >
    button.accordion-title + div.accordion-content`) into plain sections: a
    heading from the button, then the panel's content.

    Left as list items, Docling concatenates all of a panel's paragraphs into
    the item's text and emits its nested lists after it, separating each list
    from the sentence introducing it — e.g. "ISEE pari o inferiore a
    27.000 €" lost "Hai l'esenzione totale ... se ti trovi in una delle
    seguenti condizioni:", so no chunk said what the threshold was for.

    The button becomes one level below the heading preceding the list, and
    the panel's own headings one level below that, so the chunker's heading
    path keeps the page's nesting ("1° anno" > "Importi, esenzioni..." >
    "Avere l'esenzione...").
    """
    boxes = main.select("ul.accordion-box")
    # Levels are read before any change: once a box is flattened, its
    # demoted headings would be found as the "preceding heading" of the next.
    levels = []
    for box in boxes:
        previous = box.find_previous(re.compile(r"^h[1-6]$"))
        levels.append(int(previous.name[1]) if previous else 2)

    for box, level in zip(boxes, levels):
        section = soup.new_tag("div")
        for item in box.find_all("li", recursive=False):
            button = item.find("button", recursive=False)
            panel = item.find("div", class_="accordion-content", recursive=False)
            if button and button.get_text(strip=True):
                title = soup.new_tag(f"h{min(level + 1, 6)}")
                title.string = button.get_text(" ", strip=True)
                section.append(title)
            if panel:
                for heading in panel.find_all(re.compile(r"^h[1-6]$")):
                    heading.name = f"h{min(level + 2, 6)}"
                section.append(panel.extract())
        box.replace_with(section)


def preprocess_html(html: str) -> str:
    """Reduces a crawled page to its actual content before Docling sees it.

    - Keeps only <main> (present on every corsi.unibo.it page, with the <h1>
      inside and the menu/footer/cookie banner outside): the site footer
      otherwise ends up glued to the page's last paragraph.
    - Un-hides collapsed accordion panels: Docling drops anything marked
      `aria-hidden="true"`/`hidden`, which on these pages is real content
      (contact cards, admission steps, exam sessions...).
    - Drops interface controls, see `_HTML_UI_SELECTORS`.
    - Turns accordion lists into plain sections, see `_flatten_accordion_lists`.

    Args:
        html (str): Raw page HTML.

    Returns:
        str: A minimal HTML document with the page title and <main>'s
        content, or `html` unchanged if the page has no <main>.
    """
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main")
    if main is None:
        return html

    for selector in _HTML_UI_SELECTORS:
        for tag in main.select(selector):
            tag.decompose()
    for tag in main.select('[aria-hidden="true"], [hidden]'):
        if tag.get_text(strip=True):
            tag.attrs.pop("aria-hidden", None)
            tag.attrs.pop("hidden", None)
    _flatten_accordion_lists(soup, main)
    _move_header_legends_below_table(soup, main)

    title = soup.title.string if soup.title and soup.title.string else ""
    return (
        f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>{escape(title)}</title></head>'
        f"<body>{main.decode_contents()}</body></html>"
    )


def _html_stream(path: Path, base_dir: Path) -> DocumentStream:
    """Wraps a preprocessed HTML page (see `preprocess_html`) as a Docling
    input stream, named by its path relative to `base_dir`."""
    html = preprocess_html(path.read_text(encoding="utf-8", errors="ignore"))
    return DocumentStream(name=str(path.relative_to(base_dir)), stream=io.BytesIO(html.encode("utf-8")))


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

    paths = [f for f in paths if not file_already_converted(f, base_dir, output_dir)]

    # HTML pages go in as preprocessed streams (see `_html_stream`), named by
    # their path relative to `base_dir` so each result can be mapped back to
    # its raw file; other formats go in as plain paths.
    sources = [_html_stream(p, base_dir) if p.suffix.lower() == ".html" else p for p in paths]
    raw_by_input = {
        (s.name if isinstance(s, DocumentStream) else str(s)): p for s, p in zip(sources, paths)
    }

    times: list[float] = []
    t_prev = time.perf_counter()
    for result in converter.convert_all(sources, raises_on_error=False):
        now = time.perf_counter()
        times.append(now - t_prev)
        t_prev = now

        raw_file = raw_by_input[str(result.input.file)]
        stem = raw_file.stem
        dest = output_dir / raw_file.relative_to(base_dir).parent
        dest.mkdir(parents=True, exist_ok=True)

        if result.status == ConversionStatus.SUCCESS:
            doc = result.document
            # A preprocessed stream would otherwise record the hash of the
            # stream, not of the raw file `file_already_converted` compares against.
            doc.origin.binary_hash = _raw_file_hash(raw_file)
            normalize_document(doc)
            doc.save_as_markdown(dest / f"{stem}.md", image_mode=ImageRefMode.REFERENCED)
            doc.save_as_json(dest / f"{stem}.json", image_mode=ImageRefMode.REFERENCED)
        else:
            failures.append(str(raw_file))

    if times:
        print(f"Converted {len(times)} document(s) in {sum(times):.2f}s "
              f"({sum(times) / len(times):.2f}s/document on average)")

    if input_dir.is_dir():
        backfilled = backfill_url_sidecars(input_dir, output_dir)
        if backfilled:
            print(f"Backfilled {backfilled} URL sidecar(s) for previously-converted pages.")

    return {"failures": failures}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type = Path, default=settings.data_raw_dir, help="Cartella sorgente dei documenti da convertire (PDF, DOCX, PPTX, HTML). Se non specificata, usa la cartella predefinita in config.py")
    parser.add_argument("--destination", type = Path, default=settings.data_converted_dir, help="Cartella di destinazione dei documenti convertiti (Markdown + JSON). Se non specificata, usa la cartella predefinita in config.py")
    args = parser.parse_args()

    if settings.picture_description:
        print(f"Descrizione delle immagini con {check_vision_server()} ({settings.vision_base_url})")
    else:
        print("Descrizione delle immagini disattivata (PICTURE_DESCRIPTION=false)")

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