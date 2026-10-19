import re
from collections import Counter

from docling_core.types.doc import ContentLayer, DoclingDocument, SectionHeaderItem, TableItem
from docling_core.types.doc.document import TableData
from docling_core.types.doc.labels import DocItemLabel

_ARTICLE_RE = re.compile(r"art\.?\s*\d+", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"[.!?:;]\s*$")
_MERGEABLE_LABELS = {DocItemLabel.TEXT, DocItemLabel.LIST_ITEM}
_BOILERPLATE_MIN_REPEATS = 3
# Shorter repeated texts are punctuation, empty inline wrappers or link
# labels ("Studenti Online"), not running headers/footers.
_BOILERPLATE_MIN_CHARS = 15
# A real article header is a short line ("Art. 5 - Iscrizione"), not a
# paragraph that happens to open with "Art. 5 del regolamento prevede...".
_ARTICLE_HEADER_MAX_CHARS = 120


def normalize_document(doc: DoclingDocument) -> None:
    """Fixes recurring Docling layout-detection quirks on Ateneo regolamenti
    (misclassified/split/merged headers, stray boilerplate, fragmented
    paragraphs, paginated tables).

    Only paginated documents (PDF) are touched: these are layout-detection
    fixes, while HTML/DOCX already carry their structure in the markup —
    applied to web pages they deleted page titles repeated in the menu
    (with the whole page under them) and collapsed bullet lists into one line.

    Args:
        doc (DoclingDocument): Document to fix, mutated in place.
    """
    if not doc.pages:
        return
    _fix_mislabeled_article_headers(doc)
    _split_merged_part_and_article(doc)
    _merge_split_article_number_and_title(doc)
    _strip_repeated_boilerplate(doc)
    _merge_fragmented_paragraphs(doc)
    _merge_paginated_tables(doc)


def _is_in_table(doc: DoclingDocument, item) -> bool:
    """True if `item` is the content of a table cell. Docling represents a
    formatted cell (e.g. `<td><p><strong>...</strong></p></td>` in HTML) as
    a rich cell whose text items live in `doc.texts` like any paragraph —
    the text-level fixes below must leave them alone, or they get merged
    into / deleted from the surrounding text and the cell ends up empty.
    """
    parent = item.parent
    while parent is not None:
        node = parent.resolve(doc)
        if isinstance(node, TableItem):
            return True
        parent = getattr(node, "parent", None)
    return False


def _fix_mislabeled_article_headers(doc: DoclingDocument) -> None:
    """Re-labels article headers (e.g. "Art. 5") that Docling classified as
    plain text instead of a section header.

    Args:
        doc (DoclingDocument): Document to fix, mutated in place.
    """
    default_level = next((t.level for t in doc.texts if isinstance(t, SectionHeaderItem)), 1)
    for item in list(doc.texts):
        if isinstance(item, SectionHeaderItem) or _is_in_table(doc, item):
            continue
        text = item.text.strip()
        if _ARTICLE_RE.match(text) and len(text) <= _ARTICLE_HEADER_MAX_CHARS:
            doc.insert_heading(sibling=item, text=item.text, level=default_level, after=True)
            doc.delete_items(node_items=[item])


def _split_merged_part_and_article(doc: DoclingDocument) -> None:
    """Splits a header where a Part title and the following Article title
    were merged onto a single line into two separate headers.

    Args:
        doc (DoclingDocument): Document to fix, mutated in place.
    """
    for item in list(doc.texts):
        if not isinstance(item, SectionHeaderItem):
            continue
        text = item.text.strip()
        match = _ARTICLE_RE.search(text)
        if match is None or match.start() == 0:
            continue  # not merged with a preceding Part title
        part_title, article_title = text[: match.start()].strip(), text[match.start() :].strip()
        item.text = item.orig = part_title
        doc.insert_heading(sibling=item, text=article_title, level=item.level, after=True)


def _merge_split_article_number_and_title(doc: DoclingDocument) -> None:
    """Merges an article number header (e.g. "Art. 5") with its title header
    when Docling split them into two separate headers.

    Args:
        doc (DoclingDocument): Document to fix, mutated in place.
    """
    i = 0
    while i < len(doc.texts) - 1:
        cur, nxt = doc.texts[i], doc.texts[i + 1]
        cur_text = cur.text.strip()
        if (
            isinstance(cur, SectionHeaderItem)
            and isinstance(nxt, SectionHeaderItem)
            and _ARTICLE_RE.fullmatch(cur_text)
            and _ARTICLE_RE.match(nxt.text.strip()) is None
        ):
            cur.text = cur.orig = f"{cur_text} {nxt.text.strip()}"
            doc.delete_items(node_items=[nxt])
            continue  # re-check cur against the new next sibling
        i += 1


def _strip_repeated_boilerplate(doc: DoclingDocument) -> None:
    """Removes running headers/footers Docling's furniture detection missed:
    any leaf body text of `_BOILERPLATE_MIN_CHARS`+ characters repeated
    `_BOILERPLATE_MIN_REPEATS`+ times verbatim is boilerplate. Items with
    children are never deleted — that would take their whole subtree along.

    Args:
        doc (DoclingDocument): Document to fix, mutated in place.
    """
    candidates = [
        item for item in doc.texts
        if item.content_layer == ContentLayer.BODY
        and not item.children
        and len(item.text.strip()) >= _BOILERPLATE_MIN_CHARS
        and not _is_in_table(doc, item)
    ]
    counts = Counter(item.text.strip() for item in candidates)
    boilerplate = [item for item in candidates if counts[item.text.strip()] >= _BOILERPLATE_MIN_REPEATS]
    if boilerplate:
        doc.delete_items(node_items=boilerplate)


def _merge_fragmented_paragraphs(doc: DoclingDocument) -> None:
    """Merges consecutive text/list-item blocks that don't end in sentence
    punctuation back into a single paragraph — only siblings (same parent,
    same content layer), so a paragraph never absorbs a footer, a table cell
    or the first item of a list that follows it.

    Args:
        doc (DoclingDocument): Document to fix, mutated in place.
    """
    i = 0
    while i < len(doc.texts) - 1:
        cur, nxt = doc.texts[i], doc.texts[i + 1]
        if (
            cur.label in _MERGEABLE_LABELS
            and nxt.label in _MERGEABLE_LABELS
            and not _SENTENCE_END_RE.search(cur.text.strip())
            and cur.parent.cref == nxt.parent.cref
            and cur.content_layer == nxt.content_layer
            and not _is_in_table(doc, cur)
            and not _is_in_table(doc, nxt)
        ):
            cur.text = cur.orig = f"{cur.text.rstrip()} {nxt.text.lstrip()}"
            doc.delete_items(node_items=[nxt])
            continue  # re-check cur against the new next sibling
        i += 1


def _merge_paginated_tables(doc: DoclingDocument) -> None:
    """Merges a table TableFormer split across a page break (same column
    count, adjacent, no repeated header row) — fee/deadline tables must stay
    whole and literal.

    Args:
        doc (DoclingDocument): Document to fix, mutated in place.
    """
    changed = True
    while changed:
        changed = False
        children = doc.body.children
        for k in range(len(children) - 1):
            cur_ref, nxt_ref = children[k].cref, children[k + 1].cref
            if not (cur_ref.startswith("#/tables/") and nxt_ref.startswith("#/tables/")):
                continue
            cur = doc.tables[int(cur_ref.rsplit("/", 1)[-1])]
            nxt = doc.tables[int(nxt_ref.rsplit("/", 1)[-1])]
            if cur.data.num_cols != nxt.data.num_cols:
                continue
            row_offset = cur.data.num_rows
            merged_cells = list(cur.data.table_cells)
            for cell in nxt.data.table_cells:
                merged_cells.append(
                    cell.model_copy(
                        update={
                            "start_row_offset_idx": cell.start_row_offset_idx + row_offset,
                            "end_row_offset_idx": cell.end_row_offset_idx + row_offset,
                        }
                    )
                )
            merged_data = TableData(
                table_cells=merged_cells,
                num_rows=cur.data.num_rows + nxt.data.num_rows,
                num_cols=cur.data.num_cols,
            )
            doc.insert_table(sibling=cur, data=merged_data, prov=cur.prov[0] if cur.prov else None, after=True)
            doc.delete_items(node_items=[cur, nxt])
            changed = True
            break
