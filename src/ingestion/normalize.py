import re
from collections import Counter

from docling_core.types.doc import DoclingDocument, SectionHeaderItem
from docling_core.types.doc.document import TableData
from docling_core.types.doc.labels import DocItemLabel

_ARTICLE_RE = re.compile(r"art\.?\s*\d+", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"[.!?:;]\s*$")
_MERGEABLE_LABELS = {DocItemLabel.TEXT, DocItemLabel.LIST_ITEM}
_BOILERPLATE_MIN_REPEATS = 3


def normalize_document(doc: DoclingDocument) -> None:
    """Corrects recurring Docling layout-detection quirks seen on Ateneo
    regolamenti: article headers misclassified as another label, headers
    merged with the enclosing Part title, headers split from their title,
    repeated page header/footer boilerplate landing in the body instead of
    being filtered as furniture, and paragraphs/list items shredded across
    a boilerplate injection or across one text block per line. Mutates in
    place.
    """
    _fix_mislabeled_article_headers(doc)
    _split_merged_part_and_article(doc)
    _merge_split_article_number_and_title(doc)
    _strip_repeated_boilerplate(doc)
    _merge_fragmented_paragraphs(doc)
    _merge_paginated_tables(doc)


def _fix_mislabeled_article_headers(doc: DoclingDocument) -> None:
    default_level = next((t.level for t in doc.texts if isinstance(t, SectionHeaderItem)), 1)
    for item in list(doc.texts):
        if isinstance(item, SectionHeaderItem):
            continue
        if _ARTICLE_RE.match(item.text.strip()):
            doc.insert_heading(sibling=item, text=item.text, level=default_level, after=True)
            doc.delete_items(node_items=[item])


def _split_merged_part_and_article(doc: DoclingDocument) -> None:
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
    """Removes running page headers/footers (e.g. a university letterhead
    repeated on every page) that Docling's own furniture detection missed
    and left in the main body. Any text appearing verbatim on at least
    `_BOILERPLATE_MIN_REPEATS` occasions is treated as boilerplate, not
    content — legitimate body text this short and this repetitive within
    one regolamento is not a realistic false positive.
    """
    counts = Counter(item.text.strip() for item in doc.texts)
    boilerplate = [item for item in list(doc.texts) if counts[item.text.strip()] >= _BOILERPLATE_MIN_REPEATS]
    if boilerplate:
        doc.delete_items(node_items=boilerplate)


def _merge_fragmented_paragraphs(doc: DoclingDocument) -> None:
    i = 0
    while i < len(doc.texts) - 1:
        cur, nxt = doc.texts[i], doc.texts[i + 1]
        if (
            cur.label in _MERGEABLE_LABELS
            and nxt.label in _MERGEABLE_LABELS
            and not _SENTENCE_END_RE.search(cur.text.strip())
        ):
            cur.text = cur.orig = f"{cur.text.rstrip()} {nxt.text.lstrip()}"
            doc.delete_items(node_items=[nxt])
            continue  # re-check cur against the new next sibling
        i += 1


def _merge_paginated_tables(doc: DoclingDocument) -> None:
    """Stitches back together a table that TableFormer split into two
    separate TableItems because it spans a page break — the continuation
    page has no repeated header row, but shares the same column count and
    sits immediately next to the first part with nothing in between.
    Fee/deadline tables are exactly the content the Grounding Guard (§7)
    depends on staying literal, so a table silently split in half — with
    only the first half retrievable under its real heading context — is
    worth fixing here rather than downstream.
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
