import json
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup
from llama_index.core.schema import NodeRelationship, TextNode

from chunking.chunker import _readable_tables, build_parent_nodes
from chunking.metadata import (
    current_academic_year,
    extract_metadata,
    read_checksum,
    read_source_url,
)
from ingestion.converter import preprocess_html
from ingestion.web_crawler import (
    extract_links,
    is_in_scope,
    normalize_url,
    url_sidecar_path,
    url_to_local_path,
)

ROOT = "https://corsi.unibo.it/laurea/Infermieristica"


# --- crawler -------------------------------------------------------------------------------------


def test_normalize_url_resolves_relative_links_and_drops_fragment_and_query():
    url = normalize_url("../Infermieristica/tirocini?lang=it#contatti", base_url=ROOT + "/home")
    assert url == "https://corsi.unibo.it/laurea/Infermieristica/tirocini"
    assert normalize_url("https://CORSI.unibo.it/laurea") == "https://corsi.unibo.it/laurea"


def test_scope_is_the_root_path_on_the_root_host():
    assert is_in_scope(ROOT, [ROOT])
    assert is_in_scope(ROOT + "/tirocini", [ROOT])
    assert not is_in_scope("https://corsi.unibo.it/laurea/InfermieristicaBis", [ROOT])
    assert not is_in_scope("https://corsi.unibo.it/laurea/IngegneriaScienzeInformatiche", [ROOT])
    assert not is_in_scope("https://www.unibo.it/laurea/Infermieristica", [ROOT])
    assert not is_in_scope("mailto:segreteria@unibo.it", [ROOT])


def test_every_page_is_saved_as_index_html_in_a_folder_mirroring_its_url(tmp_path):
    assert url_to_local_path(ROOT + "/frequenza-delle-lezioni", tmp_path) == (
        tmp_path / "laurea" / "Infermieristica" / "frequenza-delle-lezioni" / "index.html"
    )
    assert (
        url_to_local_path(ROOT + "/index.html", tmp_path)
        == tmp_path / "laurea" / "Infermieristica" / "index.html"
    )
    unsafe = url_to_local_path("https://x.it/a%3Ab/c d", tmp_path)
    assert unsafe == tmp_path / "a_b" / "c_d" / "index.html"
    assert url_sidecar_path(tmp_path / "index.html") == tmp_path / "index.html.url"


def test_extract_links_keeps_only_unique_html_pages():
    html = """
      <a href="tirocini">Tirocini</a> <a href="tirocini#sedi">Sedi</a> <a href="#top">Su</a>
      <a href="bando.pdf">Bando</a> <a href="mailto:a@b.it">Mail</a> <a href="https://www.unibo.it/">Ateneo</a>
    """
    assert extract_links(html, ROOT + "/") == [ROOT + "/tirocini", "https://www.unibo.it/"]


# --- HTML cleaning -------------------------------------------------------------------------------


def _body(html: str) -> BeautifulSoup:
    return BeautifulSoup(preprocess_html(html), "html.parser").body


def test_only_the_main_content_is_kept():
    body = _body(
        "<html><head><title>T</title></head><body><nav>Menu</nav>"
        "<main><h1>Tirocini</h1><p>Testo</p><form>Cerca</form></main><footer>Piè</footer></body></html>"
    )
    text = body.get_text(" ", strip=True)
    assert "Tirocini" in text and "Testo" in text
    assert "Menu" not in text and "Piè" not in text and "Cerca" not in text


def test_pages_without_main_are_left_unchanged():
    html = "<html><body><p>Solo testo</p></body></html>"
    assert preprocess_html(html) == html


def test_hidden_content_is_made_visible():
    body = _body(
        '<main><div aria-hidden="true"><p>Prova il 16 settembre 2026</p></div>'
        "<div hidden></div></main>"
    )
    hidden = body.find(attrs={"aria-hidden": True})
    assert hidden is None
    assert "16 settembre 2026" in body.get_text()


def test_accordion_items_become_sections_under_the_preceding_heading():
    body = _body("""<main><h2>Esenzioni</h2><ul class="accordion-box">
        <li><button>Avere l'esenzione</button>
            <div class="accordion-content"><p>Hai l'esenzione se:</p>
                <ul><li>ISEE fino a 27.000 €</li></ul></div></li>
        </ul></main>""")
    heading = body.find("h3")
    assert heading is not None and heading.get_text() == "Avere l'esenzione"
    assert body.find("ul", class_="accordion-box") is None
    # The sentence and its list stay together, right after the new heading.
    following = heading.find_next("div", class_="accordion-content").get_text(" ", strip=True)
    assert following.startswith("Hai l'esenzione se:") and "27.000" in following


def test_header_tooltips_become_a_legend_after_the_table():
    body = _body("""<main><table><tr><th>Insegnamento</th>
        <th>CFU<span class="legend">? Crediti formativi universitari</span></th></tr>
        <tr><td>Anatomia</td><td>6</td></tr></table></main>""")
    headers = [th.get_text(strip=True) for th in body.find_all("th")]
    assert headers == ["Insegnamento", "CFU"]
    assert (
        body.find("table").find_next_sibling("p").get_text()
        == "Legenda — CFU: Crediti formativi universitari"
    )


# --- chunking ------------------------------------------------------------------------------------


def test_tables_are_rewritten_in_the_words_of_a_question():
    chunk = (
        "Scadenze 2026:\n"
        "| **SCADENZA DOMANDA** | **APPELLO DI LAUREA** |\n| - | - |\n| 04/06/2026 | 16/07/2026 |\n"
        "Entro il 04/06/2026."
    )
    assert _readable_tables(chunk) == (
        "Scadenze 2026:\n"
        "| scadenza domanda | appello di laurea |\n| - | - |\n| 4 giugno 2026 | 16 luglio 2026 |\n"
        "Entro il 04/06/2026."
    )


def test_invalid_dates_in_tables_are_left_alone():
    assert _readable_tables("| 31/13/2026 |") == "| 31/13/2026 |"


def test_sections_with_two_or_more_chunks_get_a_parent():
    meta = {"source_path": "laurea/Infermieristica/tirocini/index.json", "id": "doc-1"}
    a = TextNode(text="Primo.", metadata={**meta, "headings": ["Tirocini", "Sedi"]})
    b = TextNode(text="Secondo.", metadata={**meta, "headings": ["Tirocini", "Sedi"]})
    alone = TextNode(text="Solo.", metadata={**meta, "headings": ["Tirocini", "Valutazione"]})

    parents = build_parent_nodes([a, b, alone])

    assert len(parents) == 1
    assert parents[0].text == "Primo.\n\nSecondo."
    assert a.relationships[NodeRelationship.PARENT].node_id == parents[0].id_
    assert NodeRelationship.PARENT not in alone.relationships
    # Ids are stable across runs, so re-indexing doesn't duplicate parents.
    assert (
        build_parent_nodes(
            [TextNode(text="x", metadata=a.metadata), TextNode(text="y", metadata=b.metadata)]
        )[0].id_
        == parents[0].id_
    )


# --- metadata ------------------------------------------------------------------------------------


def test_academic_year_switches_in_september():
    assert current_academic_year(date(2026, 8, 31)) == "2025/2026"
    assert current_academic_year(date(2026, 9, 1)) == "2026/2027"


def _converted(path: Path, url: str | None = None, binary_hash: int = 255) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"origin": {"binary_hash": binary_hash}}), encoding="utf-8")
    if url:
        path.with_name(path.name + ".url").write_text(url + "\n", encoding="utf-8")
    return path


def test_course_metadata_comes_from_the_folder(tmp_path):
    doc = _converted(
        tmp_path / "laurea" / "Infermieristica" / "frequenza-delle-lezioni" / "index.json", url=ROOT
    )
    meta = extract_metadata(doc, tmp_path)
    assert (meta["ambito"], meta["tipo_laurea"], meta["corso"], meta["sezione"]) == (
        "corso",
        "laurea",
        "Infermieristica",
        "frequenza-delle-lezioni",
    )
    assert meta["source_url"] == ROOT
    assert meta["checksum"] == "00000000000000ff"
    assert meta["id"] == extract_metadata(doc, tmp_path)["id"]


def test_university_wide_documents_have_no_course(tmp_path):
    doc = _converted(tmp_path / "ateneo" / "it" / "tasse" / "index.json")
    meta = extract_metadata(doc, tmp_path)
    assert meta["ambito"] == "ateneo"
    assert meta["corso"] is None
    assert meta["source_url"] is None


def test_unreadable_checksum_and_missing_url_are_none(tmp_path):
    bad = tmp_path / "x.json"
    bad.write_text("{non è json", encoding="utf-8")
    assert read_checksum(bad) is None
    assert read_source_url(bad) is None
