from config import settings
from eval.evaluate import timing_summary
from eval.evaluate_retrieval import (
    CONFIGS,
    RetrievalConfig,
    _has_evidence,
    _matches,
    _normalize_title,
)

PAGE = {
    "text": "La frequenza è **obbligatoria**\nper il 66% delle ore.",
    "headings": ["Frequenza delle lezioni"],
    "source_file": "index.json",
    "source_url": "https://x",
    "corso": "Infermieristica",
}
OTHER_COURSE = dict(PAGE, corso="PsicologiaScolasticaComunita")
EXPECTED = [{"title": "Frequenza delle Lezioni", "corso": "Infermieristica"}]


def test_titles_compare_ignoring_case_and_spacing():
    assert _normalize_title("Iscriversi al  Corso") == _normalize_title("iscriversi al corso")


def test_a_hit_needs_the_same_page_of_the_same_course():
    assert _matches([PAGE], EXPECTED) == (True, True)
    assert _matches([OTHER_COURSE, PAGE], EXPECTED) == (False, True)
    assert _matches([OTHER_COURSE], EXPECTED) == (False, False)
    assert _matches([], EXPECTED) == (False, False)


def test_evidence_is_found_despite_markdown_and_line_breaks():
    assert _has_evidence([PAGE], ["obbligatoria per il 66%"])
    assert not _has_evidence([PAGE], ["75%"])


def test_retrieval_configs_point_to_the_right_index():
    production = RetrievalConfig(name="p")
    assert production.collection_name == settings.qdrant_collection
    assert production.docstore_path == settings.data_chunked_dir / "docstore.json"
    small = RetrievalConfig(name="s", chunk_max_tokens=256)
    assert small.collection_name == f"{settings.qdrant_collection}__tok256-BAAI-bge-m3"
    assert len({c.name for c in CONFIGS}) == len(CONFIGS)


def test_timing_summary_reports_mean_median_and_p95():
    stats = timing_summary([{"id": str(i), "generate_s": float(i)} for i in range(1, 21)])[
        "generate_s"
    ]
    assert stats["mean"] == 10.5
    assert stats["median"] == 10.5
    assert stats["p95"] == 20.0
