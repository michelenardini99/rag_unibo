from llama_index.core.schema import NodeWithScore, TextNode

from retrieval.glossary import expand_acronyms
from retrieval.query_filters import (
    closest_known_value,
    extract_academic_year,
    match_course_answer,
    mentions_course,
    names_course_heads,
    needs_course_disambiguation,
    strip_course_mention,
)
from retrieval.retriever import _dedupe_by_text, _keep_priority_course

COURSES = ["PsicologiaScolasticaComunita", "IngegneriaScienzeInformatiche", "Infermieristica"]


# --- academic year ------------------------------------------------------------------------------


def test_academic_year_is_normalized_from_any_common_form():
    assert extract_academic_year("tasse per il 2025/2026") == "2025/2026"
    assert extract_academic_year("piano di studi 2026-27") == "2026/2027"
    assert extract_academic_year("nell'anno accademico 2026 cosa cambia?") == "2026/2027"
    assert extract_academic_year("quando scade la rata?") is None


# --- course recognition --------------------------------------------------------------------------


def test_closest_known_value_tolerates_spelling_but_not_unrelated_answers():
    assert (
        closest_known_value("Psicologia Scolastica e Comunità", COURSES)
        == "PsicologiaScolasticaComunita"
    )
    assert closest_known_value("Infermieristica", COURSES) == "Infermieristica"
    assert closest_known_value("NESSUNO", COURSES) is None
    assert closest_known_value("", COURSES) is None


def test_mentions_course_matches_word_stems():
    assert mentions_course("esami di informatica", "IngegneriaScienzeInformatiche")
    assert not mentions_course(
        "se sono iscritto al primo anno in corso", "PsicologiaScolasticaComunita"
    )


def test_names_course_heads_uses_only_the_distinctive_first_word():
    # "scienze" belongs to Ingegneria's identifier, but is not its distinctive word.
    assert names_course_heads("Psicologia e Scienze della Comunità", COURSES) == {
        "PsicologiaScolasticaComunita"
    }
    assert names_course_heads("quando scade la rata?", COURSES) == set()


def test_match_course_answer_understands_short_replies():
    assert match_course_answer("ingegneria", COURSES) == "IngegneriaScienzeInformatiche"
    assert match_course_answer("Infermieristica", COURSES) == "Infermieristica"
    assert match_course_answer("no", COURSES) is None


def test_strip_course_mention_removes_the_course_but_keeps_the_topic():
    query = (
        "requisiti per la domanda di laurea nel corso di laurea magistrale "
        "in Psicologia scolastica e di comunità?"
    )
    assert (
        strip_course_mention(query, "PsicologiaScolasticaComunita")
        == "requisiti per la domanda di laurea?"
    )


def test_strip_course_mention_keeps_queries_that_would_become_too_short():
    assert strip_course_mention("Infermieristica?", "Infermieristica") == "Infermieristica?"
    assert strip_course_mention("quando scade la rata?", None) == "quando scade la rata?"


# --- course disambiguation -----------------------------------------------------------------------


def _chunk(corso, score, title="Pagina"):
    return {"corso": corso, "score": score, "headings": [title]}


def test_no_disambiguation_when_results_are_university_wide_or_single_course():
    assert not needs_course_disambiguation([_chunk(None, 0.9)])
    assert not needs_course_disambiguation([_chunk("Infermieristica", 0.9), _chunk(None, 0.95)])
    assert not needs_course_disambiguation(
        [_chunk("Infermieristica", 0.9), _chunk("Infermieristica", 0.7)]
    )


def test_disambiguation_when_two_courses_compete():
    chunks = [_chunk("Infermieristica", 0.90), _chunk("PsicologiaScolasticaComunita", 0.85)]
    assert needs_course_disambiguation(chunks)
    assert not needs_course_disambiguation(chunks, remembered_corso="Infermieristica")


def test_disambiguation_when_another_course_has_the_same_page():
    chunks = [
        _chunk("PsicologiaScolasticaComunita", 0.97, "Prova finale"),
        _chunk("Infermieristica", 0.6, "Prova finale"),
    ]
    assert needs_course_disambiguation(chunks)


def test_no_disambiguation_for_a_marginal_unrelated_course_page():
    chunks = [
        _chunk("IngegneriaScienzeInformatiche", 0.92, "Muoversi in città"),
        _chunk("Infermieristica", 0.5, "Tirocini"),
    ]
    assert not needs_course_disambiguation(chunks)


# --- query expansion -----------------------------------------------------------------------------


def test_acronyms_are_expanded_once_and_only_when_needed():
    assert (
        expand_acronyms("Come assolvo l'OFA?")
        == "Come assolvo l'OFA (obblighi formativi aggiuntivi)?"
    )
    already = "OFA (obblighi formativi aggiuntivi): come si assolvono?"
    assert expand_acronyms(already) == already
    # Case-sensitive acronyms are not expanded inside ordinary words.
    assert expand_acronyms("lo sol mi illumina") == "lo sol mi illumina"


# --- result filtering ----------------------------------------------------------------------------


def _node(corso, score, title, text=None):
    meta = {"corso": corso, "headings": [title]}
    return NodeWithScore(node=TextNode(text=text or f"{title} {corso}", metadata=meta), score=score)


def test_priority_course_drops_other_courses_when_it_answers():
    ranked = [
        _node("PsicologiaScolasticaComunita", 0.95, "Prova finale"),
        _node("IngegneriaScienzeInformatiche", 0.94, "Prova finale"),
        _node(None, 0.90, "Domanda di laurea"),
    ]
    kept = _keep_priority_course(ranked, "IngegneriaScienzeInformatiche")
    assert [n.node.metadata["corso"] for n in kept] == ["IngegneriaScienzeInformatiche", None]


def test_priority_course_keeps_pages_only_other_courses_have():
    ranked = [
        _node("IngegneriaScienzeInformatiche", 0.95, "Coordinatore"),
        _node("PsicologiaScolasticaComunita", 0.20, "Tirocinio"),
    ]
    kept = _keep_priority_course(ranked, "PsicologiaScolasticaComunita")
    assert len(kept) == 2


def test_priority_course_is_a_no_op_without_a_course():
    ranked = [_node("Infermieristica", 0.9, "A")]
    assert _keep_priority_course(ranked, None) == ranked


def test_dedupe_keeps_the_first_of_identical_texts():
    items = ["La mora è di 50 €.", "la  mora è di 50 €.", "Altro testo."]
    assert _dedupe_by_text(items, lambda t: t) == ["La mora è di 50 €.", "Altro testo."]
