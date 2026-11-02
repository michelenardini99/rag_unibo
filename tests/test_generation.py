from llama_index.core.base.llms.types import MessageRole

from generation.condense import OFF_TOPIC_REPLY, condense_question, decompose_question
from generation.generator import NO_CONTEXT_RESPONSE, generate_response
from generation.hyde import hyde_query
from generation.prompt import (
    CitationStripper,
    build_context_block,
    build_prompt,
    course_display_name,
    format_sources,
    question_for_answer,
    source_title,
    strip_citations,
)

FREQUENZA = {
    "text": "La frequenza ai corsi è obbligatoria per il 66% delle ore di lezione.",
    "headings": ["Frequenza delle lezioni"],
    "source_file": "index.json",
    "source_url": "https://corsi.unibo.it/laurea/Infermieristica/frequenza-delle-lezioni",
    "corso": "Infermieristica",
}
LEZIONI_ATENEO = {
    "text": "Ogni corso può prevedere la frequenza obbligatoria delle lezioni o meno.",
    "headings": ["Lezioni: frequenza e orari", "Frequenza"],
    "source_file": "index.json",
    "source_url": "https://www.unibo.it/it/studiare/lezioni",
    "corso": None,
}


# --- citations ----------------------------------------------------------------------------------


def test_strip_citations_removes_every_form():
    answer = "Scade il 29 ottobre 2026 [1], [2] e la mora è di 50 € [3, 4][5-6]."
    assert strip_citations(answer) == "Scade il 29 ottobre 2026 e la mora è di 50 €."


def test_strip_citations_keeps_square_brackets_that_are_not_citations():
    assert strip_citations("Il voto [in trentesimi] è 28 [1].") == "Il voto [in trentesimi] è 28."


def test_citation_stripper_never_shows_a_citation_split_across_tokens():
    stripper = CitationStripper()
    shown = "".join(stripper.feed(t) for t in ["La frequenza è del 66%", " [", "1", "2", "]", "."])
    shown += stripper.flush()
    assert shown == "La frequenza è del 66%."
    assert "[" not in shown


def test_citation_stripper_releases_text_that_turns_out_not_to_be_a_citation():
    stripper = CitationStripper()
    shown = "".join(stripper.feed(t) for t in ["Vedi [", "nota", "]"]) + stripper.flush()
    assert shown == "Vedi [nota]"


# --- sources ------------------------------------------------------------------------------------


def test_format_sources_lists_only_cited_pages_once_with_real_links():
    chunks = [FREQUENZA, LEZIONI_ATENEO, dict(FREQUENZA, text="La verifica della frequenza…")]
    sources = format_sources("Obbligatoria per il 66% [1][3].", chunks)
    assert sources.count("Frequenza delle lezioni") == 1
    assert f"({FREQUENZA['source_url']})" in sources
    assert "Lezioni: frequenza e orari" not in sources


def test_format_sources_falls_back_to_all_pages_when_nothing_is_cited():
    sources = format_sources("Obbligatoria per il 66%.", [FREQUENZA, LEZIONI_ATENEO])
    assert "Frequenza delle lezioni" in sources and "Lezioni: frequenza e orari" in sources


def test_format_sources_ignores_citations_of_blocks_that_do_not_exist():
    assert "Frequenza delle lezioni" in format_sources("Vero [1][9].", [FREQUENZA])
    assert format_sources("Niente.", []) == ""


def test_source_title_uses_page_title_for_web_pages_and_filename_for_documents():
    assert source_title(FREQUENZA) == "Frequenza delle lezioni"
    document = {"source_file": "Regolamento didattico_ corso.json", "headings": ["Art. 1"]}
    assert source_title(document) == "Regolamento didattico: corso"


# --- prompt -------------------------------------------------------------------------------------


def test_context_blocks_are_numbered_with_scope_and_section():
    block = build_context_block([FREQUENZA, LEZIONI_ATENEO])
    assert block.startswith("[1] (corso: Infermieristica) Frequenza delle lezioni\n")
    assert "[2] (generale di ateneo) Lezioni: frequenza e orari > Frequenza\n" in block
    assert "https://" not in block


def test_course_display_name_splits_camel_case():
    assert course_display_name("IngegneriaScienzeInformatiche") == "Ingegneria Scienze Informatiche"


def test_question_for_answer_lists_sub_questions_only_when_split():
    assert question_for_answer("Quando scade?", ["Quando scade?"]) == "Quando scade?"
    combined = question_for_answer("A e B?", ["A?", "B?"])
    assert "1. A?" in combined and "2. B?" in combined


def test_build_prompt_puts_context_course_and_question_after_history():
    history = [("Domanda precedente", "Risposta precedente")]
    messages = build_prompt(
        "quante lezioni devo seguire?", [FREQUENZA], history, corso="Infermieristica"
    )
    assert [m.role for m in messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
    ]
    text = messages[-1].blocks[0].text
    assert (
        text.index("CONTESTO:")
        < text.index("CORSO DELLO STUDENTE: Infermieristica")
        < text.index("DOMANDA")
    )
    assert text.rstrip().endswith("quante lezioni devo seguire?")


def test_build_prompt_omits_course_line_when_course_is_unknown():
    text = build_prompt("domanda", [FREQUENZA], []).pop().blocks[0].text
    assert "CORSO DELLO STUDENTE" not in text


# --- calls to the language model ------------------------------------------------------------------


def test_condense_does_not_call_the_model_without_history(fake_llm):
    llm = fake_llm("non deve essere usata")
    assert (
        condense_question(llm, [], "Entro quando scade la prima rata?")
        == "Entro quando scade la prima rata?"
    )
    assert llm.calls == []


def test_condense_rewrites_a_follow_up_with_the_history(fake_llm):
    llm = fake_llm("  Entro quando scade la prima rata per la magistrale?\n")
    history = [("Entro quando scade la prima rata per la triennale?", "Il 24 settembre 2026.")]
    assert condense_question(llm, history, "E per la magistrale?") == (
        "Entro quando scade la prima rata per la magistrale?"
    )
    sent = llm.calls[0][0][-1].blocks[0].text
    assert "triennale" in sent and "E per la magistrale?" in sent


def test_decompose_keeps_a_single_question_as_is(fake_llm):
    assert decompose_question(fake_llm("Quanto costa la mora?"), "Quanto costa la mora?") == (
        ["Quanto costa la mora?"],
        None,
    )


def test_decompose_splits_and_caps_sub_questions(fake_llm):
    reply = "Domanda uno?\n\n Domanda due?\nDomanda tre?\nDomanda quattro?"
    assert decompose_question(fake_llm(reply), "q", max_subquestions=3) == (
        ["Domanda uno?", "Domanda due?", "Domanda tre?"],
        None,
    )


def test_decompose_falls_back_to_the_question_on_an_empty_reply(fake_llm):
    assert decompose_question(fake_llm(""), "Quanto costa la mora?") == (
        ["Quanto costa la mora?"],
        None,
    )


def test_greetings_get_the_model_reply_and_no_search(fake_llm):
    llm = fake_llm("CORTESIA: Prego! Se hai altre domande, chiedi pure.")
    assert decompose_question(llm, "grazie mille") == (
        ["grazie mille"],
        "Prego! Se hai altre domande, chiedi pure.",
    )
    _, reply = decompose_question(fake_llm("CORTESIA"), "ciao")
    assert reply


def test_off_topic_questions_get_the_fixed_reply(fake_llm):
    _, reply = decompose_question(fake_llm("FUORI TEMA"), "chi ha vinto la Serie A?")
    assert reply == OFF_TOPIC_REPLY


def test_hyde_appends_the_hypothetical_passage_to_the_question(fake_llm):
    query = hyde_query(
        fake_llm("La prova di ammissione si svolge a settembre."), "quando c'è il test d'ingresso?"
    )
    assert query == "quando c'è il test d'ingresso?\nLa prova di ammissione si svolge a settembre."
    assert hyde_query(fake_llm(""), "domanda") == "domanda"


def test_generate_response_does_not_call_the_model_without_context(fake_llm):
    llm = fake_llm("risposta inventata")
    assert generate_response(llm, "domanda", []).strip() == NO_CONTEXT_RESPONSE
    assert llm.calls == []


def test_generate_response_is_deterministic(fake_llm):
    llm = fake_llm("Obbligatoria per il 66% [1].")
    generate_response(llm, "domanda", [FREQUENZA])
    assert llm.calls[0][1]["temperature"] == 0.0
