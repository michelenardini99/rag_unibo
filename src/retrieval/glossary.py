import re

GLOSSARY: dict[str, tuple[str, bool]] = {
    "ISEE": ("indicatore della situazione economica", False),
    "CFU": ("crediti formativi universitari", False),
    "OFA": ("obblighi formativi aggiuntivi", False),
    "SOL": ("Studenti Online", True),
    "PPV": ("Prova Pratica Valutativa", False),
    "TPV": ("Tirocinio Pratico Valutativo", False),
    "ER.GO": ("Azienda Regionale per il Diritto agli Studi Superiori", False),
    "AQ": ("Assicurazione della Qualità", True),
}


def expand_acronyms(query: str) -> str:
    """Adds the expansion right after each glossary acronym in `query`
    ("ISEE" -> "ISEE (indicatore della situazione economica)"), so search
    and reranking match pages that only spell it out, while pages using the
    acronym still match it. Acronyms whose expansion the query already
    contains are left alone.

    Args:
        query (str): A standalone (sub-)question.

    Returns:
        str: `query` with expansions added, or unchanged.
    """
    for acronym, (expansion, case_sensitive) in GLOSSARY.items():
        if expansion.lower() in query.lower():
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(acronym)}(?!\w)", 0 if case_sensitive else re.IGNORECASE)
        query = pattern.sub(lambda m: f"{m.group(0)} ({expansion})", query, count=1)
    return query
