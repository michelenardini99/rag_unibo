import re
import unicodedata

from config import settings

# Words a greeting, a thank-you or an acknowledgement is made of. A message
# counts as small talk only if every word is one of these: a single content
# word ("ciao, quando scade la rata?") makes it a question.
_THANKS = {"grazie", "ringrazio", "ringraziamo", "mille", "tante", "gentilissimo", "gentilissima", "gentile"}
_GOODBYE = {"arrivederci", "presto", "buonanotte", "addio"}
_GREETINGS = {
    "ciao", "salve", "buongiorno", "buonasera", "buon", "pomeriggio", "giorno", "sera", "hey", "ehi", "hello", "hi",
    "come", "stai", "sta",
}
_VOCABULARY = _THANKS | _GOODBYE | _GREETINGS | {
    "va", "tutto", "bene", "benissimo", "ok", "okay", "perfetto", "ottimo", "capito",
    "d", "accordo", "daccordo", "si", "no", "e", "tu", "te", "lei", "a", "ti", "molto", "anche", "allora",
    "oggi",
}
_MAX_WORDS = 6


def _words(text: str) -> list[str]:
    plain = "".join(ch for ch in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(ch))
    return re.findall(r"[a-z]+", plain)


def small_talk_reply(message: str) -> str | None:
    """A fixed reply if `message` is only a greeting, a thank-you or an
    acknowledgement, else None (it's a question for the RAG pipeline).

    Without this, "buongiorno" was searched like a question (a
    medical-specialisation page scored 0.403 and was cited as its source),
    and the greeting reply, kept in history, was then picked up by the
    condensation as the "standalone question" for a following "Come stai?",
    which went on to be decomposed, searched, and to trigger a course
    question.

    Args:
        message (str): The student's message, as typed.

    Returns:
        str | None: The reply to send, or None.
    """
    words = _words(message)
    if not words or len(words) > _MAX_WORDS or any(w not in _VOCABULARY for w in words):
        return None
    if any(w in _THANKS for w in words):
        return "Prego! Se hai altre domande sull'università, chiedi pure."
    if any(w in _GOODBYE for w in words):
        return "A presto! Torna quando vuoi per altre domande sull'università."
    if any(w in _GREETINGS for w in words):
        return (
            f"Ciao! Sono l'assistente virtuale per l'{settings.assistant_institution}: chiedimi pure di "
            f"{settings.assistant_scope}."
        )
    return "Bene! Se hai altre domande sull'università, chiedi pure."
