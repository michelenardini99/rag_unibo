import json
import urllib.request

from docling.datamodel.pipeline_options import PictureDescriptionApiOptions

from config import settings

CAPTION_PROMPT = (
    "Descrivi in italiano il contenuto di questa immagine in modo preciso e "
    "dettagliato, senza ripetere le stesse frasi. Se è un diagramma o uno "
    "schema, spiega gli elementi principali e le relazioni tra loro. Se "
    "contiene del testo leggibile, riportalo. Se è un grafico con dei dati, "
    "descrivi i valori e le tendenze principali. Se è puramente decorativa "
    "(es. un logo), dillo brevemente."
)

picture_descriptions_options = PictureDescriptionApiOptions(
    url=f"{settings.vision_base_url}/chat/completions",
    params={
        "model": settings.vision_model,
        "max_tokens": 500,
        "temperature": 0.0,
        "repetition_penalty": 1.15,
    },
    prompt=CAPTION_PROMPT,
    concurrency=2,
    # Default (0.05) assumes dense text pages where a meaningful figure takes
    # up a large fraction of the page; this corpus is mostly 720x540 slides,
    # where legitimate diagrams/screenshots are often only ~1-2% of the page
    # area (see REPORT_ARCHITETTURA.md image_grounded investigation) and would
    # otherwise be silently skipped.
    picture_area_threshold=0.005,
)


def check_vision_server() -> str:
    """Checks that the picture-description server answers before converting,
    instead of letting every image fail one by one mid-conversion.

    Returns:
        str: The model the server actually loaded (behind `settings.vision_model`).

    Raises:
        SystemExit: If the server can't be reached, with how to start it.
    """
    try:
        with urllib.request.urlopen(f"{settings.vision_base_url}/models", timeout=10) as response:
            return json.load(response)["data"][0]["root"]
    except OSError:
        raise SystemExit(
            f"Il server per la descrizione delle immagini ({settings.vision_base_url}) non risponde.\n"
            "Avvialo con `docker compose --profile indexing up -d vllm-vision` e attendi che il modello sia "
            "caricato, oppure converti senza descrizioni con PICTURE_DESCRIPTION=false."
        )

