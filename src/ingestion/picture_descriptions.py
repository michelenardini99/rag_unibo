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
    url=f"{settings.vllm_base_url}/chat/completions",
    params={
        "model": settings.generation_model,
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
