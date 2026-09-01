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

# Didascalia immagini via il server vLLM già attivo per la generazione
# (Qwen2.5-VL-7B, vision-capable) invece del piccolo SmolVLM-256M locale,
# che su diagrammi accademici produceva loop di ripetizione degeneri
# (§7 architettura, vedi PIPELINE.md §5.5 per i dettagli). Nessun modello
# aggiuntivo da caricare: riusa l'endpoint OpenAI-compatibile su GPU A.
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
)
