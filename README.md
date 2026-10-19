# RAG Unibo

Assistente conversazionale che risponde a domande su regolamenti, tasse, iscrizioni, corsi e procedure amministrative dell'Università di Bologna, basandosi solo sui documenti che gli vengono forniti (retrieval-augmented generation), con citazione delle fonti. Gira interamente in locale: nessun dato esce verso servizi cloud di terze parti.

Questa guida spiega come installare, configurare e avviare il sistema senza richiedere di conoscerne l'architettura interna. Per i dettagli tecnici, vedi `REPORT_ARCHITETTURA.md`.

## Prerequisiti

- Python 3.11–3.13
- Docker e Docker Compose
- Una o più GPU NVIDIA con almeno 16 GB di VRAM (driver NVIDIA + `nvidia-container-toolkit` per Docker), necessarie sia per il servizio di generazione (vLLM, in container) sia per i modelli di embedding/reranking (in esecuzione locale sull'host)

## 1. Installazione

```bash
git clone <url-del-repo>
cd rag_unibo

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## 2. Configurazione

Crea un file `.env` nella root del progetto con almeno queste variabili (adatta porte, modelli e ID delle GPU al tuo hardware):

```bash
# Qdrant (database vettoriale)
QDRANT_HTTP_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_COLLECTION=ateneo_docs
QDRANT_URL=http://localhost:6333

# vLLM (modello di generazione, in container)
VLLM_PORT=8000
VLLM_DEVICE_ID=0
VLLM_MODEL_ID=Qwen/Qwen2.5-14B-Instruct-AWQ
VLLM_MAX_MODEL_LEN=8192
VLLM_GPU_MEMORY_UTILIZATION=0.90
VLLM_BASE_URL=http://localhost:8000/v1

# Modelli locali (embedding, reranking; eseguiti sull'host, non in container)
EMBEDDINGS_DEVICE_ID=1
EMBEDDING_MODEL_ID=BAAI/bge-m3
RERANKER_MODEL_ID=BAAI/bge-reranker-v2-m3

# Cache pesi Hugging Face
HF_HOME=.cache/huggingface
# Necessario solo se il modello scelto è ad accesso limitato (gated)
HUGGING_FACE_HUB_TOKEN=
```

Note:
- `VLLM_DEVICE_ID` ed `EMBEDDINGS_DEVICE_ID` devono indicare GPU diverse se disponibili più di una.
- Se il modello servito da vLLM non supporta input visivi, imposta `GENERATION_MODEL_VISION=false`: il sistema salterà la descrizione automatica delle immagini nei documenti invece di fallire in fase di conversione.

## 3. Avvio dei servizi infrastrutturali

```bash
docker compose up -d
```

Avvia Qdrant (`localhost:6333`) e il server di generazione vLLM (`localhost:8000`). Attendi che entrambi siano pronti (`docker compose logs -f vllm` per seguire il download/caricamento del modello al primo avvio).

## 4. Caricamento dei documenti

Copia i documenti (PDF, DOCX, PPTX, HTML) in `datasets/raw/`, organizzati in sottocartelle secondo lo schema `tipo_laurea/corso/sezione/materia` (ogni livello di cartella diventa un metadato usato poi per filtrare le ricerche, es. `datasets/raw/magistrale/informatica/regolamenti/tasse/documento.pdf`).

In alternativa, per acquisire pagine da un sito web (una o più pagine radice, separate da virgola — ciascuna definisce il proprio ambito di crawling, anche su host diversi):

```bash
python -m ingestion.web_crawler <url-radice>[,<url-radice>...] [cartella-output] [numero-massimo-pagine]
```

Poi converti i documenti grezzi in formato strutturato:

```bash
python -m ingestion.converter --source datasets/raw --destination datasets/converted
```

E indicizzali nel database vettoriale:

```bash
python -m indexing
```

Rieseguire questo comando dopo aver aggiunto o modificato documenti aggiorna l'indice in modo incrementale (solo i file nuovi o cambiati). Per ricostruire l'indice da zero: `python -m indexing --recreate`.

## 5. Avvio dell'assistente

```bash
cd src
chainlit run app.py -w --port 8001
```

Chainlit usa di default la porta 8000, già occupata da vLLM: `--port 8001` (o un'altra porta libera) evita il conflitto. L'interfaccia conversazionale sarà raggiungibile su `http://localhost:8001`.

## 6. Valutazione (opzionale)

Per misurare la qualità del recupero documenti e del confronto tra modelli su un set di domande di test:

```bash
cd src
python -m eval.evaluate_retrieval
python -m eval.evaluate_model_comparison
```

I risultati vengono salvati in `datasets/eval/`. La valutazione end-to end delle risposte (fedeltà al contesto, correttezza) si esegue invece dal notebook `notebook/evaluation.ipynb`.
