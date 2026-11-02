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
- `GENERATION_MODEL_VISION` indica se il modello delle risposte legge anche le immagini: se `true`, le immagini dei frammenti recuperati vengono allegate al prompt. Con un modello solo testuale (come Qwen2.5-14B-Instruct) va lasciato a `false`. La descrizione delle immagini in conversione è indipendente e usa il modello di `VISION_MODEL_ID` (vedi la sezione 4).

## 3. Avvio dei servizi infrastrutturali

```bash
docker compose up -d qdrant vllm
```

Avvia Qdrant (`localhost:6333`) e il server di generazione vLLM (`localhost:8000`). Attendi che entrambi siano pronti (`docker compose logs -f vllm` per seguire il download/caricamento del modello al primo avvio).

Tutte le porte sono pubblicate solo su `127.0.0.1`: vLLM e Qdrant non hanno autenticazione e non sono raggiungibili da altre macchine. Le immagini sono fissate per versione e digest in `docker-compose.yml`.

## 4. Caricamento dei documenti

Copia i documenti (PDF, DOCX, PPTX, HTML) in `datasets/raw/`, organizzati in sottocartelle secondo lo schema `tipo_laurea/corso/sezione/materia` (ogni livello di cartella diventa un metadato usato poi per filtrare le ricerche, es. `datasets/raw/magistrale/informatica/regolamenti/tasse/documento.pdf`).

In alternativa, per acquisire pagine da un sito web (una o più pagine radice, separate da virgola — ciascuna definisce il proprio ambito di crawling, anche su host diversi):

```bash
python -m ingestion.web_crawler <url-radice>[,<url-radice>...] [cartella-output] [numero-massimo-pagine]
```

Poi converti i documenti grezzi in formato strutturato. Le immagini contenute nei documenti vengono descritte a parole da un modello multimodale, servito da un vLLM dedicato che va avviato solo per la conversione (`VISION_DEVICE_ID` in `.env` indica una GPU con memoria libera, diversa da quella del vLLM delle risposte):

```bash
docker compose --profile indexing up -d vllm-vision    # attendi che il modello sia caricato
python -m ingestion.converter --source datasets/raw --destination datasets/converted
docker compose stop vllm-vision
```

Il vLLM delle risposte resta acceso, quindi l'assistente continua a funzionare durante la conversione. Se il server delle descrizioni non risponde, la conversione si ferma subito con l'indicazione di come avviarlo; per convertire senza descrivere le immagini, imposta `PICTURE_DESCRIPTION=false`.

E indicizzali nel database vettoriale:

```bash
python -m indexing
```

Rieseguire questo comando dopo aver aggiunto o modificato documenti aggiorna l'indice in modo incrementale (solo i file nuovi o cambiati). Per ricostruire l'indice da zero: `python -m indexing --recreate`.

## 5. Avvio dell'assistente

In container, insieme agli altri servizi (la prima volta costruisce l'immagine, con le versioni esatte di `requirements.lock.txt`):

```bash
docker compose up -d --build app
```

Il container parte solo quando vLLM e Qdrant sono pronti, usa l'indice in `datasets/chunked` e i modelli in `.cache/huggingface`, e si riavvia da solo in caso di errore. L'interfaccia è su `http://localhost:8001`; da un'altra macchina si raggiunge con un tunnel SSH (`ssh -L 8001:localhost:8001 <server>`) o con l'inoltro delle porte di VS Code. Per renderla raggiungibile direttamente (`APP_BIND=0.0.0.0` in `.env`) serve prima un reverse proxy con autenticazione: l'interfaccia non ne ha una propria.

Durante lo sviluppo si può invece avviare l'applicazione sull'host, con il ricaricamento automatico del codice:

```bash
cd src
chainlit run app.py -w --port 8001
```

## Test

I test unitari coprono la logica che non richiede GPU, modelli scaricati né servizi in esecuzione (vLLM, Qdrant): pulizia delle pagine e chunking, metadati, riconoscimento del corso, costruzione del prompt, citazioni ed elenco delle fonti, metriche di valutazione. Le chiamate al modello linguistico sono sostituite da un modello finto con risposte predefinite.

```bash
pytest            # dalla radice del progetto
ruff check tests  # stile del codice dei test
```

pytest e ruff sono già compresi in `requirements.lock.txt`; con un ambiente installato da `requirements.txt` vanno aggiunti con `pip install -e ".[dev]"`.

## 6. Valutazione (opzionale)

Per misurare la qualità del recupero documenti e del confronto tra modelli su un set di domande di test:

```bash
cd src
python -m eval.evaluate_retrieval
bash eval/run_model_comparison.sh
```

Il confronto tra modelli carica in vLLM un modello alla volta per generare le risposte, poi un unico modello giudice (Qwen2.5-14B) per valutarle tutte; dura diverse ore, meglio lanciarlo dentro `tmux`. I risultati vengono salvati in `datasets/eval/`. La valutazione end-to end delle risposte (fedeltà al contesto, correttezza) si esegue invece dal notebook `notebook/evaluation.ipynb`.
