# Pipeline RAG — Guida all'Architettura Implementata

Questo documento spiega **come è costruito realmente** il sistema, componente per componente, per chi non ha mai aperto il repository. È distinto da [`Architettura di un Sistema RAG Locale per l'Assistenza Universitaria.md`](Architettura%20di%20un%20Sistema%20RAG%20Locale%20per%20l'Assistenza%20Universitaria.md), che è il documento di *progettazione* (confronto tra strumenti, scelte motivate); qui invece si descrive il codice così com'è, incluse le difficoltà incontrate durante lo sviluppo e come sono state risolte.

## 1. Obiettivo del progetto

Un assistente virtuale RAG (Retrieval-Augmented Generation) per l'assistenza universitaria, che risponde a domande su regolamenti, tasse, iscrizioni e procedure amministrative dell'Università di Bologna, basandosi **esclusivamente** sui documenti d'Ateneo indicizzati — non sulla conoscenza generale del modello linguistico. Vincoli chiave: tutto gira **in locale** (nessun servizio cloud di terze parti, per la riservatezza dei dati), su **due GPU da 16 GB**, e la **precisione numerica** (date, importi, scadenze) ha priorità sulla fluidità della risposta.

## 2. Vista d'insieme

```mermaid
flowchart TD
    subgraph ING["INGESTION (batch, offline)"]
        A1["Documenti Ateneo\nPDF / DOCX / PPTX / HTML"]
        A2["converter.py\nDocling: OCR, tabelle, layout"]
        A3["normalize.py\ncorregge quirk di parsing"]
        A1 --> A2 --> A3
    end

    subgraph CHK["CHUNKING"]
        B1["chunker.py\nHybridChunker (~512 token)"]
        B2["build_parent_nodes()\nraggruppa i chunk per sezione"]
        A3 --> B1 --> B2
    end

    subgraph EMB["EMBEDDING + STORAGE"]
        C1["embedder.py\nBGE-M3: denso + sparso + ColBERT"]
        C2[("Qdrant\nsolo le foglie sono indicizzate")]
        B1 -->|solo foglie| C1 --> C2
        B2 -->|foglie + genitori| C3[("docstore.json\nSimpleDocumentStore")]
        B1 --> C3
    end

    subgraph RET["RETRIEVAL (online, per query)"]
        D0["Domanda studente"]
        D1["condense.py\nriformula con la cronologia"]
        D2["hybrid_search.py\nricerca ibrida in Qdrant"]
        D3["reranker.py\ncross-encoder BGE"]
        D4["retriever.py\nAutoMergingRetriever"]
        D0 --> D1 --> D2
        C2 --> D2
        D2 --> D3 --> D4
        C3 --> D4
    end

    subgraph GEN["GENERAZIONE"]
        E1["generator.py + prompt.py\nLLM vision-capable via vLLM"]
        D4 --> E1
    end

    E1 --> OUT["Risposta + fonti citate"]

    subgraph EVAL["VALUTAZIONE / ABLATION"]
        F1["evaluate.py\nRAGAS: retrieving / generative / end-to-end"]
        F2["select_diagnostic_subset.py\nsottoinsieme di test per tag diagnostico"]
        F3["run_ablations.py\nconfronta N configurazioni,\nun results file per config"]
        F2 --> F3
        F3 --> F1
    end
    OUT -.-> F1
    F3 -.->|se la config cambia chunk size| B1
```

## 3. Stack tecnologico

| Stadio | Strumento | File |
|---|---|---|
| Parsing documenti | Docling (+ SmolVLM per le immagini) | `src/ingestion/converter.py` |
| Normalizzazione | correzioni custom su quirk di Docling | `src/ingestion/normalize.py` |
| Chunking | LlamaIndex `DoclingNodeParser` + `HybridChunker` | `src/chunking/chunker.py` |
| Embedding | BAAI/BGE-M3 (denso + sparso + ColBERT) | `src/embedding/embedder.py` |
| Vector DB | Qdrant | `src/embedding/qdrant_store.py` |
| Retrieval | ricerca ibrida + reranker + `AutoMergingRetriever` (LlamaIndex) | `src/retrieval/` |
| Generazione | Qwen2.5-VL-7B-Instruct-AWQ via vLLM (OpenAI-compatible) | `src/generation/` |
| Valutazione | RAGAS | `src/eval/evaluate.py` |
| Ablation / confronto configurazioni | itera chunking/retrieval/LLM diversi, riusando l'indice quando possibile | `src/eval/run_ablations.py` |
| Sottoinsieme diagnostico di test | classifica le domande per tipo di fatto richiesto, per un dev set più economico | `src/eval/select_diagnostic_subset.py` |
| UI | Chainlit | `src/app.py` |
| Config | pydantic-settings, letto da `.env` | `src/config.py` |

Infrastruttura: `docker-compose.yml` avvia Qdrant e vLLM; BGE-M3/reranker girano in un venv locale sulla seconda GPU (§6 del documento di progettazione).

## 4. La pipeline in dettaglio

### 4.1 Ingestion — `src/ingestion/converter.py`

Legge i documenti grezzi (`datasets/raw/`) e li converte con **Docling**: OCR, riconoscimento struttura tabelle (`TableStructureOptions`), e didascalie automatiche delle immagini tramite un modello VLM leggero (`smolvlm_picture_description`) così i diagrammi diventano cercabili come testo. Output: un file `.json` (rappresentazione strutturata `DoclingDocument`, usata a valle) e un `.md` per documento, in `datasets/converted/`.

**Difficoltà incontrata e soluzione — quirk di parsing.** Docling, su documenti reali (regolamenti scannerizzati, slide con layout irregolare), a volte produce output imperfetto: intestazioni di articolo classificate come testo normale, titoli di sezione e articolo uniti in una sola riga, intestazioni/piè di pagina ripetuti che finiscono nel corpo del testo invece di essere filtrati, paragrafi spezzati su più blocchi, tabelle che attraversano un cambio pagina e vengono spezzate in due tabelle separate. `src/ingestion/normalize.py` corregge tutti questi casi con euristiche mirate (es. un testo che compare identico ≥3 volte è trattato come boilerplate ripetuto e rimosso; due tabelle adiacenti con lo stesso numero di colonne vengono ricucite in una sola). Le tabelle di tasse/scadenze sono il caso critico: se una tabella viene spezzata, solo la prima metà resta recuperabile — per questo la fusione delle tabelle paginate è esplicitamente collegata al futuro "Grounding Guard" (§7, vedi sezione Limiti).

Si esegue come script standalone, `python -m ingestion.converter`, sotto `if __name__ == "__main__"`: legge `data_raw_dir`/`data_converted_dir`/`embeddings_device_id` da `settings` invece di percorsi hardcoded (vedi anche §5.7 — in precedenza la conversione partiva a livello di modulo, non solo quando lanciato come script).

Output doppio per documento: `.json` (rappresentazione strutturata, l'unico letto a valle dal chunking, §4.2) e `.md` (non consumato da nessun codice, tenuto solo per ispezione umana rapida — costa nulla in più perché deriva dallo stesso `DoclingDocument` già in memoria).

### 4.2 Chunking — `src/chunking/chunker.py`

Ogni documento convertito viene passato a `DoclingNodeParser` + `HybridChunker` (tokenizer BGE-M3, target ~512 token, `merge_peers=True` per accorpare frammenti adiacenti troppo piccoli). Ogni chunk ("nodo foglia") porta con sé metadati: `anno_accademico`, `corso`, `categoria`, `stato` (vigente/superato — vedi Limiti), `source_file`, `source_path`, e un breadcrumb `headings` (il percorso titolo→sotto-titolo nel documento originale).

`build_node_parser()`/`chunk_documents()` espongono `chunk_max_tokens` e `tokenizer_model` come parametri (default `settings.chunk_max_tokens`/`settings.embedding_model_id`, prima erano `512`/`"BAAI/bge-m3"` hardcoded nel tokenizer). Non cambiano il comportamento di default, ma permettono di rigenerare l'indice con una dimensione di chunk diversa senza toccare il codice — è quello che usa l'ablation runner (§4.11) per confrontare chunk più piccoli/più grandi.

`_resolve_image_paths()` risolve un problema specifico: `DoclingNodeParser` scarta i dati binari delle immagini quando serializza i metadati del nodo, lasciando solo un riferimento (`#/pictures/3`). Per passare l'immagine **originale** al modello generativo (non solo la sua didascalia testuale, spesso troppo generica), la funzione ricarica il `DoclingDocument` originale e risolve il riferimento nel percorso file reale.

**Chunking gerarchico (nodi genitore).** `build_parent_nodes()` raggruppa i chunk foglia che condividono la stessa sezione (`source_path` + `headings` identici) sotto un nodo "genitore" sintetico, collegato ai figli tramite le relazioni native di LlamaIndex (`NodeRelationship.PARENT`/`CHILD`). Questo non altera l'indicizzazione né il retrieval per il caso comune — serve esclusivamente all'`AutoMergingRetriever` (§4.5). Sul corpus attuale: **722 chunk foglia → 101 sezioni genitore**.

### 4.3 Embedding — `src/embedding/embedder.py`

Modello **BGE-M3**, che produce in un solo passaggio tre rappresentazioni per ogni chunk: vettore **denso** (semantica), vettore **sparso** stile BM25 (termini esatti, utile per codici corso/numeri), e **ColBERT** (multi-vettore, un embedding per token, usato per un secondo stadio di similarità più preciso). Solo i chunk foglia vengono embeddati — i nodi genitore non entrano mai in questo stadio.

### 4.4 Vector Store — `src/embedding/qdrant_store.py`

Qdrant ospita una singola collezione (`ateneo_docs`) con tre "vector space" nominati (`dense`, `colbert`, `sparse`) sullo stesso punto, più un indice sul campo payload `stato` per il filtraggio. Solo le foglie vengono inserite (`upsert_nodes`); i nodi genitore restano esclusivamente nel docstore su file (`datasets/chunked/docstore.json`), non in Qdrant.

`ensure_collection()` accetta un flag `recreate`: se `False` (default) non fa nulla quando la collezione esiste già; se `True` — usato da `reindex.py` (§4.9) — la cancella e ricrea anche se già presente, per non lasciare punti orfani con id non più prodotti dal chunking quando la struttura dei chunk cambia (es. dopo un'ablation su `chunk_max_tokens`).

### 4.5 Retrieval — `src/retrieval/`

Tre file cooperano:

- **`hybrid_search.py`**: esegue la ricerca ibrida vera e propria in Qdrant. Prima un *prefetch* di 50 candidati sia dal vettore denso sia da quello sparso (entrambi filtrati su `stato = vigente`), poi un secondo passaggio che riordina l'unione di questi candidati usando la similarità **ColBERT** (max-sim), restituendo i 20 migliori.
- **`reranker.py`**: passa questi 20 candidati a un **cross-encoder** dedicato (`BGE-Reranker-v2-m3`), che valuta query e testo del chunk insieme (più preciso ma più lento della similarità vettoriale, per questo si applica solo al pool già ridotto) e restituisce i punteggi.
- **`retriever.py`**: orchestra tutto, e implementa il **retrieval gerarchico** (vedi §5, "difficoltà" più sotto per il perché). `HybridQdrantRetriever` incapsula la ricerca ibrida + reranking come un `BaseRetriever` di LlamaIndex, così può alimentare l'`AutoMergingRetriever` **nativo** di LlamaIndex: se abbastanza chunk fratelli (stessa sezione) compaiono tra i candidati, vengono fusi nel loro nodo genitore invece di restare frammenti isolati.

  `retrieve()` espone come parametri keyword-only (dopo i 6 argomenti posizionali esistenti, per non rompere le chiamate da `app.py`/`evaluate.py`) tutto ciò che prima era costante di modulo: `use_reranker`, `use_automerging`, `prefetch_limit`, `candidate_pool`, `top_k`, `score_threshold`, `merged_fallback_threshold`. Con `use_reranker=False` il punteggio finale resta quello ColBERT della ricerca ibrida (nessun secondo passaggio cross-encoder); con `use_automerging=False` il retriever ritorna i nodi foglia grezzi, senza fusione di sezione. Servono a isolare il contributo di ciascun componente via ablation (§4.11) senza duplicare la funzione.

### 4.6 Generazione — `src/generation/`

- **`condense.py`**: se c'è cronologia di conversazione, un primo giro di LLM riformula l'ultima domanda dello studente in una domanda autonoma (es. "e quella dopo?" → "quando scade la seconda rata per Laurea Magistrale?"), usata poi per il retrieval. Se non c'è cronologia, si salta questo passaggio.
- **`prompt.py`**: costruisce il prompt finale — un blocco `CONTESTO` con ogni chunk preceduto da `[Fonte: <nome> - <headings>]`, le immagini associate (via `ImageBlock`, per un LLM vision-capable), e un system prompt che vincola il modello a rispondere solo da quel contesto, citando la fonte tra parentesi quadre per ogni affermazione, riportando date/importi *esattamente* come appaiono (senza arrotondare). Il footer "Fonti consultate" è generato deterministicamente dal codice (non lasciato al modello), così è sempre corretto anche se il modello dimentica di citare.
- **`generator.py`**: costruisce il client verso vLLM (`OpenAILike`, endpoint compatibile OpenAI) e orchestra la chiamata; se non ci sono chunk recuperati, restituisce un messaggio di fallback fisso invece di lasciar rispondere il modello a vuoto.

### 4.7 Configurazione — `src/config.py`

Un oggetto `Settings` (pydantic-settings) legge `.env` alla radice del repository e centralizza: URL/porte Qdrant, URL del modello generativo, device GPU per i modelli locali, soglie di retrieval, dimensione target dei chunk, numero di turni di storico conversazione. Usato in `app.py`, `ingestion/converter.py`, `chunking/chunker.py` e `reindex.py` (§4.1/4.2/4.9); `embedder.py` è ancora parzialmente allineato — `build_embedding_model` continua a usare `'BAAI/bge-m3'` hardcoded invece di `settings.embedding_model_id`, quindi cambiare quel campo in `.env` oggi non ha alcun effetto sul modello di embedding realmente caricato.

### 4.8 Interfaccia — `src/app.py` (Chainlit)

All'avvio di una chat (`on_chat_start`) carica una volta sola: modello di embedding, reranker, client LLM, client Qdrant, docstore. Ad ogni messaggio (`on_message`): condensa la domanda, recupera i chunk, genera la risposta in streaming, mostra le fonti come elementi laterali cliccabili, e mantiene lo storico (troncato alle ultime `max_history_turns` conversazioni).

### 4.9 Reindex — `src/reindex.py`

Script standalone che ricostruisce da zero chunk (foglie + genitori) ed embedding, e re-indicizza Qdrant. Non ri-esegue il parsing Docling (costoso, non necessario se i documenti sorgente non sono cambiati) — riparte da `datasets/converted/`. Cancella e ricrea la collezione Qdrant (`ensure_collection(..., recreate=True)`, §4.4) per evitare punti orfani con ID non più presenti nel docstore.

Accetta anche `chunk_max_tokens`/`tokenizer_model`, inoltrati a `chunk_documents()` (§4.2): passando una `chunked_dir`/`collection_name` distinte per variante si può tenere più di un indice costruito con parametri di chunking diversi senza che si sovrascrivano a vicenda — è la convenzione che usa `run_ablations.py` (§4.11).

### 4.10 Valutazione — `src/eval/evaluate.py`

Usa **RAGAS** su un set di 81 domande curate a mano (`datasets/eval/qa_test_set.json`, formato: lista piatta di `{id, question, expected_answer, verification, expected_source}`). Le metriche sono raggruppate per stadio di pipeline:

- **retrieving**: `context_precision`, `context_recall` — qualità di ciò che viene recuperato.
- **generative**: `faithfulness`, `answer_relevancy` — quanto la risposta è aderente al contesto e pertinente alla domanda.
- **end_to_end**: `answer_correctness`, `answer_similarity` — quanto la risposta finale combacia con la ground truth, dipende dall'intera pipeline insieme.

Il risultato (medie + punteggio per singolo caso) viene salvato in `datasets/eval/results.json`.

`evaluate()`/`build_eval_dataset()` accettano `collection_name` (prima era hardcoded `"ateneo_docs"` dentro `build_eval_dataset`, indipendentemente da quale collezione veniva passata — bug latente, corretto) e `retrieval_kwargs`, un dizionario inoltrato direttamente a `retrieve()` (§4.5): permette di valutare con reranker/automerging disattivati o soglie diverse senza duplicare la funzione. Se un caso del dataset porta un campo `diagnostic_tag` (§4.12), `results_to_json` lo riporta in ogni riga di `per_case`, per poter aggregare le metriche per tipo di domanda a posteriori invece di dover incrociare i risultati a mano con il dataset originale.

### 4.11 Ablation runner — `src/eval/run_ablations.py`

Valuta più configurazioni sullo stesso dataset di test, un `results_<nome>.json` per ciascuna, così sono confrontabili — pensato per il lavoro di tesi di confronto tra componenti (LLM diversi, retriever/chunker diversi), non solo per un numero finale unico.

Ogni configurazione è un `AblationConfig` (dataclass): `name`, `chunk_max_tokens`/`tokenizer_model` (§4.2), `llm_model`/`llm_base_url` (§4.6), e `retrieval_kwargs` — un dizionario libero inoltrato a `retrieve()` (§4.5), non un campo per ciascun parametro: così resta sincronizzato automaticamente se `retrieve()` guadagna nuovi parametri, senza dover toccare `AblationConfig`.

**Riuso dell'indice.** Una `index_key` derivata da `(chunk_max_tokens, tokenizer_model)` identifica quali configurazioni condividono lo stesso indice. Solo le varianti che cambiano la dimensione dei chunk richiedono un reindex (`chunk_documents` + embedding BGE-M3 + upsert Qdrant, lo step costoso); quelle che cambiano solo i flag di retrieval o l'LLM riusano l'indice già costruito, in una collezione Qdrant/`chunked_dir` dedicata per `index_key` (es. `ateneo_docs__tok512-BAAI-bge-m3`) così le varianti non si sovrascrivono. Se l'indice per una `index_key` esiste già su disco (`chunked_dir/docstore.json`) viene riusato senza ricostruirlo; `--force-reindex` lo ricostruisce comunque.

Configurazioni incluse di default in `CONFIGS`: `baseline`, `no_reranker`, `no_automerging`, `chunk_256`, `chunk_768` — le prime tre isolano il contributo di reranker e retrieval gerarchico (le scelte architetturali più specifiche del progetto), le ultime due la sensibilità alla dimensione del chunk. Non ancora coperto: dense-only vs sparse-only vs ibrido (richiederebbe un flag in `hybrid_search.search_candidates()`, non ancora implementato — vedi Limiti, §6).

Uso: `python -m eval.run_ablations [--dataset PATH] [--results-dir PATH] [--force-reindex]`.

### 4.12 Sottoinsieme diagnostico — `src/eval/select_diagnostic_subset.py`

Le 81 domande di `qa_test_set.json` sono costose da rivalutare a ogni configurazione (RAGAS chiama un LLM come giudice per quasi tutte le metriche, oltre a condense+retrieval+generazione della pipeline). Questo script produce un sottoinsieme più piccolo pensato per **massimizzare la capacità di distinguere le configurazioni tra loro**, non per essere rappresentativo in senso statistico — il set completo resta riservato alla valutazione finale.

Classifica ogni domanda in base al campo `verification`, già presente nel dataset (nessuna annotazione aggiuntiva), tramite `classify()`:

- **`multi_fact`** (14/81): la verifica richiede più fatti insieme (es. "entrambe le cifre", "almeno 3 delle 4 procedure"). Le domande più sensibili a `use_automerging` — se le informazioni sono sparse su più chunk della stessa sezione, solo la fusione parent-child le riporta in un unico contesto.
- **`qualitative_negation`** (8/81): risposta non riducibile a un singolo valore ("giudizio qualitativo", risposte che devono negare un obbligo). Più sensibili a LLM/prompt che al retrieval puro — buon gruppo di controllo per confrontare modelli generativi.
- **`single_fact`** (59/81 nel dataset completo): un singolo valore esatto (cifra, data, range, percentuale) — la maggioranza. Qui contano dimensione del chunk e reranker: un chunk troppo grande o un candidato quasi-giusto rompono il match esatto richiesto dalla verifica.

`select_diagnostic_subset()` tiene `multi_fact`/`qualitative_negation` **per intero** (sono rare e sono il segnale più diagnostico: scartarle appiattirebbe proprio le differenze che l'ablation vuole misurare), e campiona `single_fact` a una frazione configurabile per categoria tematica (prefisso dell'id, es. `tasse-`, `laurea-` — 23 categorie nel dataset attuale), così ogni argomento resta rappresentato senza portarsi dietro tutti i quasi-duplicati fattuali dello stesso tema. Con il tasso di default (35%): **52/81 domande**, tutte le 23 categorie coperte, salvate in `datasets/eval/qa_test_set_diagnostic.json` con un campo `diagnostic_tag` aggiunto a ciascun caso (letto da `evaluate.results_to_json`, §4.10).

Uso: `python -m eval.select_diagnostic_subset [--single-fact-rate 0.35]`.

## 5. Difficoltà affrontate e soluzioni

Questa sezione documenta i problemi reali incontrati lavorando sul sistema già funzionante, non ipotetici.

### 5.1 `config.py` era silenziosamente rotto

`ROOT_DIR` era calcolato con `Path(__file__).resolve().parents[2]`, che da `src/config.py` punta **fuori dal repository** (`/home/utente` invece della cartella del progetto). Il file `.env` non veniva mai trovato, quindi ogni istanziazione di `Settings()` falliva su un campo obbligatorio mancante. **Soluzione**: `parents[1]`, e allineamento dei percorsi di default alla cartella reale `datasets/` (il default preesistente puntava a una cartella `data/` mai esistita). Contestualmente, `requirements.txt` conteneva la riga letterale `pip install chainlit` (non un requirement valido, avrebbe fatto fallire `pip install -r requirements.txt`) e mancava `pydantic-settings` pur essendo già usato: entrambi corretti.

### 5.2 Troncamento nelle chiamate di grading di RAGAS (e il "worst case" che non ha retto)

Durante l'evaluate, ~8% delle chiamate LLM interne di RAGAS (usate per calcolare le metriche) fallivano con `IncompleteOutputException`, con fallback automatico a 1 sola generazione invece di 3 (riducendo l'affidabilità statistica delle metriche self-consistency). **Causa**: `llm_factory(...)` di RAGAS usa di default `max_tokens=1024` per il completamento — insufficiente per un output strutturato (verdetti multipli per chunk/frase). Il parametro non è documentato nella signature esplicita di `llm_factory` (passa via `**kwargs`) ma è confermato nel sorgente della libreria.

**Prima soluzione (rivelatasi incompleta): `max_tokens=4096`**, dimensionato stimando che il prompt di grading worst-case (contesto recuperato + domanda + risposta + template RAGAS) restasse sotto i ~3.500 token, lasciando margine entro la finestra di contesto di 8192 del modello servito da vLLM. La stima teneva per i casi osservati inizialmente, ma non è un limite reale imposto dal codice: un contesto recuperato più ampio del previsto (una sezione fusa da `AutoMergingRetriever`, §4.5, o un run con chunk più grandi come la variante `chunk_768` dell'ablation runner, §4.11) può facilmente superarla. Quando è successo, il fallimento non è più stato un troncamento silenzioso ma un errore hard `400` di vLLM (`This model's maximum context length is 8192 tokens... input_tokens=4097`) — input e `max_tokens` insieme superano la finestra del modello, e la chiamata va in eccezione invece di degradare.

**Soluzione applicata (parziale, non definitiva)**: `max_tokens` abbassato a `1024` — riduce il rischio di sforare la finestra totale con contesti recuperati grandi, perché lascia più margine all'input qualunque sia la dimensione del contesto. **Ma `1024` è esattamente il valore di default della libreria che aveva causato il problema originale di questa sezione** (`IncompleteOutputException` sulle risposte più lunghe): il fix scambia un rischio (crash hard `400` su input lunghi) con l'altro (troncamento silenzioso su verdetti/statement insolitamente lunghi), non li elimina entrambi — un budget fisso, da nessuna delle due parti, non può garantire contemporaneamente "spazio sufficiente per l'output di RAGAS" e "spazio sufficiente per l'input" dentro una finestra di 8192 token condivisa e di dimensione fissa. **Non ancora fatto, soluzione più robusta**: calcolare `max_tokens` dinamicamente come `context_window - token_effettivi_del_prompt - margine` invece di una costante fissa, oppure limitare a monte il numero/la lunghezza dei chunk passati a RAGAS. **Lezione**: un budget di token dimensionato su un caso peggiore stimato a mano si rompe silenziosamente non appena qualcosa a monte (qui: le stesse configurazioni pensate per essere confrontate, chunk size e top-k inclusi) fa crescere l'input oltre la stima — e stringere la costante sul lato opposto sposta il problema, non lo risolve.

### 5.3 Un caso di valutazione basato su un fatto inventato

Il caso `tasse-07` chiedeva "qual è la seconda soglia ISEE (40.000 €)... e cosa comporta rispetto alla prima?", con una faithfulness di 0.0. Analizzando il documento sorgente, **27.000 € e 40.000 € non sono due soglie graduate della stessa agevolazione** — sono condizioni indipendenti per categorie di esenzione scollegate (40.000 € compare solo abbinato ad altre condizioni specifiche: diventare genitore durante l'anno, essere vicini alla laurea, certificazioni di invalidità...). La domanda e la `expected_answer` presupponevano una gerarchia inesistente nel documento. **Lezione**: un punteggio di faithfulness basso non implica automaticamente un'allucinazione del modello — può essere il caso di valutazione stesso ad essere costruito su un fatto sbagliato.

### 5.4 Un'allucinazione reale, isolata con lo stesso metodo

Diversamente dal caso precedente, la domanda "se scelgo il pagamento a rate, quante ne devo pagare e quando?" ha prodotto una risposta con un errore fattuale genuino: "prima rata entro il 23 novembre 2026" — ma il documento dice che il 23 novembre è la data in cui **diventano visibili gli importi** della seconda/terza rata, non la scadenza della prima (che è il 24 settembre, o 19 novembre con mora). Il modello ha preso un numero reale dal contesto recuperato e l'ha attaccato al fatto sbagliato — un'allucinazione più subdola di un numero inventato di sana pianta, ed esattamente il tipo di errore per cui è pensato il futuro "Grounding Guard" (verifica letterale numero↔contesto, non solo "il numero esiste nel contesto da qualche parte").

### 5.5 Domande di sintesi che il retrieval non recuperava affatto

La domanda "quali sono le diverse metodologie di pagamento delle tasse?" restituiva sistematicamente "nessuna informazione pertinente", pur essendo l'informazione presente nel corpus. Investigando col retrieval isolato: il documento giusto **veniva trovato**, ma nessuno dei 20 candidati superava la soglia di reranking (0.3) — il migliore si fermava a 0.24. **Causa strutturale**: il chunking divide il documento per sezione/categoria di studente; nessun singolo chunk da 512 token è "dedicato" al tema generale "modalità di pagamento" — l'informazione è distribuita su più sezioni ciascuna centrata su altro (scadenze di una categoria specifica, importi, ecc.), quindi il cross-encoder non giudica mai un singolo chunk fortemente pertinente a una domanda ampia.

**Soluzione — retrieval gerarchico (`AutoMergingRetriever`).** Vedi §4.2 e §4.5: quando più chunk fratelli della stessa sezione compaiono tra i candidati, vengono fusi nel nodo genitore invece di restare frammenti isolati e scartati singolarmente. Tentativo naïve iniziale: applicare una soglia più permissiva alle sezioni fuse — ma testato su domande fattuali normali, faceva passare anche sezioni marginalmente pertinenti, diluendo il contesto proprio dove la precisione conta di più. **Soluzione finale**: soglia standard (0.3) invariata come primo tentativo per *tutte* le domande; solo se restituisce zero risultati si ritenta con una soglia permissiva (0.12) **applicata solo alle sezioni fuse**, mai ai chunk singoli. Verificato empiricamente: le domande fattuali continuano a essere risolte interamente dal passaggio standard (es. un punteggio di 0.98 per un caso reale), il fallback permissivo scatta solo quando altrimenti non ci sarebbe nessuna risposta.

### 5.6 Regressione sul filtro `stato`

Durante lo sviluppo, il campo `"stato": "vigente"` era stato rimosso da `metadata.py` (era comunque hardcoded, mai realmente popolato da una logica di validità temporale). Il filtro Qdrant su `stato = vigente` in `hybrid_search.py`, però, dipende da quel campo: senza, un nuovo re-indexing avrebbe fatto sì che **nessun punto** corrispondesse più al filtro, azzerando il retrieval. Ripristinato come parte del lavoro sul retrieval gerarchico, che richiedeva comunque un re-indexing completo.

### 5.7 `converter.py` rilanciava la conversione a ogni import

`convert_corpus(...)` era chiamata a livello di modulo, in fondo al file, invece che sotto `if __name__ == "__main__"` (pattern già usato correttamente in `chunker.py`). L'effetto: qualunque cosa importasse `ingestion.converter` — anche indirettamente, es. un futuro test o un altro script che importasse solo `build_converter` — avrebbe rilanciato l'intera conversione Docling (OCR, tabelle, VLM) su tutto `datasets/raw/`, con percorsi e device hardcoded (`Path("datasets/raw/")`, `device=2`) indipendenti da `settings`. **Soluzione**: spostata sotto `if __name__ == "__main__"`, con percorsi/device letti da `settings` come nel resto della pipeline (§4.1).

## 6. Limiti noti e lavoro futuro

- **Grounding Guard non implementato** (§7 del documento di progettazione): non esiste ancora uno stadio che estragga automaticamente date/importi dalla risposta generata e li verifichi letteralmente contro i chunk recuperati prima di mostrarla allo studente. È il gap più rilevante rispetto al rischio dichiarato del progetto (una data sbagliata è peggio di "non lo so") — il caso §5.4 sopra è un esempio concreto di cosa intercetterebbe.
- **`stato` è sempre `"vigente"`**: il meccanismo di filtraggio esiste ed è testato, ma non c'è ancora una logica che marchi un documento come `superato`/`bozza` quando ne arriva una versione più recente.
- **Sintesi cross-documento non coperta**: l'`AutoMergingRetriever` unisce chunk della *stessa* sezione/documento. Una domanda che richiede di aggregare informazioni sparse su documenti diversi (es. "riassumi tutto il percorso dall'iscrizione alla laurea") non è ancora gestita — servirebbe query decomposition/sub-question querying.
- **`README.md` è vuoto**: nessuna istruzione di setup, pur avendo già `.env.example` e `docker-compose.yml` pronti.
- **Nessun VLM dedicato Granite-Docling**: la descrizione automatica delle immagini usa `smolvlm_picture_description` (SmolVLM), non Granite-Docling come indicato nel documento di progettazione — scelta equivalente ma non identica a quanto originariamente pianificato.
- **Ablation dense-only/sparse-only non ancora possibile**: `run_ablations.py` (§4.11) confronta dimensione dei chunk, reranker on/off, automerging on/off e LLM diversi, ma non ancora "solo denso" vs "solo sparso" vs ibrido — richiederebbe un flag `retrieval_mode` in `hybrid_search.search_candidates()`/`HybridQdrantRetriever` (§4.5), non ancora implementato.
- **Il budget di token per il giudice RAGAS è una costante fissa, non un calcolo dinamico** (§5.2): `max_tokens=1024` riduce ma non elimina il rischio di sforare la finestra di contesto di vLLM con contesti recuperati grandi (chunk size alto, `top_k` alto, sezioni fuse da `AutoMergingRetriever`), e può a sua volta troncare verdetti insolitamente lunghi. La soluzione robusta (calcolare `max_tokens` in base ai token effettivi del prompt) non è ancora implementata.

## 7. Come eseguire il progetto (riferimento rapido)

1. `cp .env.example .env` e adattare i valori (porte, device GPU).
2. `docker compose up -d` — avvia Qdrant e vLLM.
3. Popolare `datasets/raw/` con i documenti, poi `python -m ingestion.converter` (parsing) e `python -m reindex` (chunking + embedding + indicizzazione — vedi §4.9).
4. `chainlit run src/app.py` — avvia l'interfaccia conversazionale.
5. Per la valutazione: `python -m eval.evaluate` (richiede un piccolo script che carichi modelli/dataset e chiami `evaluate()`, vedi `src/eval/evaluate.py`).
6. Per confrontare configurazioni diverse (chunk size, reranker/automerging on/off, LLM — §4.11): prima `python -m eval.select_diagnostic_subset` (una tantum, genera `datasets/eval/qa_test_set_diagnostic.json`, §4.12), poi `python -m eval.run_ablations --dataset ../datasets/eval/qa_test_set_diagnostic.json` — un file di risultati per configurazione in `datasets/eval/ablations/`. Sul set completo (senza `--dataset`) per il risultato finale da riportare in tesi.
