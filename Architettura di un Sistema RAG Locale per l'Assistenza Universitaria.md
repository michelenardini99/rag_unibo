# **Architettura di un Sistema RAG Locale per l'Assistenza Universitaria**

L'obiettivo di questo progetto è la progettazione di un assistente virtuale per il contesto universitario basato sull'architettura **Retrieval-Augmented Generation (RAG)**. Poiché le università gestiscono dati sensibili (dati degli studenti, bozze di regolamento, documenti amministrativi e materiali didattici coperti da proprietà intellettuale), l'infrastruttura deve operare **interamente in locale**, garantendo la riservatezza delle informazioni e l'assenza di costi ricorrenti verso servizi Cloud di terze parti1.

Il sistema è vincolato da tre requisiti non negoziabili, emersi durante la fase di analisi:

1. **Hardware limitato**: l'infrastruttura di calcolo disponibile è composta da **due GPU da 16 GB di VRAM ciascuna** (32 GB totali, non necessariamente aggregabili senza overhead). Questo esclude a priori i modelli linguistici di grandi dimensioni (27B+) ipotizzati in una prima bozza del progetto e impone un dimensionamento realistico di ogni componente della pipeline.
2. **Comprensione visiva dei contenuti**: una parte rilevante dell'informazione (in particolare nelle slide di lezione) è veicolata da diagrammi, grafici e schemi visivi, non da testo. Il sistema deve poter recuperare e ragionare su questi contenuti, non solo sul testo circostante.
3. **Precisione fattuale**: il sistema deve rispondere a domande su scadenze, importi e tasse (es. *"Come richiedere l'ISEU?"*). Una data o un importo errato non è un difetto accettabile ma un rischio concreto per lo studente — la precisione numerica ha priorità sulla fluidità della risposta.

Una delle principali sfide architetturali risiede inoltre nell'estrema **eterogeneità dei documenti d'Ateneo**:

> 1. **Documenti Amministrativi e Regolamenti**: Caratterizzati da alta densità testuale, rigida gerarchia normativa (titoli, articoli, commi) e tabelle complesse (es. tasse, scadenze, conversioni di voto)2.
> 2. **Slide di Lezioni (PPTX e PDF)**: Caratterizzate da un layout bidimensionale frammentato, poche frasi per pagina, presenza di grafici, schemi visivi e note del relatore4.
> 3. **Articoli Scientifici e Dispense**: Caratterizzate da strutture a più colonne, elevata presenza di formule matematiche/chimiche complesse e riferimenti bibliografici2.

Il presente documento analizza i migliori strumenti open source e gratuiti per ogni fase della pipeline RAG, delineando lo stack tecnologico raccomandato e i meccanismi introdotti per rispettare i tre vincoli sopra elencati.

## **1\. Elaborazione e Parsing dei Documenti**

La fase di parsing è il punto più critico dell'intero sistema RAG2. Un parser che altera l'ordine di lettura, corrompe le tabelle o distrugge la struttura gerarchica produce blocchi di testo (*chunk*) privi di coerenza semantica, compromettendo sia la fase di embedding sia la generazione finale dell'LLM2.

### **Confronto tra i Parser Open Source Primari**

| Caratteristica / Strumento | Marker (Datalab) | Docling (IBM Research) |
| :---- | :---- | :---- |
| **Formati Supportati** | Solo PDF (richiede conversione preventiva)2. | Multi-formato nativo: PDF, DOCX, PPTX, XLSX, Immagini, HTML4. |
| **Estrazione Formule Matematiche** | Eccellente (riconoscimento avanzato in sintassi LaTeX)2. | Buono (estrazione testo/LaTeX via modelli dedicati)4. |
| **Gestione Layout e Gerarchia** | Ottima per PDF accademici mono/doppia colonna2. | Eccellente (modelli *DocLayNet* e *TableFormer* per gerarchia e tabelle)1. |
| **Chunking Integrato** | No (restituisce Markdown piatto; chunking manuale necessario)2. | Sì (HybridChunker e HierarchicalChunker nativi con metadati gerarchici)1. |
| **Comprensione Visiva (immagini/diagrammi)** | Assente. | Sì, tramite modello VLM dedicato (Granite-Docling)4. |
| **Velocità ed Efficienza** | Lento su file complessi, rischio timeout su grandi volumi2. | Buona efficienza su CPU/GPU, ideale per elaborazioni locali continuous1. |
| **Licenza Software** | GPL-3.0 (restrittiva per ambiti commerciali o chiusi)2. | MIT (completamente permissiva e open source)1. |

### **Valutazione per il Dominio Universitario**

* **Marker**: Risulta molto efficace per la conversione di **articoli scientifici e dispense di matematica/fisica** in formato PDF grazie alla capacità di convertire equazioni complesse in blocchi LaTeX puliti2. Tuttavia, la mancanza di supporto nativo per i file .pptx (slide) e .docx, l'assenza di comprensione visiva e la licenza GPL-3.0 ne limitano l'impiego come parser unico d'Ateneo2. Questo è un compromesso consapevole: si accetta un'estrazione delle formule leggermente meno accurata nei paper scientifici in cambio di una pipeline unificata su tutte le tipologie di documento.
* **Docling**: Costituisce la scelta primaria per l'architettura d'Ateneo1. Grazie al modello di rappresentazione unificato DoclingDocument, gestisce nativamente le **slide delle lezioni (PPTX)** preservando il layout vettoriale, i **regolamenti (PDF/DOCX)** mantenendo la struttura dei paragrafi, e le **tabelle amministrative dense** mediante l'algoritmo *TableFormer*1. È inoltre l'unico dei due strumenti a offrire, tramite il modello VLM Granite-Docling, un canale nativo per la comprensione delle componenti visive (§3).

## **2\. Comprensione del Testo (Embedding)**

Il modello di embedding converte i frammenti di testo in vettori matematici per consentire la ricerca semantica. Nei contesti accademici è necessario gestire sia domande generiche in linguaggio naturale (es. *"Come richiedere l'ISEU?"*) sia query su codici di corsi ed equazioni (es. *"Proprietà dell'algoritmo QuickSort"*).

### **Confronto tra i Modelli di Embedding Open Source**

> 1. **BAAI / BGE-M3**:
>
>    * **Pro**: Standard di riferimento per il RAG accademico. Implementa una **ricerca ibrida 3-in-1** nativa (vettori densi per la semantica, vettori sparsi tipo BM25 per termini tecnici/codici corso, e rappresentazione multi-vector ColBERT). Supporta oltre 100 lingue (italiano incluso) con una finestra di contesto di 8.192 token. Dimensionalmente leggero (~570M parametri, ~2 GB in fp16): l'ingombro VRAM è trascurabile rispetto al budget disponibile.
>    * **Contro**: Genera vettori a 1024 dimensioni e richiede maggiori risorse computazionali in fase di inferenza rispetto a modelli esclusivamente densi.
>
> 2. **Tencent / KaLM-Embedding-Gemma3-12B**:
>
>    * **Pro**: Basato sulle architetture avanzate Gemma 3, offre una comprensione semantica di alto livello, risultando efficace nella comprensione di concetti scientifici astratti presenti negli articoli accademici.
>    * **Contro**: Essendo un modello da 12 miliardi di parametri, richiede una quota rilevante di VRAM GPU — **incompatibile con il budget di 2×16 GB**, che deve già ospitare il modello generativo, il reranker e il VLM di ingestion.
>
> 3. **Qwen / Qwen3-Embedding-8B**:
>
>    * **Pro**: Modello di fascia media (8B parametri) altamente ottimizzato per il retrieval e per la comprensione di dati strutturati (come le tabelle estratte dai regolamenti). Perfettamente integrabile con la famiglia di modelli LLM Qwen.
>    * **Contro**: Anche in versione quantizzata, il suo ingombro compete direttamente con lo spazio VRAM riservato al modello generativo multimodale (§6).

**Scelta**: **BGE-M3** resta la raccomandazione primaria anche dopo l'introduzione del vincolo hardware: è l'unico dei tre a poter coesistere comodamente con reranker, VLM di ingestion e LLM generativo sullo stesso budget di 32 GB, senza compromessi sulla qualità del retrieval ibrido.

## **3\. Comprensione Visiva dei Contenuti (Immagini e Diagrammi)**

Il parsing testuale da solo non è sufficiente: nelle slide di lezione l'informazione più rilevante è spesso **il diagramma stesso**, non il testo sparso attorno ad esso (poche frasi per pagina, layout bidimensionale frammentato). Ignorare questa componente significa perdere una parte sostanziale del contenuto informativo del corpus. Questo problema va risolto separatamente in due momenti distinti della pipeline, con esigenze e vincoli hardware differenti.

### **3.1 Comprensione al momento dell'Ingestion (offline, batch)**

Ogni immagine, grafico o schema estratto da Docling viene passato a un **modello VLM dedicato** che ne genera una descrizione testuale. Questa descrizione viene trattata come un chunk a sé stante, embeddato con BGE-M3 e indicizzato in Qdrant — rendendo così il contenuto visivo **ricercabile tramite query testuale**, esattamente come un paragrafo di regolamento.

* **Granite-Docling VLM (IBM, ~258M parametri)**: Modello VLM leggero, addestrato specificamente per la comprensione strutturale di documenti (OCR, tabelle, formule, layout) piuttosto che per la didascalia generica di immagini naturali. È già integrato nella pipeline Docling, ha un ingombro VRAM minimo (~1 GB) e, poiché l'ingestion è un processo batch e non concorrente con le richieste degli studenti, può essere eseguito senza competere per risorse con il motore di generazione.
* **Alternativa per contenuti visivi complessi**: se in fase di valutazione la qualità delle didascalie prodotte da Granite-Docling risultasse insufficiente per grafici scientifici densi (es. diagrammi di flusso multi-step, grafici con più serie di dati), è possibile eseguire un secondo passaggio con un VLM più capace (es. Qwen2.5-VL-32B) esclusivamente in fase di ingestion notturna, sfruttando temporaneamente entrambe le GPU — un lusso possibile solo perché l'ingestion non è vincolata dalla latenza di servizio in tempo reale.
* **Estensione possibile (fuori dal perimetro iniziale)**: in alternativa alla sola didascalia testuale, si potrebbe generare anche un embedding visivo diretto dell'immagine (es. con un encoder tipo SigLIP) per una ricerca per similarità visiva pura. Questo aggiunge complessità e un ulteriore modello da mantenere in VRAM; per la tesi si raccomanda di partire dall'approccio a didascalia testuale, più semplice, e discutere l'estensione multimodale-nativa come lavoro futuro.

### **3.2 Comprensione al momento della Generazione (online, per query)**

La sola didascalia testuale è una rappresentazione **lossy** del contenuto visivo. Quando un chunk selezionato dal retrieval è associato a un'immagine (es. uno schema di processo per l'immatricolazione), la qualità della risposta finale migliora sensibilmente se l'immagine **originale** viene passata direttamente al modello generativo, che può ragionarci sopra invece di fidarsi solo della descrizione prodotta a monte da un altro modello.

Questo richiede che il **modello generativo stesso sia vision-capable** (§6) — una scelta che condiziona direttamente la selezione dell'LLM finale, discussa nella sezione sul dimensionamento hardware.

## **4\. Database Vettoriale**

Il database vettoriale archivia gli embedding e gestisce il recupero delle informazioni in frazioni di secondo.

### **1\. Qdrant (Scelta Consigliata)**

* **Pro**: Scritto in Rust, offre elevate prestazioni computazionali, consumo di RAM contenuto e scalabilità. Supporta nativamente la ricerca ibrida (densa \+ sparsa di BGE-M3) e offre una gestione avanzata del **filtraggio sui metadati** (payload filtering) senza rallentamenti — incluso il filtraggio per **validità temporale dei documenti** (§7). Include una Web UI grafica ed esegue tecniche di **quantizzazione dei vettori** riducendo l'occupazione di memoria fino al 70%, un vantaggio ulteriore in un'infrastruttura già vincolata in risorse.
* **Contro**: Richiede l'esecuzione tramite contenitore dedicato (es. Docker).

### **2\. ChromaDB**

* **Pro**: Estremamente semplice da integrare (può essere eseguito in-memory o salvato su file locale senza configurazioni complesse).
* **Contro**: Presenta degradi prestazionali quando la mole di dati cresce (es. indicizzazione di interi dipartimenti universitari) e non possiede le funzionalità avanzate di filtraggio metadati di Qdrant.

## **5\. Orchestrazione del Flusso RAG**

L'orchestratore gestisce la catena logica tra il parsing, il database vettoriale, la gestione dei prompt e l'LLM.

### **1\. LlamaIndex (Scelta Consigliata per il Retrieval)**

* **Pro**: Specializzato nell'indicizzazione e nel recupero di dati strutturati e non strutturati. Gestisce nativamente le strutture RAG gerarchiche (*Parent-Child*, *Sub-Question Querying*) ed è integrato con il modello dati di Docling (DoclingNodeParser), preservando la gerarchia capitolo-paragrafo-comma dei regolamenti d'Ateneo.
* **Contro**: Meno flessibile di LangChain nella costruzione di agenti autonomi complessi con molti strumenti esterni.

### **2\. LangChain**

* **Pro**: Framework versatile per la costruzione di applicazioni LLM trasversali. Attraverso il sintassi LCEL (*LangChain Expression Language*), consente di concatenare prompt, modelli e tool esterni (es. integrazioni con API per la prenotazione esami o calcolatrici Python per esercizi didattici)2. Include lo strumento di tracciamento *LangSmith*.
* **Contro**: Maggiore complessità nel codice d'integrazione RAG rispetto a LlamaIndex.

## **6\. Vincoli Hardware e Motore di Inferenza Locale**

L'infrastruttura disponibile è composta da **due GPU da 16 GB di VRAM ciascuna** (32 GB totali). Questo vincolo, non presente nella bozza iniziale del progetto, esclude i modelli generativi di grandi dimensioni (27B–49B) precedentemente ipotizzati: anche fortemente quantizzati, questi modelli non lascerebbero margine sufficiente per l'embedding, il reranker, il VLM di ingestion e soprattutto per la cache KV necessaria a servire più studenti in contemporanea.

Va inoltre chiarito un possibile equivoco: un'architettura *Mixture-of-Experts* (es. Qwen3-30B-A3B, "30B totali / 3B attivi") **riduce il calcolo per token ma non l'ingombro in VRAM**, poiché tutti gli esperti devono comunque risiedere in memoria. Non è quindi una soluzione al vincolo di 16 GB per scheda, ma un ottimizzatore di velocità — utile solo su hardware con VRAM abbondante, non il nostro caso.

### **Piano di allocazione per GPU**

| GPU | Ruolo | Componenti | Note |
| :---- | :---- | :---- | :---- |
| **GPU A (16 GB)** | Generazione, servita in tempo reale | LLM generativo *vision-capable*, quantizzato 4-bit (AWQ/GPTQ), via vLLM | Deve rimanere caricata continuamente; priorità alla latenza e al numero di richieste concorrenti |
| **GPU B (16 GB)** | Retrieval + Ingestion | BGE-M3 (~2 GB) \+ BGE-Reranker-v2-m3 (~1 GB) \+ Granite-Docling VLM (~1 GB, solo durante i job di ingestion) | Ampio margine libero; i job di ingestion sono batch e non competono con il traffico studenti |

Dedicare un'intera GPU al modello generativo (anziché frammentarlo con tensor-parallelism su entrambe le schede) evita l'overhead di comunicazione inter-GPU e semplifica il deployment, mantenendo la seconda scheda interamente dedicata ai servizi "leggeri" e sempre disponibile.

### **Candidati per il Modello Generativo (vision-capable, dimensionati per 16 GB)**

> 1. **Qwen2.5-VL-7B-Instruct**: Modello nativamente multimodale, ottimo supporto per documenti/tabelle e per l'italiano, ~5–6 GB in 4-bit — lascia ampio margine di VRAM per la cache KV e per servire più studenti in parallelo.
> 2. **Gemma 3 12B (variante vision)**: Qualità di ragionamento superiore rispetto a un 7B, buona sintesi concettuale; in 4-bit occupa ~7–8 GB, margine di cache KV più ridotto ma ancora accettabile su 16 GB. Da preferire se la valutazione mostra che la qualità di sintesi del 7B è insufficiente per i regolamenti più articolati.

Entrambi sostituiscono le scelte originarie (Qwen 3.6-35B, Gemma 3 27B, Nemotron 49B), non compatibili con il budget hardware disponibile. Il costo di questa scelta è una minore capacità di ragionamento puro rispetto a un modello 27B+; il beneficio è un sistema che effettivamente gira sull'hardware disponibile, con margine per la concorrenza multi-studente e per la comprensione visiva nativa (§3.2).

### **Motore di Inferenza: vLLM**

vLLM resta lo standard di riferimento anche a questa scala: le ottimizzazioni di memoria (*PagedAttention*) e il batching continuo sono ciò che permette, a parità di 16 GB, di servire più studenti simultaneamente invece di accodare le richieste una alla volta.

## **7\. Garanzia di Precisione Fattuale (Grounding Guard)**

Un RAG "standard" (recupera → inserisci nel contesto → lascia scrivere il testo all'LLM) non offre garanzie sufficienti quando l'errore possibile è una **scadenza di pagamento sbagliata**: una risposta fluida e ben scritta ma con una data errata è l'esito peggiore possibile, peggiore di un "non lo so". Per questo motivo si introduce uno stadio dedicato, assente in un RAG generico, tra la generazione e la consegna della risposta allo studente.

1. **Preservazione letterale delle tabelle**: le tabelle estratte da TableFormer (tasse, scadenze, conversioni voto) non vengono mai riassunte o parafrasate a monte; restano in forma di markdown/HTML letterale nel chunk e vengono incluse così nel prompt, in modo che l'LLM possa citarle invece di ricostruirle a memoria.
2. **Verifica di grounding post-generazione**: prima di mostrare la risposta, un controllo automatico (estrazione di date/importi/numeri tramite regex o NER leggero) verifica che ogni valore numerico presente nella risposta compaia **letteralmente** nei chunk recuperati. In caso di mancata corrispondenza, la risposta viene segnalata, soppressa o rigenerata vincolando l'LLM a citare solo ciò che è presente nel contesto. Questo controllo è economico (non richiede un altro LLM) ed è mirato esattamente al rischio dichiarato.
3. **Citazione obbligatoria della fonte**: ogni affermazione numerica in risposta è accompagnata dal riferimento al documento e all'articolo/pagina di provenienza, così lo studente può sempre verificare direttamente sulla fonte primaria.
4. **Metadati di validità temporale**: ogni documento viene taggato con data di validità e stato (*vigente*, *superato*, *bozza*). Il retrieval filtra di default sui soli documenti vigenti, evitando che un regolamento superato — ma ancora indicizzato per finalità storiche — venga citato come attuale.

Questo modulo, oltre a essere necessario dal punto di vista della sicurezza dell'informazione, rappresenta un possibile **contributo originale della tesi**: le metriche standard di RAGAS/TruLens (fedeltà, pertinenza) non misurano specificamente se una data o un importo sono stati alterati nella riformulazione. Costruire e validare un set di valutazione dedicato a questo tipo di errore (§9) è un'estensione naturale e misurabile del lavoro.

## **8\. Architettura Logica Finale del Sistema**

Di seguito viene illustrato il flusso completo dell'infrastruttura RAG locale d'Ateneo, aggiornato per riflettere il vincolo hardware, la comprensione visiva e la garanzia di precisione fattuale.

```text
+-------------------------------------------------------------------------------------+
| 1. STADIO DI ACQUISIZIONE E PARSING (INGESTION)                — batch, GPU B       |
|    - Origine Dati: Regolamenti (PDF/DOCX), Slide (PPTX), Articoli (PDF)              |
|    - Motore Testo: Docling (DocLayNet + TableFormer)                                |
|    - Motore Visivo: Granite-Docling VLM -> didascalia testuale di immagini/grafici   |
|    - Output: Documenti Markdown/JSON strutturati + didascalie immagini indicizzabili |
+-------------------------------------------------------------------------------------+
                                          |
                                          v
+-------------------------------------------------------------------------------------+
| 2. STADIO DI ORCHESTRAZIONE E CHUNKING STRUTTURALE                                   |
|    - Framework: LlamaIndex (HybridChunker)                                          |
|    - Metadati: Facoltà, Corso, Anno Accademico, Breadcrumb Gerarchico,               |
|      Data di Validità / Stato (vigente, superato, bozza)                            |
|    - Collegamento chunk di testo <-> immagine sorgente (per la generazione, §3.2)    |
+-------------------------------------------------------------------------------------+
                                          |
                                          v
+-------------------------------------------------------------------------------------+
| 3. STADIO DI VETTORIALIZZAZIONE E ARCHIVIAZIONE                — GPU B               |
|    - Modello Embedding: BGE-M3 (Denso + Sparso + ColBERT)                            |
|    - Vector Database: Qdrant (Payload Filtering incl. validità temporale)            |
+-------------------------------------------------------------------------------------+
                                          |
                                          v
+-------------------------------------------------------------------------------------+
| 4. STADIO DI ESECUZIONE QUERY E RETRIEVAL                      — GPU B               |
|    - Input: Domanda in linguaggio naturale dello studente                           |
|    - Ricerca Ibrida in Qdrant -> Top-20 candidati (filtro: solo documenti vigenti)   |
|    - Re-Ranking di Precisione: BGE-Reranker-v2-m3 -> Top-5 frammenti                 |
|    - Recupero delle immagini/diagrammi collegati ai chunk selezionati                |
+-------------------------------------------------------------------------------------+
                                          |
                                          v
+-------------------------------------------------------------------------------------+
| 5. STADIO DI GENERAZIONE MULTIMODALE                            — GPU A              |
|    - Motore Inferenza: vLLM (quantizzazione 4-bit AWQ/GPTQ)                          |
|    - Modello: Qwen2.5-VL-7B-Instruct / Gemma 3 12B (vision-capable)                  |
|    - Input: Domanda + Top-5 testo (tabelle in forma letterale) + immagini associate  |
|    - Output: Bozza di risposta in italiano                                           |
+-------------------------------------------------------------------------------------+
                                          |
                                          v
+-------------------------------------------------------------------------------------+
| 6. STADIO DI GROUNDING GUARD (VERIFICA FATTUALE)                                     |
|    - Estrazione di date/importi/numeri dalla bozza di risposta                       |
|    - Verifica di corrispondenza letterale con i chunk recuperati                     |
|    - Mismatch -> soppressione / flag / rigenerazione vincolata                       |
|    - Allegata citazione della fonte (documento + articolo/pagina) per ogni fatto      |
+-------------------------------------------------------------------------------------+
                                          |
                                          v
+-------------------------------------------------------------------------------------+
| 7. VALUTAZIONE E OSSERVABILITÀ CONTINUA                                              |
|    - Framework: RAGAS / TruLens (fedeltà, pertinenza, assenza di allucinazioni)      |
|    - Set di Valutazione Dedicato: stress-test su fatti numerici (date, importi)      |
+-------------------------------------------------------------------------------------+
```

## **9\. Valutazione e Osservabilità Continua**

Oltre alle metriche standard di RAGAS/TruLens (fedeltà al contesto, pertinenza, assenza di allucinazioni generiche), si raccomanda la costruzione di un **set di valutazione dedicato ai fatti numerici** (date di scadenza, importi, conversioni di voto), costruito a partire dai regolamenti reali d'Ateneo. Questo set permette di misurare specificamente il tasso di errore del Grounding Guard (§7) e di dimostrare, con numeri, l'effetto della sua introduzione rispetto a un RAG senza verifica — un confronto quantitativo che si presta bene a un capitolo di valutazione sperimentale nella tesi.

## **10\. Sintesi dello Stack Tecnologico Raccomandato**

> 1. **Parsing Documentale**: **Docling (IBM Research)** — Copertura nativa di PDF normativi, presentazioni PPTX, file DOCX e schemi scientifici, con preservazione delle tabelle, licenza MIT e un canale VLM integrato per i contenuti visivi1.
> 2. **Comprensione Visiva**: **Granite-Docling VLM** in ingestion (didascalie indicizzabili) \+ modello generativo *vision-capable* in fase di risposta (§3, §6) — copre sia la ricercabilità che il ragionamento diretto sui diagrammi.
> 3. **Orchestrazione RAG**: **LlamaIndex** — Gestione nativa del chunking gerarchico e integrazione immediata con le strutture dati di Docling.
> 4. **Embedding**: **BAAI / BGE-M3** — Ricerca ibrida (semantica \+ parole chiave/codici) indispensabile nel contesto universitario, con ingombro VRAM trascurabile.
> 5. **Database Vettoriale**: **Qdrant** — Alta velocità, filtraggio dei metadati (incluso lo stato di validità temporale) e ottimizzazione della RAM tramite quantizzazione.
> 6. **Re-Ranking**: **BGE-Reranker-v2-m3** — Affina i risultati presi dal database vettoriale prima di inviarli all'LLM.
> 7. **Motore di Inferenza e LLM**: **vLLM** unito a **Qwen2.5-VL-7B-Instruct** o **Gemma 3 12B** (quantizzati 4-bit) — dimensionati per girare su una singola GPU da 16 GB, con capacità multimodale nativa.
> 8. **Garanzia Fattuale**: modulo custom di **Grounding Guard** (§7) — verifica automatica di date/importi contro i chunk recuperati, prima della consegna della risposta.
> 9. **Valutazione e Osservabilità**: **RAGAS / TruLens** \+ set di valutazione dedicato ai fatti numerici, per il monitoraggio continuo delle metriche di recupero, generazione e correttezza fattuale.

#### **Bibliografia**

> 1. Docling: An Efficient Open-Source Toolkit for AI-driven Document Conversion \- arXiv, [https://arxiv.org/html/2501.17887v1](https://arxiv.org/html/2501.17887v1)
> 2. doccrush/document-parser-benchmark \- GitHub, [https://github.com/doccrush/document-parser-benchmark](https://github.com/doccrush/document-parser-benchmark)
> 3. PDF Parsing Accuracy Benchmark: Docling vs Unstructured vs Marker vs Visual Pipeline, [https://www.ertas.ai/blog/pdf-parsing-accuracy-benchmark-docling-unstructured](https://www.ertas.ai/blog/pdf-parsing-accuracy-benchmark-docling-unstructured)
> 4. docling-project/docling: Get your documents ready for gen AI \- GitHub, [https://github.com/docling-project/docling](https://github.com/docling-project/docling)
> 5. \[Feature\]: Docling-powered document processing extension — native RAG for OpenClaw · Issue \#23200 \- GitHub, [https://github.com/openclaw/openclaw/issues/23200](https://github.com/openclaw/openclaw/issues/23200)
> 6. Chunking \- Docling \- GitHub Pages, [https://docling-project.github.io/docling/concepts/chunking/](https://docling-project.github.io/docling/concepts/chunking/)
> 7. opendataloader-project/opendataloader-bench: OpenDataLoader Benchmark \- GitHub, [https://github.com/opendataloader-project/opendataloader-bench](https://github.com/opendataloader-project/opendataloader-bench)
> 8. Docling: A Guide to Building a Document Intelligence App | DataCamp, [https://www.datacamp.com/tutorial/docling](https://www.datacamp.com/tutorial/docling)
