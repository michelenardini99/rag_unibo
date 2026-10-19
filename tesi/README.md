# Tesi — struttura iniziale

Scheletro LaTeX della tesi, con le norme redazionali già applicate nel preambolo (`main.tex`) e un placeholder
`\notaredazionale{...}` in ogni sezione che indica cosa scrivere e, dove utile, a quale materiale già pronto nel
repository attingere. Questo file è una guida per chiunque riprenda in mano il progetto — non fa parte del documento
compilato.

## Come compilare

```bash
cd tesi
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex   # due passate dopo bibtex, per aggiornare indice e riferimenti incrociati
```

Produce `main.pdf`. I file generati dalla compilazione sono già esclusi da git (`.gitignore`).

## Norme redazionali applicate

| Norma | Dove, nel preambolo di `main.tex` |
|---|---|
| Font Times New Roman o Arial | `\usepackage{mathptmx}` (Times); per Arial vedi il blocco commentato subito sotto |
| Corpo 12 pt (note 10 pt) | `\documentclass[12pt,...]` — le note a piè di pagina sono automaticamente a 10pt con la classe `report` a 12pt |
| Margini 2,5 cm su tutti i lati | `\usepackage[top=2.5cm,bottom=2.5cm,left=2.5cm,right=2.5cm]{geometry}` |
| Interlinea 1,5 | `\usepackage{setspace}` + `\onehalfspacing` |

## Struttura dei capitoli e budget di pagine

| Capitolo | File | Pagine target |
|---|---|---|
| Introduzione | `capitoli/01_introduzione.tex` | 2 |
| Stato dell'arte | `capitoli/02_stato_dell_arte.tex` | 15 |
| Architettura e sviluppo del sistema | `capitoli/03_architettura_e_sviluppo.tex` | 25–30 |
| Validazione del sistema | `capitoli/04_validazione.tex` | 30 |
| Conclusioni e sviluppi futuri | `capitoli/05_conclusioni.tex` | 3 |
| Bibliografia | `bibliografia.bib` | — |

Totale indicativo: ~75–80 pagine + bibliografia.

## Dove trovare il materiale già pronto

Molto del contenuto dei capitoli 3 e 4 **esiste già**, scritto in dettaglio, altrove nel repository — i capitoli 3 e 4
vanno riadattati in prosa da tesi più che scritti da zero:

- **`REPORT_ARCHITETTURA.md`** (root del repository) — relazione tecnica completa: analisi dei requisiti, design
  architetturale, scelte tecnologiche, design di dettaglio per componente, problematiche affrontate, valutazione
  sperimentale, limiti noti. Ogni `\notaredazionale` nei capitoli 3 e 4 rimanda al paragrafo specifico da cui partire.
- **`datasets/eval/qa_test_set_diagnostic.json`** — il dataset diagnostico di 124 domande usato per la valutazione
  (Capitolo 4).
- **`datasets/eval/model_comparison/`** e **`datasets/eval/retrieval_screening/`** — risultati grezzi degli
  esperimenti di confronto modelli e ablation del retrieval.
- **`notebook/evaluation.ipynb`** — analisi e grafici dei risultati di valutazione.

**Attenzione**: alcuni di questi risultati sperimentali sono precedenti a correzioni fatte successivamente al sistema
(fusione dei chunk corti, filtro di corso diventato bonus di priorità, decomposizione delle domande multi-parte) — il
Capitolo 4 segnala esplicitamente dove serve ri-eseguire la valutazione per numeri aggiornati prima della consegna
finale.

## Nota sul titolo

Il titolo in `main.tex` ("Un assistente RAG locale per l'assistenza universitaria: progettazione e valutazione") è
quello discusso e scelto in una sessione di lavoro precedente — cambialo liberamente se cambia il taglio della tesi.
