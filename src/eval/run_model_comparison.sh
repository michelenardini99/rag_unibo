#!/usr/bin/env bash
# Confronto tra i modelli di generazione con un giudice unico (vedi
# eval/evaluate_model_comparison.py): prima ogni modello genera le risposte,
# poi il giudice, caricato una volta sola, le valuta tutte. Va lanciato da src/,
# meglio dentro tmux: dura diverse ore.
#
#   bash eval/run_model_comparison.sh [cartella-risultati]
set -euo pipefail

RESULTS_DIR="${1:-../datasets/eval/model_comparison/giudice_unico}"
JUDGE="Qwen/Qwen2.5-14B-Instruct-AWQ"
MODELS=(
  "Qwen/Qwen2.5-14B-Instruct-AWQ:Qwen-2.5-14B"
  "casperhansen/mistral-nemo-instruct-2407-awq:Mistral-Nemo-12B"
  "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4:Llama-3.1-8B"
  "Qwen/Qwen2.5-7B-Instruct-AWQ:Qwen-2.5-7B"
)
PYTHON=../.venv/bin/python

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

load_model() {
  local model_id="$1"
  log "Carico in vLLM $model_id..."
  # The shell variable takes precedence over VLLM_MODEL_ID in .env.
  (cd .. && VLLM_MODEL_ID="$model_id" docker compose up -d --force-recreate vllm)
  for _ in $(seq 180); do
    served=$(curl -sf localhost:8000/v1/models | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin)['data'][0]['root'])" 2>/dev/null || true)
    if [[ "$served" == "$model_id" ]]; then log "vLLM pronto con $model_id."; return; fi
    sleep 10
  done
  log "vLLM non pronto dopo 30 minuti con $model_id."; exit 1
}

for entry in "${MODELS[@]}"; do
  load_model "${entry%%:*}"
  log "Generazione: ${entry##*:}"
  "$PYTHON" -m eval.evaluate_model_comparison generate --name "${entry##*:}" --results-dir "$RESULTS_DIR"
done

load_model "$JUDGE"
for entry in "${MODELS[@]}"; do
  log "Giudizio: ${entry##*:} (giudice $JUDGE)"
  "$PYTHON" -m eval.evaluate_model_comparison judge --name "${entry##*:}" --results-dir "$RESULTS_DIR" --judge-model-id "$JUDGE"
done
log "Completato. Risultati in $RESULTS_DIR; vLLM resta con $JUDGE (il modello di produzione)."
