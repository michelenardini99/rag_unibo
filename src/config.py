from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    # --- Percorsi dataset ---
    data_raw_dir: Path = ROOT_DIR / "datasets" / "raw"
    data_converted_dir: Path = ROOT_DIR / "datasets" / "converted"
    data_chunked_dir: Path = ROOT_DIR / "datasets" / "chunked"
    data_eval_dir: Path = ROOT_DIR / "datasets" / "eval"

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_grpc_port: int = 6334
    qdrant_collection: str = "ateneo_docs"

    # --- LLM generativo (vLLM, GPU A — §6 architettura) ---
    vllm_base_url: str = "http://localhost:8000/v1"
    generation_model: str = "generation-llm"
    # Deve coincidere con VLLM_MAX_MODEL_LEN del server (docker-compose.yml):
    # è la stessa finestra di contesto, solo vista dal client.
    vllm_max_model_len: int = 8192

    # --- Modelli locali (GPU B — §6 architettura) ---
    embeddings_device_id: int
    embedding_model_id: str = "BAAI/bge-m3"
    reranker_model_id: str = "BAAI/bge-reranker-v2-m3"

    # --- Retrieval (§4-5 architettura) ---
    retrieval_prefetch_limit: int = 50
    retrieval_candidate_limit: int = 20
    rerank_top_k: int = 5
    rerank_score_threshold: float = 0.6

    # --- Chunking (§2 architettura) ---
    chunk_max_tokens: int = 512

    # --- Condensazione query / storico conversazione ---
    max_history_turns: int = 5
    condense_max_tokens: int = 64
    condense_temperature: float = 0.0


settings = Settings()
