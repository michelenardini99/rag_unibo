from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    data_raw_dir: Path = ROOT_DIR / "datasets" / "raw"
    data_converted_dir: Path = ROOT_DIR / "datasets" / "converted"
    data_chunked_dir: Path = ROOT_DIR / "datasets" / "chunked"
    data_eval_dir: Path = ROOT_DIR / "datasets" / "eval"

    crawl_delay_seconds: float = 1.0
    crawl_user_agent: str = "RagUniboBot/1.0 (+progetto tesi RAG Unibo)"
    crawl_max_pages: int | None = 500
    crawl_strip_query_strings: bool = True

    qdrant_url: str = "http://localhost:6333"
    qdrant_grpc_port: int = 6334
    qdrant_collection: str = "ateneo_docs"

    vllm_base_url: str = "http://localhost:8000/v1"
    generation_model: str = "generation-llm"
    generation_model_vision: bool = True

    vllm_max_model_len: int = 8192

    embeddings_device_id: int
    embedding_model_id: str = "BAAI/bge-m3"
    reranker_model_id: str = "BAAI/bge-reranker-v2-m3"

    retrieval_prefetch_limit: int = 50
    retrieval_candidate_limit: int = 20
    rerank_top_k: int = 5
    rerank_score_threshold: float = 0.3
    merged_threshold: float = 0.12
    corso_priority_boost: float = 0.1

    chunk_max_tokens: int = 512

    metadata_schema: list[str] = ["tipo_laurea", "corso", "sezione", "materia"]
    academic_year_start_month: int = 9

    assistant_institution: str = "Università di Bologna"
    assistant_scope: str = "regolamenti, tasse, iscrizioni, corsi e procedure amministrative"
    assistant_language: str = "italiano"
    assistant_audience: str = "studente"
    source_title_separator: str = " — "

    max_history_turns: int = 5
    condense_max_tokens: int = 64
    condense_temperature: float = 0.0


settings = Settings()
