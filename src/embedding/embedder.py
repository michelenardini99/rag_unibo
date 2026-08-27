from FlagEmbedding import BGEM3FlagModel
from llama_index.core.schema import BaseNode

from config import settings

FILTER_ONLY_KEYS = [*settings.metadata_schema, "stato", "source_file", "source_path", "image_paths"]

def build_embedding_model(device_id: int) -> BGEM3FlagModel:
    """Builds the embedding model for the pipeline."""
    return BGEM3FlagModel(
        'BAAI/bge-m3',
        devices=f"cuda:{device_id}"
    )


def embed_nodes(nodes: list[BaseNode], model: BGEM3FlagModel) -> dict:
    for node in nodes:
        node.excluded_embed_metadata_keys.extend(FILTER_ONLY_KEYS)
    texts = [n.get_content(metadata_mode="embed") for n in nodes]
    return model.encode(
        texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True
    )