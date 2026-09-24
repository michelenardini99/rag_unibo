from qdrant_client import QdrantClient, models
from embedding.embedder import FILTER_ONLY_KEYS


def ensure_collection(client: QdrantClient, collection_name: str, recreate: bool = False) -> None:
    """
    Ensure that a collection exists in Qdrant. If it doesn't exist, create it.

    Args:
        client (QdrantClient): The Qdrant client instance.
        collection_name (str): The name of the collection to ensure.
        recreate (bool): If True and the collection already exists, delete and
            recreate it instead of leaving it as-is. Used by a full reindex to
            avoid orphaned points whose id is no longer present in the docstore.
    """
    if client.collection_exists(collection_name):
        if not recreate:
            _ensure_payload_indexes(client, collection_name)
            print(f"Collection '{collection_name}' already exists.")
            return
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE),
            "colbert": models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        }
    )
    _ensure_payload_indexes(client, collection_name)


# Fields queried by an exact-match filter or faceted for their vocabulary
# (see retrieval.query_filters) — "stato" for the always-on vigente filter,
# "id" for delete_by_doc_id, the rest for query-time metadata filtering.
_INDEXED_FIELDS = ("stato", "id", "tipo_laurea", "corso", "sezione", "materia", "anno_accademico")


def _ensure_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    """
    Creates the keyword payload indexes retrieval/filtering relies on
    (`_INDEXED_FIELDS`), tolerating ones that already exist so this can run
    again on a pre-existing collection that predates one of these fields.

    Args:
        client (QdrantClient): The Qdrant client instance.
        collection_name (str): The collection to index.
    """
    for field_name in _INDEXED_FIELDS:
        try:
            client.create_payload_index(
                collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD
            )
        except Exception:
            pass

def delete_by_doc_id(client: QdrantClient, collection_name: str, doc_id: str) -> None:
    """
    Deletes every point belonging to one document, keyed by its stable "id"
    payload field (see chunking.metadata.extract_metadata).

    Args:
        client (QdrantClient): The Qdrant client instance.
        collection_name (str): The collection to delete from.
        doc_id (str): The document's stable "id" metadata value.
    """
    client.delete(
        collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="id", match=models.MatchValue(value=doc_id))]
            )
        ),
    )


def nodes_to_points(nodes: list, embeddings: dict) -> list[models.PointStruct]:
    """
    Convert a list of nodes and their embeddings into Qdrant PointStructs.

    Args:
        nodes (list): A list of nodes to convert.
        embeddings (dict): A dictionary containing embeddings for the nodes.

    Returns:
        list[models.PointStruct]: A list of PointStructs ready for insertion into Qdrant.
    """
    points = []
    for i, node in enumerate(nodes):
        weights = embeddings["lexical_weights"][i]
        points.append(
            models.PointStruct(
                id=node.node_id,
                vector={
                    "dense": embeddings["dense_vecs"][i].tolist(),
                    "colbert": embeddings["colbert_vecs"][i].tolist(),
                    "sparse": models.SparseVector(
                        indices=[int(k) for k in weights],
                        values=list(weights.values())
                    )
                },
                payload={
                    "text": node.text,
                    "headings": node.metadata.get("headings"),
                    **{k: node.metadata.get(k) for k in FILTER_ONLY_KEYS},
                    "image_paths": node.metadata.get("image_paths") or [],
                }
            )
        )
    return points

def upsert_nodes(client: QdrantClient, collection_name: str, nodes: list, embeddings: dict, batch_size: int = 64) -> None:
    """
    Upsert nodes into a Qdrant collection.

    Args:
        client (QdrantClient): The Qdrant client instance.
        collection_name (str): The name of the collection to upsert into.
        nodes (list): A list of nodes to upsert.
        embeddings (dict): A dictionary containing embeddings for the nodes.
    """
    points = nodes_to_points(nodes, embeddings)
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name,
            points=points[i:i + batch_size]
        )
    