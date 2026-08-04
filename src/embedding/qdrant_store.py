from qdrant_client import QdrantClient, models
from embedding.embedder import FILTER_ONLY_KEYS


def ensure_collection(client: QdrantClient, collection_name: str) -> None:
    """
    Ensure that a collection exists in Qdrant. If it doesn't exist, create it.

    Args:
        client (QdrantClient): The Qdrant client instance.
        collection_name (str): The name of the collection to ensure.
    """
    if client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' already exists.")
        return
    else:
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
        client.create_payload_index(
            collection_name,
            field_name="stato",
            field_schema=models.PayloadSchemaType.KEYWORD
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
    