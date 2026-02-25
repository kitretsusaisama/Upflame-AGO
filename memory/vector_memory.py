from typing import List, Dict, Any, Optional
import uuid
import json

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

class VectorMemory:
    def __init__(self, collection_name: str = "upflame_ago_memory", dimension: int = 768, path: str = ":memory:"):
        self.collection_name = collection_name
        self.dimension = dimension

        if HAS_QDRANT:
            # Initialize local Qdrant instance
            self.client = QdrantClient(location=path)
            # Create collection if not exists
            if not self.client.collection_exists(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
                )
        else:
            self.memory = []
            print("Warning: qdrant-client not installed. Using simple list for VectorMemory.")

    def add(self, text: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None):
        if metadata is None:
            metadata = {}
        metadata["text"] = text

        point_id = str(uuid.uuid4())

        if HAS_QDRANT:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=metadata
                    )
                ]
            )
        else:
            self.memory.append({"id": point_id, "vector": vector, "payload": metadata})

    def search(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        if HAS_QDRANT:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            return [hit.payload for hit in results]
        else:
            # Dummy cosine similarity search (slow, strictly for fallback demo)
            # Just return top k based on simple dot product
            if not self.memory:
                return []

            # Simple dot product
            scores = []
            for item in self.memory:
                score = sum(a*b for a,b in zip(query_vector, item["vector"]))
                scores.append((score, item["payload"]))

            scores.sort(key=lambda x: x[0], reverse=True)
            return [s[1] for s in scores[:limit]]
