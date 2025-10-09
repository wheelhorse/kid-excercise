from qdrant_client import QdrantClient

client = QdrantClient(
    url="http://back.com:6333",
    api_key="ysanta-hrms-qdrant"
)

qdrant_ids = set()
offset = 0
limit = 1000

scroll_generator = client.scroll(
    collection_name="resume_hybrid_search",
    offset=offset,
    limit=limit,
    with_vectors=False,
    with_payload=False
)

for points_batch in scroll_generator:
    # points_batch might be an int (total count) or a list of PointStruct
    if isinstance(points_batch, list):
        for point in points_batch:
            qdrant_ids.add(point.id)
        offset += len(points_batch)
        print("Scroll returned non-list:", offset)
    else:
        pass
        # sometimes scroll returns just the total number of points as int
        #print("Scroll returned non-list:", points_batch)
    
print(f"Total unique points in Qdrant: {len(qdrant_ids)}")

