from qdrant_client import QdrantClient
from pprint import pprint
from qdrant_client.models import SparseVector, NamedSparseVector

client = QdrantClient(
    url="http://back.com:6333",
    api_key="ysanta-hrms-qdrant"
)

qdrant_ids = set()
offset = 0
limit = 1000

info = client.get_collection("resume_hybrid_search")
pprint(info)

info = client.scroll(collection_name="resume_hybrid_search", limit=1, with_vectors=True)
print(info)

sparse_query = SparseVector(indices=[2084, 2671, 3723, 86188, 909202, 921853, 921854, 921855, 921856, 921857, 921858], values=[3.6415367, 2.7641938, 3.9246392, 9.580978, 14.294881, 15.672969, 15.672969, 15.672969, 15.672969, 14.842136, 15.672969])

results = client.search(
    collection_name="resume_hybrid_search",
                    query_vector=NamedSparseVector(  # Fixed: Use keyword args
                    name="sparse",
                    vector=sparse_query
                ),
    limit=5,
    with_vectors=True
)
for r in results:
    print(r.id, r.score)

