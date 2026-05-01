from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

COLLECTION = "products_demo"

# 1) Connect to local Qdrant
client = QdrantClient(host="localhost", port=6333)

# 2) Create or reset collection (vector size = 4)
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)
client.create_collection(
    collection_name=COLLECTION, vectors_config=VectorParams(size=4, distance=Distance.COSINE)
)

# 3) Insert points
points = [
    PointStruct(id=1, vector=[0.9, 0.1, 0.1, 0.0], payload={"name": "red t-shirt"}),
    PointStruct(id=2, vector=[0.1, 0.9, 0.1, 0.0], payload={"name": "blue jeans"}),
    PointStruct(id=3, vector=[0.85, 0.15, 0.1, 0.0], payload={"name": "maroon t-shirt"}),
]
client.upsert(collection_name=COLLECTION, points=points)

# 4) Query nearest vectors
query_vector = [0.88, 0.12, 0.1, 0.0]
results = client.query_points(
    collection_name=COLLECTION,
    query=query_vector,
    limit=2,
).points

print("Top matches:")
for r in results:
    print(f"id={r.id}, score={r.score:.4f}, payload={r.payload}")