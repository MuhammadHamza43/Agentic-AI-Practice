import pandas as pd
import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Constants
COLLECTION_NAME = "policies_by_text_length"
CHUNK_SIZE = 150
OVERLAP = 25

# Load data
df = pd.read_csv("HR.csv") 
df = df.dropna(subset=['content'])  # Changed from 'text' to 'content'

# Initialize model and Qdrant
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
client = QdrantClient(url="http://localhost:6333")

# Recreate collection if exists
if COLLECTION_NAME in [col.name for col in client.get_collections().collections]:
    client.delete_collection(collection_name=COLLECTION_NAME)

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Chunking function
def chunk_text(text, chunk_size=150, overlap=25):
    text = str(text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks

# Process and upload
points = []
for idx, row in df.iterrows():
    content_chunks = chunk_text(row["content"], CHUNK_SIZE, OVERLAP)

    for i, chunk in enumerate(content_chunks):
        embedding = model.encode(chunk).tolist()
        metadata = {
            "chunk_index": i,
            "text_chunk": chunk,
            "title": row.get("title", ""),
            "department": row.get("department", ""),
            "year": int(row.get("year", 0)) if pd.notna(row.get("year")) else 0,
            "author": row.get("author", ""),
            "language": row.get("language", "")
        }
        point = PointStruct(
            id=uuid.uuid4().int >> 96,
            vector=embedding,
            payload=metadata
        )
        points.append(point)

# Upload to Qdrant
client.upsert(collection_name=COLLECTION_NAME, points=points)

print(f"✅ {len(points)} chunks uploaded to collection '{COLLECTION_NAME}'.")
