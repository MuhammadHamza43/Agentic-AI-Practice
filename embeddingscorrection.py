import pandas as pd
import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Set up model and Qdrant
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
client = QdrantClient(url="http://localhost:6333")
collection_name = "hr_policies"

# Recreate collection
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Load data
df = pd.read_csv("HR.csv")
df = df.fillna('')  # Replace NaNs with empty strings to avoid errors

# Create points with enriched content
points = []
for idx, row in df.iterrows():
    # Combine metadata into content for embedding
    enriched_content = f"""Title: {row['title']}
Department: {row['department']}
Author: {row['author']}
Year: {row['year']}
Language: {row['language']}

{row['content']}"""

    embedding = model.encode(enriched_content)

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding.tolist(),
        payload={
            'sr': int(row['sr']) if 'sr' in row and pd.notna(row['sr']) else idx,
            'title': str(row['title']),
            'content': str(row['content']),  # original content still preserved
            'department': str(row['department']),
            'author': str(row['author']),
            'year': int(row['year']) if pd.notna(row['year']) else None,
            'language': str(row['language']),
            'content_length': len(str(row['content'])),
            'word_count': len(str(row['content']).split()),
            'full_content': enriched_content  # 👈 this is the key you'll point to in retrieval
        }
    )
    points.append(point)

# Upload to Qdrant
client.upsert(collection_name=collection_name, points=points)

print(f"✅ Done! Embedded and stored {len(points)} documents.")
