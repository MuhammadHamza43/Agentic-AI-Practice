import pandas as pd
import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Set up model and Qdrant
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
client = QdrantClient(url="http://localhost:6333")
collection_name = "hr_policies1"

# Recreate collection
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Load data
df = pd.read_csv("HR.csv")
contents = df['content'].fillna('').astype(str).tolist()
embeddings = model.encode(contents, show_progress_bar=True)

# Create points
points = []
for idx, row in df.iterrows():
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=embeddings[idx].tolist(),
        payload={
            'sr': int(row['sr']) if 'sr' in row and pd.notna(row['sr']) else 0,
            'title': str(row['title']),
            'content': str(row['content']),
            'department': str(row['department']),
            'author': str(row['author']),
            'year': int(row['year']) if pd.notna(row['year']) else None,
            'language': str(row['language']),
            'content_length': len(str(row['content'])),
            'word_count': len(str(row['content']).split())
        }
    )
    points.append(point)

# Upload to Qdrant
client.upsert(collection_name=collection_name, points=points)

print(f" Done! Embedded and stored {len(points)} documents.")
