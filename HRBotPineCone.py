import os
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion, VectorType
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Load CSV data as list of documents (one per row)
def load_documents(file_path):
    df = pd.read_csv(file_path)
    documents = []
    for _, row in df.iterrows():
        # Combine all metadata into the document text
        content = (
            f"Department: {row.get('department', '')}\n"
            f"Language: {row.get('language', '')}\n"
            f"Year: {row.get('year', '')}\n"
            f"Title: {row.get('title', '')}\n\n"
            f"Content: {row.get('content', '')}"
        )
        metadata = {
            "department": row.get("department", ""),
            "language": row.get("language", ""),
            "year": row.get("year", ""),
            "title": row.get("title", ""),
            "content": row.get("content", ""),
        }
        documents.append({"text": content, "metadata": metadata})
    return documents


# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "hrbot-index"
dimension = 384  # For sentence-transformers/all-MiniLM-L6-v2

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
        vector_type=VectorType.DENSE
    )

index = pc.Index(index_name)

# Use HuggingFace sentence transformer
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def embed_and_upsert(documents, batch_size=100):
    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    embeddings = embedding_model.embed_documents(texts)

    vectors = [
        (f"doc-{i}", emb, meta)
        for i, (emb, meta) in enumerate(zip(embeddings, metadatas))
    ]

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch, namespace="HR-namespace")
        print(f"Upserted batch {i // batch_size + 1} with {len(batch)} vectors")

# Run the process
documents = load_documents("HR.csv")
embed_and_upsert(documents)
