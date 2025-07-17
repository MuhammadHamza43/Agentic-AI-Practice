from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter
import sys

# Load model and Qdrant
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
client = QdrantClient(url='http://localhost:6333')
collection_name = "hr_policies"

print("💬 HR Policy Chatbot (type 'exit' to quit)\n")

while True:
    question = input("You: ")
    if question.lower() in ['exit', 'quit']:
        break

    # Convert question to vector
    vector = model.encode(question).tolist()

    # Search top match
    results = client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=100  # top-1 match
    )

    print(f"\nFound {len(results)} result(s). Showing up to 15 best matches:\n")
    for idx, result in enumerate((results or [])[:30], 1):
        content = result.payload.get('content', '')[:1000]
        title = result.payload.get('title', 'Unknown')
        department = result.payload.get('department', 'Unknown')
        year = result.payload.get('year', 'Unknown')
        author = result.payload.get('author', 'Unknown')
        print(f"{idx}. 🤖 Answer: {content}\n   📄 Source: {title} ({department}) | Year: {year} | Author: {author}\n")
