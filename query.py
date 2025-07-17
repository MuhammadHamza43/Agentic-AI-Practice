from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter
import sys

# Load model and Qdrant
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
client = QdrantClient(url='http://localhost:6333')
collection_name = "policies_by_text_length"

print("💬 HR Policy by content Chatbot (type 'exit' to quit)\n")

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
        limit=1  # top-1 match
    )

    if results:
        top_result = results[0]
        print(f"\n🤖 Answer: {top_result.payload['text_chunk'][:500]}...\n")
        print(f"📄 Source: {top_result.payload['title']} ({top_result.payload['department']})\n")
    else:
        print("🤖 Sorry, I couldn't find anything relevant.\n")
