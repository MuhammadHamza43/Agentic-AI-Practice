from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
client = QdrantClient(url='http://localhost:6333')
collection_name = "hr_policies"

print("💬 HR Policy Chatbot (type 'exit' to quit)\n")

while True:
    question = input("You: ")
    if question.lower() in ['exit', 'quit']:
        break

    vector = model.encode(question).tolist()

    results = client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=5,
        with_payload=True
    )

    if results:
        top_result = results[0]
        payload = top_result.payload
        print(f"\n🤖 Answer: {payload.get('content', '')[:1000]}...\n")
        print(f"📄 Source: {payload.get('title', '')} ({payload.get('department', '')})\n")
    else:
        print("🤖 Sorry, I couldn't find anything relevant.\n")
