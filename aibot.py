import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient

load_dotenv()

client = QdrantClient(url="http://localhost:6333")

vectorstore = Qdrant(
    client=client,
    collection_name="hr_policies1",
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    content_payload_key="content"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini", 
        temperature=0.3
    ),
    retriever=retriever
)

print("HR Policy Assistant started. Type 'exit' or 'quit' to stop.")

while True:
    query = input("\nYou: ")
    if query.lower() in ['exit', 'quit']:
        break
    
    try:
        result = qa_chain.invoke({"query": query})
        answer = result["result"]
        print(f"\n🤖 {answer}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please try again with a different query.")
