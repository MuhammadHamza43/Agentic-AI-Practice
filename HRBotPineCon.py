from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.schema import Document
import os
from dotenv import load_dotenv, find_dotenv
# from langchain_pinecone import Pinecone as LangchainPinecone

from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone


load_dotenv(find_dotenv())

# Use the provided context file link as the host
pinecone_host = "https://hrbot-index-wyfr1cz.svc.aped-4627-b74a.pinecone.io"


# Initialize Pinecone client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Access the index correctly
index = pc.Index("hrbot-index")


# Create a LangChain Pinecone retriever
vectorstore = LangchainPinecone(
    index=index,
    embedding=OpenAIEmbeddings(),
    text_key="text",
    namespace="HR-namespace"
)

def query_hrbot(question, top_k=25):
    """Query Pinecone for relevant chunks given a question."""
    print(f"Query: {question}")
    results = vectorstore.similarity_search(question, k=top_k)
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(doc.page_content)
        print(f"Metadata: {doc.metadata}")

# Example usage:
if __name__ == "__main__":
    user_question = input("Ask HRBot a question: ")
    query_hrbot(user_question)