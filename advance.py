import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
import pinecone

load_dotenv()

from pinecone import Pinecone  
from langchain_pinecone import Pinecone as LangchainPinecone
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from langchain_pinecone.vectorstores import PineconeVectorStore

def get_vectorstore(option):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if option == "qdrant":
        client = QdrantClient(url="http://localhost:6333")
        return Qdrant(
            client=client,
            collection_name="hr_policies",
            embeddings=embeddings,
            content_payload_key="full_content"
        )

    elif option == "pinecone":
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = "hrbot-index"
        index = pc.Index(index_name)
        return PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="content",
        )

    else:
        raise ValueError("Invalid option")

print("Choose vector store backend:")
print("1. Qdrant")
print("2. Pinecone")
choice = input("Enter 1 or 2: ").strip()

if choice == "1":
    backend = "qdrant"
elif choice == "2":
    backend = "pinecone"
else:
    print("Invalid choice. Exiting.")
    exit(1)

vectorstore = get_vectorstore(backend)
retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an HR Policy Assistant. Use the following context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer as helpfully as possible:"
    )
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),  
        temperature=0.6
    ),
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=False,
    verbose=True  
)

print(f"\n💼 HR Policy Assistant started using {backend.capitalize()}. Type 'exit' or 'quit' to stop.")

while True:
    query = input("\n🧑 You: ")
    
    if query.lower() in ['exit', 'quit']:
        print("👋 Session ended. Goodbye!")
        break

    try:
        # Get answer from chain
        result = qa_chain.invoke({"query": query})
        answer = result["result"]
        print(f"\n🤖 {answer}")
        
    except Exception as e:
        print(f"\n⚠️ Error: {e}")
        print("Please try again with a different query.")

