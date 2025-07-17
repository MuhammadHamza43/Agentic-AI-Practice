import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from langchain.memory import ConversationBufferMemory
from langchain_postgres import PostgresChatMessageHistory
import psycopg  
import uuid
import tiktoken

#Loading Keys
load_dotenv()

#my postgress_url 
POSTGRES_URL = "postgresql://Admin:12345678@localhost:5432/mydatabase"

# Establish a synchronous connection to Postgres
sync_connection = psycopg.connect(POSTGRES_URL)

# Define table name and session id
TABLE_NAME = "chat_history"
SESSION_ID = str(uuid.uuid4()) 

# Create the table if it doesn't exist (only needs to be done once)
PostgresChatMessageHistory.create_tables(sync_connection, TABLE_NAME)

# Initialize PostgresChatMessageHistory
history = PostgresChatMessageHistory(
    TABLE_NAME,
    SESSION_ID,
    sync_connection=sync_connection
)

# Initialize Qdrant client
client = QdrantClient(url="http://localhost:6333")

# Initialize ConversationBufferMemory with Postgres history
memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=history,
    return_messages=True
)

# Initialize vector store
vectorstore = QdrantVectorStore(
    client=client,
    collection_name="hr_policies",
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    content_payload_key= "full_content"
)

# Retrieves context 
retriever = vectorstore.as_retriever(search_kwargs={"k":50})

# Chsining it all
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4.1-nano",
        temperature=0.6,
    ),
    retriever=retriever,
    memory=memory,
    return_source_documents=False 
)

#Token counting function
def count_tokens(text, model="gpt-4.1-nano"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

# Simple UI

print("HR Policy Assistant started. Type 'exit' or 'quit' to stop.")
TOTAL_TOKEN_LIMIT = 12096

while True:
    query = input("\nYou: ")
    if query.lower() in ['exit', 'quit']:
        break
    
    try:
        # Extract chat history and count tokens
        chat_history = " ".join([m.content for m in memory.chat_memory.messages])
        input_tokens = count_tokens(query) + count_tokens(chat_history)
        retrieved_docs = retriever.get_relevant_documents(query)
        total_doc_tokens = 0
        for doc in retrieved_docs:
            total_doc_tokens += count_tokens(doc.page_content)
        #Total input tokens include query, chat history, and retrieved documents
        total_input_tokens = input_tokens + total_doc_tokens
        # Check if total input tokens exceed the limit
        if total_input_tokens > TOTAL_TOKEN_LIMIT:
            print(f"\n Total input tokens ({total_input_tokens}) exceed the set limit ({TOTAL_TOKEN_LIMIT}). Please shorten your query or chat history.")
            break
        # Invoke the chain
        result = qa_chain.invoke({"question": query})
        # Extract answer and count output tokens
        answer = result["answer"]
        output_tokens = count_tokens(answer)
        chat_history = result.get("chat_history", [])
        print(f"\n🤖 {answer}")
        print(f"Input tokens: {total_input_tokens} | Output tokens: {output_tokens}  | Total: {total_input_tokens + output_tokens}")
        #Error handling
    except Exception as e:
        for message in chat_history:
            print(f"{message.type}: {message.content}")
        
    except:
        print(f"Error: {e}")
        print("Please try again with a different query.")
