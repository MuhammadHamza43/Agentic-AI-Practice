import os
from datetime import datetime
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
from psycopg import Connection

# Load environment variables
load_dotenv()


class RateLimitError(Exception):
    pass


# Postgres setup
POSTGRES_URL = "postgresql://Admin:12345678@localhost:5432/ai"
conn = Connection.connect(POSTGRES_URL)
conn.autocommit = True  
checkpointer = PostgresSaver(conn)
checkpointer.setup()

# Initialize Qdrant client and vector store
client = QdrantClient(url="http://localhost:6333")
vectorstore = QdrantVectorStore(
    client=client,
    collection_name="hr_policies",
    embedding=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ),
    content_payload_key="full_content",
)


# Define tools
@tool
def get_hr_policy(query: str) -> str:
    """Retrieve HR policy documents relevant to the user's query."""
    docs = vectorstore.as_retriever(search_kwargs={"k": 25}).invoke(query)
    return "\n".join([doc.page_content for doc in docs])


tools = [get_hr_policy]

# PostgresSaver setup (replace ConversationBufferMemory)
DB_URI = "postgresql://Admin:12345678@localhost:5432/mydatabase"
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

model = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4.1-nano",
    temperature=0.6,
)

app = create_react_agent(
    model,
    tools=tools,
    prompt="""You are an HR Policy Assistant. Your task is to help users find relevant HR policies or generate HR reports based on their queries.""",
    checkpointer=checkpointer,
)

# Unique thread id for conversation
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

print("HR Policy Assistant (LangGraph) started. Type 'exit' or 'quit' to stop.")

total_tokens_used = 0

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    input_message = HumanMessage(content=user_input)
    try:
        # Stream the response from the agent
        for event in app.stream(
            {"messages": [input_message]}, config, stream_mode="values"
        ):
            ai_message = event["messages"][-1]
            if ai_message.type != "ai":
                continue

            print(ai_message.content)
            # Extract token usage from usage_metadata only
            usage_metadata = getattr(event["messages"][-1], "usage_metadata", None)
            if usage_metadata and "total_tokens" in usage_metadata:
                input_tokens = usage_metadata.get("input_tokens", 0)
                output_tokens = usage_metadata.get("output_tokens", 0)
                total_tokens = usage_metadata.get("total_tokens", 0)
                total_tokens_used += total_tokens
        print(f"\nTokens used - input: {input_tokens}, output: {output_tokens}, total: {total_tokens}")
        print(f"Total tokens used in this session: {total_tokens_used}")

        if total_tokens_used > 50000:
            raise RateLimitError("Token limit exceeded for this chat session.")

    except RateLimitError as e:
        print(f"Error: {e}")
        break

conn.close()
