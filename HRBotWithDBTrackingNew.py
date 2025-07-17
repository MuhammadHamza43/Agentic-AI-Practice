import os
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
import json
import psycopg
import time
import HRBotWIthDbTrancking as st


# Load environment variables
load_dotenv()

# Constants
POSTGRES_URL = "postgresql://Admin:12345678@localhost:5432/ai"
DB_URI = "postgresql://Admin:12345678@localhost:5432/mydatabase"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "hr_policies"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4.1-nano"
TEMPERATURE = 0.6
SEARCH_K = 25
TOKEN_LIMIT = 50000
CONNECTION_TIMEOUT = 60
MAX_RETRIES = 2
RETRY_DELAY = 1

# Database connection configuration
CONNECTION_KWARGS = {
    "autocommit": True,
    "prepare_threshold": 0,
}

# SQL queries
CREATE_THREADS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chatbot_threads (
    thread_id UUID PRIMARY KEY,
    title TEXT NOT NULL,
    messages JSONB DEFAULT '[]',
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

UPDATE_THREAD_MESSAGES_SQL = """
UPDATE chatbot_threads 
SET messages = %s, 
    input_tokens = COALESCE(input_tokens, 0) + %s,
    output_tokens = COALESCE(output_tokens, 0) + %s
WHERE thread_id = %s
"""


class RateLimitError(Exception):
    pass


class DatabaseManager:
    """Handles all database operations with efficient connection management."""

    def __init__(self, conn, postgres_url=POSTGRES_URL):
        self.postgres_url = postgres_url
        self.conn = conn
        self.checkpointer = None
        self._setup_database()

    def get_connection(self):
        """Get or create a connection with retry logic."""
        if self._is_connection_valid():
            return self.conn

        for attempt in range(MAX_RETRIES):
            try:
                self.conn = psycopg.connect(
                    self.postgres_url, connect_timeout=CONNECTION_TIMEOUT
                )
                self.conn.autocommit = True
                return self.conn
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                raise e

    def create_thread(self, thread_id, title):
        """Create a new thread row."""
        for attempt in range(MAX_RETRIES):
            try:
                conn = self.get_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO chatbot_threads (thread_id, title) VALUES (%s, %s)",
                        (thread_id, title),
                    )
                return
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    self.conn = None  # Force reconnection
                    continue
                raise e

    def get_thread_messages(self, thread_id):
        """Fetch messages JSON for a given thread_id."""
        for attempt in range(MAX_RETRIES):
            try:
                conn = self.get_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT messages FROM chatbot_threads WHERE thread_id = %s",
                        (thread_id,),
                    )
                    result = cur.fetchone()
                    return result[0] if result else []
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    self.conn = None  # Force reconnection
                    continue
                raise e

    def update_thread_messages(
        self, thread_id, messages, input_tokens=0, output_tokens=0
    ):
        """Update messages JSON and store input/output tokens separately for a given thread_id."""
        for attempt in range(MAX_RETRIES):
            try:
                conn = self.get_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        UPDATE_THREAD_MESSAGES_SQL,
                        (
                            json.dumps(messages),
                            input_tokens,
                            output_tokens,
                            thread_id,
                        ),
                    )
                return
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    self.conn = None  # Force reconnection
                    continue
                raise e

    def get_thread_titles(self):
        """Return a list of (thread_id, title) for all threads."""
        for attempt in range(MAX_RETRIES):
            try:
                conn = self.get_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT thread_id, title FROM chatbot_threads ORDER BY timestamp DESC"
                    )
                    return cur.fetchall()
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    self.conn = None  # Force reconnection
                    continue
                raise e

    def load_thread_messages(self, thread_id):
        """Load all messages for a given thread_id as a list."""
        for attempt in range(MAX_RETRIES):
            try:
                conn = self.get_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT messages FROM chatbot_threads WHERE thread_id = %s",
                        (thread_id,),
                    )
                    result = cur.fetchone()
                    return result[0] if result else []
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    self.conn = None  # Force reconnection
                    continue
                raise e

    def _setup_database(self):
        """Setup database tables and checkpointer."""
        # Create a separate connection for setup that we'll close after use
        setup_conn = psycopg.connect(
            self.postgres_url, connect_timeout=CONNECTION_TIMEOUT
        )
        setup_conn.autocommit = True

        try:
            # Setup checkpointer with the setup connection
            self.checkpointer = PostgresSaver(setup_conn)
            self.checkpointer.setup()

            # Create tables
            with setup_conn.cursor() as cur:
                cur.execute(CREATE_THREADS_TABLE_SQL)
        finally:
            # Don't close the setup connection as PostgresSaver might need it
            pass

        # Now get our main connection
        self.conn = self.get_connection()

    def _is_connection_valid(self):
        """Check if the current connection is valid."""
        try:
            if self.conn and not self.conn.closed:
                # Test the connection with a simple query
                with self.conn.cursor() as cur:
                    cur.execute("SELECT 1")
                return True
        except Exception:
            return False
        return False


class HRPolicyAssistant:
    """Main HR Policy Assistant class."""

    def __init__(self):
        conn = psycopg.connect(POSTGRES_URL, connect_timeout=CONNECTION_TIMEOUT)
        conn.autocommit = True
        self.db_manager = DatabaseManager(conn)
        self.total_tokens_used = 0
        self._setup_components()

    def run(self):
        """Main execution loop."""
        thread_id = str(uuid.uuid4())
        title = input("Enter a title for this chat session: ")
        self.db_manager.create_thread(thread_id, title)
        config = {"configurable": {"thread_id": thread_id}}

        print(
            f"HR Policy Assistant (LangGraph) started. Session title: '{title}'. Type 'exit' or 'quit' to stop."
        )

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            try:
                self._process_user_input(user_input, thread_id, config)
            except RateLimitError as e:
                print(f"Error: {e}")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")

    def _setup_components(self):
        """Setup all components (vector store, model, tools, agent)."""
        # Initialize Qdrant client and vector store
        client = QdrantClient(url=QDRANT_URL)
        self.vectorstore = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
            content_payload_key="full_content",
        )

        # Define tools
        @tool
        def get_hr_policy(query: str) -> str:
            """Retrieve HR policy documents relevant to the user's query."""
            docs = self.vectorstore.as_retriever(search_kwargs={"k": SEARCH_K}).invoke(
                query
            )
            return "\n".join([doc.page_content for doc in docs])

        tools = [get_hr_policy]

        # Setup model
        model = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=OPENAI_MODEL,
            temperature=TEMPERATURE,
        )

        # Create agent
        self.app = create_react_agent(
            model,
            tools=tools,
            prompt="""You are an HR Policy Assistant. Your task is to help users find relevant HR policies or generate HR reports based on their queries.""",
            checkpointer=self.db_manager.checkpointer,
        )

    def _process_user_input(self, user_input, thread_id, config):
        """Process user input and generate response."""
        input_message = HumanMessage(content=user_input)
        last_ai_message = None
        last_event = None

        # Stream the response from the agent
        for event in self.app.stream(
            {"messages": [input_message]}, config, stream_mode="values"
        ):
            ai_message = event["messages"][-1]
            if ai_message.type != "ai":
                continue

            print(ai_message.content)
            last_ai_message = ai_message
            last_event = event

        # Only update thread messages and tokens once, after streaming is done
        if last_ai_message and last_event:
            self._update_thread_with_response(
                thread_id, user_input, last_ai_message, last_event
            )

    def _update_thread_with_response(self, thread_id, user_input, ai_message, event):
        """Update thread with new messages and token usage."""
        # Fetch current messages and append new ones
        messages = self.db_manager.get_thread_messages(thread_id)
        messages.append({"message": user_input})
        messages.append({"message": ai_message.content})

        # Extract token usage
        input_tokens, output_tokens, total_tokens = self._extract_token_usage(event)
        self.total_tokens_used += total_tokens

        # Update database
        self.db_manager.update_thread_messages(
            thread_id, messages, input_tokens, output_tokens
        )

        # Display token usage
        print(
            f"\nTokens used - input: {input_tokens}, output: {output_tokens}, total: {total_tokens}"
        )
        print(f"Total tokens used in this session: {self.total_tokens_used}")

        # Check token limit
        if self.total_tokens_used > TOKEN_LIMIT:
            raise RateLimitError("Token limit exceeded for this chat session.")

    def _extract_token_usage(self, event):
        """Extract token usage from event metadata."""
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

        usage_metadata = getattr(event["messages"][-1], "usage_metadata", None)
        if usage_metadata and "total_tokens" in usage_metadata:
            input_tokens = usage_metadata.get("input_tokens", 0)
            output_tokens = usage_metadata.get("output_tokens", 0)
            total_tokens = usage_metadata.get("total_tokens", 0)

        return input_tokens, output_tokens, total_tokens


# Legacy function for backward compatibility
def get_global_conn():
    """Get or create a global connection with a large timeout."""
    global conn
    try:
        if conn and not conn.closed:
            return conn
    except Exception:
        pass
    conn = psycopg.connect(POSTGRES_URL, connect_timeout=CONNECTION_TIMEOUT)
    conn.autocommit = True
    return conn


def get_session_messages(thread_id):
    """Fetch all messages for a given thread_id, ordered by timestamp."""
    conn = get_global_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT sender, message FROM chatbot_messages WHERE thread_id = %s ORDER BY timestamp ASC",
            (thread_id,),
        )
        return cur.fetchall()


# Main execution
if __name__ == "__main__":
    # Initialize global connection for backward compatibility

    # Run the assistant
    assistant = HRPolicyAssistant()
    assistant.run()
