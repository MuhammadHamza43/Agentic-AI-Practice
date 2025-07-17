import os
import uuid
import json
import time
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
import psycopg

# Load environment variables
load_dotenv()

# Constants
POSTGRES_URL = "postgresql://Admin:12345678@localhost:5432/ai"
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


class HRPolicyAssistant:
    """Simple HR Policy Assistant class with methods to get and interact."""
    
    def __init__(self):
        self.conn = None
        self.checkpointer = None
        self.vectorstore = None
        self.app = None
        self.total_tokens_used = 0
        self.setup_database()
        self.setup_components()
    
    def get_connection(self):
        """Get or create a database connection with retry logic."""
        if self.conn and not self.conn.closed:
            try:
                with self.conn.cursor() as cur:
                    cur.execute("SELECT 1")
                return self.conn
            except Exception:
                pass
        
        for attempt in range(MAX_RETRIES):
            try:
                self.conn = psycopg.connect(
                    POSTGRES_URL, connect_timeout=CONNECTION_TIMEOUT
                )
                self.conn.autocommit = True
                return self.conn
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                raise e
    
    def setup_database(self):
        """Setup database tables and checkpointer."""
        conn = self.get_connection()
        
        # Setup checkpointer
        self.checkpointer = PostgresSaver(conn)
        self.checkpointer.setup()
        
        # Create tables
        with conn.cursor() as cur:
            cur.execute(CREATE_THREADS_TABLE_SQL)
    
    def setup_components(self):
        """Setup all components (vector store, model, tools, agent)."""
        # Initialize Qdrant client and vector store
        client = QdrantClient(url=QDRANT_URL)
        
        # Initialize embeddings with proper device handling and error handling
        try:
            embedding_kwargs = {
                'model_name': EMBEDDING_MODEL,
                'model_kwargs': {'device': 'cpu', 'trust_remote_code': True},
                'encode_kwargs': {'normalize_embeddings': False}
            }
            embeddings = HuggingFaceEmbeddings(**embedding_kwargs)
        except Exception as e:
            st.error(f"Error initializing embeddings: {e}")
            # Fallback to a simpler initialization
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'}
                )
            except Exception as e2:
                st.error(f"Fallback embedding initialization failed: {e2}")
                # Use a different model as last resort
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
        
        self.vectorstore = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            content_payload_key="full_content",
        )
        
        # Define tools
        @tool
        def get_hr_policy(query: str) -> str:
            """Retrieve HR policy documents relevant to the user's query."""
            docs = self.vectorstore.as_retriever(search_kwargs={"k": SEARCH_K}).invoke(query)
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
            checkpointer=self.checkpointer,
        )
    
    def create_thread(self, title):
        """Create a new thread and return thread_id."""
        thread_id = str(uuid.uuid4())
        conn = self.get_connection()
        
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chatbot_threads (thread_id, title) VALUES (%s, %s)",
                (thread_id, title),
            )
        
        return thread_id
    
    def get_thread_titles(self):
        """Return a list of (thread_id, title) for all threads."""
        conn = self.get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT thread_id, title FROM chatbot_threads ORDER BY timestamp DESC"
            )
            return cur.fetchall()
    
    def get_thread_messages(self, thread_id):
        """Fetch messages for a given thread_id."""
        conn = self.get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT messages FROM chatbot_threads WHERE thread_id = %s",
                (thread_id,),
            )
            result = cur.fetchone()
            return result[0] if result else []
    
    def get_thread_stats(self, thread_id):
        """Get token usage statistics for a thread."""
        conn = self.get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT input_tokens, output_tokens FROM chatbot_threads WHERE thread_id = %s",
                (thread_id,),
            )
            result = cur.fetchone()
            if result:
                return {
                    'input_tokens': result[0] or 0,
                    'output_tokens': result[1] or 0,
                    'total_tokens': (result[0] or 0) + (result[1] or 0)
                }
            return {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    
    def chat(self, user_input, thread_id):
        """Process user input and return AI response with token usage."""
        config = {"configurable": {"thread_id": thread_id}}
        input_message = HumanMessage(content=user_input)
        
        # Stream response
        ai_response = ""
        last_event = None
        
        for event in self.app.stream(
            {"messages": [input_message]}, config, stream_mode="values"
        ):
            ai_message = event["messages"][-1]
            if ai_message.type == "ai":
                ai_response = ai_message.content
                last_event = event
        
        # Extract token usage
        input_tokens, output_tokens, total_tokens = self.extract_token_usage(last_event)
        
        # Update thread messages
        self.update_thread_messages(thread_id, user_input, ai_response, input_tokens, output_tokens)
        
        return {
            'response': ai_response,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens
        }
    
    def update_thread_messages(self, thread_id, user_input, ai_response, input_tokens, output_tokens):
        """Update thread with new messages and token usage."""
        # Fetch current messages and append new ones
        messages = self.get_thread_messages(thread_id)
        messages.append({"sender": "user", "message": user_input, "timestamp": datetime.now().isoformat()})
        messages.append({"sender": "assistant", "message": ai_response, "timestamp": datetime.now().isoformat()})
        
        # Update database
        conn = self.get_connection()
        with conn.cursor() as cur:
            cur.execute(
                UPDATE_THREAD_MESSAGES_SQL,
                (json.dumps(messages), input_tokens, output_tokens, thread_id),
            )
    
    def extract_token_usage(self, event):
        """Extract token usage from event metadata."""
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        
        if event and event["messages"]:
            usage_metadata = getattr(event["messages"][-1], "usage_metadata", None)
            if usage_metadata and "total_tokens" in usage_metadata:
                input_tokens = usage_metadata.get("input_tokens", 0)
                output_tokens = usage_metadata.get("output_tokens", 0)
                total_tokens = usage_metadata.get("total_tokens", 0)
        
        return input_tokens, output_tokens, total_tokens
    
    def search_policies(self, query):
        """Search HR policies directly without chat interface."""
        docs = self.vectorstore.as_retriever(search_kwargs={"k": SEARCH_K}).invoke(query)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]


# Streamlit App
def main():
    st.set_page_config(
        page_title="HR Policy Assistant",
        page_icon="👥",
        layout="wide"
    )
    
    # Initialize the assistant with error handling
    if 'assistant' not in st.session_state:
        try:
            with st.spinner("Initializing HR Policy Assistant..."):
                st.session_state.assistant = HRPolicyAssistant()
            st.success("HR Policy Assistant initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize HR Policy Assistant: {e}")
            st.error("Please check your configuration and try again.")
            st.stop()
    
    # Sidebar for thread management
    st.sidebar.title("Chat Sessions")
    
    # Create new thread
    if st.sidebar.button("New Chat Session"):
        st.session_state.current_thread = None
        st.session_state.new_thread_mode = True
    
    # Load existing threads
    threads = st.session_state.assistant.get_thread_titles()
    
    if threads:
        st.sidebar.subheader("Previous Sessions")
        for thread_id, title in threads:
            if st.sidebar.button(f"📁 {title}", key=f"thread_{thread_id}"):
                st.session_state.current_thread = thread_id
                st.session_state.new_thread_mode = False
    
    # Main content area
    st.title("👥 HR Policy Assistant")
    st.markdown("Ask questions about HR policies, get policy information, or generate HR reports.")
    
    # Handle new thread creation
    if st.session_state.get('new_thread_mode', False):
        st.subheader("Start New Chat Session")
        thread_title = st.text_input("Enter a title for this chat session:")
        
        if st.button("Create Session") and thread_title:
            thread_id = st.session_state.assistant.create_thread(thread_title)
            st.session_state.current_thread = thread_id
            st.session_state.new_thread_mode = False
            st.success(f"Created new session: {thread_title}")
            st.rerun()
    
    # Chat interface
    if st.session_state.get('current_thread'):
        thread_id = st.session_state.current_thread
        
        # Display thread statistics
        stats = st.session_state.assistant.get_thread_stats(thread_id)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Input Tokens", stats['input_tokens'])
        with col2:
            st.metric("Output Tokens", stats['output_tokens'])
        with col3:
            st.metric("Total Tokens", stats['total_tokens'])
        
        # Display chat history
        messages = st.session_state.assistant.get_thread_messages(thread_id)
        
        chat_container = st.container()
        with chat_container:
            if messages:
                for msg in messages:
                    if msg.get('sender') == 'user':
                        st.chat_message("user").write(msg['message'])
                    else:
                        st.chat_message("assistant").write(msg['message'])
        
        # Chat input
        user_input = st.chat_input("Ask about HR policies...")
        
        if user_input:
            # Display user message
            st.chat_message("user").write(user_input)
            
            # Get AI response
            with st.spinner("Thinking..."):
                result = st.session_state.assistant.chat(user_input, thread_id)
            
            # Display AI response
            st.chat_message("assistant").write(result['response'])
            
            # Show token usage
            st.info(f"Tokens used - Input: {result['input_tokens']}, Output: {result['output_tokens']}, Total: {result['total_tokens']}")
            
            # Check token limit
            if stats['total_tokens'] + result['total_tokens'] > TOKEN_LIMIT:
                st.error("Token limit exceeded for this session. Please start a new session.")
            
            st.rerun()
    
    # Policy search tab
    st.markdown("---")
    st.subheader("🔍 Direct Policy Search")
    
    search_query = st.text_input("Search HR policies directly:")
    
    if st.button("Search Policies") and search_query:
        with st.spinner("Searching policies..."):
            results = st.session_state.assistant.search_policies(search_query)
        
        if results:
            st.success(f"Found {len(results)} relevant policies")
            for i, result in enumerate(results):
                with st.expander(f"Policy Result {i+1}"):
                    st.write(result['content'])
                    if result['metadata']:
                        st.json(result['metadata'])
        else:
            st.warning("No relevant policies found.")
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    ### How to use:
    1. **New Chat Session**: Click "New Chat Session" to start a new conversation
    2. **Continue Previous**: Select from previous sessions in the sidebar
    3. **Ask Questions**: Type your HR policy questions in the chat
    4. **Direct Search**: Use the policy search for quick document retrieval
    5. **Token Tracking**: Monitor token usage to stay within limits
    """)


if __name__ == "__main__":
    main()