import os
import uuid
import json
import time
import asyncio
from typing import List, Dict, Optional
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
from langchain_tavily import TavilySearch
import psycopg
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading
import uvicorn
import re

# Load environment variables
load_dotenv()

# Constants
POSTGRES_URL = "postgresql://Admin:12345678@localhost:5432/ai"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "hr_policies"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEMPERATURE = 0.6
SEARCH_K = 6
TOKEN_LIMIT = 50000
CONNECTION_TIMEOUT = 60
MAX_RETRIES = 2
RETRY_DELAY = 1

# Available GPT models
AVAILABLE_MODELS = {
    "GPT-4.1 Nano": "gpt-4.1-nano",
    "GPT-4.0 Nano": "gpt-4.0-nano", 
    "GPT-3.5 Nano": "gpt-3.5-nano"
}

# FastAPI port
FASTAPI_PORT = 8000

# Users table
CREATE_USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chatbot_users (
    username TEXT PRIMARY KEY,
    display_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Function to create user-specific thread table
def get_user_table_name(username: str) -> str:
    """Generate a safe table name for the user."""
    # Clean username to make it safe for table names
    clean_username = re.sub(r'[^a-zA-Z0-9_]', '_', username.lower())
    return f"chatbot_threads_{clean_username}"

def create_user_threads_table_sql(table_name: str) -> str:
    """Generate SQL to create user-specific threads table."""
    return f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        thread_id UUID PRIMARY KEY,
        title TEXT NOT NULL,
        messages JSONB DEFAULT '[]',
        input_tokens INTEGER DEFAULT 0,
        output_tokens INTEGER DEFAULT 0,
        model_name TEXT DEFAULT 'gpt-4.1-nano',
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

# Function to get update thread messages SQL for user table
def get_update_thread_messages_sql(table_name: str) -> str:
    """Generate SQL to update thread messages for user table."""
    return f"""
    UPDATE {table_name} 
    SET messages = %s, 
        input_tokens = COALESCE(input_tokens, 0) + %s,
        output_tokens = COALESCE(output_tokens, 0) + %s,
        model_name = %s
    WHERE thread_id = %s
    """

# Enhanced Pydantic Models for FastAPI
class ChatRequest(BaseModel):
    message: str
    thread_id: str
    username: str
    model_name: Optional[str] = "gpt-4.1-nano"

class ChatResponse(BaseModel):
    response: str
    input_tokens: int
    output_tokens: int
    total_tokens: int

class ThreadCreate(BaseModel):
    title: str
    username: str

class ThreadResponse(BaseModel):
    thread_id: str
    title: str

class PolicySearchRequest(BaseModel):
    query: str

class PolicySearchResponse(BaseModel):
    results: List[Dict]

class UserCreate(BaseModel):
    username: str
    display_name: Optional[str] = None

class UserResponse(BaseModel):
    username: str
    display_name: str
    created_at: str
    last_login: str

# Enhanced HR Policy Assistant Class
class HRPolicyAssistant:
    """Enhanced HR Policy Assistant class with user management."""
    
    def __init__(self):
        self.conn = None
        self.checkpointer = None
        self.vectorstore = None
        self.apps = {}  # Store different agents for different models
        self.user_checkpointers = {}  # Store checkpointers for different users
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
        """Setup database tables and main checkpointer."""
        conn = self.get_connection()
        
        # Setup main checkpointer (can be used for general purposes)
        self.checkpointer = PostgresSaver(conn)
        self.checkpointer.setup()
        
        # Create users table
        with conn.cursor() as cur:
            cur.execute(CREATE_USERS_TABLE_SQL)
    
    def get_user_checkpointer(self, username: str):
        """Get or create a user-specific checkpointer."""
        if username not in self.user_checkpointers:
            # Create user-specific table
            self.ensure_user_table_exists(username)
            # For now, we'll use the main checkpointer but with user-specific thread IDs
            # In a more advanced setup, you could create separate checkpointer instances
            self.user_checkpointers[username] = self.checkpointer
        return self.user_checkpointers[username]
    
    def ensure_user_table_exists(self, username: str):
        """Ensure user-specific table exists."""
        table_name = get_user_table_name(username)
        conn = self.get_connection()
        
        with conn.cursor() as cur:
            cur.execute(create_user_threads_table_sql(table_name))
    
    def setup_components(self):
        """Setup all components (vector store, models, tools, agents)."""
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
            print(f"Error initializing embeddings: {e}")
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'}
                )
            except Exception as e2:
                print(f"Fallback embedding initialization failed: {e2}")
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
        
        tools = [get_hr_policy, TavilySearch(
            api_key=os.getenv("TAVILY_API_KEY"),
            search_type="web",
            search_kwargs={"k": SEARCH_K}
        )]

        # Create agents for all available models
        for model_display_name, model_name in AVAILABLE_MODELS.items():
            try:
                model = ChatOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model=model_name,
                    temperature=TEMPERATURE,
                )
                
                # Create agent for this model
                self.apps[model_name] = create_react_agent(
                    model,
                    tools=tools,
                    prompt="""You are an HR Policy Assistant. Your task is to help users find relevant HR policies or generate HR reports based on their queries. If needed ask for department and year of policy from user if not provided in conversation. Always call the agent for any query related to policies. If such policy does not exist say so. Follow the chat history and focus on last policy query and do not move to another policy unless user does.""",
                    checkpointer=self.checkpointer,
                )
                print(f"Initialized agent for model: {model_name}")
            except Exception as e:
                print(f"Failed to initialize model {model_name}: {e}")
    
    def get_agent(self, model_name: str):
        """Get the agent for the specified model."""
        if model_name not in self.apps:
            raise ValueError(f"Model {model_name} not available. Available models: {list(self.apps.keys())}")
        return self.apps[model_name]
    
    def create_or_get_user(self, username: str, display_name: str = None):
        """Create a new user or get existing user."""
        conn = self.get_connection()
        display_name = display_name or username
        
        with conn.cursor() as cur:
            # Try to insert new user
            try:
                cur.execute(
                    "INSERT INTO chatbot_users (username, display_name) VALUES (%s, %s)",
                    (username, display_name),
                )
            except Exception:
                # User already exists, update last login
                cur.execute(
                    "UPDATE chatbot_users SET last_login = CURRENT_TIMESTAMP WHERE username = %s",
                    (username,),
                )
        
        # Ensure user table exists
        self.ensure_user_table_exists(username)
        return username
    
    def get_all_users(self):
        """Get all users."""
        conn = self.get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT username, display_name, created_at, last_login FROM chatbot_users ORDER BY last_login DESC"
            )
            return cur.fetchall()
    
    def create_thread(self, title: str, username: str, model_name: str = "gpt-4.1-nano"):
        """Create a new thread for a user and return thread_id."""
        thread_id = str(uuid.uuid4())
        table_name = get_user_table_name(username)
        conn = self.get_connection()
        
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {table_name} (thread_id, title, model_name) VALUES (%s, %s, %s)",
                (thread_id, title, model_name),
            )
        return thread_id
    
    def get_thread_titles(self, username: str):
        """Return a list of (thread_id, title, model_name) for a user's threads."""
        table_name = get_user_table_name(username)
        conn = self.get_connection()
        
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT thread_id, title, model_name FROM {table_name} ORDER BY timestamp DESC"
            )
            return cur.fetchall()
    
    def get_thread_messages(self, thread_id: str, username: str):
        """Fetch messages for a given thread_id of a user."""
        table_name = get_user_table_name(username)
        conn = self.get_connection()
        
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT messages FROM {table_name} WHERE thread_id = %s",
                (thread_id,),
            )
            result = cur.fetchone()
            return result[0] if result else []
    
    def get_thread_stats(self, thread_id: str, username: str):
        """Get token usage statistics for a thread."""
        table_name = get_user_table_name(username)
        conn = self.get_connection()
        
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT input_tokens, output_tokens, model_name FROM {table_name} WHERE thread_id = %s",
                (thread_id,),
            )
            result = cur.fetchone()
            if result:
                return {
                    'input_tokens': result[0] or 0,
                    'output_tokens': result[1] or 0,
                    'total_tokens': (result[0] or 0) + (result[1] or 0),
                    'model_name': result[2] or 'gpt-4.1-nano'
                }
            return {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'model_name': 'gpt-4.1-nano'}
    
    def chat(self, user_input: str, thread_id: str, username: str, model_name: str = "gpt-4.1-nano"):
        """Process user input and return AI response with token usage."""
        app = self.get_agent(model_name)
        # Use username-prefixed thread_id for langgraph
        config = {"configurable": {"thread_id": f"{username}_{thread_id}"}}
        input_message = HumanMessage(content=user_input)
        
        # Stream response
        ai_response = ""
        last_event = None
        
        for event in app.stream(
            {"messages": [input_message]}, config, stream_mode="values"
        ):
            ai_message = event["messages"][-1]
            if ai_message.type == "ai":
                ai_response = ai_message.content
                last_event = event
        
        # Extract token usage
        input_tokens, output_tokens, total_tokens = self.extract_token_usage(last_event)
        
        # Update thread messages
        self.update_thread_messages(thread_id, username, user_input, ai_response, input_tokens, output_tokens, model_name)
        
        return {
            'response': ai_response,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens
        }
    
    def update_thread_messages(self, thread_id: str, username: str, user_input: str, ai_response: str, 
                             input_tokens: int, output_tokens: int, model_name: str):
        """Update thread with new messages and token usage."""
        messages = self.get_thread_messages(thread_id, username)
        messages.append({"sender": "user", "message": user_input, "timestamp": datetime.now().isoformat()})
        messages.append({"sender": "assistant", "message": ai_response, "timestamp": datetime.now().isoformat()})
        
        table_name = get_user_table_name(username)
        conn = self.get_connection()
        update_sql = get_update_thread_messages_sql(table_name)
        
        with conn.cursor() as cur:
            cur.execute(
                update_sql,
                (json.dumps(messages), input_tokens, output_tokens, model_name, thread_id),
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
    
    def search_policies(self, query: str):
        """Search HR policies directly without chat interface."""
        docs = self.vectorstore.as_retriever(search_kwargs={"k": SEARCH_K}).invoke(query)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]


# FastAPI App
app = FastAPI(title="HR Policy Assistant API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global assistant instance
assistant = None

@app.on_event("startup")
async def startup_event():
    global assistant
    assistant = HRPolicyAssistant()

@app.get("/")
async def root():
    return {"message": "HR Policy Assistant API is running"}

@app.get("/models")
async def get_available_models():
    """Get list of available models."""
    return {"models": AVAILABLE_MODELS}

@app.post("/users", response_model=UserResponse)
async def create_user(request: UserCreate):
    """Create or get a user."""
    username = assistant.create_or_get_user(request.username, request.display_name)
    users = assistant.get_all_users()
    user_data = next((u for u in users if u[0] == username), None)
    
    if user_data:
        return UserResponse(
            username=user_data[0],
            display_name=user_data[1],
            created_at=user_data[2].isoformat(),
            last_login=user_data[3].isoformat()
        )
    
    raise HTTPException(status_code=404, detail="User not found")

@app.get("/users")
async def get_all_users():
    """Get all users."""
    users = assistant.get_all_users()
    return [
        {
            "username": user[0],
            "display_name": user[1],
            "created_at": user[2].isoformat(),
            "last_login": user[3].isoformat()
        }
        for user in users
    ]

@app.post("/threads", response_model=ThreadResponse)
async def create_thread(request: ThreadCreate):
    """Create a new chat thread for a user."""
    thread_id = assistant.create_thread(request.title, request.username)
    return ThreadResponse(thread_id=thread_id, title=request.title)

@app.get("/threads/{username}")
async def get_user_threads(username: str):
    """Get all threads for a user."""
    threads = assistant.get_thread_titles(username)
    return [
        {"thread_id": thread_id, "title": title, "model_name": model_name} 
        for thread_id, title, model_name in threads
    ]

@app.get("/threads/{username}/{thread_id}/messages")
async def get_thread_messages(username: str, thread_id: str):
    """Get messages for a specific thread."""
    messages = assistant.get_thread_messages(thread_id, username)
    return {"messages": messages}

@app.get("/threads/{username}/{thread_id}/stats")
async def get_thread_stats(username: str, thread_id: str):
    """Get thread statistics."""
    stats = assistant.get_thread_stats(thread_id, username)
    return stats

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the assistant."""
    try:
        result = assistant.chat(request.message, request.thread_id, request.username, request.model_name)
        return ChatResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=PolicySearchResponse)
async def search_policies(request: PolicySearchRequest):
    """Search HR policies."""
    results = assistant.search_policies(request.query)
    return PolicySearchResponse(results=results)


# FastAPI Server Runner
def run_fastapi():
    """Run FastAPI server in a separate thread."""
    uvicorn.run(app, host="0.0.0.0", port=FASTAPI_PORT)


# Streamlit App
def main():
    st.set_page_config(
        page_title="HR Policy Assistant",
        page_icon="👥",
        layout="wide"
    )
    
    # User Authentication Section
    if 'current_user' not in st.session_state:
        st.title("👥 HR Policy Assistant - User Login")
        st.markdown("Please enter your username to access your chat sessions.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            username = st.text_input("Username:", placeholder="Enter your username")
            display_name = st.text_input("Display Name (optional):", placeholder="Enter your display name")
        
        with col2:
            st.markdown("### Recent Users")
            # Show recent users if any exist
            if 'assistant' in st.session_state:
                try:
                    users = st.session_state.assistant.get_all_users()
                    if users:
                        for user in users[:5]:  # Show last 5 users
                            if st.button(f"👤 {user[1]} ({user[0]})", key=f"user_{user[0]}"):
                                st.session_state.current_user = user[0]
                                st.session_state.current_display_name = user[1]
                                st.rerun()
                    else:
                        st.info("No previous users found.")
                except Exception:
                    st.info("Users list not available yet.")
        
        if st.button("Login / Create User") and username:
            # Initialize the assistant if not done
            if 'assistant' not in st.session_state:
                try:
                    with st.spinner("Initializing HR Policy Assistant..."):
                        st.session_state.assistant = HRPolicyAssistant()
                    st.success("HR Policy Assistant initialized successfully!")
                except Exception as e:
                    st.error(f"Failed to initialize HR Policy Assistant: {e}")
                    st.error("Please check your configuration and try again.")
                    st.stop()
            
            # Create or get user
            try:
                st.session_state.assistant.create_or_get_user(username, display_name or username)
                st.session_state.current_user = username
                st.session_state.current_display_name = display_name or username
                st.success(f"Welcome, {display_name or username}!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to create/login user: {e}")
        
        return
    
    # Initialize the assistant if not done (for existing users)
    if 'assistant' not in st.session_state:
        try:
            with st.spinner("Initializing HR Policy Assistant..."):
                st.session_state.assistant = HRPolicyAssistant()
            st.success("HR Policy Assistant initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize HR Policy Assistant: {e}")
            st.error("Please check your configuration and try again.")
            st.stop()
    
    # Start FastAPI server in background
    if 'fastapi_started' not in st.session_state:
        fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
        fastapi_thread.start()
        st.session_state.fastapi_started = True
        
    
    # Main App Interface
    current_user = st.session_state.current_user
    current_display_name = st.session_state.current_display_name
    
    # Header with user info and logout
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title(f"👥 HR Policy Assistant - Welcome {current_display_name}")
    with col2:
        if st.button("🚪 Logout"):
            # Clear session state
            for key in list(st.session_state.keys()):
                if key.startswith(('current_', 'selected_', 'new_thread_mode')):
                    del st.session_state[key]
            st.rerun()
    
    # Sidebar for thread management and model selection
    st.sidebar.title(f"Chat Sessions - {current_display_name}")
    
    # Model selection
    st.sidebar.subheader("🤖 Model Selection")
    selected_model_display = st.sidebar.selectbox(
        "Choose GPT Model:",
        options=list(AVAILABLE_MODELS.keys()),
        index=0
    )
    selected_model = AVAILABLE_MODELS[selected_model_display]
    
    # Store selected model in session state
    st.session_state.selected_model = selected_model
    st.session_state.selected_model_display = selected_model_display
    
    # Create new thread
    if st.sidebar.button("New Chat Session"):
        st.session_state.current_thread = None
        st.session_state.new_thread_mode = True
    
    # Load existing threads for the current user
    threads = st.session_state.assistant.get_thread_titles(current_user)
    
    if threads:
        st.sidebar.subheader("Previous Sessions")
        for thread_id, title, model_name in threads:
            # Show model name in button
            model_display = next((k for k, v in AVAILABLE_MODELS.items() if v == model_name), model_name)
            button_text = f"📁 {title} ({model_display})"
            if st.sidebar.button(button_text, key=f"thread_{thread_id}"):
                st.session_state.current_thread = thread_id
                st.session_state.new_thread_mode = False
    
    # Main content area
    st.markdown("Ask questions about HR policies, get policy information, or generate HR reports.")
    
    # Display current model
    st.info(f"🤖 Current Model: **{selected_model_display}** | 👤 User: **{current_display_name}**")
    
    
    # Handle new thread creation
    if st.session_state.get('new_thread_mode', False):
        st.subheader("Start New Chat Session")
        thread_title = st.text_input("Enter a title for this chat session:")
        
        if st.button("Create Session") and thread_title:
            thread_id = st.session_state.assistant.create_thread(thread_title, current_user, selected_model)
            st.session_state.current_thread = thread_id
            st.session_state.new_thread_mode = False
            st.success(f"Created new session: {thread_title} with model: {selected_model_display}")
            st.rerun()
    
    # Chat interface
    if st.session_state.get('current_thread'):
        thread_id = st.session_state.current_thread
        
        # Display thread statistics
        stats = st.session_state.assistant.get_thread_stats(thread_id, current_user)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Input Tokens", stats['input_tokens'])
        with col2:
            st.metric("Output Tokens", stats['output_tokens'])
        with col3:
            st.metric("Total Tokens", stats['total_tokens'])
        with col4:
            thread_model = stats.get('model_name', 'gpt-4.1-nano')
            thread_model_display = next((k for k, v in AVAILABLE_MODELS.items() if v == thread_model), thread_model)
            st.metric("Thread Model", thread_model_display)
        
        # Display chat history
        messages = st.session_state.assistant.get_thread_messages(thread_id, current_user)
        
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
            with st.spinner(f"Thinking with {selected_model_display}..."):
                result = st.session_state.assistant.chat(user_input, thread_id, current_user, selected_model)
            
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

# Run the Streamlit app
if __name__ == "__main__":
    main()
    