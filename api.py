import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime
import uvicorn
from dotenv import load_dotenv
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

# --- Arize Phoenix Tracing Setup ---
# Disabled for local development to avoid connection errors
# Uncomment and configure if you want tracing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use host.docker.internal for Docker container to access host services
phoenix_host = os.getenv("PHOENIX_HOST", "host.docker.internal")
phoenix_endpoint = f"http://{phoenix_host}:6006"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_endpoint

try:
    from phoenix.otel import register
    tracer_provider = register(
        project_name="default",
        endpoint=f"{phoenix_endpoint}/v1/traces",
        auto_instrument=True  # This automatically instruments CrewAI and other libraries
    )
    logging.info(f"✅ Arize Phoenix tracing successfully initialized for API server at {phoenix_endpoint}")
except ImportError as e:
    logging.warning(f"⚠️  Phoenix module not found: {e}. Install with: pip install arize-phoenix")
except Exception as e:
    logging.warning(f"⚠️  Could not initialize Arize Phoenix tracing: {e}")

# --- End of Tracing Setup ---

# Ensure the project root is in the Python path
import sys
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import your existing crew creation function
from src.rag_system.crew import create_rag_crew

# Import conversation memory
from src.utils.conversation_memory import get_conversation_memory, start_conversation, add_message, get_conversation_context

# Load environment variables
load_dotenv()

# Initialize the FastAPI app
app = FastAPI(
    title="CrewAI RAG API",
    description="An API server for the agentic RAG pipeline.",
    version="1.0.0",
)

# Add CORS middleware to allow requests from OpenWebUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Define the request model to be compatible with OpenAI's format
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    conversation_id: Optional[str] = None  # Add conversation ID support

@app.get("/v1/models")
def list_models():
    """
    OpenAI-compatible endpoint to list available models.
    This is required for OpenWebUI to discover available models.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "crew-ai-rag",
                "object": "model", 
                "created": 1677652288,
                "owned_by": "crew-ai-rag",
                "permission": [],
                "root": "crew-ai-rag",
                "parent": None,
                "max_tokens": 8192,         # Updated to match gemma3:1b max tokens
                "context_length": 8192       # Updated to match gemma3:1b context length
            }
        ]
    }

@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible endpoint to interact with the CrewAI RAG pipeline with conversation history.
    """
    
    # Extract the last user message as the query
    user_message = next((msg["content"] for msg in reversed(request.messages) if msg["role"] == "user"), None)

    if not user_message:
        return {"error": "No user message found"}

    print(f"Received query for API: {user_message}")

    # Handle conversation history
    conversation_id = request.conversation_id or start_conversation()
    
    # Add user message to conversation history
    add_message(conversation_id, "user", user_message)
    
    # Get conversation context
    conversation_context = get_conversation_context(conversation_id)
    
    # Create enhanced query with conversation context
    enhanced_query = create_contextual_query(user_message, conversation_context)
    
    print(f"Enhanced query with context: {enhanced_query}")

    # Kick off the CrewAI crew with the enhanced query
    rag_crew = create_rag_crew(enhanced_query, conversation_context)
    result = rag_crew.kickoff()
    
    # Add assistant response to conversation history
    add_message(conversation_id, "assistant", str(result))
    
    # Format the response to be compatible with the OpenAI API standard
    response = {
        "id": f"chatcmpl-{conversation_id[:8]}", # Use conversation ID for response ID
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()), # Current timestamp
        "model": request.model,
        "conversation_id": conversation_id,  # Include conversation ID in response
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": str(result), # Ensure the result is a string
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0, # You can implement token counting if needed
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }
    return response


def create_contextual_query(user_message: str, conversation_context: Dict[str, Any]) -> str:
    """
    Create an enhanced query that includes conversation context
    
    Args:
        user_message: The current user message
        conversation_context: Conversation context and history
        
    Returns:
        Enhanced query with context
    """
    if not conversation_context or conversation_context.get('message_count', 0) <= 1:
        # First message in conversation, no context needed
        return user_message
    
    # Build context from recent messages
    recent_messages = conversation_context.get('recent_messages', [])
    context_summary = conversation_context.get('context_summary', '')
    
    if not recent_messages:
        return user_message
    
    # Create context string
    context_parts = []
    for msg in recent_messages[-3:]:  # Last 3 messages
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')[:150]  # Truncate long content
        context_parts.append(f"{role}: {content}")
    
    context_string = "\n".join(context_parts)
    
    # Enhanced query with context
    enhanced_query = f"""Previous conversation context:
{context_string}

Current question: {user_message}

Please answer the current question considering the conversation context above. If the current question references something from the previous conversation, use that context to provide a more complete answer."""
    
    return enhanced_query


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    """
    Get conversation history and context
    """
    try:
        memory = get_conversation_memory()
        context = memory.get_conversation_context(conversation_id)
        history = memory.get_conversation_history(conversation_id)
        
        if not context:
            return {"error": "Conversation not found"}
        
        return {
            "conversation_id": conversation_id,
            "context": context,
            "history": history
        }
    except Exception as e:
        return {"error": str(e)}


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    """
    Delete a conversation
    """
    try:
        memory = get_conversation_memory()
        success = memory.delete_conversation(conversation_id)
        
        if success:
            return {"message": f"Conversation {conversation_id} deleted successfully"}
        else:
            return {"error": "Conversation not found"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/conversations")
def list_conversations():
    """
    List all conversations (for debugging/admin purposes)
    """
    try:
        memory = get_conversation_memory()
        conversations = memory.get_all_conversations()
        
        # Return summary of conversations
        conversation_summaries = []
        for conv_id, conv_data in conversations.items():
            conversation_summaries.append({
                "conversation_id": conv_id,
                "created_at": conv_data.get("created_at"),
                "last_updated": conv_data.get("last_updated"),
                "message_count": len(conv_data.get("messages", [])),
                "recent_message": conv_data.get("messages", [])[-1] if conv_data.get("messages") else None
            })
        
        return {
            "conversations": conversation_summaries,
            "total_count": len(conversation_summaries)
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/conversations")
def create_conversation():
    """
    Create a new conversation
    """
    try:
        conversation_id = start_conversation()
        return {
            "conversation_id": conversation_id,
            "message": "New conversation created"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # This allows you to run the API server directly for testing
    uvicorn.run(app, host="0.0.0.0", port=8001)
