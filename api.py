import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from dotenv import load_dotenv

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
    OpenAI-compatible endpoint to interact with the CrewAI RAG pipeline.
    """
    # Extract the last user message as the query
    user_message = next((msg["content"] for msg in reversed(request.messages) if msg["role"] == "user"), None)

    if not user_message:
        return {"error": "No user message found"}

    print(f"Received query for API: {user_message}")

    # Kick off the CrewAI crew with the user's query
    rag_crew = create_rag_crew(user_message)
    result = rag_crew.kickoff()
    
    # Format the response to be compatible with the OpenAI API standard
    response = {
        "id": "chatcmpl-123", # Dummy ID
        "object": "chat.completion",
        "created": 1677652288, # Dummy timestamp
        "model": request.model,
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

if __name__ == "__main__":
    # This allows you to run the API server directly for testing
    uvicorn.run(app, host="0.0.0.0", port=8000)
