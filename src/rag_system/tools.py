import os
import requests
from dotenv import load_dotenv
from urllib.parse import urlparse
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from crewai.tools import tool
from typing import Dict, Union, Any


# Load environment variables to get the database URL
load_dotenv()

def warm_up_ollama(base_url: str, model_name: str):
    """Pre-warm the Ollama model to avoid cold start delays."""
    try:
        response = requests.post(f"{base_url}/api/generate", 
                                json={"model": model_name, "prompt": "test"}, 
                                timeout=5)
        if response.status_code == 200:
            print(f"✅ Ollama model {model_name} warmed up successfully")
    except Exception as e:
        print(f"⚠️ Could not warm up Ollama model {model_name}: {e}")

@tool("Document Retrieval Tool")  
def document_retrieval_tool(query: Union[str, Dict[str, Any]]) -> str:
    """
    Retrieves relevant context from a collection of policy and standards documents.
    Use this tool to search for information in policy documents, manuals, and standards.
    
    Args:
        query: The search query string or dictionary containing the query
        
    Returns:
        A formatted string containing relevant document chunks with source information
    """
    try:
        # Parse the query
        if isinstance(query, dict):
            search_query = query.get("query", str(query))
        else:
            search_query = str(query)
        
        # Get database URL from environment
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return "Error: DATABASE_URL environment variable not set"
        
        print(f"DEBUG: Using database URL: {database_url}")
        
        # Parse database URL
        db_url_parts = urlparse(database_url)
        
        # Use the correct table name from ingestion
        table_name = "document_embeddings"
        
        # Get provider configuration
        llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        
        # Initialize embedding model based on provider
        if llm_provider == "openai" and openai_api_key:
            embed_model = OpenAIEmbedding(
                api_key=openai_api_key,
                base_url=openai_base_url
            )
            embed_dim = 1536  # OpenAI embeddings are 1536-dimensional
        else:
            # Fallback to Ollama
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            warm_up_ollama(ollama_base_url, "nomic-embed-text")
            embed_model = OllamaEmbedding(
                model_name="nomic-embed-text",
                base_url=ollama_base_url,
                request_timeout=120.0
            )
            embed_dim = 768  # Ollama embeddings are 768-dimensional
        
        # Create vector store (disable hybrid search to avoid text_search_tsv column requirement)
        vector_store = PGVectorStore.from_params(
            host=db_url_parts.hostname,
            port=db_url_parts.port,
            database=db_url_parts.path.lstrip('/'),
            user=db_url_parts.username,
            password=db_url_parts.password,
            table_name=table_name,
            embed_dim=embed_dim,
            hybrid_search=False,  # Disable hybrid search to avoid text_search_tsv column
        )

        # Create a LlamaIndex VectorStoreIndex object from the vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )

        # Create a query engine with vector search only (no hybrid search)
        query_engine = index.as_query_engine(
            similarity_top_k=5
        )
        
        # Query using vector search
        response = query_engine.query(search_query)
        retrieved_nodes = response.source_nodes
        
        if not retrieved_nodes:
            return "No relevant documents found for this query."
        
        # Format the retrieved context with source metadata and contextual information
        formatted_chunks = []
        for i, node in enumerate(retrieved_nodes, 1):
            chunk_text = node.get_content()
            source_file = node.metadata.get('source_file', 'Unknown')
            ai_context = node.metadata.get('ai_context', '')
            
            formatted_chunk = f"--- Chunk {i} ---\n"
            formatted_chunk += f"Source: {source_file}\n"
            if ai_context:
                formatted_chunk += f"Context: {ai_context}\n"
            formatted_chunk += f"Content: {chunk_text}\n"
            formatted_chunks.append(formatted_chunk)
        
        return "\n".join(formatted_chunks)
        
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"