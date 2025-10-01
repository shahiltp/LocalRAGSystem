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

# Import database checker for validation
from src.utils.database_checker import DatabaseChecker, quick_database_check


# Load environment variables to get the database URL
load_dotenv()

def _extract_query_text(query):
    """Coerce agent payload (str or nested dict) into a plain query string."""
    if isinstance(query, str):
        return query.strip()
    if isinstance(query, dict):
        # Common shapes we saw from the agent:
        # {"query": "text"} OR {"query": {"description": "text"}} OR {"description": "text"}
        candidates = []
        q = query.get("query")
        if isinstance(q, str):
            candidates.append(q)
        elif isinstance(q, dict):
            for k in ("description", "text", "q", "prompt", "content"):
                if k in q and q[k]:
                    candidates.append(str(q[k]))
        for k in ("description", "text", "q", "prompt", "content"):
            if k in query and query[k]:
                candidates.append(str(query[k]))
        if candidates:
            return next((c.strip() for c in candidates if str(c).strip()), str(query))
        return str(query)
    return str(query or "").strip()

def _resolve_table_name(preferred=None):
    """Pick a table that actually exists / works without re-ingest."""
    # 1) explicit env override
    if preferred:
        return preferred
    env_name = os.getenv("RAG_TABLE_NAME")
    if env_name:
        return env_name
    # 2) common names we’ve seen in your runs
    return "data_document_embeddings"  # first try; fallback in try/except below


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
        # Parse the query (handle nested dict payloads from agents)
        search_query = _extract_query_text(query)
        if not search_query:
            return "Error: Empty query provided to Document Retrieval Tool"
        
        # Get database URL from environment
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return "Error: DATABASE_URL environment variable not set"
        
        print(f"DEBUG: Using database URL: {database_url}")
        
        # Parse database URL
        db_url_parts = urlparse(database_url)
        
        # Use the correct table name from ingestion - match your actual table
        table_name = "data_data_document_embeddings"
        
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
            warm_up_ollama(ollama_base_url, "embeddinggemma")
            embed_model = OllamaEmbedding(
                model_name="embeddinggemma",
                base_url=ollama_base_url,
                request_timeout=120.0
            )
            embed_dim = 768  # Ollama embeddings are 768-dimensional
        
        # Generate query embedding
        query_embedding = embed_model.get_text_embedding(search_query)
        
        # Use direct SQL query instead of LlamaIndex VectorStoreIndex
        # This bypasses the LlamaIndex PGVectorStore bug
        import psycopg2
        
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        # Convert embedding to PostgreSQL vector format
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # Direct similarity search query
        cur.execute(f"""
            SELECT text, metadata_, 
                   1 - (embedding <=> %s::vector) as similarity
            FROM {table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT 5
        """, (embedding_str, embedding_str))
        
        results = cur.fetchall()
        conn.close()
        
        # Convert results to LlamaIndex format
        retrieved_nodes = []
        for text, metadata, similarity in results:
            # Create a TextNode-like object
            class Node:
                def __init__(self, content, meta, score):
                    self._content = content
                    self._metadata = meta or {}
                    self._score = score
                
                def get_content(self):
                    return self._content
                
                @property
                def metadata(self):
                    return self._metadata
                
                @property
                def score(self):
                    return self._score
            
            node = Node(text, metadata, similarity)
            retrieved_nodes.append(node)
        
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