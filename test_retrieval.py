#!/usr/bin/env python3
"""
Test script to debug the Document Retrieval Tool
"""

import os
import sys
from dotenv import load_dotenv
from urllib.parse import urlparse

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

def test_retrieval():
    """Test the document retrieval functionality"""
    print("Testing Document Retrieval Tool...")
    
    # Load environment
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        return False
    
    print(f"SUCCESS: Database URL: {database_url}")
    
    # Parse database URL
    db_url_parts = urlparse(database_url)
    
    # Initialize embedding model
    print("Initializing embedding model...")
    try:
        embed_model = OllamaEmbedding(
            model_name="embeddinggemma",
            base_url="http://localhost:11434",
            request_timeout=120.0
        )
        print("SUCCESS: Embedding model initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize embedding model: {e}")
        return False
    
    # Create vector store
    print("Creating vector store...")
    try:
        vector_store = PGVectorStore.from_params(
            host=db_url_parts.hostname,
            port=db_url_parts.port,
            database=db_url_parts.path.lstrip('/'),
            user=db_url_parts.username,
            password=db_url_parts.password,
            table_name="data_data_document_embeddings",
            embed_dim=768,
            hybrid_search=False,
        )
        print("SUCCESS: Vector store created")
    except Exception as e:
        print(f"ERROR: Failed to create vector store: {e}")
        return False
    
    # Create index
    print("Creating index...")
    try:
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )
        print("SUCCESS: Index created")
    except Exception as e:
        print(f"ERROR: Failed to create index: {e}")
        return False
    
    # Test query
    test_queries = [
        "HR Bylaws salary deduction",
        "disciplinary penalty",
        "employee benefits",
        "procurement policy"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        try:
            query_engine = index.as_query_engine(similarity_top_k=3)
            response = query_engine.query(query)
            
            print(f"   Response: {response}")
            print(f"   Source nodes: {len(response.source_nodes)}")
            
            if response.source_nodes:
                for i, node in enumerate(response.source_nodes[:2]):
                    print(f"   Node {i+1}: {node.get_content()[:100]}...")
            else:
                print("   ERROR: No source nodes found")
                
        except Exception as e:
            print(f"   ERROR: Query failed: {e}")
    
    return True

if __name__ == "__main__":
    test_retrieval()
