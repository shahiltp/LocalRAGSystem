#!/usr/bin/env python3
"""
Enhanced PGVector Indexing with Contextual RAG
Implements Anthropic's Contextual Retrieval approach using LlamaIndex and Ollama
"""

import os
import sys
import logging
import json
import copy
from datetime import datetime
from urllib.parse import quote_plus
from typing import List, Dict, Any

from sqlalchemy import create_engine, text
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from llama_index.core.schema import Document, TextNode
import asyncio
import nest_asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'contextual_rag_index_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
MD_DIR = "/Users/kiwitech/Documents/agentic-rag-poc/data/processed/md"

# Database configuration
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "rag_db"
DB_USER = "kiwitech"
DB_PASSWORD = "zakirnagar"

# Construct DATABASE_URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

TABLE_NAME = "doc_md_contextual_20250830"
EMBED_DIM = 768

# Contextual RAG Configuration
CONTEXT_LLM_MODEL = "gemma3:4b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Prompts for contextual retrieval
CONTEXT_PROMPT_TEMPLATE = """You are analyzing a procurement policy document. Your task is to provide context for a specific chunk.

<document>
{WHOLE_DOCUMENT}
</document>

<chunk>
{CHUNK_CONTENT}
</chunk>

Provide a brief context (1-2 sentences) explaining:
1. Which section/topic this chunk relates to
2. How it connects to the overall procurement process
3. Its relationship to other procedures

Respond with only the context, nothing else."""

def check_database_connection():
    """Test database connectivity and check for pgvector extension"""
    try:
        logger.info(f"Testing connection to database: {DB_NAME} at {DB_HOST}:{DB_PORT}")
        
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as conn:
            # Test basic connection
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            logger.info(f"✅ PostgreSQL connected: {version}")
            
            # Check pgvector extension
            result = conn.execute(text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"))
            has_pgvector = result.fetchone()[0]
            
            if has_pgvector:
                logger.info("✅ pgvector extension is installed")
            else:
                logger.error("❌ pgvector extension not found")
                return False
                
            return True
            
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def test_ollama_connection():
    """Test Ollama service connectivity and model availability"""
    try:
        import requests
        
        # Test Ollama service
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if response.status_code != 200:
            logger.error(f"❌ Ollama service not accessible at {OLLAMA_BASE_URL}")
            return False
        
        # Check if required models are available
        models = response.json().get('models', [])
        model_names = [model['name'] for model in models]
        
        required_models = [CONTEXT_LLM_MODEL, "nomic-embed-text:v1.5"]
        missing_models = [model for model in required_models if model not in model_names]
        
        if missing_models:
            logger.error(f"❌ Missing Ollama models: {missing_models}")
            logger.info(f"Available models: {model_names}")
            return False
        
        logger.info(f"✅ Ollama connected with required models: {required_models}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ollama connection failed: {e}")
        return False

def clear_existing_table():
    """Clear existing table data"""
    try:
        logger.info(f"Clearing existing data from table: {TABLE_NAME}")
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as conn:
            # Check if table exists
            result = conn.execute(text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'data_{TABLE_NAME}'
                )
            """))
            table_exists = result.fetchone()[0]
            
            if table_exists:
                # Clear the table
                conn.execute(text(f"DELETE FROM data_{TABLE_NAME}"))
                conn.commit()
                logger.info(f"✅ Cleared table data_{TABLE_NAME}")
            else:
                logger.info(f"Table data_{TABLE_NAME} doesn't exist yet - will be created")
                
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to clear table: {e}")
        return False

def extract_page_number_from_text(text: str, chunk_index: int) -> int:
    """Extract page number from text content or estimate based on chunk position"""
    # Simple heuristic: assume ~2000 characters per page
    # You can enhance this by looking for page markers in markdown
    estimated_page = max(1, (chunk_index * 800) // 2000 + 1)  # 800 chars per chunk, 2000 per page
    return estimated_page

def create_contextual_nodes(nodes: List[TextNode], whole_document: str) -> List[TextNode]:
    """Create contextual nodes using Ollama LLM"""
    logger.info(f"Creating contextual nodes for {len(nodes)} chunks...")
    
    # Initialize LLM for context generation
    context_llm = Ollama(
        model=CONTEXT_LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0
    )
    
    enhanced_nodes = []
    
    for i, node in enumerate(nodes):
        try:
            # Create a deep copy of the node
            enhanced_node = copy.deepcopy(node)
            
            # Generate context using LLM
            context_prompt = CONTEXT_PROMPT_TEMPLATE.format(
                WHOLE_DOCUMENT=whole_document[:8000],  # Limit document size for context
                CHUNK_CONTENT=node.text
            )
            
            # Get context from LLM
            context_response = context_llm.complete(context_prompt)
            context = str(context_response).strip()
            
            # Add context to metadata
            enhanced_node.metadata["context"] = context
            
            # Add page number estimation
            page_num = extract_page_number_from_text(node.text, i)
            enhanced_node.metadata["page_number"] = page_num
            
            enhanced_nodes.append(enhanced_node)
            
            # Log progress every 50 nodes
            if (i + 1) % 50 == 0:
                logger.info(f"Generated context for {i + 1}/{len(nodes)} nodes")
                
        except Exception as e:
            logger.warning(f"Failed to generate context for node {i}: {e}")
            # Fallback: use original node with page number
            fallback_node = copy.deepcopy(node)
            fallback_node.metadata["context"] = f"Part of {node.metadata.get('file_name', 'document')}"
            fallback_node.metadata["page_number"] = extract_page_number_from_text(node.text, i)
            enhanced_nodes.append(fallback_node)
    
    logger.info(f"✅ Created {len(enhanced_nodes)} contextual nodes")
    return enhanced_nodes

def load_documents():
    """Load markdown documents from the directory"""
    try:
        logger.info(f"Loading documents from: {MD_DIR}")
        
        if not os.path.exists(MD_DIR):
            logger.error(f"❌ Directory not found: {MD_DIR}")
            return None
        
        # Load documents
        reader = SimpleDirectoryReader(input_dir=MD_DIR, recursive=True)
        documents = reader.load_data()
        
        logger.info(f"✅ Loaded {len(documents)} documents")
        
        # Log document info
        for doc in documents:
            file_name = doc.metadata.get('file_name', 'Unknown')
            text_length = len(doc.text)
            logger.info(f"Document: {file_name}, Length: {text_length:,} characters")
        
        return documents
        
    except Exception as e:
        logger.error(f"❌ Failed to load documents: {e}")
        return None

def main():
    """Main indexing function with contextual RAG"""
    logger.info("="*80)
    logger.info("Starting Contextual RAG PGVector Indexing Process")
    logger.info("="*80)
    
    # Step 1: Check database connection
    if not check_database_connection():
        logger.error("Aborting: Database connection issues")
        return
    
    # Step 2: Check Ollama
    if not test_ollama_connection():
        logger.error("Aborting: Ollama connection issues")
        return
    
    # Step 3: Clear existing table
    if not clear_existing_table():
        logger.error("Aborting: Failed to clear existing table")
        return
    
    # Step 4: Load documents
    documents = load_documents()
    if not documents:
        logger.error("Aborting: No documents to index")
        return
    
    try:
        # Step 5: Configure embedding model
        logger.info("Configuring embedding model...")
        Settings.embed_model = OllamaEmbedding(
            model_name="nomic-embed-text:v1.5",
            base_url=OLLAMA_BASE_URL,
        )
        
        # Step 6: Create vector store
        logger.info("Creating vector store...")
        vector_store = PGVectorStore.from_params(
            database=DB_NAME,
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            table_name=TABLE_NAME,
            embed_dim=EMBED_DIM,
            hybrid_search=True,
            text_search_config="english",
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40,
                "hnsw_dist_method": "vector_cosine_ops"
            }
        )
        logger.info("✅ Vector store created")
        
        # Step 7: Use async parallel processing for optimal performance
        logger.info("🚀 Setting up async parallel processing for contextual enhancement...")
        
        # Enable async processing
        nest_asyncio.apply()
        
        # Process all documents with optimized async approach
        all_enhanced_nodes = []
        
        async def process_documents_async():
            """Async function to process documents with parallel contextual enhancement"""
            
            # Create ingestion pipeline for standard processing
            pipeline = IngestionPipeline(
                transformations=[
                    MarkdownNodeParser(include_metadata=True),
                    TokenTextSplitter(
                        chunk_size=800,
                        chunk_overlap=200,
                        separator=" "
                    ),
                    Settings.embed_model,  # Embedding generation
                ]
            )
            
            all_nodes = []
            
            for doc_idx, document in enumerate(documents):
                logger.info(f"Processing document {doc_idx + 1}/{len(documents)}: {document.metadata.get('file_name', 'Unknown')}")
                
                # Run async parallel processing for chunking and embedding
                processed_nodes = await pipeline.arun(documents=[document], num_workers=4)
                
                # Apply contextual enhancement to processed nodes
                enhanced_nodes = create_contextual_nodes(processed_nodes, document.text)
                all_nodes.extend(enhanced_nodes)
                
                logger.info(f"✅ Processed document {doc_idx + 1} with {len(enhanced_nodes)} enhanced chunks using async pipeline")
            
            return all_nodes
        
        # Run the async processing
        loop = asyncio.get_event_loop()
        all_enhanced_nodes = loop.run_until_complete(process_documents_async())
        
        # Step 8: Create index and add all enhanced nodes
        logger.info(f"Creating vector store index and adding {len(all_enhanced_nodes)} enhanced nodes...")
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=Settings.embed_model,
        )
        
        # Add nodes to the index in batches
        batch_size = 100
        for i in range(0, len(all_enhanced_nodes), batch_size):
            batch = all_enhanced_nodes[i:i + batch_size]
            index.insert_nodes(batch)
            logger.info(f"Progress: {min(i + batch_size, len(all_enhanced_nodes))}/{len(all_enhanced_nodes)} nodes inserted")
        
        logger.info(f"✅ All {len(all_enhanced_nodes)} enhanced nodes inserted")
        
        # Step 9: Verify indexing
        logger.info("Verifying indexed data...")
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM data_{TABLE_NAME}"))
            total_count = result.fetchone()[0]
            logger.info(f"✅ Total indexed chunks: {total_count}")
            
            # Check contextual metadata
            result = conn.execute(text(f"""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN metadata_->>'context' IS NOT NULL THEN 1 END) as with_context,
                    COUNT(CASE WHEN metadata_->>'page_number' IS NOT NULL THEN 1 END) as with_page_num
                FROM data_{TABLE_NAME}
            """))
            stats = result.fetchone()
            logger.info(f"✅ Chunks with context: {stats[1]}/{stats[0]}")
            logger.info(f"✅ Chunks with page numbers: {stats[2]}/{stats[0]}")
        
        logger.info("="*80)
        logger.info("🎉 CONTEXTUAL RAG INDEXING COMPLETED SUCCESSFULLY!")
        logger.info(f"📊 Enhanced {len(all_enhanced_nodes)} chunks with contextual information")
        logger.info(f"🔍 Table: data_{TABLE_NAME}")
        logger.info(f"🧠 Context LLM: {CONTEXT_LLM_MODEL}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Indexing failed: {e}")
        raise

if __name__ == "__main__":
    main()
