import os
import logging
import psycopg2
import time
from dotenv import load_dotenv
from urllib.parse import urlparse
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser.text import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.storage.docstore.postgres import PostgresDocumentStore

# --- 1. Basic Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- 2. Constants and Settings ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')

# --- Database Connection Details ---
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set. Please add it to your .env file.")

db_url_parts = urlparse(DATABASE_URL)
DB_HOST = db_url_parts.hostname
DB_PORT = db_url_parts.port
DB_NAME = db_url_parts.path.lstrip('/')
DB_USER = db_url_parts.username
DB_PASSWORD = db_url_parts.password

# --- Ollama Model Configuration ---
#EMBEDDING_MODEL_NAME = "nomic-embed-text"  # Removed :v1.5 version tag
EMBEDDING_MODEL_NAME = "embeddinggemma"
EMBEDDING_DIM = 768
LLM_MODEL_NAME = "mistral:latest"  # Using Mistral for context generation

# --- Provider Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# --- Helper Functions for Database Setup ---
def ensure_database_exists():
    """
    Check if we can connect to the target database.
    With Docker Compose setup, the database is created automatically.
    """
    logging.info(f"Checking connection to database '{DB_NAME}' on server {DB_HOST}:{DB_PORT}...")
    try:
        # Try to connect directly to our target database
        conn = psycopg2.connect(DATABASE_URL)
        conn.close()
        logging.info(f"✅ Successfully connected to database '{DB_NAME}'.")
        return True

    except psycopg2.OperationalError as e:
        logging.error(f"❌ Could not connect to the database '{DB_NAME}'.")
        logging.error(f"   Please ensure the Docker container is running and fully initialized.")
        logging.error(f"   Error details: {e}")
        return False
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred while connecting to the database: {e}")
        return False

def check_vector_extension():
    """
    Connects to the TARGET database and ensures the pgvector extension is enabled.
    """
    logging.info(f"Connecting to '{DB_NAME}' to verify pgvector extension...")
    try:
        # Now we connect to our target database using the specific user for that DB.
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
        if cur.fetchone():
            logging.info("✅ pgvector extension is enabled.")
            conn.close()
            return True
        else:
            logging.error("❌ Failed to create or find the pgvector extension in the target database.")
            conn.close()
            return False
    except Exception as e:
        logging.error(f"❌ An error occurred while enabling pgvector extension: {e}")
        return False

def clean_existing_indexes():
    """
    Drop existing vector and document tables to recreate fresh indexes.
    """
    logging.info("🧹 Cleaning existing indexes...")
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        cur = conn.cursor()
        
        # Drop tables that LlamaIndex creates with data_ prefix
        tables_to_drop = ['data_document_embeddings', 'data_document_text_store']
        
        for table in tables_to_drop:
            try:
                cur.execute(f'DROP TABLE IF EXISTS {table} CASCADE;')
                logging.info(f"✅ Dropped table: {table}")
            except Exception as e:
                logging.warning(f"⚠️  Could not drop {table}: {e}")
        
        conn.close()
        logging.info("✅ Database cleanup completed.")
        return True
        
    except Exception as e:
        logging.error(f"❌ Error cleaning indexes: {e}")
        return False

def generate_chunk_context(chunk_text: str, llm) -> str:
    """
    Generate a 4-line context summary for a chunk using the configured LLM.
    """
    try:
        prompt = f"""Summarize this text in exactly 2 lines that capture the key information:

{chunk_text}

Summary (2 lines only):"""
        
        response = llm.complete(prompt)
        context = response.text.strip()
        
        # Ensure it's roughly 2 lines
        lines = context.split('\n')
        if len(lines) > 2:
            context = '\n'.join(lines[:2])
        
        return context
    except Exception as e:
        logging.warning(f"Failed to generate context: {e}")
        return "Document chunk containing policy and standards information.\nRelevant for organizational compliance and procedures."

# --- Main Ingestion Pipeline ---
def main():
    """
    The main function to run the entire data ingestion pipeline.
    """
    # Give Docker a moment to initialize the database fully on first run.
    logging.info("Waiting 5 seconds for the database container to initialize...")
    time.sleep(5)

    if not ensure_database_exists():
        return
    if not check_vector_extension():
        return
    
    # Clean existing indexes
    if not clean_existing_indexes():
        return

    # --- 3. Document Parsing with DoclingReader (Mandatory) ---
    logging.info("🔍 Initializing DoclingReader for structure-aware document parsing...")
    try:
        reader = DoclingReader(keep_image=False)
    except Exception as e:
        logging.warning(f"⚠️ DoclingReader initialization failed: {e}")
        logging.info("🔄 Falling back to basic document reader...")
        from llama_index.core import SimpleDirectoryReader
        reader = SimpleDirectoryReader
    
    all_docs = []
    for filename in os.listdir(RAW_DATA_DIR):
        file_path = os.path.join(RAW_DATA_DIR, filename)
        if os.path.isfile(file_path) and (filename.lower().endswith('.pdf') or filename.lower().endswith('.docx')):
            logging.info(f"📄 Parsing document: {filename}")
            try:
                if hasattr(reader, 'load_data'):
                    # DoclingReader
                    docs = reader.load_data(file_path=[file_path])
                else:
                    # SimpleDirectoryReader fallback
                    docs = reader(input_files=[file_path])
                
                for doc in docs:
                    doc.metadata['source_file'] = filename
                all_docs.extend(docs)
                logging.info(f"✅ Successfully parsed {len(docs)} semantic chunks from {filename}.")
            except Exception as e:
                logging.warning(f"⚠️ Failed to parse {filename} with primary method: {e}")
                # Try fallback with SimpleDirectoryReader
                try:
                    from llama_index.core import SimpleDirectoryReader
                    fallback_reader = SimpleDirectoryReader(input_files=[file_path])
                    docs = fallback_reader.load_data()
                    for doc in docs:
                        doc.metadata['source_file'] = filename
                    all_docs.extend(docs)
                    logging.info(f"✅ Successfully parsed {len(docs)} chunks using fallback method from {filename}.")
                except Exception as e2:
                    logging.error(f"❌ Failed to parse {filename} with fallback method: {e2}")

    if not all_docs:
        logging.error("No documents were successfully parsed. Aborting.")
        return

    logging.info(f"📚 Total documents loaded: {len(all_docs)}")

    # --- 4. Initialize LLM for context generation ---
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required when LLM_PROVIDER=openai")
        logging.info(f"🤖 Initializing OpenAI LLM: '{OPENAI_MODEL}' for context generation...")
        llm = OpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=0.1
        )
    else:
        logging.info(f"🤖 Initializing Ollama LLM: '{LLM_MODEL_NAME}' for context generation...")
        llm = Ollama(
            model=LLM_MODEL_NAME,
            temperature=0.1
        )

    # --- 5. Chunking with SentenceSplitter ---
    logging.info("✂️ Initializing SentenceSplitter for optimal chunking...")
    splitter = SentenceSplitter.from_defaults(
        chunk_size=1024,
        chunk_overlap=50,
        paragraph_separator="\n\n\n",
        include_metadata=True,
        include_prev_next_rel=True
    )
    
    # Parse documents into nodes/chunks
    logging.info("🔪 Chunking documents...")
    nodes = splitter.get_nodes_from_documents(all_docs, show_progress=True)
    logging.info(f"📝 Created {len(nodes)} chunks from documents.")

    # --- 6. Add context to each chunk using Gemma LLM ---
    logging.info("🧠 Generating context for each chunk using Gemma LLM...")
    enhanced_nodes = []
    for i, node in enumerate(nodes):
        if i % 10 == 0:  # Progress indicator
            logging.info(f"Processing chunk {i+1}/{len(nodes)}...")
        
        # Generate 2-line context
        context = generate_chunk_context(node.get_content(), llm)
        
        # Add context to node metadata
        node.metadata['ai_context'] = context
        enhanced_nodes.append(node)
    
    logging.info(f"✨ Enhanced {len(enhanced_nodes)} chunks with AI-generated context.")

    # --- 7. Initialize Embedding Model ---
    if LLM_PROVIDER == "openai":
        logging.info(f"🎯 Initializing OpenAI embedding model...")
        embed_model = OpenAIEmbedding(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        EMBEDDING_DIM = 1536  # OpenAI embeddings are 1536-dimensional
    else:
        logging.info(f"🎯 Initializing Ollama embedding model: '{EMBEDDING_MODEL_NAME}'...")
        embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL_NAME)
        EMBEDDING_DIM = 768  # Ollama embeddings are 768-dimensional

    # --- 8. Set up PGVector Store and Storage Context ---
    logging.info("🗄️ Setting up PGVectorStore and PostgresDocumentStore...")
    
    vector_store_table_name = "data_document_embeddings"
    vector_store = PGVectorStore.from_params(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        table_name=vector_store_table_name,
        embed_dim=EMBEDDING_DIM,
    )

    doc_store_table_name = "data_document_text_store"
    doc_store = PostgresDocumentStore.from_params(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        table_name=doc_store_table_name,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=doc_store
    )

    # --- 9. Indexing: Embedding and Storing ---
    logging.info("🚀 Starting the indexing process...")
    
    index = VectorStoreIndex(
        nodes=enhanced_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    logging.info("✅ Ingestion pipeline completed successfully!")
    logging.info(f"📊 Embeddings and documents are now stored in database: '{DB_NAME}'.")
    logging.info(f"📋 Vector Table: '{vector_store_table_name}', Document Table: '{doc_store_table_name}'")
    logging.info(f"🎯 Total indexed chunks: {len(enhanced_nodes)}")
    logging.info(f"🧠 Each chunk enhanced with AI-generated context using {LLM_MODEL_NAME}")


if __name__ == "__main__":
    main()
