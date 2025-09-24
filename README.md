# Agentic RAG local - Intelligent Document Q&A System

<div align="center">

[![CrewAI](https://img.shields.io/badge/CrewAI-blue?style=for-the-badge&logo=docker&logoColor=white)](https://crewai.com)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-green?style=for-the-badge&logo=python&logoColor=white)](https://llamaindex.ai)
[![OpenAI](https://img.shields.io/badge/OpenAI-green?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![Ollama](https://img.shields.io/badge/Ollama-purple?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.ai)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-blue?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)

**An intelligent agentic RAG system with multi-agent workflows, powered by CrewAI and flexible LLM support (OpenAI/Ollama)**

</div>

---

## Features

### **AI-Powered Intelligence**
- **Multi-Agent Workflow** - Specialized agents for research & synthesis
- **Flexible LLM Support** - OpenAI (cloud) or Ollama (local) models
- **Contextual Responses** - AI-generated context for better understanding
- **Smart Document Processing** - PDF/DOCX parsing with DoclingReader

### **Performance & Scale**
- **Vector Search** - PostgreSQL + pgvector for lightning-fast retrieval
- **Async Processing** - Concurrent document processing
- **Optimized Retrieval** - Vector similarity search
- **Efficient Storage** - Compressed embeddings and metadata

### **User Experience**
- **OpenWebUI Integration** - Beautiful chat interface
- **OpenAI-Compatible API** - Standard REST endpoints
- **Source Attribution** - Transparent document references
- **Rich Formatting** - Professional response presentation

---

## Architecture

```mermaid
graph TB
    subgraph "Data Ingestion Pipeline"
        RAW["Raw Documents<br/>PDFs, Word, etc."]
        DOC["Docling Converter<br/>PDF → JSON/Markdown"]
        CTX["Contextual RAG<br/>LLM + Context Generation"]
        CHUNK["Smart Chunking<br/>Markdown Parser + Token Splitter"]
    end
    
    subgraph "User Interface"
        UI["OpenWebUI<br/>Web Chat Interface"]
        API["FastAPI Server<br/>OpenAI Compatible"]
    end
    
    subgraph "Agent Layer"
        CR["CrewAI Orchestrator"]
        A1["Document Researcher<br/>Retrieval Agent"]
        A2["Insight Synthesizer<br/>Response Agent"]
    end
    
    subgraph "Retrieval Layer"
        VDB["PostgreSQL + pgvector<br/>Vector Database<br/>1536-dim (OpenAI) or 768-dim (Ollama)"]
        LLM["Flexible LLM<br/>OpenAI gpt-4o-mini<br/>or Ollama gemma3:1b"]
        EMB["Flexible Embeddings<br/>OpenAI text-embedding-3-small<br/>or Ollama nomic-embed-text"]
    end
    
    subgraph "Evaluation & Monitoring"
        RAGAS["RAGAS Evaluation<br/>Faithfulness, Relevancy<br/>Context Precision/Recall"]
        TRACE["CrewAI Tracing<br/>Execution Monitoring"]
    end
    
    %% Data Flow
    RAW --> DOC
    DOC --> CTX
    CTX --> CHUNK
    CHUNK --> EMB
    EMB --> VDB
    
    %% Query Flow
    UI --> API
    API --> CR
    CR --> A1 & A2
    A1 --> VDB
    A2 --> LLM
    VDB --> LLM
    
    %% Evaluation Flow
    CR --> RAGAS
    VDB -.-> RAGAS
    
    %% Monitoring
    API -.-> TRACE
    CR -.-> TRACE
    
    style RAW fill:#ffebee
    style DOC fill:#f3e5f5
    style CTX fill:#e8f5e8
    style CHUNK fill:#fff3e0
    style UI fill:#e1f5fe
    style API fill:#f3e5f5
    style CR fill:#fff3e0
    style VDB fill:#e8f5e8
    style RAGAS fill:#e3f2fd
    style EMB fill:#f1f8e9
```

---

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.8+
- OpenAI API Key (for OpenAI) OR Ollama installed (for local)
- 8GB+ RAM recommended

### Installation & Setup

#### **Clone & Setup Environment**
```bash
git clone <repository-url>
cd agentic-rag-local
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

#### **Configure Environment**
Create `.env` file:

**Option 1: OpenAI (Cloud)**
```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_db
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

**Option 2: Ollama (Local)**
```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_db
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
```

#### **Start Database**
```bash
docker-compose up -d postgres
```

#### **Ingest Documents**
```bash
# Place your PDFs/DOCX files in data/raw/
# For Ollama: Ensure models are pulled first
ollama pull gemma3:1b
ollama pull nomic-embed-text

# Run ingestion
python src/data_ingestion/ingest.py
```

#### **Start API Server**
```bash
python -c "import uvicorn; uvicorn.run('api:app', host='0.0.0.0', port=8001)"
```

#### **Start Web Interface (Optional)**
```bash
docker-compose up -d open-webui
# Access at http://localhost:3000
```

---

## Usage

### **Via Web Interface**
1. Open `http://localhost:3000`
2. Configure OpenAI API connection: `http://host.docker.internal:8001/v1`
3. Select "crew-ai-rag" model
4. Ask questions about your documents

### **Via API**
```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "crew-ai-rag",
    "messages": [
      {"role": "user", "content": "What are the procurement standards?"}
    ]
  }'
```

### **Via Python**
```python
from src.rag_system.crew import create_rag_crew

crew = create_rag_crew('What are the key requirements?')
result = crew.kickoff()
print(result)
```

---

## Switching Between LLM Providers

### **Using OpenAI (Fast - Requires API Key)**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1 #optional because agent already know fron name openai
```

### **Using Ollama (Local - Free)**
```bash
# Install Ollama
ollama pull gemma3:1b
ollama pull nomic-embed-text

# Set environment
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434

# Re-ingest documents (required due to different embedding dimensions)
python -c "from src.data_ingestion.ingest import clean_existing_indexes; clean_existing_indexes()"
python src/data_ingestion/ingest.py
```

**Advantages:**
- Completely free
- Runs locally (privacy)
- No internet required
- Customizable models

---

## Evaluation with RAGAS

### **When to Run RAGAS**
- **After initial setup** - Verify system works correctly
- **After model changes** - Compare OpenAI vs Ollama performance
- **After document updates** - Ensure new content is properly indexed
- **Before production** - Quality assurance and benchmarking
- **Performance optimization** - Identify bottlenecks and improvements

### **Running RAGAS Evaluation**
```bash
# Ensure your RAG system is running first
python src/evaluation/run_ragas_eval.py
```

### **Metrics Evaluated**
- **Faithfulness**: How factually accurate are responses?
- **Answer Relevancy**: Does the answer address the question?
- **Context Recall**: Did retrieval find all relevant information?
- **Context Precision**: How precise is the retrieved context?

---

## Follow-up Questions Feature

### **What Creates Follow-up Questions**
The system **does not automatically generate follow-up questions**. However, you can:

#### **Agent-Based Follow-ups**
- The **Insight Synthesizer** agent can suggest related topics
- Responses include contextual information that naturally leads to follow-ups
- Source attribution helps identify related documents

#### **Manual Follow-ups**
- Ask related questions based on the response
- Use source information to explore specific documents
- Chain questions for deeper exploration

#### **Custom Implementation**
To add automatic follow-up questions, you could:
```python
# In agents.py - modify the Insight Synthesizer
def generate_follow_up_questions(context, response):
    """Generate relevant follow-up questions"""
    prompt = f"""
    Based on this response: {response}
    And context: {context}
    
    Suggest 2-3 follow-up questions that would help the user explore this topic further.
    """
    return llm.complete(prompt)
```

---

## Configuration

### **Environment Variables**
| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `LLM_PROVIDER` | `openai` or `ollama` | `ollama` |
| `OPENAI_API_KEY` | OpenAI API key | Required for OpenAI |
| `OPENAI_MODEL` | OpenAI model name | `gpt-4o-mini` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |

### **Project Structure**
```
Local-agentic-rag/
├── data/
│   ├── raw/                 # Input documents (PDF/DOCX)
│   └── processed/           # Processed data
├── src/
│   ├── config/             # Configuration settings
│   ├── data_ingestion/     # Document parsing and ingestion
│   ├── evaluation/         # RAGAS evaluation scripts
│   └── rag_system/         # Core RAG implementation
│       ├── agents.py       # CrewAI agent definitions
│       ├── crew.py         # Workflow orchestration
│       └── tools.py        # Document retrieval tool
├── api.py                  # FastAPI server
├── docker-compose.yml      # PostgreSQL setup
└── requirements.txt        # Python dependencies
```

---

## Daily Startup Checklist

### **After PC Restart**

**For OpenAI:**
```bash
# 1. Start database
docker-compose up -d postgres

# 2. Activate environment
.\venv\Scripts\activate

# 3. Set environment variables
$env:DATABASE_URL="postgresql://postgres:password@localhost:5432/rag_db"
$env:LLM_PROVIDER="openai"
$env:OPENAI_API_KEY="sk-your-key"

# 4. Start API
python -c "import uvicorn; uvicorn.run('api:app', host='0.0.0.0', port=8001)"

# 5. Start web UI (optional)
docker-compose up -d open-webui
```

**For Ollama:**
```bash
# 1. Start Ollama
ollama serve

# 2. Start database
docker-compose up -d postgres

# 3. Activate environment
.\venv\Scripts\activate

# 4. Set environment variables
$env:DATABASE_URL="postgresql://postgres:password@localhost:5432/rag_db"
$env:LLM_PROVIDER="ollama"
$env:OLLAMA_BASE_URL="http://localhost:11434"

# 5. Start API
python -c "import uvicorn; uvicorn.run('api:app', host='0.0.0.0', port=8001)"

# 6. Start web UI (optional)
docker-compose up -d open-webui
```

---

## Troubleshooting

### **Common Issues**

#### **Port 8000/8001 Already in Use**
```bash
# Check what's using the port
netstat -ano | findstr :8001
# Kill the process or use different port
```

#### **Database Connection Failed**
```bash
# Check if PostgreSQL is running
docker ps | grep postgres
# Restart if needed
docker-compose restart postgres
```

#### **Models Not Showing in OpenWebUI**
- Verify API is accessible: `curl http://localhost:8001/v1/models`
- Check OpenWebUI connection settings
- Restart OpenWebUI: `docker-compose restart open-webui`
- For Ollama: Ensure Ollama is running and models are pulled

#### **PDF Parsing Errors**
- Update transformers: `pip install --upgrade transformers`
- Use simpler PDFs or convert to DOCX format

#### **Ollama Connection Issues**
- Check if Ollama is running: `ollama list`
- Verify models are pulled: `ollama pull gemma3:1b`
- Check Ollama base URL in environment variables

---

## Monitoring & Debugging

### **Real-Time Logs**
```bash
# API logs
python -c "import uvicorn; uvicorn.run('api:app', host='0.0.0.0', port=8001)"

# Container logs
docker logs rag_postgres
docker logs open-webui
```

### **Performance Monitoring**
- **CrewAI Traces**: View execution flow at provided URLs
- **RAGAS Evaluation**: Run periodic quality assessments
- **API Metrics**: Monitor response times and token usage

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---


## Acknowledgments

- [CrewAI](https://crewai.com) - Multi-agent orchestration
- [LlamaIndex](https://llamaindex.ai) - Document processing & retrieval
- [OpenAI](https://openai.com) - Cloud LLM and embeddings
- [Ollama](https://ollama.ai) - Local LLM and embeddings
- [PostgreSQL](https://postgresql.org) + [pgvector](https://github.com/pgvector/pgvector) - Vector database
- [OpenWebUI](https://openwebui.com) - Beautiful chat interface

---

<div align="center">

**Star this repo if it helped you!**

[Report Bug](https://github.com/your-repo/issues) • [Request Feature](https://github.com/your-repo/issues) • [Discussions](https://github.com/your-repo/discussions)

</div>
