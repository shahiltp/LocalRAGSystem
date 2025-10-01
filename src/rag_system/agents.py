import os
from crewai import Agent, LLM
from .tools import document_retrieval_tool

# Initialize the LLM based on environment for easy switching between Ollama and OpenAI
# Defaults to Ollama; set LLM_PROVIDER=openai to switch.
llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()

if llm_provider == "openai":
    # OpenAI configuration (minimal change). Requires OPENAI_API_KEY in env.
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url = os.getenv("OPENAI_BASE_URL")  # optional (e.g., Azure/OpenRouter)
    ollama_llm = LLM(
        model=openai_model,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=openai_base_url,
        temperature=0.5,
        timeout=300,
        verbose=True,
        max_tokens=8192,  # sensible default for OpenAI; override via model settings server-side
    )
else:
    # Ollama configuration (default)
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_llm = LLM(
        model="ollama/mistral:latest",  # Using mistral:7b model for better instruction following
        base_url=ollama_base_url,
        temperature=0.2,
        timeout=500,
        verbose=True,  # Enable verbose logging for debugging
        # Token configuration for mistral:7b
        max_tokens=1024,   # Smaller cap for faster responses and fewer timeouts
        num_ctx=8192,      # Context window size for mistral:7b
    )

# --- AGENT 1: The Specialist Retriever ---
# This agent's only job is to call the retrieval tool correctly.
document_researcher = Agent(
    role='Document Researcher',
    goal='Use the Document Retrieval Tool to find information relevant to a user\'s query from the knowledge base.',
    backstory=(
   "You are an information retrieval specialist. Your role is strictly limited to:, "
   "1) Analyze the user's query to understand intent, "
   "2) Retrieve relevant text chunks using the Document Retrieval Tool, "
   "3) Return only the raw retrieved context - no interpretation or answers. "
   "DO NOT answer questions using your general knowledge. "
   "DO NOT provide explanations, summaries, or interpretations. "
   "ONLY return the exact text chunks retrieved from the tool for the next agent to use."
),
    tools=[document_retrieval_tool],
    llm=ollama_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,  # Limit iterations to prevent infinite loops
)

# --- AGENT 2: The Specialist Synthesizer ---
# This agent's only job is to write the final answer based on the context it receives.
insight_synthesizer = Agent(
    role='Insight Synthesizer',
    goal='Create clear, professional responses that directly answer user questions based on the provided context.',
    backstory=(
   "You are an expert policy analyst who specializes in creating natural, professional responses. "
   "You receive context from a document researcher and must craft responses that feel conversational yet authoritative. "
   
   "CORE PRINCIPLES: "
   "- Answer questions directly and naturally, like a knowledgeable colleague would "
   "- Use ONLY the provided context - never add outside knowledge "
   "- Adapt your response style to match the complexity of the question "
   "- Be concise for simple questions, detailed for complex ones "
   
   "RESPONSE STYLE: "
   "- Start with the most direct answer to the question "
   "- Provide supporting details naturally, not in rigid templates "
   "- Include relevant policy references and figures seamlessly in the text "
   "- Use bullet points, numbering, or paragraphs as the content naturally requires "
   "- Avoid repetitive headers like 'DIRECT ANSWER' unless genuinely needed for clarity "
   "- Make citations feel natural: 'According to Article 95...' rather than 'SOURCE REFERENCE:' "
   "- If the question is simple, keep the answer simple "
   
   "QUALITY CHECKS: "
   "- If context is insufficient, clearly state what information is missing "
   "- Ensure accuracy by staying strictly within the provided context "
   "- Maintain professional tone while being conversational "
),
    llm=ollama_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,  # Limit iterations to prevent infinite loops
    # This agent does not need tools; it only processes text.
    tools=[]
)
