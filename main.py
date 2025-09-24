import os
import sys
import logging
from phoenix.otel import register

# --- Arize Phoenix Tracing Setup ---
# This block configures the tracer to send data to your local Phoenix instance.
# It should be at the very top of your application's entry point.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"

try:
    tracer_provider = register(
      project_name="default",
      endpoint="http://localhost:6006/v1/traces",
      auto_instrument=True # This automatically instruments CrewAI and other libraries
    )
    logging.info("✅ Arize Phoenix tracing successfully initialized.")
except Exception as e:
    logging.warning(f"⚠️  Could not initialize Arize Phoenix tracing: {e}")
# --- End of Tracing Setup ---





import sys
from src.rag_system.crew import create_rag_crew

def main():
    """
    Main function to run the Agentic RAG pipeline.
    Accepts a query from the command line or prompts the user.
    """
    # Check if a query was passed as a command-line argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # Prompt the user for a query if none was provided
        query = input("Please enter your question about the documents: ")

    if not query:
        print("No query provided. Exiting.")
        return

    print(f"\n🚀 Kicking off the RAG Crew for your query: '{query}'")
    print("--------------------------------------------------")

    # Create and run the RAG crew
    rag_crew = create_rag_crew(query)
    result = rag_crew.kickoff()

    print("\n--------------------------------------------------")
    print("✅ Crew execution finished. Here is the final answer:")
    print(result)

if __name__ == "__main__":
    main()
