from crewai import Crew, Process, Task
from .agents import document_researcher, insight_synthesizer

def create_rag_crew(query: str):
    """
    Creates and configures a two-agent RAG crew to process a query.
    - The Document Researcher finds relevant information.
    - The Insight Synthesizer formulates the final answer based on the retrieved context.
    """

    # Task for the Document Researcher agent
    # This task focuses exclusively on using the tool to find information.
    research_task = Task(
        description=f"Find relevant information in the policy and standards documents for the query: '{query}'.",
        expected_output="A block of text containing chunks of the most relevant document sections and their source file names.",
        agent=document_researcher
    )

    # Task for the Insight Synthesizer agent
    # This task takes the context from the first task and focuses on crafting the answer.
    synthesis_task = Task(
        description=f"Analyze the provided document context from {research_task} and formulate a comprehensive and accurate answer to the user's original question: '{query}'.",
        expected_output="""A professional, well-structured response that directly answers the user's question. Format the response naturally and appropriately based on the content:

Guidelines for response formatting:
- Start with a clear, direct answer to the question
- Provide supporting details, explanations, or calculations only when relevant
- Include specific references to policy articles, sections, or documents when citing sources
- Use natural language flow rather than rigid templates
- Adapt the structure to fit the content (simple answers for simple questions, detailed breakdowns for complex ones)
- Use proper formatting (bullet points, numbering, or paragraphs) as appropriate for the content
- Ensure professional tone and clarity
- Include precise figures, timeframes, and regulatory references where applicable

The response should feel conversational yet authoritative, avoiding repetitive headers unless the content genuinely requires structured breakdown.""",
        agent=insight_synthesizer,
        context=[research_task] # This ensures it uses the output from the research_task
    )

    # Create the crew with a sequential process
    rag_crew = Crew(
        agents=[document_researcher, insight_synthesizer],
        tasks=[research_task, synthesis_task],
        process=Process.sequential, # The tasks will be executed one after the other
        verbose=True
    )

    return rag_crew
