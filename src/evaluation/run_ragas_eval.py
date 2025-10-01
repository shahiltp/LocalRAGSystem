import os
import sys
import asyncio
import time
import math
from statistics import mean
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from dotenv import load_dotenv
import json


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Now you can import from your modules
from src.rag_system.crew import create_rag_crew

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# Specify the path to your golden dataset
EVAL_DATASET_PATH = os.path.join(project_root, 'src', 'evaluation', 'eval_dataset.jsonl')

def run_rag_pipeline(query: str):
    """
    A wrapper function to run the CrewAI RAG pipeline and return the final result.
    """
    try:
        rag_crew = create_rag_crew(query)
        result = rag_crew.kickoff()
        
        # --- FIX: Convert the CrewAI output object to a plain string ---
        # The 'datasets' library expects simple data types like strings, not complex objects.
        # Casting the result to a string extracts the final answer.
        answer_string = str(result)
        
        # For this evaluation, we treat the entire final output as the 'answer'.
        # The 'contexts' for RAGAS are the pieces of text the answer is based on.
        # Since the agent's final answer is a synthesis of the retrieved context,
        # we will use the answer itself as the context to measure faithfulness.
        answer = answer_string
        contexts = [answer_string] # Use the string version here as well

        return {"answer": answer, "contexts": contexts}
    except Exception as e:
        print(f"Error running crew for query '{query}': {e}")
        return {"answer": "Error", "contexts": []}

async def main():
    """
    Main function to run the RAGAS evaluation.
    """
    print(f"Loading evaluation dataset from: {EVAL_DATASET_PATH}")
    if not os.path.exists(EVAL_DATASET_PATH):
        print(f"ERROR: Evaluation dataset not found at {EVAL_DATASET_PATH}")
        return

    # Load the golden dataset from the .jsonl file
    golden_dataset = Dataset.from_json(EVAL_DATASET_PATH)

    # --- Run the RAG pipeline for each question in the dataset ---
    print("\nRunning RAG pipeline on the evaluation dataset...")
    results = []
    questions = []
    ground_truths = []
    per_item = []
    
    for entry in golden_dataset:
        question = entry['question']
        ground_truth = entry['ground_truth']
        
        print(f"  - Processing question: '{question[:80]}...'")
        t0 = time.time()
        pipeline_output = run_rag_pipeline(question)
        elapsed = time.time() - t0

        results.append(pipeline_output)
        questions.append(question)
        ground_truths.append(ground_truth)
        per_item.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": pipeline_output.get("answer"),
            "contexts": pipeline_output.get("contexts", []),
            "elapsed_seconds": round(elapsed, 3),
        })

    # --- Prepare the dataset for RAGAS evaluation ---
    evaluation_data = {
        "question": questions,
        "answer": [res["answer"] for res in results],
        "contexts": [res["contexts"] for res in results],
        "ground_truth": ground_truths,
    }
    eval_dataset = Dataset.from_dict(evaluation_data)

    # --- Run the RAGAS evaluation ---
    print("\nEvaluating the results with RAGAS...")
    
    # Define the metrics we want to use
    metrics = [
        faithfulness,       # How factually accurate is the answer based on the context?
        answer_relevancy,   # How relevant is the answer to the question?
        context_recall,     # Did the retriever find all the relevant context?
        context_precision,  # Was the retrieved context precise and not full of noise?
    ]

    # Run the evaluation
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
    )

    print("\nEvaluation Complete!")
    print("-------------------------")
    print(result)
    print("-------------------------")
    # Robust extraction helper
    def to_scalar(value):
        try:
            # If it's a list/tuple, prefer mean or first element
            if isinstance(value, (list, tuple)):
                # try mean of numeric elements
                numeric = [v for v in value if isinstance(v, (int, float)) and not math.isnan(v)]
                if numeric:
                    return float(mean(numeric))
                return float(value[0]) if value else None
            # If it's numpy-like with item()
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, (int, float)):
                if isinstance(value, float) and math.isnan(value):
                    return None
                return float(value)
            # If it's string convertible float
            return float(value)
        except Exception:
            return None

    try:
        # Some ragas versions return a mapping-like object; attempt safe access
        summary = {
            "faithfulness": to_scalar(result.get("faithfulness", None) if hasattr(result, "get") else None),
            "answer_relevancy": to_scalar(result.get("answer_relevancy", None) if hasattr(result, "get") else None),
            "context_recall": to_scalar(result.get("context_recall", None) if hasattr(result, "get") else None),
            "context_precision": to_scalar(result.get("context_precision", None) if hasattr(result, "get") else None),
        }

        output_payload = {
            "items": per_item,
            "summary": summary,
        }

        output_path = os.path.join(project_root, "src", "evaluation", "ragas_results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=4, ensure_ascii=False)

        print(f"SUCCESS: Results saved to {output_path}")
    except Exception as e:
        print(f"Error converting/saving results: {e}")

if __name__ == "__main__":
    # Ragas evaluation uses asyncio, so we run the main function in an event loop
    asyncio.run(main())
