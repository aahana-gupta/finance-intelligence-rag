"""
Ragas evaluation script for finance-intelligence-rag.

Hits the live /ask endpoint, collects (question, answer, contexts) for a small
golden test set, and scores the pipeline on faithfulness, answer relevancy,
and context precision.

Judge model: Groq (Llama 3.1), accessed via langchain_openai's ChatOpenAI client
pointed at Groq's OpenAI-compatible endpoint. Ragas' metrics need an LLM judge;
this project doesn't use OpenAI anywhere else, so Groq is used here too instead
of introducing a second LLM provider.

IMPORTANT VERSION NOTE (found via testing, not assumption):
Newer ragas releases (0.4.x) have a broken import chain against recent
langchain-community versions (a hard import of a since-removed VertexAI
submodule). Pin these exact versions to avoid it:
    ragas==0.2.15
    langchain-community==0.3.19
    langchain-openai==0.3.9

ALSO (found via testing): ragas.evaluate() defaults to OpenAI embeddings
for answer_relevancy/context_precision, which fails outright since this
project has no OPENAI_API_KEY. We pass the same local all-MiniLM-L6-v2
model embed.py already uses instead (see get_judge_embeddings()).
"""

import os
import requests
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/ask")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_judge_llm():
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file before running eval.py."
        )
    return ChatOpenAI(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        temperature=0,
    )


def get_judge_embeddings():
    # ragas.evaluate() defaults to OpenAI embeddings, which this project has no
    # key for. Reuse the same local model embed.py already uses for retrieval.
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

# --- Golden test set ---
# Replace these with real questions matched to whatever documents you've
# actually uploaded, and their real, verified ground-truth answers.
test_queries = [
    {
        "question": "What was the revenue growth mentioned in the earnings call?",
        "ground_truth": "Revenues grew 0.6% sequentially and 1.7% year-on-year in constant currency terms.",
    },
    {
        "question": "What risks did management highlight?",
        "ground_truth": "Management highlighted tariff-related uncertainty weighing on the Manufacturing vertical (especially Auto), soft and slow discretionary client spending, geopolitical uncertainty affecting the Retail/CPG and Communications sectors, and continued cost pressure in Hi-Tech.",
    },
    {
        "question": "What was the marketing budget for the fiscal year?",
        # Deliberately out-of-scope question — tests that the system correctly
        # refuses rather than hallucinating (per the "only use context" prompt rule).
        "ground_truth": "This information is not available in the provided documents.",
    },
]


def collect_responses(queries):
    questions, answers, contexts, ground_truths = [], [], [], []
    for item in queries:
        response = requests.post(API_URL, json={"question": item["question"]}, timeout=30)
        if response.status_code != 200:
            print(f"Failed to query API for: {item['question']} (status {response.status_code})")
            continue
        data = response.json()
        questions.append(data["question"])
        answers.append(data["raw_answer"] if "raw_answer" in data else data["answer"])
        contexts.append(data["contexts"])
        ground_truths.append(item["ground_truth"])
    return questions, answers, contexts, ground_truths


def main():
    print(f"Gathering responses from {API_URL} ...")
    questions, answers, contexts, ground_truths = collect_responses(test_queries)

    if not questions:
        print("No successful responses collected — check that your API is running and reachable.")
        return

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print("Running Ragas evaluation (this calls the judge LLM once per metric per row)...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=get_judge_llm(),
        embeddings=get_judge_embeddings(),
    )

    print("\n=== RAG EVALUATION RESULTS ===")
    print(result)


if __name__ == "__main__":
    main()
