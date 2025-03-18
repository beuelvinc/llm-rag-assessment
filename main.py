import os
import json
import time
import numpy as np
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
import logging
# Initialize Sentence-BERT model
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
import nest_asyncio
import asyncio

# Apply nest_asyncio to allow running within an existing loop
nest_asyncio.apply()
# Constants
DATA_DIR = "./data"  # Path to data
QA_FILE = "./qa.json"  # Path to QA file
OUTPUT_FILE = "./benchmark_results.json"  # Output file for saving results

# Custom encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):  # Check if it's a numpy object
            return obj.item()  # Convert to Python native type
        return super(NumpyEncoder, self).default(obj)



# Helper Functions for Accuracy Measurement
def exact_match_accuracy(predicted_answer, reference_answer):
    return predicted_answer.strip().lower() == reference_answer.strip().lower()


def fuzzy_match_accuracy(predicted_answer, reference_answer):
    return fuzz.ratio(predicted_answer, reference_answer)


def cosine_similarity_score(predicted_answer, reference_answer):
    embeddings = sentence_model.encode([predicted_answer, reference_answer])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
    return similarity[0][0]


# Initialize LightRAG for different models
async def initialize_rag(model_name="gemma2:2b"):
    """Initialize the LightRAG instance with a specified LLM backend."""
    rag = LightRAG(
        working_dir="./working_dir",
        llm_model_func=ollama_model_complete,
        llm_model_name=model_name,
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {"num_ctx": 32768},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts,
                embed_model="nomic-embed-text",
                host="http://localhost:11434"
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


# Function to process and benchmark the queries
async def benchmark_rag(rag, qa_data, model_name):
    results = []
    for _, item in qa_data.items():
        question = item.get("question", "")
        reference_answer = item.get("answer", "")

        if not question:
            continue

        logging.info(f"Querying RAG ({model_name}) with question: {question}")

        # Measure the time taken by Hybrid search
        start_time = time.time()
        hybrid_result = rag.query(question, param=QueryParam(mode="hybrid"))
        hybrid_time = time.time() - start_time

        # Compute fuzzy match and cosine similarity
        fuzzy_score = fuzzy_match_accuracy(hybrid_result, reference_answer)
        semantic_score = cosine_similarity_score(hybrid_result, reference_answer)

        # Record the results
        record = {
            "model_name": model_name,
            "question": question,
            "reference_answer": reference_answer,
            "hybrid_answer": hybrid_result,
            "hybrid_time": hybrid_time,
            "fuzzy_score": fuzzy_score,
            "semantic_score": semantic_score,
        }
        results.append(record)

    return results


# Function to load QA data
def load_qa_data():
    if not os.path.exists(QA_FILE):
        logging.error(f"QA file not found: {QA_FILE}")
        return {}

    with open(QA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


# Benchmarking for multiple models
async def benchmark_models(models):
    qa_data = load_qa_data()
    all_results = []

    for model_name in models:
        rag = await initialize_rag(model_name)
        model_results = await benchmark_rag(rag, qa_data, model_name)
        all_results.extend(model_results)

    # Save the results to a JSON file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        # Ensure out_f is a text stream by using the correct file mode and encoding
        json.dump(all_results, out_f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    logging.info(f"Benchmark results written to {OUTPUT_FILE}")


def main():
    model_1 = "gemma2:2b"  # 2b
    model_2 = "moondream"  # 1.4b
    model_3 = "llama3.2"  # 3b

    models = [model_1,model_2,model_3]  # List of model names
    asyncio.run(benchmark_models(models))


if __name__ == "__main__":
    main()
