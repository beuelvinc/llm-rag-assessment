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
import requests
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
import nest_asyncio
import asyncio
from dotenv import load_dotenv

load_dotenv()

nest_asyncio.apply()
# Constants

CONTEXT_SIZE = 16384/4
DATA_DIR = "./subdata"  # Path to data
QA_FILE = "./qa.json"  # Path to QA file
OUTPUT_FILE = f"./benchmark_results_llama3_{str(CONTEXT_SIZE)}.json"  # Output file for saving results

# Custom encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):  # Check if it's a numpy object
            return obj.item()  # Convert to Python native type
        return super(NumpyEncoder, self).default(obj)



def fuzzy_match_accuracy(predicted_answer, reference_answer):
    return fuzz.ratio(predicted_answer, reference_answer)


def cosine_similarity_score(predicted_answer, reference_answer):
    embeddings = sentence_model.encode([predicted_answer, reference_answer])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
    return similarity[0][0]

def get_llm_accuracy_score(question, real_answer, model_answer):

    return  "data from LLM"
    url = "https://api-inference.huggingface.co/models/google/gemma-3-4b-it"
    headers = {"Authorization": f"Bearer {os.getenv('api_key')}"}

    prompt = (
        f"Given the following question, real answer, and model's answer, "
        f"please calculate the accuracy of the model's answer compared to the real answer. "
        f"Accuracy should be between 0 and 1, where 1 indicates a perfect match, and 0 indicates no match. "
        f"Return only accuracy nothing else.\n\n"
        f"Question: {question}\n"
        f"Real Answer: {real_answer}\n"
        f"Model Answer: {model_answer}\n"
        f"Accuracy: "
    )
    payload = {
        "inputs": prompt
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        result = response.json()
        print(result)
        accuracy_score = result[0].get("generated_text", "").strip()
        print(result,accuracy_score)
        try:
            accuracy_score = float(accuracy_score)
            return max(0.0, min(1.0, accuracy_score))
        except ValueError:
            logging.error(f"Invalid accuracy score returned: {accuracy_score}")
            return 0.0
    else:
        logging.error(f"Hugging Face API error: {response.status_code}")
        return 0.0

async def insert_data_from_folder(rag, folder_path):
    """
    Recursively traverse `folder_path`, read each file,
    and insert its text content into RAG.
    """
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if str(filename) == ".DS_Store":
                continue

            filepath = os.path.join(root, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()
                    if text_content:
                        logging.info(f"inserting content from file: {filepath}")

                        rag.insert(text_content)
                        logging.info(f"Inserted content from file: {filepath}")
            except Exception as e:
                logging.error(f"Could not read file {filepath}: {e}")



# Initialize LightRAG for different models
async def initialize_rag(model_name,context_size=32768):
    """Initialize the LightRAG instance with a specified LLM backend."""
    rag = LightRAG(
        working_dir=f"./working_dir",
        llm_model_func=ollama_model_complete,
        llm_model_name=model_name,
        llm_model_max_async=4,
        llm_model_max_token_size=context_size,
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {"num_ctx": context_size},
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
        llm_score = get_llm_accuracy_score(question, reference_answer, hybrid_result)

        # Record the results
        record = {
            "model_name": model_name,
            "question": question,
            "reference_answer": reference_answer,
            "hybrid_answer": hybrid_result,
            "hybrid_time": hybrid_time,
            "fuzzy_score": fuzzy_score,
            "semantic_score": semantic_score,
            "llm_score": llm_score,

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

    for model in models:
        model_name = model.get("name")
        context_size = model.get("context_size")
        rag = await initialize_rag(model_name,context_size)
        await insert_data_from_folder(rag, DATA_DIR)
        model_results = await benchmark_rag(rag, qa_data, model_name)
        all_results.extend(model_results)

    # Save the results to a JSON file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        # Ensure out_f is a text stream by using the correct file mode and encoding
        json.dump(all_results, out_f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    logging.info(f"Benchmark results written to {OUTPUT_FILE}")


def main():
    # model_1 = "gemma2:2b"  # 2b
    # model_2 = "moondream"  # 1.4b
    model_3 = "llama3.2"  # 3b

    model_data = [
        # {"name":model_1,"context_size":16384},
        # {"name":model_2,"context_size":x},
        {"name":model_3,"context_size":CONTEXT_SIZE},
        ]
    asyncio.run(benchmark_models(model_data))


if __name__ == "__main__":
    main()
