import sys
import os
import json
import time
import shutil
import numpy as np
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
import logging
import asyncio
import nest_asyncio
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()
nest_asyncio.apply()
logging.basicConfig(level=logging.INFO)

sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
DATA_DIR = "./subdata"
QA_FILE = "./test.json"
WORKING_DIR = "./working_dir"
llm_port = os.getenv("LLM_PORT")

# --- Parse arguments ---
model_name = sys.argv[1]
context_size = int(sys.argv[2])

# --- Cleanup ---
if os.path.exists(WORKING_DIR):
    shutil.rmtree(WORKING_DIR)
    logging.info(f" Deleted working dir: {WORKING_DIR}")

# --- Helpers ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        return super(NumpyEncoder, self).default(obj)

def fuzzy_match_accuracy(predicted_answer, reference_answer):
    return fuzz.ratio(predicted_answer, reference_answer)

def cosine_similarity_score(predicted_answer, reference_answer):
    embeddings = sentence_model.encode([predicted_answer, reference_answer])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
    return similarity[0][0]

def get_llm_accuracy_score(q, r, m):
    return "data from LLM"

async def insert_data_from_folder(rag, folder_path, batch_size=100):
    contents = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename == ".DS_Store":
                continue
            filepath = os.path.join(root, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    contents.append(content)

    logging.info(f" Total documents to insert: {len(contents)}")

    for i in range(0, len(contents), batch_size):
        batch = contents[i:i + batch_size]
        rag.insert(batch)
        logging.info(f" Inserted batch {i // batch_size + 1}/{(len(contents) - 1) // batch_size + 1}")

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name=model_name,
        llm_model_max_async=4,
        llm_model_max_token_size=context_size,
        llm_model_kwargs={
            "host": f"http://host.docker.internal:{llm_port}",
            "options": {"num_ctx": context_size},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts,
                embed_model="nomic-embed-text",
                host=f"http://host.docker.internal:{llm_port}"
            ),
        ),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

async def benchmark():
    with open(QA_FILE, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    rag = await initialize_rag()
    await insert_data_from_folder(rag, DATA_DIR)

    results = []
    for _, item in qa_data.items():
        question = item.get("question", "")
        ref_answer = item.get("answer", "")
        if not question:
            continue

        start = time.time()
        hybrid_result = rag.query(question, param=QueryParam(mode="hybrid"))
        elapsed = time.time() - start

        fuzzy = fuzzy_match_accuracy(hybrid_result, ref_answer)
        semantic = cosine_similarity_score(hybrid_result, ref_answer)
        llm = get_llm_accuracy_score(question, ref_answer, hybrid_result)

        results.append({
            "model_name": model_name,
            "context_size": context_size,
            "question": question,
            "reference_answer": ref_answer,
            "hybrid_answer": hybrid_result,
            "hybrid_time": elapsed,
            "fuzzy_score": fuzzy,
            "semantic_score": semantic,
            "llm_score": llm,
        })

    output_file = os.path.join("/app", f"benchmark_results_{model_name.replace(':', '_')}_{context_size}.json")
    try:
        print(" Writing to", output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        print(" File written:", output_file)
    except Exception as e:
        print(" Failed to write file:", output_file)
        print("Error:", str(e))

    logging.info(f" Results saved to {output_file}")

def main():
    asyncio.run(benchmark())

if __name__ == "__main__":
    main()

