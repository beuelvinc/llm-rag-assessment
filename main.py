import os
import json
import asyncio
import nest_asyncio
import logging
import inspect

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

nest_asyncio.apply()

# Adjust paths and working directory as needed
DATA_DIR = "./data"      # This is where your 50-55 folders/files reside
QA_FILE = "./qa.json"    # JSON with questions/answers
OUTPUT_FILE = "./gemma2.json"
WORKING_DIR = "./working_dir"  # RAG working directory

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

# -----------------------------------------------------------------------------------
# 1) Initialize RAG
# -----------------------------------------------------------------------------------
async def initialize_rag():
    """Initialize the LightRAG instance with Ollama backend (modify as needed)."""
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="gemma2:2b",
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        llm_model_kwargs={
            "host": "http://localhost:11434",  # Adjust if different
            "options": {"num_ctx": 32768},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts,
                embed_model="nomic-embed-text",
                host="http://localhost:11434"  # Adjust if different
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


# -----------------------------------------------------------------------------------
# Helper: Print streamed output
# -----------------------------------------------------------------------------------
async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)


# -----------------------------------------------------------------------------------
# 2) Insert data from files into RAG
# -----------------------------------------------------------------------------------
async def insert_data_from_folder(rag, folder_path):
    """
    Recursively traverse `folder_path`, read each file,
    and insert its text content into RAG.
    """
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            # Depending on your files, you may need more robust reading/parsing here
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()
                    if text_content:
                        rag.insert(text_content)
                        logging.info(f"Inserted content from file: {filepath}")
            except Exception as e:
                logging.error(f"Could not read file {filepath}: {e}")


# -----------------------------------------------------------------------------------
# 3) Read QA from qa.json, query RAG, and save results
# -----------------------------------------------------------------------------------
async def process_qa_and_save(rag):
    """
    1. Read questions from `QA_FILE`.
    2. For each question:
        - Use Naive search
        - Use Hybrid search
    3. Store results alongside original "answer" in gemma2.json
    """
    if not os.path.exists(QA_FILE):
        logging.error(f"QA file not found: {QA_FILE}")
        return

    with open(QA_FILE, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    results = []
    for _, item in qa_data.items():
        question = item.get("question", "")
        reference_answer = item.get("answer", "")

        if not question:
            continue

        logging.info(f"Querying RAG with question: {question}")

        # ----------------------------------
        # Naive Search
        # ----------------------------------
        naive_result = rag.query(question, param=QueryParam(mode="naive"))
        logging.debug(f"Naive result: {naive_result}")

        # ----------------------------------
        # Hybrid Search
        # ----------------------------------
        hybrid_result = rag.query(question, param=QueryParam(mode="hybrid"))
        logging.debug(f"Hybrid result: {hybrid_result}")

        # Gather final record
        record = {
            "question": question,
            "answer": reference_answer,  # from qa.json
            "gemma_naive_answer": naive_result,
            "gemma_hybrid_answer": hybrid_result,
        }
        results.append(record)

    # -----------------------------------------------------------------------------------
    # 4) Write results to gemma2.json
    # -----------------------------------------------------------------------------------
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=4)

    logging.info(f"Results written to {OUTPUT_FILE}")


# -----------------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------------
async def run_all():
    # 1) Initialize the RAG instance
    rag = await initialize_rag()

    # 2) Insert data from the data folder
    await insert_data_from_folder(rag, DATA_DIR)

    # 3) Process QA (naive & hybrid) and save to gemma2.json
    await process_qa_and_save(rag)

def main():
    asyncio.run(run_all())

if __name__ == "__main__":
    main()
