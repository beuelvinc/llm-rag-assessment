#!/bin/bash


MODEL_1="gemma2:2b" #2b
MODEL_2="moondream" #1.4b
MODEL_3="llama3.2" #3b
MODEL_4="nomic-embed-text"

OLLAMA_CLI="ollama"

echo "Starting to pull models..."

echo "Pulling $MODEL_1..."
$OLLAMA_CLI pull $MODEL_1

echo "Pulling $MODEL_2..."
$OLLAMA_CLI pull $MODEL_2

echo "Pulling $MODEL_3..."
$OLLAMA_CLI pull $MODEL_3

echo "Pulling $MODEL_4..."
$OLLAMA_CLI pull $MODEL_4
echo "Pull completed for all models!"
