#!/bin/bash


VENV_NAME=".venv"
MODEL_1="gemma2:2b"        # 2b
MODEL_2="smollm2:1.7b"        # 1.7b
MODEL_3="llama3.2"         # 3b
MODEL_4="nomic-embed-text" # embed model
OLLAMA_CLI="ollama"

# Detect OS (for reference, can be expanded if needed)
detect_os() {
  echo "Detecting OS..."
  unameOut="$(uname -s)"
  case "${unameOut}" in
      Linux*)     OS=Linux;;
      Darwin*)    OS=Mac;;
      CYGWIN*|MINGW*|MSYS*)    OS=Windows;;
      *)          OS="UNKNOWN:${unameOut}"
  esac
  echo "Operating System Detected: $OS"
}

# Create Python virtual environment
create_venv() {
  echo "Creating virtual environment in $VENV_NAME..."

  # Use python3 or fallback to python
  if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
  elif command -v python >/dev/null 2>&1; then
    PYTHON=python
  else
    echo "❌ Python is not installed."
    exit 1
  fi

  $PYTHON -m venv "$VENV_NAME"

  # shellcheck disable=SC2317
  if [ $? -eq 0 ]; then
    echo "✅ Virtual environment created at $VENV_NAME"
  else
    echo "❌ Failed to create virtual environment"
    exit 1
  fi
}

# Activate virtual environment
activate_venv() {
  echo "Activating virtual environment..."

  if [ "$OS" = "Windows" ]; then
    source "$VENV_NAME/Scripts/activate"
  else
    source "$VENV_NAME/bin/activate"
  fi

  echo "✅ Virtual environment activated"
}

# Install requirements
install_requirements() {
  if [ -f "req.txt" ]; then
    echo "Installing packages from req.txt..."
    pip install -r req.txt
  else
    echo "⚠️ req.txt not found, skipping installation"
  fi
}



# Run all functions
configure_venv() {
  detect_os
  create_venv
  activate_venv
  install_requirements

}




ollama_pull_models() {
  echo "🚀 Starting to pull models..."

  echo "⬇️ Pulling $MODEL_1..."
  $OLLAMA_CLI pull $MODEL_1

  echo "⬇️ Pulling $MODEL_2..."
  $OLLAMA_CLI pull $MODEL_2

  echo "⬇️ Pulling $MODEL_3..."
  $OLLAMA_CLI pull $MODEL_3

  echo "⬇️ Pulling $MODEL_4..."
  $OLLAMA_CLI pull $MODEL_4

  echo "✅ Pull completed for all models!"
}


run_all_tests() {
  echo " Starting to test all models..."
  python test_runner.py
  echo " test completed for all models!"
}

run_all() {
  echo " Run to test all models..."
  python runner.py
  echo " Run completed for all models!"
}



main() {
  if [ $# -eq 0 ]; then
    echo "No function specified. Available options: detect_os, create_venv, activate_venv, install_requirements, your_custom_function"
    exit 1
  fi

  for func in "$@"; do
    if declare -f "$func" > /dev/null; then
      echo "➡️ Running function: $func"
      "$func"
    else
      echo "❌ Function '$func' not found."
    fi
  done
}

main "$@"

#./run.sh ollama_pull_models
#./run.sh configure_venv
#./run.sh
