import subprocess

models = ["gemma2:2b", "smollm2:1.7b", "llama3.2"]
context_sizes = [32768, 16384, 4096]
#context_sizes = [32768]


for model in models:
    for ctx_size in context_sizes:
        print(f"\n🚀 Test Running {model} with context size {ctx_size}")
        result = subprocess.run(
            ["python", "test.py", model, str(ctx_size)],
            capture_output=True, text=True,check=True
        )

        print("Output:\n", result.stdout)
        print("Errors:\n", result.stderr)