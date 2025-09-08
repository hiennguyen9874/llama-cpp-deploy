# Llama cpp

## Install

- Download from [github.com/ggml-org/llama.cpp/releases](https://github.com/ggml-org/llama.cpp/releases)

## Models

### GLM-4.5-Air

- https://huggingface.co/unsloth/GLM-4.5-Air-GGUF/discussions/1
- https://huggingface.co/unsloth/GLM-4.5-Air-GGUF/discussions/9
- https://github.com/ggml-org/llama.cpp/pull/15186
- If you want to disable thinking, add /nothink (correct, no underscore) at the end of your prompt.

```
curl http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "glm45air-q4kxl",
  "prompt": "two steps to build a house:"
}'
```

### Seed OSS

- https://huggingface.co/unsloth/Seed-OSS-36B-Instruct-GGUF

- Chat command:

```bash
curl http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "seed-oss-36b-q5kx",
  "prompt": "two steps to build a house:"
}'
```

### Gemma 3

- https://huggingface.co/unsloth/gemma-3-27b-it-GGUF

```bash
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/build/bin/llama-server \
  -hf unsloth/gemma-3-27b-it-GGUF:Q5_K_XL \
  --alias gemma-3-27b-q5kx \
  --host 0.0.0.0 --port 8080 \
  -fa on -ngl 999 --device CUDA0 \
  -c 16384 \
  -b 4096 -ub 1024 \
  -ctk q8_0 -ctv q8_0 \
  --parallel 4 \
  --threads 32 --threads-http -1 \
  --metrics --slots
```

### Qwen3

- https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF
- https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF
- https://huggingface.co/BasedBase/Qwen3-30B-A3B-Thinking-2507-Deepseek-v3.1-Distill

```bash
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/build/bin/llama-server \
  -hf BasedBase/Qwen3-30B-A3B-Thinking-2507-Deepseek-v3.1-Distill:Q5_K_M \
  --alias qwen3-30b-a3b-thinking \
  --host 0.0.0.0 --port 8080 \
  -fa on -ngl 999 --device CUDA0 \
  -c 16384 \
  -b 4096 -ub 1024 \
  -ctk q8_0 -ctv q8_0 \
  --parallel 4 \
  --threads 32 --threads-http -1 \
  --jinja \
  --metrics --slots \
  --temp 0.6 --min-p 0.0 --top-p 0.95 --top-k 20 --presence-penalty 1.0 \
  --api-key llama-cpp-api-key
```

### Qwen3-Coder

- https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
