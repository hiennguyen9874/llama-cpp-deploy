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
  --host 0.0.0.0 --port 8000 \
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
  --host 0.0.0.0 --port 8000 \
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

```bash
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/build/bin/llama-server \
  -hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q5_K_XL \
  --alias Qwen3-30B-A3B-Instruct-2507 \
  --host 0.0.0.0 --port 8000 \
  -fa on -ngl 999 --device CUDA0 \
  -c 16384 \
  -b 4096 -ub 1024 \
  -ctk q8_0 -ctv q8_0 \
  --parallel 4 \
  --threads 32 --threads-http -1 \
  --jinja \
  --metrics --slots \
  --temp 0.7 --min-p 0.0 --top-p 0.80 --top-k 20 --presence-penalty 1.0 \
  --api-key llama-cpp-api-key
```

```bash
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/build/bin/llama-server \
  -hf unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF:Q5_K_XL \
  --alias Qwen3-30B-A3B-Thinking-2507 \
  --host 0.0.0.0 --port 8000 \
  -fa on -ngl 999 --device CUDA0 \
  -c 16384 \
  -b 4096 -ub 1024 \
  -ctk q8_0 -ctv q8_0 \
  --parallel 4 \
  --threads 32 --threads-http -1 \
  --jinja \
  --metrics --slots \
  --temp 0.6 --min-p 0.0 --top-p 0.80 --top-k 20 --presence-penalty 1.0 \
  --api-key llama-cpp-api-key
```

### Qwen3-Coder

- https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF

### embeddinggemma-300M

```bash
CUDA_VISIBLE_DEVICES=7 ./llama.cpp/llama-server \
  -hf unsloth/embeddinggemma-300m-GGUF:Q8_0 \
  --alias embeddinggemma-300m \
  --embeddings \
  --host 0.0.0.0 --port 8001 \
  --api-key llama-cpp-api-key \
  --threads -1 \
  -c 4096 \
  -b 512
```

```bash
curl --request POST \
    --url http://localhost:8001/v1/embeddings \
    --header "Content-Type: application/json" \
    --header "Authorization: Bearer llama-cpp-api-key" \
    --data '{"input": "Hello embeddings"}' \
    --silent
```

### Qwen3-Embedding

- https://github.com/ggml-org/llama.cpp/pull/14029
- https://github.com/ggml-org/llama.cpp/issues/14234#issuecomment-2979796913
- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

```bash
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/build/bin/llama-server \
  -hf Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 \
  --alias qwen3-embedding-0.6b \
  --embeddings --pooling last -ub 8192 \
  --host 0.0.0.0 --port 8081 \
  --verbose-prompt \
  --api-key llama-cpp-api-key \
  -fa on -ngl 999 --device CUDA0 \
  --parallel 4 \
  --threads 32 --threads-http -1 \
  --metrics --slots
```

```bash
curl --request POST \
    --url http://localhost:8081/v1/embeddings \
    --header "Content-Type: application/json" \
    --header "Authorization: Bearer llama-cpp-api-key" \
    --data '{"input": "Hello embeddings"}' \
    --silent
```

```
"embedding_model_name": "Qwen/Qwen3-Embedding-4B",
"max_context_tokens": 32768,
"embedding_dimension": 2560,

  self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["embedding_model_name"], padding_side='left')
  self.model = AutoModel.from_pretrained(CONFIG["embedding_model_name"])
  self.model.to(self.device)
  if self.device == "cuda":
      self.model = self.model.half()  # Convert to float16

  self.max_length = CONFIG["max_context_tokens"]

  # Task description from pair embedding generator
  self.task_description = 'Given this project documentation, create a comprehensive embedding that focuses on project purpose and scope of work, technical details and implementation, and domain-specific information'
  instruction_template = f'Instruct: {self.task_description}\nQuery:'
  instruction_tokens = len(self.tokenizer.encode(instruction_template))
  self.effective_max_tokens = self.max_length - instruction_tokens
```

```
"input": ["test<|endoftext|>"],
```

- To ensure embeddings follow instructions during downstream tasks, we concatenate the instruction
  and the query into a single input context, while leaving the document unchanged before processing
  with LLMs. The input format for queries is as follows:

```
{Instruction} {Query}<|endoftext|>
```

### Qwen3-Reranker

```bash
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/build/bin/llama-server \
  -hf prithivMLmods/Qwen3-Reranker-0.6B-seq-cls-GGUF:Q5_K_M \
  --reranking --host 0.0.0.0 \
  --port 8082 \
  --api-key llama-cpp-api-key
```

Current error same as https://huggingface.co/Mungert/Qwen3-Reranker-4B-GGUF/discussions/1

### NVIDIA-Nemotron-Nano-12B-v2

```bash
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/build/bin/llama-server \
  -hf Mungert/NVIDIA-Nemotron-Nano-12B-v2-GGUF:Q5_K_M \
  --alias NVIDIA-Nemotron-Nano-12B-v2 \
  --host 0.0.0.0 --port 8000 \
  -fa on -ngl 999 --device CUDA0 \
  -b 4096 -ub 1024 \
  -ctk q8_0 -ctv q8_0 \
  --metrics --slots \
  --temp 0.6 --top-p 0.9 \
  --ctx-size 16384 \
  --parallel 1 \
  --threads 16 --threads-http -1 \
  --api-key llama-cpp-api-key
```

```bash
curl http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer llama-cpp-api-key" \
  -d '{
  "model": "NVIDIA-Nemotron-Nano-12B-v2",
  "prompt": "two steps to build a house:"
}'
```

### InternVL3_5-30B-A3B

```bash
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/build/bin/llama-server \
  -hf bartowski/OpenGVLab_InternVL3_5-30B-A3B-GGUF:Q5_K_M \
  --alias NVIDIA-InternVL3_5-30B-A3 \
  --host 0.0.0.0 --port 8000 \
  -fa on -ngl 999 --device CUDA0 \
  -b 4096 -ub 1024 \
  -ctk q8_0 -ctv q8_0 \
  --metrics --slots \
  --ctx-size 8192 \
  --parallel 1 \
  --threads 16 --threads-http -1 \
  --api-key llama-cpp-api-key
```

```bash
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer llama-cpp-api-key' \
  -d '{
  "model": "NVIDIA-InternVL3_5-30B-A3",
  "prompt": "two steps to build a house:"
}'
```


### granite-4.0-h-tiny-GGUF

- https://huggingface.co/unsloth/granite-4.0-h-tiny-GGUF

```bash
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/build/bin/llama-server \
  -hf unsloth/granite-4.0-h-tiny-GGUF:Q5_K_XL \
  --alias granite-4.0-h-tiny \
  --host 0.0.0.0 --port 8000 \
  -fa on -ngl 999 --device CUDA0 \
  -c 16384 \
  --parallel 4 \
  --threads 32 --threads-http -1 \
  --jinja \
  --metrics --slots \
  --temp 0.0 --top-p 1.0 --top-k 0 \
  --api-key llama-cpp-api-key
```

```bash
curl http://hydra-jupyterlab-2:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer llama-cpp-api-key" \
  -d '{
  "model": "granite-4.0-h-tiny",
  "prompt": "Step by step to build a house"
}'
```

### Nemotron-3-Nano-30B-A3B
[https://docs.unsloth.ai/models/nemotron-3#run-nemotron-3-nano-30b-a3b](https://docs.unsloth.ai/models/nemotron-3#run-nemotron-3-nano-30b-a3b)

#### Build
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git fetch origin pull/18058/head:MASTER && git checkout MASTER && cd ..
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
```

#### Run
```bash
CUDA_VISIBLE_DEVICES=7 ./llama.cpp/llama-server \
  -hf unsloth/Nemotron-3-Nano-30B-A3B-GGUF:UD-Q4_K_XL \
  --alias Nemotron-3-Nano-30B-A3B \
  --host 0.0.0.0 --port 8000 \
  -fa on -ngl 999 --device CUDA0 \
  -c 16384 \
  --threads -1 \
  --jinja \
  --metrics --slots \
  --api-key llama-cpp-api-key \
  --prio 3 \
  --min_p 0.01 \
  --temp 1.0 --top-p 1.0
```

#### Test
```bash
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer llama-cpp-api-key" \
  -d '{
  "model": "Nemotron-3-Nano-30B-A3B",
  "prompt": "Step by step to build a house"
}'
```