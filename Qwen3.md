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
