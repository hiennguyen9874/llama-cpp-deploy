## Qwen3.5-35B-A3B

```bash
export LLAMA_CACHE="unsloth/Qwen3.5-35B-A3B-GGUF"
CUDA_VISIBLE_DEVICES=7 ./llama.cpp/llama-server \
    -hf unsloth/Qwen3.5-35B-A3B-GGUF:MXFP4_MOE \
    --alias Qwen3.5-35B-A3B \
    --host 0.0.0.0 --port 8000 \
    -fa on -ngl 999 --device CUDA0 \
    --ctx-size 32768 \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --api-key llama-cpp-api-key
```

## Qwen3.5-27B

```bash
export LLAMA_CACHE="unsloth/Qwen3.5-27B-GGUF"
CUDA_VISIBLE_DEVICES=7 ./llama.cpp/llama-server \
    -hf unsloth/Qwen3.5-27B-GGUF:UD-Q4_K_XL \
    --host 0.0.0.0 --port 8000 \
    --alias Qwen3.5-27B \
    -fa on -ngl 999 --device CUDA0 \
    --ctx-size 32768 \
    --temp 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --min-p 0.00 \
    --chat-template-kwargs '{"enable_thinking":false}' \
    --api-key llama-cpp-api-key
```

## Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF

```bash
export LLAMA_CACHE="Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF"
./llama.cpp/llama-cli \
    -hf Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF:Q4_K_M \
    --ctx-size 32768 \
    --temp 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --min-p 0.00 \
    --chat-template-kwargs "{\"enable_thinking\": false}"
```
