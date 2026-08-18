## DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF

```bash
export LLAMA_CACHE="DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF"
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/llama-server \
  -hf DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF:Q8_0 \
  --alias Qwen3-VL-Embedding-2B \
  --embedding --pooling none --no-warmup \
  --host 0.0.0.0 --port 8001 \
  --api-key llama-cpp-api-key \
  -fa on -ngl 10 --device CUDA0 \
  --ctx-size 32768

curl http://localhost:4000/embeddings \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Follow the white rabbit.",
    "model": "Qwen3-VL-Embedding-2B",
    "encoding_format": "float"
  }'
```

## Qwen3-VL-Reranker-2B

```bash
export LLAMA_CACHE="mradermacher/Qwen3-VL-Reranker-2B-GGUF"
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/llama-server \
  -hf mradermacher/Qwen3-VL-Reranker-2B-GGUF:Q8_0 \
  --reranking \
  --host 0.0.0.0 --port 8082 \
  --api-key llama-cpp-api-key \
  -fa on -ngl 999 --device CUDA0 \
  --ctx-size 32768

curl http://localhost:8082/rerank \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer llama-cpp-api-key" \
  -d '{
    "query": "ping",
    "documents": ["pong"],
    "model": "mradermacher/Qwen3-VL-Reranker-2B-GGUF"
  }'
```
