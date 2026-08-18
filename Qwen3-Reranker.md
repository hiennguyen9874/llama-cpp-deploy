### Qwen3-Reranker

```bash
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/build/bin/llama-server \
  -hf prithivMLmods/Qwen3-Reranker-0.6B-seq-cls-GGUF:Q5_K_M \
  --reranking --host 0.0.0.0 \
  --port 8082 \
  --api-key llama-cpp-api-key
```

Current error same as https://huggingface.co/Mungert/Qwen3-Reranker-4B-GGUF/discussions/1
