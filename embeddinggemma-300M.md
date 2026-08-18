### embeddinggemma-300M

```bash
./llama.cpp/llama-server \
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
