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
