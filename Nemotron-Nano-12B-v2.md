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
