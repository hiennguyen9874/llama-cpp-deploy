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
