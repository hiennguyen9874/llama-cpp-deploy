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
