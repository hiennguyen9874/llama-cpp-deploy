```bash
./llama.cpp/llama-server \
  --hf-repo prithivMLmods/chandra-ocr-2-GGUF \
  --hf-file chandra-ocr-2.Q5_K_M.gguf \
  --mmproj-url https://huggingface.co/prithivMLmods/chandra-ocr-2-GGUF/resolve/main/chandra-ocr-2.mmproj-f16.gguf \
  --alias chandra-ocr-2 \
  --ctx-size 32768 \
  --host 0.0.0.0 --port 8080 \
  --batch-size 2048 --ubatch-size 512 \
  --reasoning off \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --n-gpu-layers all \
  --flash-attn on \
  --parallel 1 \
  --temp 0
```

```bash
./llama.cpp/llama-cli \
  --hf-repo prithivMLmods/chandra-ocr-2-GGUF \
  --hf-file chandra-ocr-2.Q5_K_M.gguf \
  --mmproj-url https://huggingface.co/prithivMLmods/chandra-ocr-2-GGUF/resolve/main/chandra-ocr-2.mmproj-f16.gguf \
  --image scripts/test.png \
  -p "document parsing." \
  -n 8192 \
  --temp 0
```