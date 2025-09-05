# 1) Bối cảnh & yêu cầu phần cứng

- **GLM-4.5-Air** là MoE (106B tổng tham số, \~12B kích hoạt), thiết kế “agent-native”, hỗ trợ reasoning & coding tốt. ([docs.z.ai][1], [Hugging Face][2])
- Bản **GGUF Q4_K_XL** có **khoảng 72.9 GB** (kích thước file trên Hugging Face), vì vậy **RAM hệ thống** nên ≥ 96 GB (tốt nhất 128 GB) để vừa chứa trọng số + overhead + cache/OS khi mmap; VRAM 40 GB sẽ offload được nhiều layer nhưng **không cần** toàn bộ trọng số nằm trên GPU. ([Hugging Face][3])
- GLM-4.5-Air có template hội thoại kiểu ChatGLM. Trang Unsloth ghi chú “**use `--jinja`**” khi chạy với llama.cpp (để server đọc template từ metadata). ([Hugging Face][3])

---

# 2) Nguyên tắc tối ưu cho A100 40GB (MoE, Q4)

**Mục tiêu:** tối ưu **tốc độ/token** và **độ ổn định**, vẫn giữ **context dài vừa phải (≈32K)**.

1. **Offload tối đa lên GPU**
   Dùng `--n-gpu-layers 999` để “đẩy hết mức có thể” vào VRAM; phần còn lại ở RAM. Với MoE, **các tensor “exps” (expert activations)** rất tốn VRAM — thường **đẩy về CPU** bằng `--override-tensor exps=CPU` là trade-off phổ biến, giúp khởi chạy ổn trên 40 GB. ([Medium][4], [Medium][5])

2. **Flash-Attention**
   Bật `--flash-attn` (A100 có Tensor Core mạnh). Theo tài liệu, FA giúp hiệu năng/VRAM tốt hơn so với attention thường. ([GitHub][6])

3. **KV-cache quantization**

   - Mặc định KV là **FP16** → tốn VRAM lớn khi context dài.
   - **`--cache-type-k q8_0 --cache-type-v q8_0`**: thường **giảm \~1/2 VRAM** cho KV so với FP16, ảnh hưởng chất lượng rất nhỏ; nếu cần dư VRAM hơn nữa dùng `q4_0` (đỡ 2/3 VRAM nhưng chất lượng có thể giảm). ([smcleod.net][7])
   - KV-cache tăng **tuyến tính theo context**; tham chiếu bảng Llama-3.1 cho thấy ở FP16, 70B cần \~0.313 GB/1k token; với **q8_0** xấp xỉ **một nửa**; GLM-4.5-Air sẽ khác đôi chút theo kiến trúc, nhưng xu hướng tương tự. ([Hugging Face][8], [GitHub][9])

4. **Batching & u-batch**

   - Với chat realtime, bắt đầu **`-b 2048`** và **`-ub 1024`**.
   - Với throughput (dịch vụ nhiều người dùng + continuous batching), thử **`-b 4096 -ub 2048`** rồi tăng/giảm theo VRAM thực tế (KV-cache là nút thắt).

5. **Context & RoPE**

   - Khởi đầu **`--ctx-size 32768`** (an toàn cho 40 GB VRAM với KV q8_0).
   - Nếu cần > 32K, nâng dần; ưu tiên để **llama.cpp dùng metadata** của model; chỉ bật **`--rope-scaling`** khi bạn phải cưỡng bức vượt train-ctx. (GLM-4.5-Air bản dịch vụ hỗ trợ 128K, nhưng GGUF/quant có thể khác; đo và điều chỉnh). ([together.ai][10])

6. **Chat template**

   - Dùng template kèm model: **`--jinja`** (llama.cpp sẽ lấy từ metadata). Hoặc chỉ rõ **`--chat-template chatglm4`** (template có trong danh sách hỗ trợ). ([Hugging Face][3], [GitHub][11])

7. **Mmap & mlock**

   - **Giữ mặc định mmap** (không thêm `--no-mmap`) để hệ thống nạp theo nhu cầu. Chỉ dùng `--mlock` nếu bạn có **nhiều RAM** và muốn tránh page-out.

---

# 3) Cấu hình khuyến nghị (chuẩn production, 1× A100 40GB)

```bash
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/build/bin/llama-server \
  -hf unsloth/GLM-4.5-Air-GGUF:Q4_K_XL \
  --alias glm45air-q4kxl \
  --host 0.0.0.0 --port 8080 \
  -dev CUDA0 \
  --n-gpu-layers 999 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --ctx-size 32768 \
  --batch-size 2048 --ubatch-size 1024 \
  --threads -1 --parallel 1 \
  --jinja \
  --override-tensor exps=CPU \
  --metrics --slots \
  --chat-template-file templates/glm_4_5.jinja
```

**Giải thích nhanh các flag “đinh”:**

- `-hf …:Q4_K_XL` – nạp đúng biến thể 4-bit K_XL (\~72.9 GB). ([Hugging Face][3])
- `--n-gpu-layers 999` – offload tối đa lên A100.
- `--flash-attn` – tăng tốc attention, tối ưu cho GPU. ([GitHub][6])
- `--cache-type-k/v q8_0` – cắt 1/2 VRAM KV so với FP16; nếu thiếu VRAM, thử `q4_0`. ([smcleod.net][7])
- `--ctx-size 32768` – điểm khởi đầu cân bằng VRAM/tốc độ.
- `--override-tensor exps=CPU` – ép MoE expert tensors chạy CPU để giảm áp lực VRAM; thường giúp ổn định trên 40 GB. ([Medium][4], [Medium][5])
- `--jinja` – dùng chat template đi kèm model (khuyến nghị bởi Unsloth). ([Hugging Face][3])

---

# 4) Biến thể cấu hình theo mục tiêu

### A. Long-context (64K–128K, hy sinh tốc độ)

- Đổi KV: `--cache-type-k q4_0 --cache-type-v q4_0` (hoặc `k=q8_0, v=q4_0`).
- Tăng context: `--ctx-size 65536` rồi 81920/131072 nếu còn VRAM.
- Giảm batch: `-b 1024 -ub 512`.
- (Gợi ý) Giữ `exps=CPU` để nhường VRAM cho KV.
- Lưu ý: KV-cache tăng rất mạnh theo context; số liệu FP16 tham chiếu cho 8B/70B cho thấy mức tiêu thụ **nhảy bậc** khi lên 128K; q8_0 \~ 1/2, q4_0 \~ 1/3. ([Hugging Face][8], [GitHub][9], [smcleod.net][7])

### B. Throughput cao (nhiều phiên song song, continuous batching)

- Bật mặc định `--cont-batching` (đã enabled).
- `-b 4096 -ub 2048`, `--parallel 2` (nếu prompt ngắn).
- Giữ `--ctx-size 32768` để KV không bùng nổ.
- Theo dõi Prometheus qua `--metrics` để điều chỉnh.

### C. Chất lượng lấy ổn định (ít “điên”)

- Sampler gợi ý: `--temp 0.6 --top-p 0.95 --repeat-penalty 1.1`.
- Tránh `--reasoning-format` đặc thù (DeepSeek…) → dùng `--reasoning-format none` hoặc để mặc định **auto** để không “cắt” thought tags lạ. (GLM có hybrid reasoning riêng). ([together.ai][10])

---

# 5) Khuyến nghị vận hành

- **RAM hệ thống:** vì Q4_K_XL \~72.9 GB, hãy đảm bảo **≥ 96–128 GB** RAM để tránh swapping khi mmap + KV trên CPU. ([Hugging Face][3])
- **Giám sát**: bật `--metrics` + scrape Prometheus để theo dõi **tokens/s**, **VRAM**, **failures**.
- **An toàn API**: dùng `--api-key` hoặc `--api-key-file` khi mở cổng public.
- **Template**: nếu `/chat/completions` không áp được template tuỳ biến, xem danh sách template có sẵn (có **chatglm4**) hoặc dùng `/completions` và tự áp prompt. ([GitHub][11])
- **Nâng cao (tuỳ chọn)**: fork **ik_llama.cpp** tối ưu MoE (DeepSeek/GLM), có nhiều tuỳ chọn split/quant nâng cao; dùng khi cần vắt hiệu năng và bạn chấp nhận khác biệt với mainline. ([GitHub][12])

---

# 6) Checklist “đỡ tốn thời gian”

- [ ] Driver + CUDA phù hợp; build llama.cpp có **CUDA=ON**.
- [ ] Máy có **RAM ≥ 96–128 GB**. ([Hugging Face][3])
- [ ] Dùng **`--jinja`** (hoặc `--chat-template chatglm4`). ([Hugging Face][3], [GitHub][11])
- [ ] `--n-gpu-layers 999`, `--flash-attn`. ([GitHub][6])
- [ ] `--cache-type-k/v q8_0` (thiếu VRAM → `q4_0`). ([smcleod.net][7])
- [ ] `--override-tensor exps=CPU` cho MoE. ([Medium][4], [Medium][5])
- [ ] Bắt đầu `--ctx-size 32768`, nâng dần nếu ổn. ([Hugging Face][8])

---

# 7) Ví dụ biến thể lệnh (tham khảo)

**Long-context 64K an toàn VRAM:**

```bash
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/build/bin/llama-server \
  -hf unsloth/GLM-4.5-Air-GGUF:Q4_K_XL \
  --alias glm45air-64k \
  -dev CUDA0 -ngl 999 -fa --jinja \
  --ctx-size 65536 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  -b 1024 -ub 512 \
  -ot exps=CPU --metrics
```

**Throughput nhiều user (CBatch):**

```bash
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/build/bin/llama-server \
  -hf unsloth/GLM-4.5-Air-GGUF:Q4_K_XL \
  --alias glm45air-tput \
  -dev CUDA0 -ngl 999 -fa --jinja \
  --ctx-size 32768 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -b 4096 -ub 2048 --parallel 2 \
  -ot exps=CPU --metrics
```

---

# 8) Gỡ rối nhanh

- **CUDA OOM khi ingest prompt dài** → hạ `--ctx-size`, giảm `-b/ -ub`, chuyển KV sang `q4_0`. ([smcleod.net][7])
- **Token/s thấp bất thường** → kiểm tra có bật **FA**, đảm bảo `--n-gpu-layers` cao; tránh chạy “draft model” cùng **KV q8_0** (một số báo cáo giảm \~16% với speculative decoding + KV-q8). ([GitHub][13])
- **Trả lời “kỳ lạ” khi bật reasoning-format** → để **`--reasoning-format none`** (GLM không dùng định dạng DeepSeek). ([GitHub][11])
- **Sai định dạng hội thoại** → đảm bảo `--jinja` hoặc `--chat-template chatglm4`. ([Hugging Face][3], [GitHub][11])

---

Nếu bạn muốn, mình có thể **xuất ra file `.env`/script** (hoặc docker compose) đúng theo profile bạn chọn (realtime/long-context/throughput) để bạn chạy thẳng.

[1]: https://docs.z.ai/guides/llm/glm-4.5?utm_source=chatgpt.com "GLM-4.5"
[2]: https://huggingface.co/docs/transformers/main/en/model_doc/glm4v_moe?utm_source=chatgpt.com "Glm4vMoe"
[3]: https://huggingface.co/unsloth/GLM-4.5-Air-GGUF "unsloth/GLM-4.5-Air-GGUF · Hugging Face"
[4]: https://medium.com/%40david.sanftenberg/gpu-poor-how-to-configure-offloading-for-the-qwen-3-235b-a22b-moe-model-using-llama-cpp-13dc15287bed?utm_source=chatgpt.com "How to run big MoE models like Qwen-3–235B-A22B in Llama.cpp ..."
[5]: https://mychen76.medium.com/advanced-llama-cpp-6a02707b4bc3?utm_source=chatgpt.com "Llamacpp 128K GGUF, RoPE Scaling, Tensor Override for Local ..."
[6]: https://github.com/gpustack/gguf-parser-go?utm_source=chatgpt.com "gpustack/gguf-parser-go - GitHub"
[7]: https://smcleod.net/2024/12/bringing-k/v-context-quantisation-to-ollama/?utm_source=chatgpt.com "Bringing K/V Context Quantisation to Ollama"
[8]: https://huggingface.co/blog/llama31?utm_source=chatgpt.com "Llama 3.1 - 405B, 70B & 8B with multilinguality and long ..."
[9]: https://github.com/huggingface/blog/issues/2345?utm_source=chatgpt.com "Llama3.1 inference memory requirements · Issue #2345"
[10]: https://www.together.ai/models/glm-4-5-air?utm_source=chatgpt.com "GLM-4.5-Air API"
[11]: https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template?utm_source=chatgpt.com "Templates supported by llama_chat_apply_template"
[12]: https://github.com/ikawrakow/ik_llama.cpp/discussions/258?utm_source=chatgpt.com "Quick-start Guide coming over from llama.cpp and ktransformers! #258"
[13]: https://github.com/ggerganov/llama.cpp/issues/10552?utm_source=chatgpt.com "Misc. bug: [server] Using q8_0 for KV cache reduces ..."
