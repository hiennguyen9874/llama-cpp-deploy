# Hướng dẫn chạy Gemma 4 QAT bằng `llama.cpp`

Tài liệu này tổng hợp từ:

- Model card `unsloth/gemma-4-31B-it-qat-GGUF` (Hugging Face)
- Unsloth docs: [Gemma 4](https://unsloth.ai/docs/models/gemma-4), [Gemma 4 QAT](https://unsloth.ai/docs/models/gemma-4/qat), [MTP](https://unsloth.ai/docs/models/mtp)
- Bản trợ giúp `llama-cpp.md` của binary `llama.cpp` đang dùng trong thư mục này (dùng để kiểm tra tên cờ có tồn tại, ví dụ `--spec-type draft-mtp`).

Các con số tốc độ/bộ nhớ lấy từ nguồn Unsloth chỉ mang tính tham khảo: phiên bản `llama.cpp`, driver, GPU, prompt và độ dài context thực tế đều làm kết quả thay đổi. Những chỗ là **đề xuất của người viết** (không có trong nguồn) được ghi rõ.

## 1. Cấu hình nên dùng ngay

Máy hiện tại: 8× NVIDIA A100 40 GB. Gemma 4 31B QAT chỉ nặng ~17.3 GB nên **vừa thoải mái một GPU**, còn dư nhiều VRAM cho KV cache, context dài và MTP.

### 1.1. `llama-server` — 31B QAT, text + MTP (khuyến nghị)

```bash
./llama.cpp/llama-server \
  -hf unsloth/gemma-4-31B-it-qat-GGUF:UD-Q4_K_XL \
  --no-mmproj \
  --device CUDA0 \
  --host 0.0.0.0 --port 8080 \
  --alias gemma-4-31b-qat \
  --n-gpu-layers all \
  --ctx-size 65536 --parallel 1 --kv-unified \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --batch-size 2048 --ubatch-size 512 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --jinja \
  --chat-template-kwargs '{"enable_thinking":true}' \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --cache-prompt --cache-ram 16384
```

Ghi chú:

- `UD-Q4_K_XL` là **quant duy nhất** của repo QAT (xem mục 2). Không có Q6/Q8 để "nâng chất lượng".
- `-hf` tự tải model, `mmproj` và tự nhận file MTP `mtp-gemma-4-31B-it.gguf` ở repo root — **không cần `--model-draft`** (mục 4).
- `--ctx-size 65536` là điểm bắt đầu an toàn (đề xuất). Model hỗ trợ tới 262,144 token; tăng dần và theo dõi `nvidia-smi` (mục 6).
- `--cache-type-k/v q8_0` là lựa chọn an toàn (đề xuất, giống cách tuning ở `Qwen3.8-27B.md`). Nếu muốn chất lượng chuẩn và còn VRAM, bỏ hai cờ này (mặc định `f16`).
- Không cần `--cache-type-k-draft/-v-draft`: theo model card, drafter **dùng chung KV cache của target**.
- Thinking bật bằng `--chat-template-kwargs '{"enable_thinking":true}'`; tắt bằng `false` (mục 3).

### 1.2. `llama-server` — 31B QAT có vision

```bash
./llama.cpp/llama-server \
  -hf unsloth/gemma-4-31B-it-qat-GGUF:UD-Q4_K_XL \
  --device CUDA0 \
  --host 0.0.0.0 --port 8080 \
  --alias gemma-4-31b-qat \
  --n-gpu-layers all \
  --ctx-size 65536 --parallel 1 --kv-unified \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --batch-size 2048 --ubatch-size 512 \
  --image-min-tokens 280 --image-max-tokens 1120 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --jinja \
  --chat-template-kwargs '{"enable_thinking":true}' \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --cache-prompt --cache-ram 16384
```

- Bỏ `--no-mmproj` để `-hf` tự lấy projector. Cũng có thể tải tay và chỉ định `--mmproj <file>` (mục 5).
- Gemma 4 hỗ trợ token budget cho ảnh: **70, 140, 280, 560, 1120**. Thấp = nhanh (phân loại, caption, video nhiều frame); cao = chi tiết (OCR, đọc chữ nhỏ, parse tài liệu). Việc ánh xạ sang `--image-min-tokens/--image-max-tokens` ở trên là đề xuất của người viết; với OCR đặt `--image-min-tokens 560 --image-max-tokens 1120`.
- **Đặt ảnh trước text** trong prompt.
- Bắt đầu với `--parallel 1` khi dùng vision + MTP; chỉ tăng sau khi kiểm tra ổn định.

### 1.3. Chạy nhanh bằng `llama-cli` (thử nghiệm)

```bash
export LLAMA_CACHE="unsloth/gemma-4-31B-it-qat-GGUF"
./llama.cpp/llama-cli \
  -hf unsloth/gemma-4-31B-it-qat-GGUF:UD-Q4_K_XL \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --spec-type draft-mtp --spec-draft-n-max 2
```

- `LLAMA_CACHE` ép `llama.cpp` lưu model vào thư mục chỉ định.
- Không cần đặt context: nguồn nói `llama.cpp` tự dùng lượng cần thiết.
- Tắt thinking nên dùng `llama-server` vì `llama-cli` **có thể không hoạt động ổn định** với `enable_thinking:false`.

### 1.4. Các kích thước khác (thay repo/alias)

| Model           | Repo                                    | Dung lượng QAT | RAM+VRAM tối thiểu (QAT) | Context tối đa |
| --------------- | --------------------------------------- | -------------: | -----------------------: | -------------- |
| **E2B**         | `unsloth/gemma-4-E2B-it-qat-GGUF`       |        2.62 GB |                     3 GB | 128K           |
| **E4B**         | `unsloth/gemma-4-E4B-it-qat-GGUF`       |        4.22 GB |                     5 GB | 128K           |
| **12B Unified** | `unsloth/gemma-4-12b-it-qat-GGUF`       |        6.72 GB |                     7 GB | 256K           |
| **26B A4B**     | `unsloth/gemma-4-26B-A4B-it-qat-GGUF`   |        14.2 GB |                    15 GB | 256K           |
| **31B**         | `unsloth/gemma-4-31B-it-qat-GGUF`       |        17.3 GB |                    18 GB | 256K           |

Lệnh giữ nguyên, chỉ đổi `-hf`, `--alias`, `--mmproj`/`--model-draft` (nếu tải tay) cho đúng kích thước. Riêng E2B/E4B có thêm quant mobile `UD-Q2_K_XL` (mục 2.2).

## 2. Chọn model và quant

### 2.1. QAT là gì và vì sao chỉ có một quant

QAT (Quantization-Aware Training) là biến thể Gemma 4 của Google DeepMind được huấn luyện có tính đến lượng tử hóa 4-bit, nên giữ chất lượng gần BF16 nhưng giảm khoảng **72%** bộ nhớ:

| Gemma 4     | QAT int4 GGUF | BF16 gốc | Giảm    |
| ----------- | ------------: | -------: | ------: |
| **E2B**     |       2.62 GB |  9.31 GB |  71.86% |
| **E4B**     |       4.22 GB |  15.1 GB |  72.05% |
| **12B**     |       6.72 GB |  23.8 GB |  71.76% |
| **26B A4B** |       14.2 GB |  50.5 GB |  71.88% |
| **31B**     |       17.3 GB |  61.4 GB |  71.82% |

Unsloth chỉ upload **một** file `UD-Q4_K_XL` cho mỗi model, vì độ chính xác ở các mức cao hơn không cải thiện mà còn giảm. Lý do:

- Chuyển thẳng QAT BF16 sang `Q4_0` của `llama.cpp` **không lossless**: `llama.cpp` dùng scale F16 còn QAT BF16 dùng scale BF16, và scale không được chọn tối ưu.
- Chuyển đổi ngây thơ chỉ khớp byte 24.77% với BF16 QAT; Unsloth Dynamic đẩy lên **99.96%**, đồng thời file nhỏ hơn (không cần Q6_K cho embedding).

| Model | Phương pháp | Dung lượng (GB) | Mean KLD | Top-1 % |
| ----- | ----------- | --------------: | -------: | ------: |
| E2B   | Unsloth     |            2.62 |  0.00173 |   98.16 |
| E2B   | Q4_0 thường |            3.35 |  0.05109 |   89.29 |
| E4B   | Unsloth     |            4.22 |  0.00121 |   98.54 |
| E4B   | Q4_0 thường |            5.15 |  0.03778 |   90.94 |
| 12B   | Unsloth     |            6.72 |  0.13288 |   88.76 |
| 12B   | Q4_0 thường |            6.98 |  0.50702 |   74.08 |
| 26B   | Unsloth     |           14.25 |  0.09788 |   85.63 |
| 26B   | Q4_0 thường |           14.44 |  0.36094 |   70.20 |
| **31B** | **Unsloth** |       **17.29** |**0.01403** |**96.67** |
| 31B   | Q4_0 thường |           17.65 |  0.09349 |   87.91 |

Với 31B, `UD-Q4_K_XL` gần như đồng nhất với BF16 (Top-1 96.67%), là bản tốt nhất cho Gemma 4 31B nếu chỉ có ~18–20 GB. Nếu cần quant khác (Q5/Q6/Q8) thì dùng repo **không phải QAT** `unsloth/gemma-4-31B-it-GGUF` (mục 2.3).

### 2.2. Quant mobile (E2B, E4B)

Google có thêm bản "mobile mixture QAT" cho E2B/E4B; Unsloth chuyển sang `UD-Q2_K_XL` (lớp 2-bit dùng `TQ2_0`):

|                    | E2B mobile          | E4B mobile          |
| ------------------ | ------------------- | ------------------- |
| Dung lượng         | 2.19 GB             | 3.22 GB             |
| Tensor 2-bit       | 61 (gồm deep MLP)   | 2 (chỉ embedding)   |
| Mean KLD vs BF16   | 0.00409             | 0.00102             |
| Top-1 %            | 97.82%              | 98.76%              |

Tải bằng `--include "*UD-Q2_K_XL*"` (mục 5). Không áp dụng cho 12B/26B/31B.

### 2.3. Nên chọn 26B-A4B hay 31B?

- **26B-A4B** (MoE, 4B tham số active): cân bằng tốc độ/chất lượng, phù hợp khi RAM/VRAM hạn chế (~15 GB) hoặc cần throughput cao.
- **31B** (dense): mạnh nhất họ Gemma 4 (MMLU Pro 85.2%, AIME 2026 89.2%, LiveCodeBench v6 80.0%, GPQA Diamond 84.3%, Tau2 76.9%), chậm hơn 26B-A4B. MTP giúp dense 31B hưởng lợi nhiều nhất (mục 4).
- **12B Unified**: kiến trúc không có encoder, hỗ trợ text + ảnh + audio, 256K context.
- **E2B/E4B**: thiết bị biên/laptop; có thêm audio; context 128K.

Gemma 4 31B và 26B-A4B **không hỗ trợ audio**; audio chỉ có ở E2B, E4B, 12B (tối đa 30 giây). Video tối đa 60 giây (1 frame/giây).

Bản QAT vs bản thường: nếu cần Q8_0/BF16 để đối chiếu chất lượng, dùng `unsloth/gemma-4-31B-it-GGUF` (bản không QAT; cần 34–38 GB cho 8-bit, 62 GB cho BF16).

## 3. Thinking, sampler và multi-turn

### 3.1. Sampler chính thức

Dùng chung cho mọi use case, cả QAT:

```text
temperature = 1.0    top_p = 0.95    top_k = 64
```

Khác với Qwen, không có cấu hình riêng cho non-thinking.

### 3.2. Bật/tắt thinking

Gemma 4 dùng role chuẩn `system`/`user`/`assistant`. Thinking do token `<|think|>` ở **đầu system prompt** điều khiển, và `llama-server` bọc việc này qua `chat-template-kwargs`:

```bash
# Bật thinking
--chat-template-kwargs '{"enable_thinking":true}'
# Tắt thinking
--chat-template-kwargs '{"enable_thinking":false}'
```

Trên Windows PowerShell: `--chat-template-kwargs "{\"enable_thinking\":false}"`.

Định dạng output khi bật thinking:

```text
<|channel>thought
[internal reasoning]
<channel|>
[final answer]
```

Khi tắt thinking, các model lớn (trừ E2B/E4B) **vẫn có thể phát khối thought rỗng** (`<|channel>thought\n<channel|>`) — client cần xử lý được.

Có thể đặt cùng `--reasoning-format deepseek` để `llama-server` tách phần thought vào `message.reasoning_content`; mặc định là `auto`. Nếu thấy thought lẫn vào `content`, đặt cờ này hoặc kiểm tra `--jinja`.

Có thể giới hạn thinking bằng `--reasoning-budget N` (đề xuất, ví dụ `8192`; mặc định `-1` không giới hạn) kèm `--reasoning-budget-message`. Không có `--reasoning-effort low/medium/xhigh` như Qwen3.8 — không có trong nguồn Gemma; đừng sao chép cờ đó.

### 3.3. Multi-turn

Trong hội thoại nhiều lượt, lịch sử chỉ được chứa **câu trả lời cuối** của model. **Không đưa lại khối thought** của các lượt trước vào prompt kế tiếp. Vì vậy không thêm `--reasoning-preserve` mặc định như cấu hình Qwen; chỉ bật khi đã kiểm tra template GGUF thật sự hỗ trợ và có lý do rõ ràng.

Nguồn Unsloth có nhắc "Preserved Thinking" nhưng trong các trang được đọc không mô tả cách bật cho Gemma, nên không nên suy đoán.

## 4. MTP (speculative decoding)

### 4.1. Nguyên lý và lợi ích

Google DeepMind huấn luyện MTP riêng cho Gemma 4 (model "assistant"), gồm cả bản QAT. Drafter đề xuất nhiều token, model chính xác minh song song; chỉ token đã xác minh được giữ, nên **kết quả không đổi** và chất lượng không giảm.

- GGUF chạy nhanh khoảng **1.4×–2.2×**; Gemma 4 QAT + MTP được Unsloth đo **1.5×–2.2×**.
- Model dense như 31B hưởng lợi nhiều nhất (>1.4×). Thiết bị băng thông bộ nhớ thấp (Mac cũ) lợi ít hơn.
- MTP tốn thêm ~**2 GB** RAM/VRAM.

### 4.2. Cách hoạt động với repo QAT

Repo `unsloth/gemma-4-31B-it-qat-GGUF` có sẵn `mtp-gemma-4-31B-it.gguf` ở root (drafter "smart Q4_0" gần lossless). Cả hai đều đã được "smart 4-bit recovery" giống quant chính. Có thư mục `MTP/` với các độ chính xác khác. Unsloth chỉ đưa ra 8-bit và 16-bit (BF16/F16) cho bản non-QAT; còn QAT dùng 4-bit thông minh.

Build `llama.cpp` đủ mới sẽ tự tìm file này khi dùng `-hf`:

```bash
./build/bin/llama-server \
  -hf unsloth/gemma-4-31B-it-qat-GGUF:UD-Q4_K_XL \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  -ngl 999 -fa on
```

Khác với Qwen (cần repo `-MTP-GGUF` riêng), Gemma 4 **không cần repo draft riêng** — chỉ tải repo GGUF bình thường.

Khi khởi động, log có thể in một số dòng lỗi liên quan đến MTP; theo Unsloth có thể bỏ qua, miễn là drafter được nạp và tốc độ tăng.

### 4.3. Chọn `--spec-draft-n-max`

- Model card dùng `4`; hướng dẫn MTP của Unsloth dùng `2` là điểm bắt đầu tốt. **Đừng giả định `2` tối ưu**: thử từng giá trị `1`–`6` (hoặc ít nhất `2`, `3`, `4`) và chọn giá trị nhanh nhất cho phần cứng và prompt của bạn.
- So sánh với `--spec-type none` bằng cùng prompt và seed để kiểm tra MTP thật sự có lợi.
- Cờ hữu ích: `--spec-draft-n-min`, `--spec-draft-p-min` (mặc định `0.00`), `--gpu-layers-draft all`, `--device-draft`. Các cờ cũ `--draft`, `--draft-max` đã bị gỡ; dùng nhóm `--spec-*`.

### 4.4. Chỉ định file MTP thủ công

Khi tải tay (không dùng `-hf`) phải trỏ tới drafter:

```bash
./llama.cpp/llama-server \
  --model unsloth/gemma-4-31B-it-qat-GGUF/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf \
  --mmproj unsloth/gemma-4-31B-it-qat-GGUF/mmproj-BF16.gguf \
  --model-draft unsloth/gemma-4-31B-it-qat-GGUF/mtp-gemma-4-31B-it.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --alias gemma-4-31b-qat --port 8001 \
  --chat-template-kwargs '{"enable_thinking":true}'
```

Tên `mmproj` có thể là `mmproj-BF16.gguf` hoặc `mmproj-F16.gguf` tùy repo (các hướng dẫn Unsloth dùng cả hai); kiểm tra danh sách file thực tế của repo trước khi chạy. Tên file drafter cũng đổi theo kích thước (ví dụ `mtp-gemma-4-12B-it.gguf`).

### 4.5. Cần build `llama.cpp` mới

Không giả định bản `llama.cpp` cũ hỗ trợ MTP. Kiểm tra:

```bash
./llama.cpp/llama-server --help | grep -A2 "spec-type"
# Phải thấy draft-mtp trong danh sách
```

Bản `llama-cpp.md` của thư mục này liệt kê `draft-mtp` nên binary hiện tại có hỗ trợ.

## 5. Cài đặt, tải model

### 5.1. Build `llama.cpp`

```bash
apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j --clean-first \
    --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
```

- `-DGGML_CUDA=OFF` nếu chỉ chạy CPU. Trên Mac Metal cũng đặt `OFF` (Metal bật mặc định).
- Luôn dùng bản mới nhất: MTP và các sửa lỗi Gemma 4 vào rất nhanh.

### 5.2. Tải model thủ công

```bash
pip install huggingface_hub hf_transfer
hf download unsloth/gemma-4-31B-it-qat-GGUF \
    --local-dir unsloth/gemma-4-31B-it-qat-GGUF \
    --include "*mmproj-*" \
    --include "mtp-*" \
    --include "*UD-Q4_K_XL*"
```

- Chỉ tải phần cần thiết: bỏ `mmproj` nếu text-only, bỏ `mtp-*` nếu không dùng MTP.
- Với E2B/E4B mobile dùng `--include "*UD-Q2_K_XL*"`.
- Nếu tải bị treo, xem hướng dẫn debug Hugging Face Hub/XET của Unsloth.

### 5.3. Chạy bằng Docker (`docker-compose.yml` hiện tại)

`docker-compose.yml` trong thư mục này dùng `ghcr.io/ggml-org/llama.cpp:server-cuda` và mount `./models`. Ví dụ (đề xuất), sau khi tải model vào `./models/gemma-4-31B-it-qat-GGUF/` như trên:

```yaml
    command: >
      -m /models/gemma-4-31B-it-qat-GGUF/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf
      --mmproj /models/gemma-4-31B-it-qat-GGUF/mmproj-BF16.gguf
      --model-draft /models/gemma-4-31B-it-qat-GGUF/mtp-gemma-4-31B-it.gguf
      --spec-type draft-mtp --spec-draft-n-max 2
      --port 8000 --host 0.0.0.0
      --alias gemma-4-31b-qat
      -fa on -ngl all --device CUDA0
      -c 65536 -b 2048 -ub 512
      -ctk q8_0 -ctv q8_0
      --parallel 1 --kv-unified
      --jinja
      --chat-template-kwargs '{"enable_thinking":true}'
      --temp 1.0 --top-p 0.95 --top-k 64
      --metrics --slots
      --api-key llama-cpp-api-key
```

Image `server-cuda` phải đủ mới để hỗ trợ `draft-mtp`; kiểm tra bằng `docker run --rm --entrypoint /app/llama-server <image> --help | grep draft-mtp` (đường dẫn binary có thể khác theo image) và `docker compose pull` trước khi chạy.

## 6. Bộ nhớ, context và KV cache

### 6.1. Yêu cầu bộ nhớ

Bảng tổng RAM + VRAM (hoặc unified memory) khuyến nghị từ Unsloth:

| Gemma 4     | QAT 4-bit | Non-QAT 4-bit | 8-bit    | BF16/FP16 | 4-bit + MTP |
| ----------- | --------: | ------------: | -------: | --------: | ----------: |
| **E2B**     |      3 GB |          4 GB |   5–8 GB |     10 GB |        5 GB |
| **E4B**     |      5 GB |      5.5–6 GB |  9–12 GB |     16 GB |    6.5–7 GB |
| **12B**     |      7 GB |        7–8 GB | 13–14 GB |     25 GB |      8–9 GB |
| **26B A4B** |     15 GB |      16–18 GB | 28–30 GB |     52 GB |    17–18 GB |
| **31B**     |     18 GB |      17–20 GB | 34–38 GB |     62 GB |    18–21 GB |

Quy tắc: tổng bộ nhớ khả dụng phải lớn hơn kích thước file model. Nếu thiếu, `llama.cpp` vẫn chạy với offload RAM/đĩa nhưng chậm hơn. Context lớn cần thêm bộ nhớ cho KV cache.

### 6.2. Context

- E2B/E4B: tối đa 128K. 12B, 26B A4B, 31B: tối đa **262,144** token.
- Gemma 4 dùng attention lai: các lớp sliding window (512 token cho E2B/E4B, 1024 cho 12B/26B/31B) xen kẽ với lớp global (lớp cuối luôn global), Key/Value hợp nhất ở lớp global và p-RoPE. Do đó KV cache nhẹ hơn nhiều so với một Transformer thông thường có cùng context.
- Chạy 31B trên A100 40 GB: bắt đầu `65536`, tăng lên `131072`/`262144` nếu còn VRAM. Luôn giữ headroom 1–2 GiB cho spike/batch/ảnh.
- Không tự override `--rope-*`/`--yarn-*` khi GGUF đã có metadata đúng; context nạp được không chứng minh truy xuất cuối context còn chính xác.

### 6.3. KV cache

- `f16` (mặc định): chất lượng chuẩn, tốn nhiều VRAM nhất.
- `q8_0`: gần lossless, nên là lựa chọn đầu tiên khi cần tiết kiệm.
- `q5_*`/`q4_*`: chỉ dùng khi cần context rất lớn; phải test dài hạn. Một số build cần `-DGGML_CUDA_FA_ALL_QUANTS=ON` để Flash Attention không rơi về CPU (xem ghi chú trong `Qwen3.8-27B.md`).
- K và V có thể khác nhau nhưng bắt đầu cùng kiểu để dễ chẩn đoán.
- `--swa-full` giữ cache SWA đầy đủ, tốn nhiều VRAM; để mặc định trừ khi cần tái sử dụng prefix mạnh.

### 6.4. Xử lý OOM

Thứ tự đề xuất: giảm `--ctx-size` → giảm `--batch-size`/`--ubatch-size` (256/128) → tắt vision (`--no-mmproj`) → tắt MTP (`--spec-type none`) → KV `q8_0` xuống `q5`/`q4` → cuối cùng mới offload CPU (`-ot`, `--fit`). Không có quant thấp hơn `UD-Q4_K_XL` cho 12B/26B/31B; nếu quá chật hãy chuyển sang kích thước nhỏ hơn (31B → 26B-A4B).

## 7. CLI argument quan trọng

Các cờ dưới đây dùng được với binary trong `llama-cpp.md`. Xem `Qwen3.8-27B.md` mục 4–5 để có danh mục đầy đủ hơn; ở đây chỉ nêu các cờ liên quan trực tiếp đến Gemma 4 QAT.

### 7.1. Model, GPU, bộ nhớ

| Argument                                      | Công dụng                                                                                         |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `-hf repo[:quant]`                            | Tải/nạp từ Hugging Face, tự lấy `mmproj` và MTP. Dùng `:UD-Q4_K_XL`.                              |
| `-m`, `--model FILE`                          | GGUF cục bộ.                                                                                      |
| `--mmproj FILE`, `--no-mmproj`                | Chỉ định projector vision / chạy text-only (tiết kiệm VRAM).                                      |
| `--mmproj-offload` / `--no-mmproj-offload`    | Projector trên GPU/CPU; CPU tiết kiệm VRAM nhưng xử lý ảnh chậm hơn.                              |
| `-dev`, `--device`; `--list-devices`          | Chọn GPU, ví dụ `CUDA0`.                                                                          |
| `-ngl`, `--n-gpu-layers all`                  | Đưa toàn bộ layer lên GPU.                                                                        |
| `-fa`, `--flash-attn on`                      | Nên bật với NVIDIA và context dài.                                                                |
| `-ctk`, `-ctv`                                | Kiểu KV cache K/V.                                                                                |
| `-fit`, `--fit on/off`                        | Tự điều chỉnh vừa VRAM. Lần đầu `on`; khi đã biết cấu hình vừa, `off`.                            |
| `-sm`, `-ts`, `-mg`                           | Multi-GPU; 31B QAT một GPU 40 GB không cần.                                                       |
| `-ot`, `--override-tensor`                    | Offload chọn lọc tensor sang CPU. Với MoE 26B-A4B có thể dùng `--cpu-moe`/`--n-cpu-moe` khi thiếu VRAM. |

`--cpu-moe`, `--n-cpu-moe` chỉ có ý nghĩa với **26B-A4B** (MoE); 31B, 12B, E2B, E4B là dense.

### 7.2. Context, batch, cache

| Argument                                  | Công dụng / khuyến nghị                                                           |
| ----------------------------------------- | --------------------------------------------------------------------------------- |
| `-c`, `--ctx-size N`                      | Tổng context. `0` lấy theo metadata model.                                        |
| `-b`, `--batch-size`; `-ub`, `--ubatch-size` | Batch logic/vật lý; 2048/512, giảm ubatch để cứu VRAM.                           |
| `-np`, `--parallel N`                     | Số slot; 1 khi chạy context lớn hoặc vision.                                      |
| `--kv-unified`                            | Một buffer KV dùng chung giữa các slot.                                           |
| `--cache-prompt`, `--cache-ram MiB`       | Tái sử dụng prefix, cache prompt trong RAM (không phải VRAM).                     |
| `--ctx-checkpoints N`                     | Checkpoint context; giảm nếu gặp vấn đề với SWA.                                  |
| `-t`, `-tb`                               | Thread CPU; chạy full GPU để auto hoặc theo core vật lý.                          |

### 7.3. MTP

| Argument                                      | Ý nghĩa                                                    |
| --------------------------------------------- | ---------------------------------------------------------- |
| `--spec-type draft-mtp`                       | Bật MTP. `none` để tắt.                                    |
| `--spec-draft-n-max N`                        | Số token draft tối đa; bắt đầu `2`, sweep `1`–`6`.         |
| `--spec-draft-n-min N`, `--spec-draft-p-min P`| Ngưỡng số/xác suất draft; mặc định `0.00`.                 |
| `--model-draft FILE`                          | Chỉ định drafter khi không dùng `-hf`.                     |
| `--gpu-layers-draft all`, `--device-draft`    | Đưa drafter lên GPU/chọn thiết bị.                         |

### 7.4. Chat, reasoning, vision

| Argument                                    | Ý nghĩa                                                                   |
| ------------------------------------------- | ------------------------------------------------------------------------- |
| `--jinja`                                   | Dùng template trong GGUF (mặc định bật).                                  |
| `--chat-template-kwargs JSON`               | `{"enable_thinking":true/false}`.                                         |
| `--reasoning on\|off\|auto`                 | Bật/tắt thinking theo template; Gemma nên điều khiển qua kwargs ở trên.   |
| `--reasoning-format none\|deepseek\|…`      | Cách trả về phần thought qua API.                                         |
| `--reasoning-budget N`                      | Giới hạn token thinking; `-1` không giới hạn.                             |
| `--image-min-tokens`, `--image-max-tokens`  | Token ảnh (dynamic resolution).                                           |
| `--mtmd-batch-max-tokens N`                 | Số image token tối đa mỗi batch encode.                                   |
| `--grammar`, `-j/--json-schema`             | Ép output theo grammar/JSON schema, hữu ích cho tool/agent.               |

### 7.5. HTTP và bảo mật

`--host`, `--port`, `--alias`, `--api-key`/`--api-key-file`, `--metrics`, `--slots`, `--ssl-key-file`/`--ssl-cert-file`. Không bind `0.0.0.0` khi thiếu firewall/API key. Không bật `--agent`, `--tools`, MCP trên mạng không tin cậy vì chúng có thể thực thi lệnh hoặc đọc/ghi file.

## 8. Gợi ý prompt

- **Thứ tự modality**: đặt **ảnh trước text**; với audio (chỉ E2B/E4B/12B) đặt audio **sau text** theo model card. Video: gửi chuỗi frame trước rồi mới đến chỉ dẫn.
- **OCR/tài liệu**: budget ảnh 560 hoặc 1120.
  ```text
  [image first]
  Extract all text from this receipt. Return line items, total, merchant, and date as JSON.
  ```
- **So sánh nhiều ảnh**:
  ```text
  [image 1]
  [image 2]
  Compare these two screenshots and tell me which one is more likely to confuse a new user.
  ```
- **ASR** (E2B/E4B/12B):
  ```text
  Transcribe the following speech segment in {LANGUAGE} into {LANGUAGE} text.
  Follow these specific instructions for formatting the answer:
  * Only output the transcription, with no newlines.
  * When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.
  ```
- **Dịch giọng nói**:
  ```text
  Transcribe the following speech segment in {SOURCE_LANGUAGE}, then translate it into {TARGET_LANGUAGE}.
  When formatting the answer, first output the transcription in {SOURCE_LANGUAGE}, then one newline, then output the string '{TARGET_LANGUAGE}: ', then the translation in {TARGET_LANGUAGE}.
  ```
- Function calling được hỗ trợ native; giữ `--jinja` để template xử lý tool call.

## 9. Quy trình tuning an toàn

1. Cập nhật `llama.cpp`, build CUDA, chạy `--list-devices`, xác nhận `--help` có `draft-mtp`.
2. Khởi động với `--no-mmproj`, `--spec-type none`, context 32K/64K, `--parallel 1`, KV `q8_0`. Kiểm tra log: toàn bộ layer/KV trên CUDA.
3. Kiểm tra `curl http://127.0.0.1:8080/health` và một request `chat/completions` ngắn có và không có thinking.
4. Bật MTP `n=2`, so tốc độ với `none`, rồi sweep `3`, `4`.
5. Tăng context từng bước (64K → 128K → 256K), theo dõi `nvidia-smi`.
6. Khi cấu hình đã chắc chắn vừa VRAM, chuyển `--fit off`.
7. Thêm vision (bỏ `--no-mmproj`), thử budget ảnh 280/560/1120.
8. Đánh giá bằng workload thật: prefill dài, code, nhiều turn, tool call, truy xuất cuối context; không chỉ nhìn tok/s.

Lệnh theo dõi:

```bash
watch -n 0.5 nvidia-smi
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-31b-qat","messages":[{"role":"user","content":"Xin chào"}]}'
```

Nếu prefill hoặc decode chậm bất thường trong khi GPU rảnh, nghi ngờ tensor/KV/Flash Attention rơi về CPU hoặc model không thực sự nằm hết trên VRAM.

## 10. Cách khác: Unsloth Studio

Unsloth Studio (macOS, Windows, Linux, WSL) tự đặt tham số suy luận và cấu hình MTP tối ưu cho phần cứng của bạn, đồng thời hỗ trợ tool calling tự sửa lỗi, web search và chạy code:

```bash
curl -fsSL https://unsloth.ai/install.sh | sh
unsloth studio -H 0.0.0.0 -p 8888
```

Mở `http://127.0.0.1:8888`, đặt mật khẩu lần đầu, vào tab Chat, tìm Gemma 4 rồi tải model/quant. MTP của Gemma 4 được bật tự động, chỉ cần tải GGUF Gemma 4 thông thường. Dùng `unsloth studio --secure` để mở qua HTTPS bằng Cloudflare tunnel. Studio phù hợp thử nhanh; triển khai production trong thư mục này nên dùng `llama-server` như các mục trên.

## 11. Tóm tắt nhanh

| Mục tiêu                             | Chọn                                                      |
| ------------------------------------ | --------------------------------------------------------- |
| Chất lượng cao nhất trong họ Gemma 4 | **31B QAT** `UD-Q4_K_XL` + MTP                            |
| Nhanh, ít bộ nhớ, còn khá mạnh       | **26B-A4B QAT** (MoE, ~15 GB)                              |
| Đa phương thức có audio, laptop      | **12B** hoặc **E4B** QAT                                  |
| Điện thoại/thiết bị biên             | **E2B/E4B** `UD-Q2_K_XL` (mobile) hoặc `UD-Q4_K_XL`       |
| Cần quant cao hơn 4-bit              | Dùng repo non-QAT `unsloth/gemma-4-*-it-GGUF` (Q8_0, BF16) |

Các tham số nên nhớ: `temp 1.0`, `top_p 0.95`, `top_k 64`, `--spec-type draft-mtp --spec-draft-n-max 2` (sweep), `--chat-template-kwargs '{"enable_thinking":true|false}'`, ảnh trước text, không đưa thought cũ vào lịch sử.
