# Chạy Unlimited-OCR bằng llama.cpp

> [https://github.com/ggml-org/llama.cpp/pull/24975](https://github.com/ggml-org/llama.cpp/pull/24975)

Unlimited-OCR là mô hình OCR/VLM đa ngôn ngữ khoảng 3B tham số của Baidu, hướng tới phân tích tài liệu dài trong một lần sinh. Bản GGUF dùng kiến trúc DeepSeek-OCR và cần **hai file**:

1. mô hình ngôn ngữ GGUF (chọn một mức lượng tử hóa);
2. vision projector `mmproj-Unlimited-OCR-F16.gguf` (~774 MiB).

Hướng dẫn này dùng `llama.cpp/llama-server` có sẵn trong workspace và API tương thích OpenAI.

## 1. Kiểm tra llama.cpp

```bash
./llama.cpp/llama-server --version
./llama.cpp/llama-server --help
```

Unlimited-OCR cần bản llama.cpp có hỗ trợ DeepSeek-OCR/MTMD. Model card GGUF cảnh báo các bản llama.cpp cũ không thể nạp mô hình. Binary hiện có trong workspace là build mới; nếu gặp lỗi `unknown model architecture` hoặc lỗi nạp projector, hãy cập nhật và build lại llama.cpp:

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build -j --target llama-server
```

Với NVIDIA CUDA, thêm `-DGGML_CUDA=ON` vào lệnh `cmake`; binary sau khi build nằm tại `llama.cpp/build/bin/llama-server`.

## 2. Tải model

`Q4_K_M` (~1.82 GiB) là lựa chọn mặc định cân bằng giữa dung lượng và chất lượng. Projector F16 dùng chung cho mọi quant.

```bash
mkdir -p models/Unlimited-OCR

hf download sahilchachra/Unlimited-OCR-GGUF \
  Unlimited-OCR-Q4_K_M.gguf \
  mmproj-Unlimited-OCR-F16.gguf \
  --local-dir models/Unlimited-OCR
```

Nếu máy chỉ có lệnh cũ:

```bash
huggingface-cli download sahilchachra/Unlimited-OCR-GGUF \
  --include 'Unlimited-OCR-Q4_K_M.gguf' 'mmproj-Unlimited-OCR-F16.gguf' \
  --local-dir models/Unlimited-OCR
```

Các lựa chọn đáng chú ý:

- `Q8_0` (2.91 GiB): gần như không mất chất lượng;
- `Q6_K` (2.43 GiB), `Q5_K_M` (2.07 GiB): ưu tiên chất lượng;
- `Q4_K_M` (1.82 GiB): khuyến nghị chung;
- `IQ4_XS` (1.53 GiB): nhỏ hơn, vẫn có chất lượng khá;
- quant 2–3 bit: chỉ nên dùng khi rất thiếu RAM/VRAM vì OCR giảm độ chính xác.

Dung lượng RAM/VRAM thực tế còn phải cộng projector, KV cache và bộ đệm xử lý ảnh.

### Chạy trực tiếp từ Hugging Face, không cần tải thủ công

`llama-server` có thể tự tải model vào cache từ Hugging Face. Với tên projector chuẩn trong repository, llama.cpp sẽ tự tìm và tải cả `mmproj`:

```bash
./llama.cpp/llama-server \
  --hf-repo sahilchachra/Unlimited-OCR-GGUF \
  --hf-file Unlimited-OCR-Q5_K_M.gguf \
  --alias Unlimited-OCR \
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

Lệnh này không cần bước `hf download`, nhưng model vẫn được tải một lần vào cache cục bộ. Các lần chạy sau sẽ dùng lại cache.

Nếu llama.cpp không tự tìm được projector, chỉ rõ URL của projector:

```bash
./llama.cpp/llama-server \
  --hf-repo sahilchachra/Unlimited-OCR-GGUF \
  --hf-file Unlimited-OCR-Q5_K_M.gguf \
  --mmproj-url https://huggingface.co/sahilchachra/Unlimited-OCR-GGUF/resolve/main/mmproj-Unlimited-OCR-F16.gguf \
  --alias Unlimited-OCR \
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

Có thể đổi `Unlimited-OCR-Q5_K_M.gguf` sang quant khác có trong repository. Với repository riêng tư, đặt token trước khi chạy:

```bash
export HF_TOKEN=hf_...
```

Nếu muốn cấm mọi truy cập mạng và chỉ dùng file đã có trong cache, thêm `--offline`.

## 3. Khởi động server

### GPU

```bash
./llama.cpp/llama-server \
  --model models/Unlimited-OCR/Unlimited-OCR-Q5_K_M.gguf \
  --mmproj models/Unlimited-OCR/mmproj-Unlimited-OCR-F16.gguf \
  --alias Unlimited-OCR \
  --ctx-size 32768 \
  --n-predict 16384 \
  --gpu-layers all \
  --flash-attn auto \
  --temp 0 \
  --host 0.0.0.0 \
  --port 8080
```

`--temp 0` phù hợp OCR vì cho kết quả ổn định. `--ctx-size 32768` bám theo cấu hình suy luận gốc; nếu thiếu bộ nhớ có thể giảm xuống `8192` hoặc `16384`, nhưng tài liệu dày có thể bị cắt kết quả. Có thể giảm `--n-predict` nếu chỉ OCR ảnh ngắn.

### Chỉ dùng CPU

```bash
./llama.cpp/llama-server \
  -m models/Unlimited-OCR/Unlimited-OCR-Q5_K_M.gguf \
  --mmproj models/Unlimited-OCR/mmproj-Unlimited-OCR-F16.gguf \
  --alias Unlimited-OCR \
  -c 8192 -n 4096 \
  --device none \
  --threads "$(nproc)" \
  --temp 0 \
  --host 127.0.0.1 --port 8080
```

OCR trên CPU sẽ chậm hơn đáng kể. Kiểm tra server sau khi khởi động:

```bash
curl http://127.0.0.1:8080/health
```

### OCR trực tiếp một ảnh bằng `llama-cli`

Không cần khởi động HTTP server nếu chỉ muốn xử lý ảnh `scripts/test.png` một lần:

```bash
./llama.cpp/llama-cli \
  --hf-repo sahilchachra/Unlimited-OCR-GGUF \
  --hf-file Unlimited-OCR-Q5_K_M.gguf \
  --image scripts/test.png \
  -p "document parsing." \
  -n 8192 \
  --temp 0
```

Bỏ `<|grounding|>` hoặc đổi prompt thành `document parsing.` nếu không cần các thẻ bounding box.

## 4. OCR một ảnh qua API

Ví dụ dưới đây dùng base64 data URL:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Unlimited-OCR",
    "temperature": 0,
    "max_tokens": 8192,
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "document parsing."},
        {"type": "image_url", "image_url": {
          "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png"
        }}
      ]
    }]
  }'
```

Trên macOS, thay `base64 -w0 document.png` bằng `base64 < document.png | tr -d '\n'`.

### Các prompt hữu ích

| Nhu cầu                                          | Prompt                           |
| ------------------------------------------------ | -------------------------------- |
| Parse tài liệu theo prompt gốc của Unlimited-OCR | `document parsing.`              |
| Văn bản thuần                                    | `Free OCR.`                      |
| Chuyển tài liệu thành Markdown, có vùng tọa độ   | `<                               |
| OCR kèm vùng tọa độ                              | `<                               |
| Parse biểu đồ/hình vẽ                            | `Parse the figure.`              |
| Mô tả ảnh                                        | `Describe this image in detail.` |
| Tìm vị trí một chuỗi                             | `<                               |

Khi dùng `<|grounding|>`, kết quả có dạng:

```text
<|det|>title [37, 64, 464, 132]<|/det|>INVOICE
<|det|>text [37, 483, 329, 543]<|/det|>Total Due: $44.00
```

Bốn số là `[x1, y1, x2, y2]`, từ góc trên-trái đến góc dưới-phải trong hệ tọa độ ảnh đầu vào. Không thêm `<|grounding|>` nếu chỉ cần text/Markdown và không cần bounding box.

## 5. Gọi bằng Python

Không cần OpenAI SDK; ví dụ sau chỉ cần `requests`:

```python
import base64
import mimetypes
from pathlib import Path

import requests

image_path = Path("document.png")
mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
data = base64.b64encode(image_path.read_bytes()).decode()

response = requests.post(
    "http://127.0.0.1:8080/v1/chat/completions",
    json={
        "model": "Unlimited-OCR",
        "temperature": 0,
        "max_tokens": 8192,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "document parsing."},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64,{data}"
                }},
            ],
        }],
    },
    timeout=1200,
)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
```

## 6. PDF và tài liệu nhiều trang

Bản Transformers gốc có `infer_multi()` và chế độ `base` cho nhiều trang. Với llama.cpp, cách ổn định là đổi PDF thành ảnh, OCR **từng trang**, rồi ghép kết quả theo thứ tự. Ví dụ cài PyMuPDF:

```bash
python -m pip install pymupdf requests
```

```python
import base64
import fitz
import requests

URL = "http://127.0.0.1:8080/v1/chat/completions"
doc = fitz.open("document.pdf")
results = []

for page_number, page in enumerate(doc, 1):
    # 200 DPI thường nhẹ hơn; tăng lên 300 DPI nếu chữ quá nhỏ.
    pix = page.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72), alpha=False)
    image = base64.b64encode(pix.tobytes("png")).decode()
    response = requests.post(URL, json={
        "model": "Unlimited-OCR",
        "temperature": 0,
        "max_tokens": 8192,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "document parsing."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image}"
                }},
            ],
        }],
    }, timeout=1200)
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"]
    results.append(f"<!-- page {page_number} -->\n{text}")

with open("document.md", "w", encoding="utf-8") as f:
    f.write("\n\n".join(results))
```

Chạy từng trang cũng tránh việc nhiều ảnh cạnh tranh context và dễ chạy lại riêng trang lỗi.

## 7. Tinh chỉnh và xử lý lỗi

- **Lặp nội dung:** giữ `temperature: 0`; thử thêm `--repeat-penalty 1.05` khi khởi động server. Cơ chế `no_repeat_ngram_size=35` chuyên biệt trong mã Transformers/SGLang gốc không tương đương hoàn toàn với repeat penalty của llama.cpp.
- **Kết quả bị dừng sớm:** tăng cả `--ctx-size`, `--n-predict` của server và `max_tokens` trong request; tổng token ảnh, prompt và output vẫn phải nằm trong context.
- **Hết VRAM:** giảm `--ctx-size`, dùng quant nhỏ hơn, hoặc thay `--gpu-layers all` bằng một số lớp cụ thể/`--gpu-layers auto`.
- **Không thấy ảnh:** kiểm tra MIME (`image/png`, `image/jpeg`), base64 không có xuống dòng, và chắc chắn đã nạp đúng `--mmproj`.
- **Lỗi kiến trúc/projector:** binary llama.cpp quá cũ hoặc không có hỗ trợ DeepSeek-OCR; cập nhật/build lại trước khi đổi model.
- **OCR chữ nhỏ kém:** crop vùng cần đọc hoặc render PDF ở DPI cao hơn; đổi từ Q4 sang Q6/Q8 nếu tài nguyên cho phép.
- **Bảo mật:** chỉ dùng `--host 0.0.0.0` trong mạng tin cậy. Khi public server, thêm `--api-key YOUR_KEY` và gửi header `Authorization: Bearer YOUR_KEY`.

## Nguồn

- Model gốc: [https://huggingface.co/baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)
- GGUF: [https://huggingface.co/sahilchachra/Unlimited-OCR-GGUF](https://huggingface.co/sahilchachra/Unlimited-OCR-GGUF)
- llama.cpp: [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
