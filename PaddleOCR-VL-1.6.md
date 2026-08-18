# Chạy PaddleOCR-VL-1.6 bằng `llama.cpp`

PaddleOCR-VL-1.6 là mô hình phân tích tài liệu thị giác-ngôn ngữ cỡ 0.9B. Theo model card, phiên bản 1.6 đạt 96,33% trên OmniDocBench v1.6 và tương thích kiến trúc với PaddleOCR-VL-1.5. Repository GGUF chính thức gồm:


| File                                | Dung lượng     | Vai trò                               |
| ----------------------------------- | -------------- | ------------------------------------- |
| `PaddleOCR-VL-1.6-GGUF.gguf`        | khoảng 893 MiB | language model                        |
| `PaddleOCR-VL-1.6-GGUF-mmproj.gguf` | khoảng 841 MiB | vision projector, bắt buộc để đọc ảnh |


Có hai cách dùng:

1. **PaddleOCR document parser + `llama-server`**: phù hợp để phân tích cả trang/tài liệu và lưu JSON/Markdown.
2. **Gọi model trực tiếp** bằng `llama-cli` hoặc OpenAI-compatible API: phù hợp để nhận dạng một loại phần tử như text, công thức, bảng, biểu đồ hoặc con dấu.

Các lệnh dưới đây dùng binary trong repository này: `./llama.cpp/llama-server` và `./llama.cpp/llama-cli`. Nếu chưa build, xem [README.md](README.md). Nên dùng bản `llama.cpp` mới có hỗ trợ PaddleOCR-VL và multimodal `mtmd`.

## 1. Chạy nhanh từ Hugging Face

`llama-server` có thể tải model bằng `--hf-repo`; `mmproj` được tự tìm và tải nếu có. Chỉ rõ `--hf-file` để tránh chọn nhầm file:

```bash
./llama.cpp/llama-server \
  --hf-repo PaddlePaddle/PaddleOCR-VL-1.6-GGUF \
  --hf-file PaddleOCR-VL-1.6-GGUF.gguf \
  --alias paddleocr-vl-1.6 \
  --host 0.0.0.0 --port 8080 \
  --n-gpu-layers all \
  --flash-attn on \
  --parallel 1 \
  --temp 0
```

Nếu việc tự tìm projector không hoạt động, chỉ rõ URL:

```bash
./llama.cpp/llama-server \
  --hf-repo PaddlePaddle/PaddleOCR-VL-1.6-GGUF \
  --hf-file PaddleOCR-VL-1.6-GGUF.gguf \
  --mmproj-url https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF/resolve/main/PaddleOCR-VL-1.6-GGUF-mmproj.gguf \
  --alias paddleocr-vl-1.6 \
  --host 0.0.0.0 --port 8080 \
  --n-gpu-layers all --flash-attn on \
  --parallel 1 --temp 0
```

Ghi chú:

- `--n-gpu-layers all` offload tối đa lên GPU. Kiểm tra thiết bị bằng `./llama.cpp/llama-server --list-devices`.
- Nếu chỉ dùng CPU, thay bằng `--device none --n-gpu-layers 0` và có thể thêm `--threads "$(nproc)"`.
- `--parallel 1` dành context cho một request, thích hợp khi xử lý trang có nhiều token ảnh.
- OCR nên dùng `temperature=0` để kết quả ổn định.
- Không dùng `--no-mmproj`: projector là thành phần bắt buộc.
- Không cần đặt `--ctx-size`, `--image-min-tokens` hoặc `--image-max-tokens` ngay từ đầu; mặc định được đọc từ metadata model. Chỉ điều chỉnh sau khi đã kiểm thử chất lượng và bộ nhớ.

## 2. Tải model về máy và chạy cục bộ

Cách này ổn định hơn khi triển khai production hoặc cần sửa metadata cho tác vụ `Spotting:`.

```bash
python3 -m pip install -U 'huggingface_hub[cli]'
mkdir -p models/PaddleOCR-VL-1.6

hf download PaddlePaddle/PaddleOCR-VL-1.6-GGUF \
  PaddleOCR-VL-1.6-GGUF.gguf \
  PaddleOCR-VL-1.6-GGUF-mmproj.gguf \
  chat_template.jinja \
  --local-dir models/PaddleOCR-VL-1.6
```

Khởi động server:

```bash
./llama.cpp/llama-server \
  --model models/PaddleOCR-VL-1.6/PaddleOCR-VL-1.6-GGUF.gguf \
  --mmproj models/PaddleOCR-VL-1.6/PaddleOCR-VL-1.6-GGUF-mmproj.gguf \
  --alias paddleocr-vl-1.6 \
  --host 0.0.0.0 --port 8080 \
  --n-gpu-layers all --flash-attn on \
  --parallel 1 --temp 0
```

Có thể kiểm tra server:

```bash
curl http://127.0.0.1:8080/health
```

## 3. Phân tích tài liệu bằng PaddleOCR

Đây là cách sử dụng được model card chính thức khuyến nghị cho document parsing. Cài PaddlePaddle 3.2.1 trở lên và PaddleOCR 3.6.0 trở lên. Ví dụ sau dành cho CUDA 12.6:

```bash
python3 -m pip install \
  paddlepaddle-gpu==3.2.1 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python3 -m pip install -U 'paddleocr[doc-parser]>=3.6.0'
```

Với CPU hoặc CUDA khác, chọn gói theo trang cài đặt PaddlePaddle. Trên macOS, model card khuyên dùng Docker để dựng môi trường.

Sau khi `llama-server` ở mục 1 hoặc 2 đã chạy, gọi CLI:

```bash
paddleocr doc_parser \
  -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png \
  --pipeline_version v1.6 \
  --vl_rec_backend llama-cpp-server \
  --vl_rec_server_url http://127.0.0.1:8080/v1
```

Thay URL sau `-i` bằng đường dẫn ảnh hoặc tài liệu cần xử lý nếu pipeline hỗ trợ định dạng đó.

Python API:

```python
from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL(
    pipeline_version="v1.6",
    vl_rec_backend="llama-cpp-server",
    vl_rec_server_url="http://127.0.0.1:8080/v1",
)

output = pipeline.predict(
    "https://paddle-model-ecology.bj.bcebos.com/"
    "paddlex/imgs/demo_image/paddleocr_vl_demo.png"
)

for result in output:
    result.print()
    result.save_to_json(save_path="output")
    result.save_to_markdown(save_path="output")
```

Lưu ý URL phải có hậu tố `/v1`, vì PaddleOCR dùng API tương thích OpenAI của `llama-server`.

## 4. Nhận dạng trực tiếp bằng `llama-cli`

Model hỗ trợ sáu prompt tác vụ:


| Tác vụ                        | Prompt chính xác       |
| ----------------------------- | ---------------------- |
| OCR text                      | `OCR:`                 |
| Công thức                     | `Formula Recognition:` |
| Bảng                          | `Table Recognition:`   |
| Biểu đồ                       | `Chart Recognition:`   |
| Con dấu                       | `Seal Recognition:`    |
| Phát hiện và nhận dạng vị trí | `Spotting:`            |


Ví dụ OCR một ảnh:

```bash
./llama.cpp/llama-cli \
  --model models/PaddleOCR-VL-1.6/PaddleOCR-VL-1.6-GGUF.gguf \
  --mmproj models/PaddleOCR-VL-1.6/PaddleOCR-VL-1.6-GGUF-mmproj.gguf \
  --image test_image.jpg \
  --prompt 'OCR:' \
  --temp 0
```

Để nhận dạng bảng, chỉ cần đổi thành:

```bash
--prompt 'Table Recognition:'
```

Không tự viết lại hoặc dịch các prompt trên nếu muốn bám sát chế độ nhận dạng mà model được huấn luyện.

## 5. Gọi OpenAI-compatible API trực tiếp

Ví dụ với ảnh có URL công khai:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "paddleocr-vl-1.6",
    "messages": [{
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png"
          }
        },
        {"type": "text", "text": "OCR:"}
      ]
    }],
    "temperature": 0,
    "stream": false
  }'
```

Với ảnh cục bộ, mã hóa thành data URL:

```python
import base64
import json
import mimetypes
import sys
import urllib.request
from pathlib import Path

path = Path(sys.argv[1])
mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
data_url = "data:%s;base64,%s" % (
    mime,
    base64.b64encode(path.read_bytes()).decode(),
)

payload = {
    "model": "paddleocr-vl-1.6",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": "OCR:"},
        ],
    }],
    "temperature": 0,
    "stream": False,
}

request = urllib.request.Request(
    "http://127.0.0.1:8080/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=3600) as response:
    result = json.load(response)

print(result["choices"][0]["message"]["content"])
```

Chạy bằng:

```bash
python3 ocr.py test_image.jpg
```

Nếu server bật `--api-key SECRET`, thêm `Authorization: Bearer SECRET` vào request.

## 6. Cấu hình riêng cho `Spotting:`

Theo model card, chế độ `Spotting:` cần đặt `clip.vision.image_max_pixels` thành `1605632`. Lệnh này sửa trực tiếp file projector, vì vậy nên giữ một bản sao hoặc dùng một projector riêng cho spotting:

```bash
python3 -m pip install gguf
cp models/PaddleOCR-VL-1.6/PaddleOCR-VL-1.6-GGUF-mmproj.gguf \
   models/PaddleOCR-VL-1.6/PaddleOCR-VL-1.6-spotting-mmproj.gguf

python3 ./llama.cpp/gguf-py/gguf/scripts/gguf_set_metadata.py \
  models/PaddleOCR-VL-1.6/PaddleOCR-VL-1.6-spotting-mmproj.gguf \
  clip.vision.image_max_pixels 1605632 --force
```

Khởi động lại server với `PaddleOCR-VL-1.6-spotting-mmproj.gguf`, sau đó dùng prompt `Spotting:`. Muốn đưa projector về giá trị mặc định được model card nêu là `1003520`:

```bash
python3 ./llama.cpp/gguf-py/gguf/scripts/gguf_set_metadata.py \
  models/PaddleOCR-VL-1.6/PaddleOCR-VL-1.6-spotting-mmproj.gguf \
  clip.vision.image_max_pixels 1003520 --force
```

Không nhầm `clip.vision.image_max_pixels` trong metadata với option `--image-max-tokens`: một giá trị tính theo pixel, giá trị còn lại tính theo token ảnh.

## 7. Tối ưu và xử lý lỗi

### Không tải được projector

Dùng cặp `--model` và `--mmproj` cục bộ ở mục 2. Kiểm tra đúng tên file; repository chính thức dùng hậu tố `-GGUF-mmproj.gguf`.

### `unknown model architecture`, lỗi tensor hoặc lỗi vision

Cập nhật và build lại `llama.cpp`. Hỗ trợ PaddleOCR-VL-1.5/1.6 là tương đối mới; binary cũ có thể đọc GGUF nhưng chưa biết kiến trúc vision này.

### CUDA out of memory

Thử theo thứ tự:

1. Giữ `--parallel 1`.
2. Giảm `--ctx-size` nếu trước đó đã đặt quá cao.
3. Giảm `--ubatch-size`, ví dụ `--ubatch-size 256` hoặc `128`.
4. Thêm `--no-mmproj-offload` để chạy projector trên CPU.
5. Chạy một phần/toàn bộ model trên CPU bằng `--n-gpu-layers 0`.

Dung lượng hai file GGUF không phải toàn bộ VRAM cần dùng; token ảnh, KV cache và compute buffer cũng chiếm bộ nhớ.

### Kết quả thiếu chữ nhỏ hoặc sai bảng/công thức

- Dùng ảnh rõ, đúng chiều, crop viền thừa và deskew trước OCR.
- Mỗi trang nên là một ảnh riêng; không ghép nhiều trang thành một ảnh dài.
- Không giảm `--image-max-tokens` trước khi benchmark.
- Chọn đúng prompt (`Table Recognition:`, `Formula Recognition:`...) hoặc dùng pipeline document parser thay vì prompt `OCR:` cho mọi loại nội dung.
- OCR/VLM vẫn có thể bỏ sót hoặc hallucinate; luôn kiểm tra thủ công với dữ liệu quan trọng.

## 8. Nguồn

- GGUF và hướng dẫn chính thức: [https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF)
- Model gốc: [https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)
- PaddleOCR-VL pipeline: [https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
- Các option `llama-server` như `--hf-repo`, `--hf-file`, `--mmproj`, `--n-gpu-layers`, `--image-max-tokens` và `--media-path` được đối chiếu với [llama-cpp.md](llama-cpp.md) trong repository này.

