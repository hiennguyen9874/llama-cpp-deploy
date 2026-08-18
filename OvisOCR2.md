# Chạy OvisOCR2 bằng `llama.cpp`

OvisOCR2 là mô hình vision-language 0.8B dựa trên Qwen3.5-0.8B, dùng để chuyển một ảnh trang tài liệu thành Markdown theo thứ tự đọc tự nhiên. Mô hình có thể xuất văn bản, công thức LaTeX, bảng HTML và thẻ ảnh chứa bounding box. Model gốc công bố điểm 96.58 trên OmniDocBench v1.6 và Avg3 75.06 trên PureDocBench.

Tài liệu này dùng `llama-server` hiện có trong repository. Bản kiểm tra khi viết tài liệu là `llama.cpp` build 10480 (`01818e495`). Các README cũ có thể nhắc `llama-minicpmv-cli` hoặc `llama-llava-cli`; với build mới nên dùng `llama-server` hoặc `llama-mtmd-cli` và cơ chế multimodal `mtmd`.

## 1. Nên dùng bản nào?

Bốn repository được khảo sát không phải bốn model được huấn luyện khác nhau. Chúng đều dựa trên cùng model `ATH-MaaS/OvisOCR2`; khác nhau chủ yếu ở định dạng và cách lượng tử hóa.

| Repository | Nội dung | Điểm đáng chú ý |
| --- | --- | --- |
| `ATH-MaaS/OvisOCR2` | Model Transformers gốc, native BF16 | Nguồn chính thức và chất lượng chuẩn; không nạp trực tiếp bằng `llama.cpp` nếu chưa chuyển sang GGUF |
| `Abiray/OvisOCR2-GGUF` | BF16/F16, Q8, Q6, Q5, Q4, Q3 và projector F32/F16/BF16 | Bộ file đơn giản, đầy đủ, dễ dùng; card khuyên Q6_K hoặc Q5_K_M tùy tài nguyên |
| `bartowski/ATH-MaaS_OvisOCR2-GGUF` | GGUF lượng tử hóa bằng imatrix, nhiều mức từ BF16 đến IQ2 | Nhiều lựa chọn nhất; có các bản `*_L` giữ embedding/output ở Q8_0; card được tạo bằng llama.cpp b10068 |
| `prithivMLmods/OvisOCR2-F32-GGUF` | Có thêm model F32 và projector Q8_0 | Hữu ích nếu cần projector nhỏ; F32 3.02 GB thường không đem lại lợi ích thực tế tương xứng so với BF16/F16 |

### Khuyến nghị

1. **Ưu tiên độ chính xác OCR:** `bartowski ... Q6_K_L` + `mmproj ... f16`.
   - Model khoảng 0.73 GB, projector khoảng 0.21 GB.
   - `Q6_K_L` dùng Q8_0 cho embedding/output và được model card đánh dấu “near perfect, recommended”. Với model chỉ 0.8B, phần RAM tiết kiệm được khi hạ xuống Q3/Q2 không đáng để đánh đổi lỗi ký tự, bảng hoặc công thức.
2. **Cân bằng chất lượng/dung lượng:** `Q5_K_M` + projector F16.
   - Chọn bản Bartowski imatrix hoặc Abiray. Đây là mức mặc định hợp lý cho CPU/máy ít RAM.
3. **Chất lượng gần model gốc:** `Q8_0` hoặc BF16 + projector BF16/F16.
   - BF16 là precision native, khoảng 1.52 GB; Q8_0 khoảng 0.81 GB thường đã rất gần BF16.
   - Nên benchmark BF16/Q8 với chính tập tài liệu của mình nếu OCR số liệu, công thức hoặc bảng là tác vụ quan trọng.
4. **Chỉ dùng Q4 khi thật sự cần tiết kiệm:** ưu tiên `Q4_K_L` hoặc `Q4_K_M`. Tránh Q3/Q2 cho dữ liệu cần độ chính xác cao.
5. **Không khuyến nghị F32:** file lớn gấp đôi BF16 nhưng không khôi phục thông tin đã mất sau quá trình huấn luyện BF16 và tốn RAM/VRAM hơn.

Projector là bắt buộc. Projector F16 là lựa chọn mặc định tốt: nhỏ hơn F32 gần một nửa mà thường không cần đánh đổi đáng kể. Không trộn file model/projector từ nhiều repository nếu chưa kiểm tra; dùng cặp do cùng người chuyển đổi cung cấp là an toàn nhất.

> Các nhận xét chất lượng trên dựa vào loại quant và model card, không phải benchmark OCR đối đầu giữa ba bộ GGUF. Với tài liệu quan trọng, hãy tạo một tập kiểm thử gồm scan xấu, tiếng Việt, bảng và công thức rồi đo lại trước khi triển khai.

## 2. Chạy nhanh từ Hugging Face

`-hf/--hf-repo` của binary hiện tại chọn quant không phân biệt hoa thường và tự tải `mmproj` nếu repository có file phù hợp.

Cấu hình khuyến nghị, ưu tiên chất lượng:

```bash
./llama.cpp/llama-server \
  -hf bartowski/ATH-MaaS_OvisOCR2-GGUF:Q6_K_L \
  --mmproj-url https://huggingface.co/bartowski/ATH-MaaS_OvisOCR2-GGUF/resolve/main/mmproj-ATH-MaaS_OvisOCR2-f16.gguf \
  --alias ovisocr2 \
  --host 0.0.0.0 --port 8080 \
  --n-gpu-layers all \
  --flash-attn on \
  --ctx-size 32768 --parallel 1 \
  --batch-size 2048 --ubatch-size 512 \
  --reasoning off \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --temp 0
```

Nếu chỉ chạy CPU:

```bash
./llama.cpp/llama-server \
  -hf bartowski/ATH-MaaS_OvisOCR2-GGUF:Q5_K_M \
  --alias ovisocr2 \
  --host 127.0.0.1 --port 8080 \
  --device none --n-gpu-layers 0 \
  --ctx-size 32768 --parallel 1 \
  --threads "$(nproc)" \
  --reasoning off \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --temp 0
```

Ghi chú:

- Dùng `./llama.cpp/llama-server --list-devices` để xem tên thiết bị. Có thể thêm `--device CUDA0` nếu cần chọn GPU cụ thể.
- `--n-gpu-layers all` đưa tối đa layer lên GPU. Nếu OOM, để `--fit on` (mặc định), giảm `--ctx-size`, giảm `--ubatch-size`, hoặc dùng `--no-mmproj-offload` để chuyển projector sang CPU.
- `--parallel 1` dành toàn bộ context cho một yêu cầu và dễ dự đoán bộ nhớ hơn.
- OCR nên giải mã tất định với `temperature=0`; request API bên dưới cũng đặt lại giá trị này.
- Model gốc cho phép sinh tối đa 16.384 token trong ví dụ. Context 32K tạo khoảng trống cho token ảnh, prompt và output. Trang rất dài có thể cần context lớn hơn nếu metadata model hỗ trợ.
- Không đặt `--no-mmproj`: OvisOCR2 cần projector để đọc ảnh.

Nếu tự động chọn nhầm projector hoặc muốn triển khai hoàn toàn xác định, hãy tải file về máy và dùng cách ở mục tiếp theo.

## 3. Tải và chạy file cục bộ

Ví dụ dùng cặp Bartowski khuyến nghị:

```bash
python3 -m pip install -U 'huggingface_hub[cli]'
mkdir -p models/OvisOCR2

huggingface-cli download bartowski/ATH-MaaS_OvisOCR2-GGUF \
  ATH-MaaS_OvisOCR2-Q6_K_L.gguf \
  mmproj-ATH-MaaS_OvisOCR2-f16.gguf \
  --local-dir models/OvisOCR2
```

Khởi động server:

```bash
./llama.cpp/llama-server \
  --model models/OvisOCR2/ATH-MaaS_OvisOCR2-Q6_K_L.gguf \
  --mmproj models/OvisOCR2/mmproj-ATH-MaaS_OvisOCR2-f16.gguf \
  --alias ovisocr2 \
  --host 0.0.0.0 --port 8080 \
  --n-gpu-layers all --flash-attn on \
  --ctx-size 32768 --parallel 1 \
  --reasoning off \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --temp 0
```

Đổi sang Abiray nếu muốn tên file ngắn và quant tiêu chuẩn:

```bash
huggingface-cli download Abiray/OvisOCR2-GGUF \
  OvisOCR2-Q6_K.gguf mmproj-F16.gguf \
  --local-dir models/OvisOCR2-Abiray
```

## 4. Gọi OpenAI-compatible API với ảnh

Script sau mã hóa ảnh cục bộ thành data URL và gọi `/v1/chat/completions`. Không cần bật `--media-path` trên server.

```python
import base64
import json
import mimetypes
import sys
import urllib.request
from pathlib import Path

PROMPT = """Extract all readable content from the image in natural human reading order and output the result as a single Markdown document. For charts or images, represent them using an HTML image tag: <img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as HTML: <table>...</table>. Transcribe all other text as standard Markdown. Preserve the original text without translation or paraphrasing."""

path = Path(sys.argv[1])
mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
data_url = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()

payload = {
    "model": "ovisocr2",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": PROMPT},
        ],
    }],
    "temperature": 0,
    "max_tokens": 16384,
    "stream": False,
}

request = urllib.request.Request(
    "http://127.0.0.1:8080/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=3600) as response:
    result = json.load(response)

markdown = result["choices"][0]["message"]["content"].strip()
Path("output.md").write_text(markdown, encoding="utf-8")
print(markdown)
```

Chạy:

```bash
python3 ocr.py trang-tai-lieu.jpg
```

Nếu server có `--api-key SECRET`, thêm header:

```python
"Authorization": "Bearer SECRET"
```

Có thể bỏ câu yêu cầu thẻ `<img ...>` khỏi prompt nếu chỉ cần text/bảng/công thức. Không nên dịch hoặc rút gọn prompt: câu “Preserve the original text without translation or paraphrasing” giúp tránh model tự dịch nội dung OCR.

## 5. Ảnh đầu vào và vùng hình

Model gốc xử lý mỗi trang như một ảnh và cấu hình vLLM tham khảo giới hạn ảnh trong khoảng `448×448` đến `2880×2880` pixel. Đây là giới hạn pixel của pipeline gốc, **không tương đương trực tiếp** với `--image-min-tokens`/`--image-max-tokens` trong `llama.cpp`.

Khuyến nghị thực tế:

- Chuyển PDF thành từng ảnh trang; không ghép nhiều trang vào một ảnh.
- Giữ đúng chiều ảnh, crop viền dư và deskew trước OCR.
- Với ảnh rất lớn, resize sao cho cạnh dài không quá khoảng 2880 px để bắt đầu; đừng nén JPEG quá mạnh.
- Không đặt cứng `--image-min-tokens`/`--image-max-tokens` khi chưa cần; trước tiên dùng metadata/default của model. Nếu thiếu VRAM mới giảm `--image-max-tokens`, sau đó kiểm tra lại chữ nhỏ.

Các tọa độ trong thẻ:

```html
<img src="images/bbox_120_80_760_540.jpg" />
```

được scale về `[0, 1000)`, không phải pixel. Đổi về pixel bằng:

```text
x_pixel = round(x_scaled * image_width  / 1000)
y_pixel = round(y_scaled * image_height / 1000)
```

Nếu không muốn các thẻ vùng hình, có thể xóa những block bắt đầu bằng `<img src="images/bbox_` sau khi sinh. Model chỉ xuất tham chiếu/bounding box; ứng dụng phải tự crop ảnh gốc và lưu đúng đường dẫn nếu muốn Markdown render được.

## 6. Xử lý lỗi

### Server không tìm thấy hoặc không nạp projector

Dùng `--model` và `--mmproj` với hai đường dẫn cục bộ thay vì `-hf`. Kiểm tra log phải cho thấy model multimodal/projector đã được nạp. Không dùng `--no-mmproj`.

### `unknown model architecture`, lỗi tensor hoặc lỗi vision

OvisOCR2/Qwen3.5 và GGUF này cần bản `llama.cpp` mới. Cập nhật binary nếu build cũ hơn bản dùng để tạo quant (Bartowski ghi b10068), rồi thử lại. Không dùng hướng dẫn CLI cũ `llama-llava-cli`/`llama-minicpmv-cli` cho binary mới.

### CUDA OOM

Xử lý theo thứ tự:

1. Giữ `--parallel 1`.
2. Giảm `--ctx-size` từ 32768 xuống 24576 hoặc 16384; đồng thời giảm `max_tokens` trong request.
3. Giảm `--ubatch-size` xuống 256/128.
4. Thêm `--no-mmproj-offload` để projector chạy trên CPU.
5. Hạ Q6 xuống Q5/Q4.

Model weight dưới 2 GB nhưng projector, token ảnh, KV cache, compute buffer và context cũng dùng RAM/VRAM; không ước lượng bộ nhớ chỉ từ kích thước file GGUF.

### Output lặp ở cuối hoặc bị cắt

- Tăng `--ctx-size` và/hoặc `max_tokens` nếu dừng vì giới hạn token.
- Giữ `temperature=0` và tắt thinking.
- README chính thức có bước hậu xử lý xóa chuỗi lặp ở phần đuôi đối với output rất dài (thường từ khoảng 8.000 ký tự). Với pipeline production nên phát hiện đoạn ngắn lặp ít nhất năm lần ở cuối rồi cắt, nhưng phải lưu output thô để đối chiếu.
- Nếu lỗi xuất hiện thường xuyên ở quant thấp, thử Q6/Q8/BF16 trước khi thay sampler.

### OCR thiếu chữ nhỏ

Dùng ảnh rõ hơn, giảm crop viền dư, không giảm `--image-max-tokens` và thử Q8/BF16. Quant model không phải nguyên nhân duy nhất: độ phân giải ảnh và vision projector có thể ảnh hưởng nhiều hơn.

## 7. Nguồn

- Model gốc: <https://huggingface.co/ATH-MaaS/OvisOCR2>
- GGUF Abiray: <https://huggingface.co/Abiray/OvisOCR2-GGUF>
- GGUF Bartowski: <https://huggingface.co/bartowski/ATH-MaaS_OvisOCR2-GGUF>
- GGUF Prithiv: <https://huggingface.co/prithivMLmods/OvisOCR2-F32-GGUF>
- Các argument `llama-server` được đối chiếu với `llama-cpp.md` trong repository này.

OvisOCR2 vẫn có thể trả kết quả sai, thiếu nội dung, hỏng cấu trúc bảng/công thức hoặc sai thứ tự đọc. Luôn kiểm tra thủ công khi dùng cho hồ sơ, tài chính, y tế hoặc dữ liệu quan trọng.
