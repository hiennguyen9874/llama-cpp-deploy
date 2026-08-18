# Chạy LFM2.5-VL-3B với llama.cpp

LFM2.5-VL-3B là mô hình vision-language 3.1B của Liquid AI, gồm backbone LFM2.5-2.6B và vision encoder SigLIP2 NaFlex 400M. Mô hình hỗ trợ văn bản, nhiều ảnh, OCR, grounding/object detection và 16 ngôn ngữ (có tiếng Việt), với context tối đa 32.768 token.

Mô hình phù hợp với tác vụ một lượt, cần độ trễ thấp như hỏi đáp ảnh, OCR tài liệu, đọc biểu đồ và dịch biển báo. Không nên dùng cho suy luận dài hoặc câu hỏi kỹ thuật phức tạp về bản vẽ.

## 1. Yêu cầu

Dùng bản `llama.cpp` mới có hỗ trợ LFM2.5-VL và multimodal projector (`mmproj`). Kiểm tra binary:

```bash
./llama.cpp/llama-server --version
./llama.cpp/llama-server --list-devices
```

Các binary trong repository này nằm tại:

```text
./llama.cpp/llama-cli
./llama.cpp/llama-server
```

Nếu cần build lại với CUDA, xem phần cài đặt trong [README.md](README.md). Khi dùng `-hf/--hf-repo`, llama.cpp tự tải model và `mmproj` từ Hugging Face; không cần tải thủ công.

## 2. Chọn bản GGUF

Hướng dẫn này mặc định sử dụng GGUF của Unsloth vì repository cung cấp nhiều mức quantization:

```text
unsloth/LFM2.5-VL-3B-GGUF
```

Khuyến nghị bắt đầu với `Q4_K_M` để cân bằng chất lượng, tốc độ và bộ nhớ. Unsloth còn cung cấp các bản như `IQ4_XS`, `Q3_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0` và các quant `UD-*`.

Repository GGUF chính thức của LiquidAI vẫn có thể dùng thay thế:

```text
LiquidAI/LFM2.5-VL-3B-GGUF
```

Cú pháp `:Q4_K_M` chọn quantization. Nếu bỏ phần này, llama.cpp mặc định tìm `Q4_K_M`. `mmproj` được phát hiện và tải tự động; chỉ dùng `--no-mmproj` khi muốn tắt xử lý ảnh.

## 3. Chạy bằng llama-cli

### Chat tương tác

```bash
./llama.cpp/llama-cli \
  -hf unsloth/LFM2.5-VL-3B-GGUF:Q6_K_XL \
  --ctx-size 32768 \
  --temp 0.2 \
  --top-k 50 \
  --repeat-penalty 1.0 \
  -n 1024
```

Trong phiên chat, thêm ảnh rồi đặt câu hỏi:

```text
/image /duong/dan/toi/anh.jpg
Hãy mô tả chi tiết ảnh này.
```

### Gửi ảnh ngay từ command line

```bash
./llama.cpp/llama-cli \
  -hf unsloth/LFM2.5-VL-3B-GGUF:Q6_K_XL \
  --image scripts/test.png \
  -p "Đọc toàn bộ chữ trong ảnh và giữ nguyên bố cục hợp lý." \
  --ctx-size 32768 \
  --temp 0.2 \
  --top-k 50 \
  --repeat-penalty 1.0 \
  -n 2048
```

Có thể truyền nhiều ảnh bằng danh sách phân cách bởi dấu phẩy:

```bash
--image page-1.png,page-2.png
```

### Dùng GGUF chính thức của LiquidAI

Nếu muốn dùng repository chính thức, chỉ cần đổi `-hf`:

```bash
./llama.cpp/llama-cli \
  -hf LiquidAI/LFM2.5-VL-3B-GGUF:Q4_K_M \
  --image scripts/test.png \
  -p "Mô tả ảnh này bằng tiếng Việt." \
  --ctx-size 32768 \
  --temp 0.2 --top-k 50 --repeat-penalty 1.0 \
  -n 1024
```

## 4. Chạy OpenAI-compatible server

### GPU

```bash
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/llama-server \
  -hf unsloth/LFM2.5-VL-3B-GGUF:Q6_K_XL \
  --alias LFM2.5-VL-3B \
  --host 0.0.0.0 --port 8000 \
  --ctx-size 32768 \
  --parallel 1 \
  --n-gpu-layers all \
  --flash-attn on \
  --temp 0.2 \
  --top-k 50 \
  --repeat-penalty 1.0 \
  --api-key llama-cpp-api-key
```

`CUDA_VISIBLE_DEVICES=0` làm GPU đó xuất hiện dưới tên `CUDA0` đối với tiến trình. Nếu cần chỉ định backend rõ ràng, kiểm tra tên bằng `--list-devices`, sau đó thêm ví dụ `--device CUDA0`.

### Chỉ dùng CPU

```bash
./llama.cpp/llama-server \
  -hf unsloth/LFM2.5-VL-3B-GGUF:Q6_K_XL \
  --alias LFM2.5-VL-3B \
  --host 0.0.0.0 --port 8000 \
  --ctx-size 32768 \
  --parallel 1 \
  --device none \
  --threads "$(nproc)" \
  --temp 0.2 \
  --top-k 50 \
  --repeat-penalty 1.0 \
  --api-key llama-cpp-api-key
```

Nếu thiếu RAM/VRAM, giảm `--ctx-size` xuống `8192` hoặc `16384`, chọn quant nhỏ hơn, hoặc giảm số layer offload thay vì dùng `--n-gpu-layers all`.

## 5. Gọi API với ảnh

`/v1/chat/completions` nhận nội dung multimodal theo định dạng OpenAI. `image_url.url` có thể là URL từ xa, data URI base64 hoặc đường dẫn local đã được server cho phép.

### Ảnh từ URL

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer llama-cpp-api-key' \
  -d '{
    "model": "LFM2.5-VL-3B",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {"url": "https://placecats.com/300/200"}
          },
          {
            "type": "text",
            "text": "Mô tả ảnh này bằng tiếng Việt."
          }
        ]
      }
    ],
    "temperature": 0.2,
    "top_k": 50,
    "repeat_penalty": 1.0,
    "max_tokens": 1024
  }'
```

### Ảnh local

Khởi động server với thư mục media được phép:

```bash
mkdir -p images
cp scripts/test.png images/test.png

./llama.cpp/llama-server \
  -hf unsloth/LFM2.5-VL-3B-GGUF:Q6_K_XL \
  --alias LFM2.5-VL-3B \
  --host 0.0.0.0 --port 8000 \
  --media-path "$PWD/images" \
  --ctx-size 32768 \
  --n-gpu-layers all --flash-attn on
```

Trong request, dùng đường dẫn tương đối với `--media-path`:

```json
{
  "type": "image_url",
  "image_url": {"url": "file://test.png"}
}
```

Không mở `--media-path` tới thư mục chứa dữ liệu nhạy cảm khi server có thể được truy cập từ mạng.

## 6. OCR tài liệu có thông tin bố cục

LFM2.5-VL-3B có thể trả về từng vùng tài liệu cùng nhãn và bounding box. Tọa độ là số nguyên đã chuẩn hóa trong khoảng `[0, 1000]`:

```text
image_index=<n> <label> [xmin, ymin, xmax, ymax]
<content>
```

Prompt ngắn để thử:

```text
Parse this document into its layout regions. For every region in reading order,
return image_index, label, normalized bounding box [xmin, ymin, xmax, ymax],
and content. Separate regions with one blank line. Return only parsed regions.
```

Các nhãn có thể gồm `text`, `title`, `list`, `table`, `image`, `chart`, `equation`, `code`, `page_header`, `page_footer`, `page_number`, v.v. Định dạng layout annotation hiện vẫn mang tính thử nghiệm.

## 7. Tham số đáng chú ý

- `--ctx-size`: context của model; tối đa 32.768 token. Context lớn dùng thêm RAM/VRAM.
- `--image-min-tokens`, `--image-max-tokens`: giới hạn token cho mỗi ảnh động; mặc định đọc từ model.
- `--mtmd-batch-max-tokens`: số image token tối đa mỗi batch khi encode ảnh; mặc định `1024`.
- `--mmproj-offload` / `--no-mmproj-offload`: bật/tắt offload vision projector lên GPU; mặc định bật.
- `--n-gpu-layers all`: offload tối đa layer lên GPU; bỏ tùy chọn này hoặc giảm giá trị khi thiếu VRAM.
- `--parallel 1`: một slot server, giúp dự đoán mức dùng context/bộ nhớ dễ hơn.
- Sampling được model card khuyến nghị: `temperature=0.2`, `top_k=50`, `repetition_penalty=1.0`.

## 8. Xử lý lỗi thường gặp

- **Không nhận ảnh hoặc báo thiếu projector:** cập nhật llama.cpp; không dùng `--no-mmproj`. Có thể chỉ định thủ công bằng `--mmproj <file>` hoặc `--mmproj-url <URL>`.
- **CUDA out of memory:** giảm `--ctx-size`, dùng quant nhỏ hơn, giảm `--n-gpu-layers`, hoặc thêm `--no-mmproj-offload` để giữ projector trên CPU.
- **Request ảnh local bị từ chối:** thêm `--media-path` và dùng `file://` với đường dẫn tương đối nằm trong thư mục đó.
- **Tải model thất bại do repo có quyền truy cập:** đặt token bằng `HF_TOKEN` hoặc truyền `--hf-token`; hai repo nêu trên hiện là public.
- **Muốn chạy hoàn toàn offline:** chạy một lần để model được cache, sau đó thêm `--offline`.

## Nguồn

- [LiquidAI/LFM2.5-VL-3B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-VL-3B-GGUF)
- [unsloth/LFM2.5-VL-3B-GGUF](https://huggingface.co/unsloth/LFM2.5-VL-3B-GGUF)
- [LFM2.5-VL-3B model card](https://huggingface.co/LiquidAI/LFM2.5-VL-3B)
- Help của binary trong repository: [llama-cpp.md](llama-cpp.md)

Lưu ý giấy phép: model dùng giấy phép `LFM1.0`; hãy đọc `LICENSE` trong repository model trước khi phân phối hoặc triển khai thương mại.
