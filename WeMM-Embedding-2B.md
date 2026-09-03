# Chạy WeMM-Embedding-2B với llama.cpp

WeMM-Embedding-2B của Tencent là mô hình embedding đa phương thức 2B, được xây dựng từ Qwen3.5-2B (kiến trúc `qwen3_5`, **không phải** Qwen3-VL). Mô hình đưa văn bản, ảnh, video, visual document hoặc nội dung trộn nhiều modality vào cùng một không gian vector. Audio không được hỗ trợ.

Thông số chính từ model card chính thức (`tencent/WeMM-Embedding-2B`):

- embedding: 2.048 chiều, đã L2-normalize sẵn;
- hỗ trợ Matryoshka Representation Learning (MRL) với các chiều `64, 128, 256, 512, 1024, 2048` (`model.config.matryoshka_dimensions`); bản 256 chiều giữ 98,7% hiệu năng image+video MMEB-v2;
- context tối đa: 262.144 token (`qwen35.context_length` trong GGUF);
- ngôn ngữ chính: tiếng Trung và tiếng Anh;
- pooling là last-token tại vị trí token `<embedding>` (xem mục 5);
- license: Apache-2.0.

Model card báo MMEB-v2 tổng là `77.9` (Image `79.6`, Video `70.8`, VisDoc `80.7`), cao hơn Qwen3-VL-Embedding-2B (`73.2`). Đây là mô hình embedding, không phải model chat/generation.

## 1. Yêu cầu

Dùng bản llama.cpp mới có hỗ trợ kiến trúc Qwen3.5, embedding và multimodal projector (`mmproj`). Bản llama.cpp trong repository này (`01818e495`) đã có `src/models/qwen35.cpp`, tức hỗ trợ text arch `qwen35`. Kiểm tra binary và thiết bị:

```bash
./llama.cpp/llama-server --version
./llama.cpp/llama-server --list-devices
```

Các option liên quan trong [llama-cpp.md](llama-cpp.md):

- `--embedding`/`--embeddings`: chỉ bật use case embedding;
- `--pooling last`: lấy hidden state của token cuối (`<embedding>`) làm vector; GGUF đã gắn metadata `qwen35.pooling_type=3` (last-token) nên có thể bỏ option này để dùng default, nhưng ghi rõ giúp tránh nhầm;
- `--embd-normalize 2`: chuẩn hóa L2, đây cũng là mặc định;
- `-hf REPO:QUANT`: tải GGUF từ Hugging Face;
- `--mmproj`, `--mmproj-url`: chỉ định vision projector;
- `--image-min-tokens`, `--image-max-tokens`: giới hạn token của ảnh có dynamic resolution;
- `--mtmd-batch-max-tokens`: số image token tối đa mỗi batch, mặc định `1024`.

Lưu ý quan trọng:

- `/v1/embeddings` yêu cầu pooling khác `none`. Không dùng `--pooling none` với `/v1/embeddings`.
- Đừng nhầm WeMM với Qwen3-VL: text arch là `qwen35`, vision projector đi kèm lại khai báo `clip.projector_type=qwen3vl_merger` (đã kiểm tra trực tiếp GGUF header). Đường text embedding hoạt động theo metadata chuẩn; đường multimodal (ảnh/video) phụ thuộc `libmtmd` của bản binary đang dùng — hãy xác nhận model có capability `multimodal` qua `GET /v1/models` sau khi khởi động, và cập nhật llama.cpp nếu projector không load.

## 2. Chọn GGUF

Repository `[DreamBlooms/WeMM-Embedding-2B-GGUF](https://huggingface.co/DreamBlooms/WeMM-Embedding-2B-GGUF)` là bản quantize của model chính thức, gồm model và projector riêng (kích thước đo từ Content-Length thực tế):

| File | Dung lượng | Gợi ý |
| ---- | ---------: | ----- |
| `WeMM-Embedding-2B-Q4_K_M.gguf` | ~1,56 GB | nhẹ nhất, dùng cho CPU/RAM hạn chế |
| `WeMM-Embedding-2B-Q8_0.gguf` | ~2,55 GB | chất lượng quant cao, lựa chọn mặc định của tài liệu này |
| `WeMM-Embedding-2B-BF16.gguf` | ~4,79 GB | gần như nguyên bản, thường không cần thiết |
| `mmproj-WeMM-Embedding-2B-BF16.gguf` | ~671 MB | projector duy nhất của repo, bắt buộc cho ảnh/video |

Repo GGUF cũng ghi rõ metadata `qwen35.pooling_type=3` đã được inject, và GGUF đã chứa đầy đủ special token (`<embedding>` id `248077`, `<|image_pad|>` `248056`, `<|video_pad|>` `248057`, `<|vision_start|>` `248053`, `<|vision_end|>` `248054` — đã kiểm tra trực tiếp vocabulary trong file Q8_0).

## 3. Chạy server trên GPU

```bash
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/llama-server \
  -hf DreamBlooms/WeMM-Embedding-2B-GGUF:Q8_0 \
  --mmproj-url https://huggingface.co/DreamBlooms/WeMM-Embedding-2B-GGUF/resolve/main/mmproj-WeMM-Embedding-2B-BF16.gguf \
  --alias WeMM-Embedding-2B \
  --embedding \
  --pooling last \
  --embd-normalize 2 \
  --host 0.0.0.0 --port 8001 \
  --ctx-size 32768 \
  --parallel 1 \
  --flash-attn on \
  --n-gpu-layers all \
  --device CUDA0 \
  --api-key llama-cpp-api-key
```

Ghi chú:

- Model hỗ trợ tới 262.144 token, nhưng `--ctx-size 32768` là điểm khởi đầu hợp lý; chỉ tăng khi VRAM cho phép vì attention và image token tốn bộ nhớ nhanh.
- `CUDA_VISIBLE_DEVICES=0` làm GPU đã chọn xuất hiện dưới tên `CUDA0` trong tiến trình. Luôn kiểm tra tên backend thực tế bằng `--list-devices`.
- Nếu thiếu VRAM: đổi model sang `Q4_K_M`, giảm `--ctx-size` xuống `8192`, hoặc thêm `--no-mmproj-offload` để giữ projector trên CPU.

## 4. Chạy chỉ bằng CPU

```bash
./llama.cpp/llama-server \
  -hf DreamBlooms/WeMM-Embedding-2B-GGUF:Q4_K_M \
  --mmproj-url https://huggingface.co/DreamBlooms/WeMM-Embedding-2B-GGUF/resolve/main/mmproj-WeMM-Embedding-2B-BF16.gguf \
  --alias WeMM-Embedding-2B \
  --embedding --pooling last --embd-normalize 2 \
  --host 0.0.0.0 --port 8001 \
  --ctx-size 32768 \
  --parallel 1 \
  --device none \
  --threads "$(nproc)" \
  --api-key llama-cpp-api-key
```

Nếu thiếu RAM, giảm `--ctx-size` xuống `8192` hoặc `16384`.

## 5. Embedding văn bản qua API tương thích OpenAI

Theo model card và code chính thức (`modeling_wemm_embedding.py`), embedding được lấy tại **vị trí token cuối cùng** (token `<embedding>` được append sau message) rồi L2-normalize. Template embedding (`embedding_chat_template.jinja` / named template `sentence_transformers` trong GGUF) render message rồi thêm `<embedding>`, với `add_generation_prompt=false`.

Điểm khác biệt quan trọng so với Qwen3-VL-Embedding: **WeMM không dùng system instruction** kiểu `Represent the user's input.`. Config sentence-transformers của model gốc có `prompts: {}`, tức query và document dùng chung một template; với document dạng media, model card dùng text đi kèm như `Represent this image.` / `Represent this video.` đặt **sau** ảnh/video.

### Vì sao phải tự format input?

`/v1/embeddings` (và `/embedding`) nhận `input`/`content` rồi tokenize trực tiếp; đường xử lý này không gọi chat-template engine. Ngoài ra, template **mặc định** (`tokenizer.chat_template`) trong GGUF là template chat generative của Qwen3.5 (có `<think>`), **không phải** template embedding — template embedding nằm ở named template `sentence_transformers`. Vì vậy với WeMM, cách đúng và đơn giản nhất là gửi chuỗi đã format thủ công, kết thúc bằng `<embedding>`:

```bash
curl http://localhost:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer llama-cpp-api-key' \
  -d @- <<'JSON'
{
  "model": "WeMM-Embedding-2B",
  "input": "<|im_start|>user\nFollow the white rabbit.<|im_end|><embedding>",
  "encoding_format": "float"
}
JSON
```

Có thể gửi batch bằng mảng `input`:

```json
{
  "model": "WeMM-Embedding-2B",
  "input": [
    "<|im_start|>user\nFirst text<|im_end|><embedding>",
    "<|im_start|>user\nSecond text<|im_end|><embedding>"
  ],
  "encoding_format": "float"
}
```

**Không được bỏ `<embedding>` ở cuối**: pooling last-token lấy hidden state của token cuối; thiếu token này vector sẽ tương ứng vị trí `<|im_end|>` và chất lượng retrieval giảm. Với retrieval, giữ cách format nhất quán giữa lúc lập chỉ mục và lúc truy vấn; nếu cần instruction cho query (ví dụ tiền tố mô tả task), phải áp dụng giống nhau cho toàn bộ query.

## 6. Embedding ảnh và text + ảnh

API OpenAI `/v1/embeddings` chuẩn chỉ mô tả input văn bản. llama.cpp hỗ trợ multimodal embedding qua endpoint riêng `/embedding`, với `content.prompt_string` chứa media marker và `content.multimodal_data` chứa dữ liệu base64 theo đúng thứ tự marker.

Theo template embedding của model gốc, **media đặt trước text** trong message user: ảnh render thành `<|vision_start|><|image_pad|><|vision_end|>`, video render thành `<|video_pad|>` trần.

**Không hard-code `<__media__>` trên llama.cpp mới.** Lấy marker thực tế từ `GET /props` (không cần bật option `--props` cho GET). Các ví dụ dưới đây cần `jq`:

```bash
MEDIA_MARKER="$(curl -fsS http://localhost:8001/props \
  -H 'Authorization: Bearer llama-cpp-api-key' | jq -er '.media_marker')"
printf 'Media marker: %s\n' "$MEDIA_MARKER"
```

Lấy lại marker sau mỗi lần restart server.

### Chỉ ảnh

```bash
IMAGE_B64="$(base64 -w 0 scripts/test.png)"
MEDIA_MARKER="$(curl -fsS http://localhost:8001/props \
  -H 'Authorization: Bearer llama-cpp-api-key' | jq -er '.media_marker')"

curl http://localhost:8001/embedding \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer llama-cpp-api-key' \
  -d @- <<JSON
{
  "content": {
    "prompt_string": "<|im_start|>user\n${MEDIA_MARKER}<|im_end|><embedding>",
    "multimodal_data": ["$IMAGE_B64"]
  },
  "embd_normalize": 2
}
JSON
```

### Text + ảnh

```bash
IMAGE_B64="$(base64 -w 0 scripts/test.png)"
MEDIA_MARKER="$(curl -fsS http://localhost:8001/props \
  -H 'Authorization: Bearer llama-cpp-api-key' | jq -er '.media_marker')"

curl http://localhost:8001/embedding \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer llama-cpp-api-key' \
  -d @- <<JSON
{
  "content": {
    "prompt_string": "<|im_start|>user\n${MEDIA_MARKER}Represent this image.<|im_end|><embedding>",
    "multimodal_data": ["$IMAGE_B64"]
  },
  "embd_normalize": 2
}
JSON
```

Mỗi media marker phải có đúng một phần tử base64 tương ứng. Không viết marker thành `<\__media__>` hoặc `<\_...>`: `\_` không phải escape hợp lệ trong JSON và sẽ gây `json.exception.parse_error.101`. Dùng đúng chuỗi mà `/props` trả về, không thêm backslash. Trước khi gửi multimedia, gọi `/v1/models` và kiểm tra model có capability `multimodal`. Trên macOS, thay `base64 -w 0` bằng `base64 < scripts/test.png | tr -d '\n'`.

Model gốc hỗ trợ cả video (`fps: 2`, `min_frames: 4`, `max_frames: 768` trong `processor_config.json`), nhưng khả năng video thực tế còn phụ thuộc phiên bản `libmtmd`/llama.cpp và projector `qwen3vl_merger` đi kèm; nên xác nhận trên bản binary đang triển khai trước khi đưa vào production.

## 7. Kích thước vector và tính similarity

Server trả vector 2.048 chiều và chuẩn hóa L2 khi dùng `--embd-normalize 2` (model gốc cũng L2-normalize trong `model.embedding()`). Vì vậy cosine similarity có thể tính trực tiếp bằng dot product:

```python
score = sum(a * b for a, b in zip(query_embedding, document_embedding))
```

Model hỗ trợ MRL với các chiều trong `matryoshka_dimensions` (`64, 128, 256, 512, 1024, 2048`), nhưng help hiện tại của llama.cpp không có option server để đặt output dimension. Nếu client cần vector ngắn hơn, lấy `N` chiều đầu rồi chuẩn hóa L2 lại trước khi lưu hoặc so sánh; dùng cùng một `N` cho toàn bộ index và query:

```python
import math
short = emb[:256]
n = math.sqrt(sum(x * x for x in short))
short = [x / n for x in short]
```

Không nhầm quantization của trọng số GGUF với quantization hậu xử lý của vector embedding: llama.cpp mặc định trả vector float qua API.

## 8. Xử lý lỗi thường gặp

- **`/v1/embeddings` báo pooling `none`:** bỏ `--pooling none`, dùng `--pooling last` (GGUF đã có `qwen35.pooling_type=3`, cũng có thể bỏ option để dùng default).
- **Vector chất lượng retrieval kém:** kiểm tra có token `<embedding>` ở cuối input chưa; không dùng system instruction kiểu Qwen3-VL-Embedding; không dùng `/apply-template` mặc định vì default template của GGUF này là template chat generative, không phải template embedding.
- **`number of media markers ... does not match number of bitmaps`:** lấy lại `.media_marker` từ `/props`; số marker trong `prompt_string` phải bằng số phần tử trong `multimodal_data`.
- **`forbidden character after backslash` gần `<\_`:** JSON không hỗ trợ escape `\_`; bỏ các backslash khỏi marker.
- **Không nhận ảnh/báo thiếu projector:** cập nhật llama.cpp và chỉ định đúng `--mmproj-url` tới `mmproj-WeMM-Embedding-2B-BF16.gguf`; không dùng `--no-mmproj`. Nếu vẫn lỗi, đây có thể là giới hạn hỗ trợ multimodal của Qwen3.5 trên bản llama.cpp hiện tại — đường text embedding vẫn hoạt động độc lập với projector.
- **CUDA out of memory:** giảm `--ctx-size`, dùng quant `Q4_K_M`, hoặc `--no-mmproj-offload` để giữ projector trên CPU.
- **Tải model lỗi:** đặt `HF_TOKEN`/`--hf-token` nếu cần; các repository nêu trên hiện là public.
- **Chạy offline:** tải/cache model và projector trước, sau đó thêm `--offline`.

## Nguồn

- [tencent/WeMM-Embedding-2B](https://huggingface.co/tencent/WeMM-Embedding-2B)
- [DreamBlooms/WeMM-Embedding-2B-GGUF](https://huggingface.co/DreamBlooms/WeMM-Embedding-2B-GGUF)
- Help của binary trong repository: [llama-cpp.md](llama-cpp.md)
