# Chạy Qwen3-VL-Embedding-2B với llama.cpp

Qwen3-VL-Embedding-2B là mô hình embedding đa phương thức 2B, được xây dựng từ Qwen3-VL-2B-Instruct. Mô hình đưa văn bản, ảnh, screenshot, video hoặc nội dung trộn nhiều modality vào cùng một không gian vector.

Thông số chính từ model card chính thức:

- context tối đa: 32.768 token;
- embedding đầy đủ: 2.048 chiều;
- hỗ trợ Matryoshka Representation Learning (MRL), có thể rút gọn còn 64–2.048 chiều;
- hơn 30 ngôn ngữ;
- instruction-aware: instruction phù hợp thường cải thiện kết quả khoảng 1–5%; Qwen khuyến nghị viết instruction bằng tiếng Anh;
- license: Apache-2.0.

Model card báo điểm tổng MMEB-V2 là `73.2` và MMTEB Mean (Task) là `63.87`. Đây là mô hình embedding, không phải model chat/generation.

## 1. Yêu cầu

Dùng bản llama.cpp mới có hỗ trợ kiến trúc Qwen3-VL, embedding và multimodal projector (`mmproj`). Kiểm tra binary và thiết bị:

```bash
./llama.cpp/llama-server --version
./llama.cpp/llama-server --list-devices
```

Các option liên quan trong [llama-cpp.md](llama-cpp.md):

- `--embedding`/`--embeddings`: chỉ bật use case embedding;
- `--pooling last`: lấy hidden state của token cuối làm vector cho Qwen embedding;
- `--embd-normalize 2`: chuẩn hóa L2, đây cũng là mặc định;
- `-hf REPO:QUANT`: tải GGUF từ Hugging Face;
- `--mmproj`, `--mmproj-url`: chỉ định vision projector;
- `--mmproj-offload`/`--no-mmproj-offload`: đặt projector trên GPU/CPU;
- `--image-min-tokens`, `--image-max-tokens`: giới hạn token của ảnh có dynamic resolution;
- `--mtmd-batch-max-tokens`: số image token tối đa mỗi batch, mặc định `1024`.

Lưu ý quan trọng: `/v1/embeddings` yêu cầu pooling khác `none`. Cấu hình cũ `--pooling none` trả embedding cho từng token và chỉ phù hợp với endpoint không tương thích OpenAI `/embeddings`; không dùng cấu hình đó với `/v1/embeddings`.

## 2. Chọn GGUF

### mradermacher static quants

Repository `[mradermacher/Qwen3-VL-Embedding-2B-GGUF](https://huggingface.co/mradermacher/Qwen3-VL-Embedding-2B-GGUF)` có model và projector trong cùng repo:


| Quant           | Dung lượng model | Gợi ý                                                  |
| --------------- | ---------------- | ------------------------------------------------------ |
| Q4_K_S / Q4_K_M | 1,2 GB           | nhanh, được model card khuyến nghị                     |
| Q5_K_M          | 1,4 GB           | chất lượng cao hơn Q4                                  |
| Q6_K            | 1,5 GB           | chất lượng rất tốt, lựa chọn mặc định của tài liệu này |
| Q8_0            | 1,9 GB           | nhanh, chất lượng quant cao nhất                       |
| f16             | 3,5 GB           | thường không cần thiết                                 |


Projector có hai lựa chọn: `mmproj-Q8_0` khoảng 0,5 GB và `mmproj-f16` khoảng 0,9 GB. Projector nằm trong static repo ngay cả khi model lấy từ repo imatrix.

### mradermacher imatrix quants

`[mradermacher/Qwen3-VL-Embedding-2B-i1-GGUF](https://huggingface.co/mradermacher/Qwen3-VL-Embedding-2B-i1-GGUF)` cung cấp weighted/imatrix quants từ IQ1 đến Q6. Các lựa chọn đáng chú ý:

- `i1-Q4_K_S`: cân bằng kích thước/tốc độ/chất lượng;
- `i1-Q4_K_M`: nhanh, được khuyến nghị;
- `i1-Q6_K`: khoảng 1,5 GB, gần chất lượng static Q6_K;
- tránh IQ1/Q2 trừ khi bộ nhớ rất hạn chế.

Repo imatrix không chứa projector; phải lấy `mmproj` từ static repo như command bên dưới.

### DevQuasar

`[DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF](https://huggingface.co/DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF)` cũng có Q2_K–Q8_0, f16 và `mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf`. Model card chỉ xác nhận đây là bản quantize của model chính thức và không đưa hướng dẫn/chỉ số riêng. Tài liệu này dùng bản mradermacher vì model card liệt kê rõ kích thước và khuyến nghị cho từng quant.

## 3. Chạy server trên GPU

```bash
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/llama-server \
  -hf mradermacher/Qwen3-VL-Embedding-2B-i1-GGUF:i1-Q6_K \
  --mmproj-url https://huggingface.co/mradermacher/Qwen3-VL-Embedding-2B-GGUF/resolve/main/Qwen3-VL-Embedding-2B.mmproj-f16.gguf \
  --alias Qwen3-VL-Embedding-2B \
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

Có thể thay projector f16 bằng bản Q8_0 để giảm bộ nhớ:

```bash
--mmproj-url https://huggingface.co/mradermacher/Qwen3-VL-Embedding-2B-GGUF/resolve/main/Qwen3-VL-Embedding-2B.mmproj-Q8_0.gguf
```

Nếu dùng static quant, đổi phần model thành:

```bash
-hf mradermacher/Qwen3-VL-Embedding-2B-GGUF:Q6_K
```

Vì model và `mmproj` cùng repo, llama.cpp có thể tự tìm projector khi dùng `-hf`. Tuy nhiên, chỉ định `--mmproj-url` rõ ràng giúp chọn chính xác f16 hay Q8_0.

`CUDA_VISIBLE_DEVICES=0` làm GPU đã chọn xuất hiện dưới tên `CUDA0` trong tiến trình. Luôn kiểm tra tên backend thực tế bằng `--list-devices`.

## 4. Chạy chỉ bằng CPU

```bash
./llama.cpp/llama-server \
  -hf mradermacher/Qwen3-VL-Embedding-2B-GGUF:Q4_K_M \
  --mmproj-url https://huggingface.co/mradermacher/Qwen3-VL-Embedding-2B-GGUF/resolve/main/Qwen3-VL-Embedding-2B.mmproj-Q8_0.gguf \
  --alias Qwen3-VL-Embedding-2B \
  --embedding --pooling last --embd-normalize 2 \
  --host 0.0.0.0 --port 8001 \
  --ctx-size 32768 \
  --parallel 1 \
  --device none \
  --threads "$(nproc)" \
  --api-key llama-cpp-api-key
```

Nếu thiếu RAM/VRAM, giảm `--ctx-size` xuống `8192` hoặc `16384`, dùng quant nhỏ hơn, chọn projector Q8_0, hoặc thêm `--no-mmproj-offload` để giữ projector trên CPU.

## 5. Embedding văn bản qua API tương thích OpenAI

Theo model card chính thức, input mặc định phải được bọc bằng system instruction `Represent the user's input.` và chat template có generation prompt.

### Vì sao llama.cpp không tự áp dụng template?

Checkpoint Hugging Face **có** `chat_template.jinja`; template này thêm system prompt mặc định nói trên và, khi `add_generation_prompt=true`, kết thúc bằng `<|im_start|>assistant\n`. Converter llama.cpp đọc cả file Jinja độc lập này, và GGUF mradermacher cũng có metadata `tokenizer.chat_template` chứa đúng template đó. Vì vậy đây không phải lỗi thiếu template khi convert/quantize.

Tuy nhiên, template trong GGUF không đồng nghĩa mọi endpoint đều tự dùng nó. Trong phiên bản llama.cpp được kiểm tra (`01818e495`, build 10480):

- `/v1/chat/completions` nhận `messages`, gọi `oaicompat_chat_params_parse()`, rồi `common_chat_templates_apply()`;
- `/v1/embeddings` và `/embedding` nhận `input`/`content`, rồi gọi thẳng `tokenize_input_prompts(..., add_special=true, parse_special=true)`; đường xử lý này không gọi chat-template engine.

Đây là chủ ý của API: embedding input có thể là text thô, chuỗi đã format, token ID, batch, hoặc nội dung multimodal; endpoint không có `messages`/role để biết đâu là system instruction. Tự động bọc mọi input bằng chat template cũng sẽ làm sai các embedding model không được huấn luyện theo chat format và bọc hai lần input đã format. Option `--chat-template` chỉ chọn/ghi đè template cho các đường xử lý chat và `/apply-template`; nó không ép `/v1/embeddings` dùng template.

Có hai cách đúng cho text:

1. gửi chuỗi đã format trực tiếp như ví dụ dưới đây;
2. gọi `/apply-template` trước, rồi gửi trường `prompt` trả về sang `/v1/embeddings`.

Ví dụ cách 2, template tự thêm system prompt mặc định vì message đầu tiên là `user`:

```bash
PROMPT="$(curl -fsS http://localhost:8001/apply-template \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer llama-cpp-api-key' \
  -d '{
    "messages": [{"role": "user", "content": "Follow the white rabbit."}],
    "add_generation_prompt": true
  }' | jq -er '.prompt')"

jq -n --arg model 'Qwen3-VL-Embedding-2B' --arg input "$PROMPT" \
  '{model: $model, input: $input, encoding_format: "float"}' |
curl -fsS http://localhost:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer llama-cpp-api-key' \
  -d @-
```

Với endpoint embedding thô của llama.cpp, gửi chuỗi đã format để giữ đúng cách model được huấn luyện:

```bash
curl http://localhost:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer llama-cpp-api-key' \
  -d @- <<'JSON'
{
  "model": "Qwen3-VL-Embedding-2B",
  "input": "<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\nFollow the white rabbit.<|im_end|>\n<|im_start|>assistant\n",
  "encoding_format": "float"
}
JSON
```

Có thể gửi batch bằng mảng `input`:

```json
{
  "model": "Qwen3-VL-Embedding-2B",
  "input": [
    "<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\nFirst text<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\nSecond text<|im_end|>\n<|im_start|>assistant\n"
  ],
  "encoding_format": "float"
}
```

Với retrieval, nên dùng instruction mô tả task cho query, ví dụ `Retrieve relevant documents for the query.`; document có thể giữ instruction mặc định `Represent the user's input.` như ví dụ chính thức. Hãy giữ cách format từng loại input nhất quán giữa lúc lập chỉ mục và lúc truy vấn. Model card khuyến nghị viết instruction tùy biến bằng tiếng Anh.

## 6. Embedding ảnh và text + ảnh

API OpenAI `/v1/embeddings` chuẩn chỉ mô tả input văn bản. llama.cpp hỗ trợ multimodal embedding qua endpoint riêng `/embedding`, với `content.prompt_string` chứa media marker và `content.multimodal_data` chứa dữ liệu base64 theo đúng thứ tự marker.

**Không hard-code `<__media__>` trên llama.cpp mới.** [Issue #19947](https://github.com/ggml-org/llama.cpp/issues/19947) xác nhận marker cố định có thể xung đột với text của người dùng; bản sửa cho [issue #21955](https://github.com/ggml-org/llama.cpp/issues/21955) chuyển server sang marker ngẫu nhiên lúc khởi động. Lấy marker thực tế từ `GET /props` (không cần bật option `--props` cho GET). Các ví dụ dưới đây cần `jq`:

```bash
MEDIA_MARKER="$(curl -fsS http://localhost:8001/props \
  -H 'Authorization: Bearer llama-cpp-api-key' | jq -er '.media_marker')"
printf 'Media marker: %s\n' "$MEDIA_MARKER"
```

Lấy lại marker sau mỗi lần restart server. Bản llama.cpp cũ chưa có marker ngẫu nhiên dùng `<__media__>`, nhưng nên cập nhật thay vì hard-code để command hoạt động với bản hiện tại.

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
    "prompt_string": "<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\n${MEDIA_MARKER}<|im_end|>\n<|im_start|>assistant\n",
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
    "prompt_string": "<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\n${MEDIA_MARKER}A document about llama.cpp deployment.<|im_end|>\n<|im_start|>assistant\n",
    "multimodal_data": ["$IMAGE_B64"]
  },
  "embd_normalize": 2
}
JSON
```

Mỗi media marker phải có đúng một phần tử base64 tương ứng. Không viết marker thành `<\__media__>` hoặc `<\_...>`: `\_` không phải escape hợp lệ trong JSON và sẽ gây `json.exception.parse_error.101`. Dùng đúng chuỗi mà `/props` trả về, không thêm backslash. Trước khi gửi multimedia, có thể gọi `/v1/models` và kiểm tra model có capability `multimodal`. Trên macOS, thay `base64 -w 0` bằng `base64 < scripts/test.png | tr -d '\n'`.

Model gốc hỗ trợ cả video, nhưng khả năng và định dạng video thực tế còn phụ thuộc phiên bản `libmtmd`/llama.cpp; nên xác nhận trên bản binary đang triển khai trước khi đưa vào production.

## 7. Kích thước vector và tính similarity

Server trả vector 2.048 chiều và chuẩn hóa L2 khi dùng `--embd-normalize 2`. Vì vậy cosine similarity có thể tính trực tiếp bằng dot product:

```python
score = sum(a * b for a, b in zip(query_embedding, document_embedding))
```

Model hỗ trợ MRL 64–2.048 chiều, nhưng help hiện tại của llama.cpp không có option server để đặt output dimension. Nếu client cần vector ngắn hơn, lấy `N` chiều đầu rồi chuẩn hóa L2 lại trước khi lưu hoặc so sánh; dùng cùng một `N` cho toàn bộ index và query.

Không nhầm quantization của trọng số GGUF với quantization hậu xử lý của vector embedding: model card nói model hỗ trợ cả hai, còn llama.cpp mặc định trả vector float qua API.

## 8. Xử lý lỗi thường gặp

- **`/v1/embeddings` báo pooling `none`:** bỏ `--pooling none`, dùng `--pooling last` hoặc để model metadata chọn mặc định.
- **`number of media markers ... does not match number of bitmaps`:** lấy lại `.media_marker` từ `/props`; số marker trong `prompt_string` phải bằng số phần tử trong `multimodal_data`.
- **`forbidden character after backslash` gần `<\_`:** JSON không hỗ trợ escape `\_`; bỏ các backslash khỏi marker.
- **Không nhận ảnh/báo thiếu projector:** cập nhật llama.cpp và chỉ định đúng `--mmproj-url`; không dùng `--no-mmproj`.
- **Vector có chất lượng retrieval kém:** kiểm tra instruction, chat template và dấu generation prompt; không gửi query thô nếu pipeline chuẩn dùng instruction. `/v1/embeddings` không tự áp dụng `tokenizer.chat_template`; hãy format trước hoặc dùng `/apply-template`.
- **CUDA out of memory:** giảm context, quant model/projector, số image token hoặc số layer GPU; có thể dùng `--no-mmproj-offload`.
- **Tải model lỗi:** đặt `HF_TOKEN`/`--hf-token` nếu cần; các repository nêu trên hiện là public.
- **Chạy offline:** tải/cache model và projector trước, sau đó thêm `--offline`.

## Nguồn

- [Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
- [DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF](https://huggingface.co/DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF)
- [mradermacher/Qwen3-VL-Embedding-2B-i1-GGUF](https://huggingface.co/mradermacher/Qwen3-VL-Embedding-2B-i1-GGUF)
- [mradermacher/Qwen3-VL-Embedding-2B-GGUF](https://huggingface.co/mradermacher/Qwen3-VL-Embedding-2B-GGUF)
- llama.cpp: [`handle_embeddings_impl`](llama.cpp/tools/server/server-context.cpp) và [`oaicompat_chat_params_parse`](llama.cpp/tools/server/server-common.cpp)
- Help của binary trong repository: [llama-cpp.md](llama-cpp.md)
