# Chạy Qwen3-VL-Reranker-2B với llama.cpp

Qwen3-VL-Reranker-2B là reranker đa phương thức 2B, được xây dựng từ Qwen3-VL-2B-Instruct. Mỗi lần chấm điểm, model nhận một cặp `(query, document)` và trả mức độ liên quan. Query và document của model gốc có thể là văn bản, ảnh, video hoặc nội dung trộn nhiều modality.

Thông số chính từ model card chính thức:

- 2B tham số, 28 layer;
- context tối đa 32.768 token;
- hỗ trợ hơn 30 ngôn ngữ;
- instruction-aware; instruction phù hợp thường cải thiện kết quả khoảng 1–5% và Qwen khuyến nghị viết instruction bằng tiếng Anh;
- license Apache-2.0;
- đây là reranker/cross-encoder, không phải model embedding hoặc chat.

Model này không có classifier được huấn luyện riêng. Code gốc lấy hidden state cuối của Qwen3-VL, lấy hai hàng `yes`/`no` từ LM head và tính `sigmoid(logit_yes - logit_no)`. Vì vậy ảnh có thể ảnh hưởng điểm: vision tower biến ảnh thành embedding, embedding được trộn vào cùng chuỗi query/document, và hidden state cuối đã attend tới cả text lẫn ảnh.

GGUF đúng cho llama.cpp chuyển hai hàng LM-head đó thành `cls.output.weight`; rank pooling lấy token cuối, nhân với head hai lớp rồi softmax. `softmax([logit_yes, logit_no])[0]` chính là `sigmoid(logit_yes - logit_no)`, nên server trả `relevance_score = P("yes")` trong `[0, 1]` mà không cần model sinh token.

## 1. Yêu cầu llama.cpp

Dùng llama.cpp **b10000 hoặc mới hơn**. Hỗ trợ text-rerank Qwen3-VL được thêm ở commit `4d99d4508` (`model: qwen3vl reranker text support`, thuộc tag b10000): converter nhận diện tên/README reranker, graph dùng token cuối và softmax rank head cho kiến trúc `QWEN3VL`. Checkout đã kiểm tra trong thư mục `llama.cpp/` là `b10472-8-g01818e495`, có commit này. Chưa có binary build sẵn trong checkout.

Kiểm tra binary và thiết bị sau khi build:

```bash
./llama.cpp/llama-server --version
./llama.cpp/llama-server --list-devices
```

Các option liên quan trong [llama-cpp.md](llama-cpp.md):

- `--rerank`/`--reranking`: bật endpoint reranking;
- `--pooling rank`: dùng rank head của model;
- `-hf REPO` và `--hf-file FILE`: tải đúng GGUF từ Hugging Face;
- `--no-mmproj`: không tải vision projector khi chỉ rerank văn bản;
- `--mmproj`/`--mmproj-url`: chỉ định vision projector cho pipeline multimodal;
- `--mmproj-offload`/`--no-mmproj-offload`: đặt projector trên GPU/CPU;
- `--image-min-tokens`, `--image-max-tokens` và `--mtmd-batch-max-tokens`: điều chỉnh xử lý ảnh;
- `--alias`: đặt tên model được dùng trong API;
- `--parallel`: số server slot.

Nên chỉ định cả `--reranking` và `--pooling rank` để cấu hình rõ ràng. `--embedding` không cần thiết với help hiện tại; đó là chế độ chỉ dành cho embedding model.

## 2. Chọn GGUF: rank head là bắt buộc

### Bản được khuyến nghị

Repository [bealore/Qwen3-VL-Reranker-2B-GGUF](https://huggingface.co/bealore/Qwen3-VL-Reranker-2B-GGUF) cung cấp bản chuyển đổi f16 đã được kiểm tra cho rank pooling:

| File | Dung lượng | Mục đích |
| --- | ---: | --- |
| `Qwen3-VL-Reranker-2B.f16.gguf` | 3,4 GB | language model và rank head, bắt buộc |
| `Qwen3-VL-Reranker-2B.Q8_0.gguf` | khoảng 1,9 GB | quant từ chính f16 đã xác minh; có cùng rank head |
| `Qwen3-VL-Reranker-2B.mmproj-f16.gguf` | 822 MB | vision tower/projector, chỉ cần cho multimodal |

Các marker được model card xác nhận:

- `qwen3vl.pooling_type = 4` (`RANK`);
- `qwen3vl.classifier.output_labels = ["yes", "no"]`;
- tensor `cls.output.weight` có shape `(2048, 2)`;
- có `tokenizer.chat_template.rerank` chứa template `{query}`/`{document}`.

### Vì sao không mặc định dùng các bản mradermacher

Hai repository sau có nhiều quant nhỏ hơn:

- [mradermacher/Qwen3-VL-Reranker-2B-GGUF](https://huggingface.co/mradermacher/Qwen3-VL-Reranker-2B-GGUF): static quants Q2_K–Q8_0/f16 và projector;
- [mradermacher/Qwen3-VL-Reranker-2B-i1-GGUF](https://huggingface.co/mradermacher/Qwen3-VL-Reranker-2B-i1-GGUF): imatrix quants IQ1–Q6; projector phải lấy từ static repo.

Tuy nhiên, README của hai repo này chỉ liệt kê quant và không xác nhận `pooling_type`, `cls.output.weight`, output labels hay rerank template. Model card của bealore cảnh báo một số community GGUF được xuất như model Qwen3-VL sinh văn bản thông thường: chúng có thể load với `--pooling rank` mà không báo lỗi nhưng trả score gần như hằng số và vô nghĩa.

Vì vậy không nên chọn GGUF chỉ dựa trên tên file hoặc việc server khởi động thành công. Chỉ dùng một quant khác sau khi xác nhận đủ bốn marker trên và chạy smoke test phân biệt rõ document liên quan/không liên quan. Các command bên dưới cố ý dùng bản bealore đã được xác minh.

## 3. Chạy server rerank văn bản trên GPU

Text-only không cần projector:

```bash
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/llama-server \
  -hf bealore/Qwen3-VL-Reranker-2B-GGUF \
  --hf-file Qwen3-VL-Reranker-2B.f16.gguf \
  --no-mmproj \
  --alias Qwen3-VL-Reranker-2B \
  --reranking \
  --pooling rank \
  --host 0.0.0.0 --port 8082 \
  --ctx-size 32768 \
  --parallel 1 \
  --flash-attn on \
  --n-gpu-layers all \
  --device CUDA0 \
  --api-key llama-cpp-api-key
```

`CUDA_VISIBLE_DEVICES=0` làm GPU được chọn xuất hiện dưới tên `CUDA0` trong tiến trình. Luôn dùng `--list-devices` để kiểm tra tên backend thực tế.

Nếu thiếu VRAM, giảm `--ctx-size` xuống `8192` hoặc `16384`, hoặc giảm số layer offload. Context cần chứa prompt rerank cùng query và từng document; không nhất thiết cấp đủ 32K nếu dữ liệu ngắn.

## 4. Chạy chỉ bằng CPU

```bash
./llama.cpp/llama-server \
  -hf bealore/Qwen3-VL-Reranker-2B-GGUF \
  --hf-file Qwen3-VL-Reranker-2B.f16.gguf \
  --no-mmproj \
  --alias Qwen3-VL-Reranker-2B \
  --reranking --pooling rank \
  --host 0.0.0.0 --port 8082 \
  --ctx-size 8192 \
  --parallel 1 \
  --device none \
  --threads "$(nproc)" \
  --api-key llama-cpp-api-key
```

Repo bealore hiện có cả f16 và Q8_0 được quant trực tiếp từ f16 đã xác minh. Có thể đổi `--hf-file` sang `Qwen3-VL-Reranker-2B.Q8_0.gguf` để giảm RAM. Không đổi sang Q4/Q6 community chưa kiểm tra chỉ dựa vào tên quant.

## 5. Gọi API rerank

llama.cpp hỗ trợ các alias `/rerank`, `/reranking`, `/v1/rerank` và `/v1/reranking`. Request chuẩn dùng `query`, mảng chuỗi `documents` và tùy chọn `top_n`:

```bash
curl http://localhost:8082/v1/rerank \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer llama-cpp-api-key' \
  -d @- <<'JSON' | jq
{
  "model": "Qwen3-VL-Reranker-2B",
  "query": "What is the capital of France?",
  "top_n": 3,
  "documents": [
    "Paris is the capital and most populous city of France.",
    "Bananas are a yellow tropical fruit rich in potassium.",
    "The Eiffel Tower is a famous landmark located in Paris, France."
  ]
}
JSON
```

Kết quả có dạng `results[]`, mỗi phần tử chứa ít nhất `index` và `relevance_score`, được xếp theo score giảm dần. `index` trỏ về vị trí trong mảng `documents`; không giả định thứ tự đầu ra giống thứ tự đầu vào.

Có thể gửi `texts` thay cho `documents` để nhận response theo kiểu Text Embeddings Inference (`score` thay cho `relevance_score`), nhưng nên giữ `documents` nếu client dùng API rerank thông thường.

### Smoke test bắt buộc

Một model đúng phải cho document trực tiếp trả lời truy vấn cao hơn rõ rệt so với nội dung không liên quan. Model card bealore đã đo ví dụ trên lần lượt khoảng `0.733`, `0.270`, `0.514`. Giá trị cụ thể có thể thay đổi theo build, nhưng score gần như giống nhau cho mọi document là dấu hiệu GGUF thiếu hoặc sai rank head.

## 6. Instruction và template

Rerank template trong bản bealore có instruction mặc định:

> Given a web search query, retrieve relevant passages that answer the query

Model chính thức hỗ trợ instruction tùy biến và Qwen khuyến nghị instruction tiếng Anh theo từng task. Tuy nhiên API rerank văn bản được mô tả trong llama.cpp chỉ nhận `query` và `documents`; không có trường `instruction` chuẩn. Không tự nối instruction vào query nếu pipeline cần tương thích điểm số. Muốn thay instruction một cách chính xác cần dùng rerank template đã chỉnh, hoặc tự dựng đầy đủ prompt khi dùng low-level `/embedding` như phần tiếp theo, rồi kiểm thử lại.

## 7. Rerank document có ảnh qua API

Có hai lớp API cần phân biệt:

1. **`/v1/rerank` chỉ hỗ trợ text:** trong checkout hiện tại, `server-context.cpp::post_rerank` buộc `query` là string và deserialize `documents` thành `std::vector<std::string>`. Vì vậy endpoint này không nhận `image_url`, content parts hay `multimodal_data`.
2. **Low-level `/embedding` hỗ trợ multimedia:** `handle_embeddings_impl()` chuyển `content` qua `tokenize_input_prompts()`, nơi object `{ "prompt_string": ..., "multimodal_data": [...] }` được xử lý bởi mtmd. Khi server dùng `--pooling rank`, vector đầu ra là các xác suất classifier; với Qwen3-VL-Reranker có labels `["yes", "no"]`, phần tử đầu chính là relevance score mà `/v1/rerank` trả về.

Vì vậy có thể rerank ảnh trên stock server mà không sửa C++, nhưng client phải tự dựng **toàn bộ rerank template** và tự sắp xếp score.

### 7.1. Chạy server vision-rerank

Không dùng `--no-mmproj`. Load projector và tắt normalization của output classifier:

```bash
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/llama-server \
  -hf bealore/Qwen3-VL-Reranker-2B-GGUF \
  --hf-file Qwen3-VL-Reranker-2B.f16.gguf \
  --mmproj-url https://huggingface.co/bealore/Qwen3-VL-Reranker-2B-GGUF/resolve/main/Qwen3-VL-Reranker-2B.mmproj-f16.gguf \
  --alias Qwen3-VL-Reranker-2B \
  --reranking --pooling rank --embd-normalize -1 \
  --host 0.0.0.0 --port 8082 \
  --ctx-size 32768 --parallel 1 \
  --flash-attn on --n-gpu-layers all --device CUDA0 \
  --api-key llama-cpp-api-key
```

`--embd-normalize -1` là bắt buộc với đường `/embedding`: graph đã softmax rank head thành `[P("yes"), P("no")]`; L2-normalize lần nữa sẽ làm sai xác suất. `/v1/rerank` dùng trực tiếp phần tử đầu nên không gặp bước normalization này.

Model text-only như Qwen3-Reranker-0.6B không thể nhận ảnh chỉ bằng cách gắn thêm projector; phải dùng checkpoint Qwen3-**VL**-Reranker cùng `mmproj` tương ứng.

### 7.2. Gọi `/embedding` với document ảnh

Media marker của server là ngẫu nhiên và thay đổi sau mỗi lần restart. Luôn lấy marker hiện tại từ `/props`:

```bash
MEDIA_MARKER="$(curl -fsS http://localhost:8082/props \
  -H 'Authorization: Bearer llama-cpp-api-key' | jq -er '.media_marker')"
IMAGE_B64="$(base64 -w 0 document.png)"
```

Trên macOS, thay command `base64` bằng:

```bash
IMAGE_B64="$(base64 < document.png | tr -d '\n')"
```

Dựng prompt đúng template `tokenizer.chat_template.rerank` của GGUF. `/embedding` không tự áp dụng template:

```bash
QUERY='What is the capital of China?'
DOCUMENT_TEXT='Use the information shown in this document image.'

PROMPT="$(printf '%s' \
"<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>
<|im_start|>user
<Instruct>: Given a web search query, retrieve relevant passages that answer the query
<Query>: ${QUERY}
<Document>: ${MEDIA_MARKER}${DOCUMENT_TEXT}<|im_end|>
<|im_start|>assistant
<think>

</think>

")"
```

Gửi ảnh dưới dạng base64 thô, không thêm prefix `data:image/...;base64,`:

```bash
jq -n --arg prompt "$PROMPT" --arg image "$IMAGE_B64" \
  '{
    content: {
      prompt_string: $prompt,
      multimodal_data: [$image]
    },
    embd_normalize: -1
  }' |
curl -fsS http://localhost:8082/embedding \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer llama-cpp-api-key' \
  -d @- | jq
```

Response native có dạng:

```json
[
  {
    "index": 0,
    "embedding": [[0.733, 0.267]]
  }
]
```

Vì labels là `["yes", "no"]`, lấy relevance score bằng:

```bash
jq '.[0].embedding[0][0]'
```

Số cụ thể ở trên chỉ minh họa; cần dùng smoke test để xác nhận model và projector thực tế.

### 7.3. Batch và sắp xếp nhiều document

`content` có thể là mảng các object. Mỗi object phải chứa một prompt query-document hoàn chỉnh; document có ảnh phải có số media marker bằng đúng số phần tử `multimodal_data`. Server trả một phần tử theo từng input, nhưng không tạo response rerank hay áp dụng `top_n`. Client có thể chuyển và sắp xếp như sau:

```bash
jq '
  map({
    index: .index,
    relevance_score: .embedding[0][0]
  })
  | sort_by(-.relevance_score)
'
```

Nếu cần schema chuẩn `/v1/rerank` nhận multimedia trực tiếp, vẫn phải mở rộng `post_rerank` để nhận content object, chèn media vào đúng `{document}` trong rerank template và chuyển bitmap qua mtmd. Low-level `/embedding` là workaround dùng được với code hiện tại. Với video, cần xác nhận decoder và frame sampling của phiên bản libmtmd đang triển khai.

## 8. Xử lý lỗi thường gặp

- **`This server does not support reranking`:** thêm `--reranking` và khởi động lại server.
- **Pooling sai hoặc không có score:** thêm `--pooling rank`; kiểm tra GGUF có metadata `RANK` và tensor `cls.output.weight`.
- **Score gần như hằng số:** nhiều khả năng đang dùng conversion generative thiếu rank head; chuyển sang bản bealore đã xác minh.
- **Score từ `/embedding` không còn có tổng bằng 1:** đặt `--embd-normalize -1` trên server và `"embd_normalize": -1` trong request; không L2-normalize xác suất rank head.
- **`number of media markers ... does not match number of bitmaps`:** lấy lại `.media_marker` từ `/props`; mỗi marker phải có đúng một phần tử base64 tương ứng.
- **Có projector nhưng ảnh không ảnh hưởng score:** kiểm tra đang dùng checkpoint Qwen3-VL-Reranker, đúng `mmproj`, và media marker nằm bên trong phần `<Document>` của full rerank template.
- **CUDA out of memory:** bỏ projector cho text-only, giảm context/parallel hoặc giảm GPU offload.
- **Tên model không khớp:** gửi giá trị của `--alias`, ở đây là `Qwen3-VL-Reranker-2B`.
- **Query/document quá dài:** giảm độ dài input hoặc tăng `--ctx-size`, tối đa theo model là 32.768 token.
- **Tải model lỗi:** đặt `HF_TOKEN`/`--hf-token` nếu cần; repo nêu trên hiện public. Khi model đã có trong cache, có thể thêm `--offline`.

## Nguồn

- [Qwen/Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B)
- [bealore/Qwen3-VL-Reranker-2B-GGUF](https://huggingface.co/bealore/Qwen3-VL-Reranker-2B-GGUF)
- [mradermacher/Qwen3-VL-Reranker-2B-i1-GGUF](https://huggingface.co/mradermacher/Qwen3-VL-Reranker-2B-i1-GGUF)
- [mradermacher/Qwen3-VL-Reranker-2B-GGUF](https://huggingface.co/mradermacher/Qwen3-VL-Reranker-2B-GGUF)
- llama.cpp: [`post_rerank` và `handle_embeddings_impl`](llama.cpp/tools/server/server-context.cpp), [`tokenize_input_subprompt`](llama.cpp/tools/server/server-common.cpp), rank graph trong [`llama-graph.cpp`](llama.cpp/src/llama-graph.cpp), và rerank template trong [`conversion/qwen.py`](llama.cpp/conversion/qwen.py)
- Help của binary trong repository: [llama-cpp.md](llama-cpp.md)
