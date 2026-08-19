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

Model chính thức hỗ trợ instruction tùy biến và Qwen khuyến nghị instruction tiếng Anh theo từng task. Tuy nhiên API rerank văn bản được mô tả trong llama.cpp chỉ nhận `query` và `documents`; không có trường `instruction` chuẩn. Không tự nối instruction vào query nếu pipeline cần tương thích điểm số. Muốn thay instruction một cách chính xác cần dùng rerank template đã chỉnh hoặc pipeline mtmd tùy biến và kiểm thử lại.

## 7. Vì sao model rerank được ảnh, nhưng `/v1/rerank` hiện chưa làm được?

Có hai lớp khả năng khác nhau:

1. **Model/graph có khả năng:** Qwen3-VL-Reranker vẫn là toàn bộ Qwen3-VL conditional-generation backbone. `scripts/qwen3_vl_reranker.py` của model gốc chạy processor ảnh/video, gọi `lm.model(...).last_hidden_state[:, -1]`, rồi chấm bằng hiệu hai hàng `yes` và `no` của LM head. Trong GGUF, LM và rank head nằm ở file chính; vision tower nằm trong `mmproj`. Sau khi mtmd chèn embedding ảnh vào chuỗi, graph rank dùng hidden state cuối nên điểm phụ thuộc vào ảnh.
2. **HTTP endpoint chưa biểu diễn được media:** trong checkout hiện tại, `server-context.cpp::post_rerank` buộc `query` là string và deserialize `documents` thành `std::vector<std::string>`. Vì vậy object `image_url`, content parts hoặc `multimodal_data` đều không thể đi qua endpoint này. `format_prompt_rerank()` có nhận `mtmd_context`, nhưng với đầu vào string nó chỉ tokenize text; load projector không thay đổi schema request.

Do đó stock `/v1/rerank` hiện chỉ rerank text. Không có curl ảnh hợp lệ cho endpoint này. Muốn rerank ảnh phải sửa server để endpoint nhận content parts/media, dựng prompt có media marker rồi đưa bitmap qua mtmd trước khi đọc rank score, hoặc viết pipeline trực tiếp bằng libmtmd/llama.cpp. Khi đó load projector:

```bash
--mmproj-url https://huggingface.co/bealore/Qwen3-VL-Reranker-2B-GGUF/resolve/main/Qwen3-VL-Reranker-2B.mmproj-f16.gguf
```

Pipeline phải giữ đúng thứ tự template `<Instruct>`, `<Query>`, `<Document>` và assistant prefix trước token được pool. Với video còn phải xác nhận decoder/frame sampling của libmtmd. Nói ngắn gọn: **weights và inference graph hỗ trợ vision-ranking; API rerank hiện tại là nút thắt.**

## 8. Xử lý lỗi thường gặp

- **`This server does not support reranking`:** thêm `--reranking` và khởi động lại server.
- **Pooling sai hoặc không có score:** thêm `--pooling rank`; kiểm tra GGUF có metadata `RANK` và tensor `cls.output.weight`.
- **Score gần như hằng số:** nhiều khả năng đang dùng conversion generative thiếu rank head; chuyển sang bản bealore đã xác minh.
- **CUDA out of memory:** bỏ projector cho text-only, giảm context/parallel hoặc giảm GPU offload.
- **Tên model không khớp:** gửi giá trị của `--alias`, ở đây là `Qwen3-VL-Reranker-2B`.
- **Query/document quá dài:** giảm độ dài input hoặc tăng `--ctx-size`, tối đa theo model là 32.768 token.
- **Tải model lỗi:** đặt `HF_TOKEN`/`--hf-token` nếu cần; repo nêu trên hiện public. Khi model đã có trong cache, có thể thêm `--offline`.

## Nguồn

- [Qwen/Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B)
- [bealore/Qwen3-VL-Reranker-2B-GGUF](https://huggingface.co/bealore/Qwen3-VL-Reranker-2B-GGUF)
- [mradermacher/Qwen3-VL-Reranker-2B-i1-GGUF](https://huggingface.co/mradermacher/Qwen3-VL-Reranker-2B-i1-GGUF)
- [mradermacher/Qwen3-VL-Reranker-2B-GGUF](https://huggingface.co/mradermacher/Qwen3-VL-Reranker-2B-GGUF)
- Help của binary trong repository: [llama-cpp.md](llama-cpp.md)
