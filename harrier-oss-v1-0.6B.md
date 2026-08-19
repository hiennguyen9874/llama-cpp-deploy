# harrier-oss-v1-0.6b với llama.cpp

## Nguồn model

- [microsoft/harrier-oss-v1-0.6b](https://huggingface.co/microsoft/harrier-oss-v1-0.6b)
- [SuperPauly/harrier-oss-v1-0.6b-gguf](https://huggingface.co/SuperPauly/harrier-oss-v1-0.6b-gguf)
- [mradermacher/harrier-oss-v1-0.6b-GGUF](https://huggingface.co/mradermacher/harrier-oss-v1-0.6b-GGUF)
- [lei-zh/harrier-oss-v1-0.6b-Q8_0-GGUF](https://huggingface.co/lei-zh/harrier-oss-v1-0.6b-Q8_0-GGUF)

Model tạo vector 1.024 chiều, dùng **last-token pooling**, chuẩn hóa L2 và hỗ trợ context tối đa 32.768 token.

## Chat template có được dùng cho embedding không?

**Không. Không nên bọc input Harrier bằng chat template.**

Checkpoint Microsoft có `chat_template.jinja` theo format Qwen3 (`<|im_start|>...`). GGUF của SuperPauly và mradermacher cũng chứa template này trong metadata `tokenizer.chat_template`; GGUF lei-zh được kiểm tra không chứa nó. Tuy nhiên, việc có hay không có metadata này không thay đổi cách gọi embedding:

- ví dụ Sentence Transformers và Transformers chính thức token hóa trực tiếp text, không gọi `apply_chat_template()`;
- query retrieval phải có instruction dạng `Instruct: ...\nQuery: ...`;
- document/passage được gửi dưới dạng text thô, không cần instruction;
- `chat_template.jinja` là metadata kế thừa từ kiến trúc Qwen3, không phải format embedding mà model card Harrier yêu cầu.

Trong llama.cpp (`01818e495`, build 10480), `/v1/embeddings` lấy `input` rồi gọi trực tiếp `tokenize_input_prompts(..., add_special=true, parse_special=true)`. Chỉ các đường xử lý chat như `/v1/chat/completions` và `/apply-template` mới gọi chat-template engine. Đây là hành vi đúng cho Harrier.

Không gọi `/apply-template` trước khi gọi `/v1/embeddings`: việc thêm role token và generation prompt `<|im_start|>assistant\n` sẽ làm thay đổi token cuối dùng để pooling và không khớp pipeline chính thức. Option `--chat-template` cũng không ép endpoint embedding dùng template.

Các GGUF nêu trên được kiểm tra có metadata:

- `qwen3.embedding_length = 1024`;
- `qwen3.pooling_type = 3` (`LAST`);
- `tokenizer.ggml.add_bos_token = false`;
- `tokenizer.ggml.add_eos_token = true`.

Vì vậy không tự thêm BOS/EOS hoặc các token `<|im_...|>` vào input.

## Chạy llama-server

```bash
./llama.cpp/llama-server \
  -hf SuperPauly/harrier-oss-v1-0.6b-gguf:Q8_0 \
  --alias harrier-oss-v1-0.6b \
  --embeddings \
  --host 0.0.0.0 \
  --port 8001 \
  --api-key llama-cpp-api-key \
  --threads -1 \
  --ctx-size 4096 \
  --batch-size 512
```

GGUF đã khai báo last-token pooling, nên bình thường không cần `--pooling last`. Có thể thêm option này để ghi đè khi dùng một GGUF khác bị thiếu metadata. Tăng `--ctx-size` nếu thực sự cần input dài hơn; model hỗ trợ tối đa 32.768 token nhưng context lớn tốn thêm bộ nhớ.

## Embedding query retrieval

Mỗi query cần một instruction một câu mô tả task. Đây là format chính thức:

```bash
curl -fsS http://localhost:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer llama-cpp-api-key' \
  -d @- <<'JSON'
{
  "model": "harrier-oss-v1-0.6b",
  "input": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: summit define",
  "encoding_format": "float",
  "embd_normalize": 2
}
JSON
```

Các prompt có sẵn trong `config_sentence_transformers.json`:

- web search: `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: `;
- semantic similarity: `Instruct: Retrieve semantically similar text\nQuery: `;
- bitext: `Instruct: Retrieve parallel sentences\nQuery: `.

Model card cho phép instruction tùy chỉnh, nhưng nên viết một câu ngắn mô tả đúng task.

## Embedding document

Document không cần instruction và cũng không cần chat template:

```bash
curl -fsS http://localhost:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer llama-cpp-api-key' \
  -d @- <<'JSON'
{
  "model": "harrier-oss-v1-0.6b",
  "input": "Definition of summit: the highest point of a mountain.",
  "encoding_format": "float",
  "embd_normalize": 2
}
JSON
```

Có thể gửi batch bằng mảng `input`. Không trộn query có instruction và document không instruction nếu ứng dụng không theo mô hình retrieval bất đối xứng này.

`embd_normalize: 2` là chuẩn hóa Euclidean/L2 của llama.cpp và khớp module `Normalize` trong pipeline Sentence Transformers. Đây là extension của llama.cpp; mặc định server hiện cũng là `2`. Sau chuẩn hóa, có thể dùng dot product hoặc cosine similarity để so sánh vector.

## Lỗi thường gặp

- **Retrieval kém khi query chỉ là text thô:** thêm `Instruct: <task>\nQuery: ` theo model card.
- **Retrieval kém sau khi dùng `/apply-template`:** bỏ chat template và gửi đúng chuỗi instruction/text thô.
- **Document bị prefix instruction:** chỉ query retrieval cần instruction; document không cần.
- **Pooling sai hoặc vector sai kích thước:** xác nhận server báo pooling `last` và vector có 1.024 phần tử; thử `--pooling last` nếu GGUF không có metadata.
- **Vector chưa chuẩn hóa:** gửi `"embd_normalize": 2` hoặc chuẩn hóa L2 ở client.
- **Input bị cắt:** tăng `--ctx-size` nhưng không vượt giới hạn model 32.768 token.

## Kết luận

Khác Qwen3-VL-Embedding, Harrier không dùng system message hay generation prompt cho embedding. Việc llama.cpp không tự áp dụng `tokenizer.chat_template` tại `/v1/embeddings` vừa là thiết kế chung của endpoint, vừa khớp chính xác cách dùng chính thức của Harrier: **query có instruction dạng `Instruct/Query`, document là text thô**.
