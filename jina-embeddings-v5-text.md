# jina-embeddings-v5-text

## Tổng quan

`jina-embeddings-v5-text` là họ mô hình embedding đa ngôn ngữ của Jina AI, hỗ trợ bốn tác vụ:

- `retrieval`: tìm kiếm query–document;
- `text-matching`: đo độ tương đồng ngữ nghĩa đối xứng;
- `clustering`: gom cụm văn bản;
- `classification`: tạo đặc trưng cho bộ phân loại phía sau.

Các mô hình được huấn luyện bằng embedding distillation từ Qwen3-Embedding-4B kết hợp contrastive loss riêng cho từng tác vụ. Bản `small` đạt 71.7 trên MTEB English v2 và 67.7 trên MMTEB; bản `nano` đạt lần lượt 71.0 và 65.5.

| Thuộc tính | `small` | `nano` |
|---|---:|---:|
| Số tham số | 677M | 239M |
| Base model | Qwen3-0.6B-Base | EuroBERT-210m |
| Số chiều đầy đủ | 1024 | 768 |
| Context tối đa theo model card | 32,768 | 8,192 |
| Matryoshka dimensions | 32, 64, 128, 256, 512, 768, 1024 | 32, 64, 128, 256, 512, 768 |
| Pooling | last token | last token |
| Giấy phép | CC BY-NC 4.0 | CC BY-NC 4.0 |

`small` hỗ trợ hơn 119 ngôn ngữ. Cả hai model card đều mô tả embedding bền vững khi giảm số chiều và binary quantization.

> **Lưu ý giấy phép:** CC BY-NC 4.0 không cho phép sử dụng thương mại nếu chưa có thỏa thuận riêng với Jina AI.

## Chọn biến thể

Model gốc chứa adapter cho cả bốn tác vụ và phù hợp với `transformers`/`sentence-transformers`:

- `jinaai/jina-embeddings-v5-text-small`
- `jinaai/jina-embeddings-v5-text-nano`

Với llama.cpp, vLLM, TEI hoặc ONNX, nên dùng model đã merge adapter theo tác vụ:

| Nhu cầu | Repository `small` |
|---|---|
| Semantic search, RAG | `jinaai/jina-embeddings-v5-text-small-retrieval` |
| Duplicate detection, sentence similarity | `jinaai/jina-embeddings-v5-text-small-text-matching` |
| Gom cụm | `jinaai/jina-embeddings-v5-text-small-clustering` |
| Đặc trưng cho classification | `jinaai/jina-embeddings-v5-text-small-classification` |

Các repository trên có GGUF từ F16 đến nhiều mức quantization. Riêng retrieval còn có repository chuyên biệt `jinaai/jina-embeddings-v5-text-small-retrieval-GGUF`.

Không dùng một model task-specific cho tác vụ khác. Model classification chỉ sinh vector đặc trưng; nó không trực tiếp trả về nhãn.

## Chạy bản GGUF bằng llama.cpp

### Build

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j "$(nproc)"
```

Bỏ `-DGGML_CUDA=ON` nếu chỉ chạy CPU. Có thể kiểm tra backend bằng:

```bash
./llama.cpp/build/bin/llama-server --list-devices
```

### Khởi chạy retrieval server

Cấu hình dưới đây tải trực tiếp bản Q8_0 từ Hugging Face, dùng last-token pooling và cung cấp API tương thích OpenAI:

```bash
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/build/bin/llama-server \
  -hf jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:Q8_0 \
  --alias jina-embeddings-v5-text-small-retrieval \
  --embeddings \
  --pooling last \
  --embd-normalize 2 \
  -c 32768 -b 32768 -ub 512 \
  -fa on -ngl all --device CUDA0 \
  --parallel 1 \
  --host 0.0.0.0 --port 8080 \
  --api-key llama-cpp-api-key \
  --metrics --slots
```

Các tùy chọn quan trọng:

- `--embeddings`: chỉ bật use case embedding;
- `--pooling last`: bắt buộc dùng last-token pooling của Jina v5;
- `--embd-normalize 2`: chuẩn hóa Euclidean/L2, cũng là mặc định của llama.cpp;
- `-c`: context tối đa;
- `-b`: logical batch size;
- `-ub`: physical micro-batch, giảm giá trị này nếu thiếu VRAM;
- `-fa on`: bật Flash Attention;
- `-ngl all`: offload tối đa số layer sang GPU;
- `--parallel`: số server slot. Tăng concurrency sẽ tăng nhu cầu bộ nhớ.

`-c 32768` là mức tối đa của bản small, không phải cấu hình bắt buộc. Nếu dữ liệu chỉ là các đoạn ngắn, có thể dùng `-c 8192 -b 8192` để giảm tài nguyên. Nếu `-b 32768` không phù hợp với bản llama.cpp hoặc phần cứng đang dùng, giảm `-b`; luôn benchmark với độ dài dữ liệu thực tế.

Chạy CPU:

```bash
./llama.cpp/build/bin/llama-server \
  -hf jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:Q8_0 \
  --alias jina-embeddings-v5-text-small-retrieval \
  --embeddings --pooling last --embd-normalize 2 \
  -c 8192 -b 8192 -ub 256 \
  --device none --threads -1 \
  --host 0.0.0.0 --port 8080
```

### Prefix đầu vào

llama.cpp không gọi preprocessing tùy biến từ model Python. Vì vậy phải tự thêm prefix đúng như lúc model được huấn luyện:

| Model/tác vụ | Vai trò | Đầu vào gửi tới llama.cpp |
|---|---|---|
| retrieval | query | `Query: {text}` |
| retrieval | document | `Document: {text}` |
| text-matching | mọi văn bản | `Document: {text}` |
| clustering | mọi văn bản | `Document: {text}` |
| classification | mọi văn bản | `Document: {text}` |

Không thay `Document:` bằng `Passage:`. Bài viết tối ưu GGUF của Jina dùng `Passage:` cho jina-embeddings-v4; model card v5 retrieval quy định `Query:` và `Document:`.

### Gọi OpenAI-compatible API

```bash
curl --request POST \
  --url http://127.0.0.1:8080/v1/embeddings \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer llama-cpp-api-key' \
  --data '{
    "model": "jina-embeddings-v5-text-small-retrieval",
    "input": [
      "Query: Hành tinh nào được gọi là Hành tinh Đỏ?",
      "Document: Sao Hỏa được gọi là Hành tinh Đỏ vì bề mặt có màu đỏ.",
      "Document: Sao Kim có kích thước gần giống Trái Đất."
    ]
  }'
```

Kết quả chứa một vector 1024 chiều cho mỗi phần tử `input`. Khi lập chỉ mục RAG:

1. thêm `Document:` cho toàn bộ corpus;
2. thêm `Query:` cho truy vấn;
3. lưu và so sánh vector bằng cosine similarity hoặc dot product trên vector đã L2-normalize;
4. không trộn vector từ task model, quantization hoặc quy tắc prefix khác nhau trong cùng một index.

### Đổi tác vụ hoặc quantization

Ví dụ text matching Q8_0:

```bash
./llama.cpp/build/bin/llama-server \
  -hf jinaai/jina-embeddings-v5-text-small-text-matching-GGUF:Q8_0 \
  --alias jina-embeddings-v5-text-small-text-matching \
  --embeddings --pooling last --embd-normalize 2 \
  -c 32768 -b 32768 -ub 512 \
  -fa on -ngl all \
  --host 0.0.0.0 --port 8080
```

Request:

```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "input": [
      "Document: Một hoàng hôn tuyệt đẹp trên bãi biển",
      "Document: A beautiful sunset over the beach"
    ]
  }'
```

Các quantization được công bố gồm F16, Q8_0, Q6_K, Q5_K_S/M, Q4_K_M, IQ4_NL/XS, Q3_K_M, Q2_K và IQ1/IQ2. Điểm khởi đầu thực dụng:

- `Q8_0`: ưu tiên chất lượng và vẫn muốn giảm dung lượng so với F16;
- `Q4_K_M`: cân bằng dung lượng/tốc độ/chất lượng;
- F16: baseline để kiểm tra sai lệch;
- IQ1/IQ2: chỉ dùng sau khi đã đánh giá recall/chất lượng trên dữ liệu thật.

Kết quả benchmark quantization trong bài viết Jina là của **jina-embeddings-v4 3B**, không nên xem là số liệu chất lượng hay tốc độ của v5-small 677M.

### Chạy kiểm tra một lần

```bash
./llama.cpp/build/bin/llama-embedding \
  -hf jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:Q8_0 \
  --pooling last \
  -p 'Query: jina embeddings là gì?' \
  --embd-output-format json
```

`llama-embedding` phù hợp cho sanity check. Với dịch vụ hoặc bulk embedding, dùng `llama-server` và gửi batch nhiều chuỗi trong một request.

## Matryoshka: giảm số chiều

Bản small chỉ được huấn luyện tại các kích thước `32, 64, 128, 256, 512, 768, 1024`. Nếu cần vector 256 chiều, lấy 256 phần tử đầu rồi chuẩn hóa L2 lại:

```python
import numpy as np

full = np.asarray(embedding, dtype=np.float32)
short = full[:256]
short /= np.linalg.norm(short)
```

Chỉ dùng các kích thước Matryoshka đã công bố. Không nên chọn một kích thước tùy ý như 300. Database và query phải dùng cùng số chiều, cùng cách cắt và cùng bước chuẩn hóa.

## Dùng model gốc qua Transformers

Model gốc tự chọn adapter và tự áp dụng prompt theo `task`/`prompt_name`, nên không thêm prefix thủ công:

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v5-text-small",
    trust_remote_code=True,
    dtype=torch.bfloat16,
).to("cuda")

query = model.encode(
    texts=["Hành tinh nào được gọi là Hành tinh Đỏ?"],
    task="retrieval",
    prompt_name="query",
)
documents = model.encode(
    texts=[
        "Sao Hỏa được gọi là Hành tinh Đỏ.",
        "Sao Kim có kích thước gần giống Trái Đất.",
    ],
    task="retrieval",
    prompt_name="document",
)
```

Yêu cầu theo model card của model gốc: `transformers>=4.57.0`, `torch>=2.8.0`, `peft>=0.15.2`; Flash Attention và `sentence-transformers` là tùy chọn.

## Ghi chú vận hành

- Chất lượng phụ thuộc mạnh vào đúng task adapter và đúng prefix.
- Last-token pooling là bắt buộc; không sao chép `--pooling mean` từ hướng dẫn jina-embeddings-v4.
- `-ub` là số token xử lý vật lý mỗi lượt. Giảm `-ub` để hạ peak VRAM; tăng dần và benchmark để tìm throughput tốt nhất.
- Context dài có chi phí attention và KV cache lớn. Chunk tài liệu nếu không thật sự cần 32K token trong một vector.
- Theo dõi `/metrics` và `/slots` khi đã bật `--metrics --slots`.
- Cố định phiên bản llama.cpp và quantization trong production; re-embed corpus nếu thay model, prefix, pooling, normalization hoặc số chiều.
- Bài viết của Jina mô tả một fork tối ưu cho decoder-only embedding và các lỗi cũ quanh logical/physical batch. Trước khi dùng fork, hãy kiểm tra upstream hiện tại; help đi kèm repository này đã cho phép cấu hình `-b` và `-ub` độc lập.

## Nguồn

- [jina-embeddings-v5-text-small](https://huggingface.co/jinaai/jina-embeddings-v5-text-small)
- [jina-embeddings-v5-text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano)
- [Small text-matching](https://huggingface.co/jinaai/jina-embeddings-v5-text-small-text-matching)
- [Small retrieval](https://huggingface.co/jinaai/jina-embeddings-v5-text-small-retrieval)
- [Small clustering](https://huggingface.co/jinaai/jina-embeddings-v5-text-small-clustering)
- [Small classification](https://huggingface.co/jinaai/jina-embeddings-v5-text-small-classification)
- [Small retrieval GGUF](https://huggingface.co/jinaai/jina-embeddings-v5-text-small-retrieval-GGUF)
- [Optimizing GGUFs for Decoder-Only Embedding Models](https://jina.ai/news/optimizing-ggufs-for-decoder-only-embedding-models/)
