# Chạy jina-reranker-v3.5 và bản GGUF với llama.cpp

## Tổng quan

`jina-reranker-v3.5` là reranker **listwise** đa ngôn ngữ 0,6B tham số của Jina AI, xây dựng trên Qwen3-0.6B. Model nhận một query cùng nhiều document trong một forward pass rồi xếp hạng chúng theo mức độ liên quan.

Các đặc điểm chính:

- 28 layer, hidden size 1024;
- hybrid attention `3L2G`: các layer sliding-window xen kẽ global attention, cửa sổ 1024 token;
- projector MLP `1024 → 512 → 512`;
- context tối đa theo model card là 131K token;
- hỗ trợ đa ngôn ngữ, domain chuyên ngành và dữ liệu có cấu trúc;
- giao diện listwise LBNL (*last-but-not-late*), tương thích với `jina-reranker-v3`;
- giấy phép **CC BY-NC 4.0**, không dùng thương mại nếu chưa có thỏa thuận riêng với Jina AI.

So với v3, model card công bố mức tăng 1,10 điểm BEIR, 2,6% tương đối trên MIRACL, 4,3% trên RTEB và 24,8% trên Struct-IR. Đây là reranker, không phải model chat và cũng không phải embedding model dùng để lập index.

## Điểm đặc biệt của bản GGUF

Repository [jinaai/jina-reranker-v3.5-GGUF](https://huggingface.co/jinaai/jina-reranker-v3.5-GGUF) không phải một GGUF có thể chạy trực tiếp bằng endpoint rerank chuẩn của `llama-server`.

Pipeline đầy đủ gồm bốn artifact:

- `jina-reranker-v3.5-*.gguf`: backbone Qwen3 đã lượng tử hóa;
- `projector.safetensors`: scoring MLP, **không được nhúng vào GGUF**;
- `tokenizer.json`: dùng để tokenize, truncate và chia block trong Python;
- `rerank.py`: dựng prompt listwise, gọi `llama-embedding`, chạy projector và tính điểm cosine.

`rerank.py` đặt một `<|embed_token|>` sau mỗi document và hai `<|rerank_token|>` cho query. Binary chỉ xuất hidden state tại các token đặc biệt này; Python sau đó đưa hidden state qua projector và tính cosine similarity. Khi danh sách dài, script chia document thành các block rồi hợp nhất query embedding bằng weighted fusion.

Vì vậy:

- không dùng `llama-server --reranking --pooling rank` cho model này;
- không chỉ tải file GGUF rồi bỏ qua `projector.safetensors`;
- điểm `relevance_score` là cosine score, không phải xác suất softmax bắt buộc nằm trong `[0, 1]`;
- API `/v1/rerank` của llama.cpp không tự thực hiện pipeline Python này.

## Yêu cầu llama.cpp

Bản GGUF hiện yêu cầu:

1. chế độ encoder non-causal;
2. hỗ trợ đúng hybrid sliding-window attention của model;
3. option tùy biến `--output-token-ids` của `llama-embedding`.

Các chức năng này đang được đề xuất trong [ggml-org/llama.cpp#26286](https://github.com/ggml-org/llama.cpp/pull/26286). Help trong [llama-cpp.md](llama-cpp.md) của checkout upstream hiện có các option chung như `--pooling none`, `--embd-normalize -1`, `--ctx-size`, `--ubatch-size`, `--flash-attn`, `--n-gpu-layers` và `--fit`, nhưng **không có `--output-token-ids`**. Do đó binary upstream hiện có trong repository này không đủ để chạy `rerank.py`.

Cho tới khi PR được merge, build fork được model card chỉ định:

### CUDA

```bash
git clone https://github.com/littlewine/llama.cpp llama.cpp-jina
cmake -S llama.cpp-jina -B llama.cpp-jina/build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp-jina/build --config Release \
  -j "$(nproc)" --target llama-embedding
```

### CPU

```bash
git clone https://github.com/littlewine/llama.cpp llama.cpp-jina
cmake -S llama.cpp-jina -B llama.cpp-jina/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp-jina/build --config Release \
  -j "$(nproc)" --target llama-embedding
```

Trên macOS có thể thay option CUDA bằng `-DGGML_METAL=ON`.

Kiểm tra binary:

```bash
./llama.cpp-jina/build/bin/llama-embedding --version
./llama.cpp-jina/build/bin/llama-embedding --help | grep output-token-ids
```

Lệnh thứ hai phải tìm thấy `--output-token-ids`. Nếu không có, đang dùng nhầm binary upstream hoặc build cũ.

## Chọn quantization

| Quantization | Dung lượng công bố | Gợi ý |
|---|---:|---|
| BF16 | 1,2 GB | baseline chất lượng |
| Q8_0 | 610 MB | ưu tiên chất lượng, giảm gần một nửa dung lượng |
| Q6_K | 473 MB | chất lượng cao, nhỏ hơn Q8_0 |
| Q5_K_M / Q5_K_S | 424 / 417 MB | mức trung gian |
| Q4_K_M / Q4_K_S | 379 / 366 MB | điểm khởi đầu thực dụng |
| IQ4_NL / IQ4_XS | 366 / 353 MB | 4-bit nhỏ hơn |
| IQ3_S / IQ3_XS / IQ3_XXS | 309 / 299 / 267 MB | cần đánh giá chất lượng trên dữ liệu thật |
| IQ2_S / IQ2_XS / IQ2_XXS | 243 / 231 / 219 MB | ưu tiên dung lượng |
| IQ1_M / IQ1_S | 207 / 199 MB | cực nhỏ, chỉ dùng sau khi benchmark recall/ranking |

Nên bắt đầu bằng `Q4_K_M`, so sánh với `Q8_0` hoặc BF16 trên tập đánh giá của ứng dụng rồi mới quyết định. Dung lượng trên model card được làm tròn; file thực tế có thể hiển thị khác tùy đơn vị MB/MiB.

## Tải model và cài Python

Cài công cụ và dependency:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U huggingface_hub numpy safetensors torch tokenizers
```

Tải một quant cùng ba file bắt buộc:

```bash
mkdir -p models/jina-reranker-v3.5
hf download jinaai/jina-reranker-v3.5-GGUF \
  jina-reranker-v3.5-Q4_K_M.gguf \
  projector.safetensors tokenizer.json rerank.py \
  --local-dir models/jina-reranker-v3.5
```

Có thể đổi `Q4_K_M` thành quantization khác trong bảng. Không cần tải `imatrix.dat` để inference; file đó được dùng khi tạo các quant mức thấp.

## Smoke test bằng script chính thức

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  models/jina-reranker-v3.5/rerank.py \
  --model models/jina-reranker-v3.5/jina-reranker-v3.5-Q4_K_M.gguf \
  --projector models/jina-reranker-v3.5/projector.safetensors \
  --tokenizer models/jina-reranker-v3.5/tokenizer.json \
  --llama-embedding ./llama.cpp-jina/build/bin/llama-embedding
```

Script chạy ví dụ tích hợp sẵn về thủ đô nước Pháp. Document nói Paris là thủ đô phải đứng trên nội dung không liên quan. Không đối chiếu score với ngưỡng xác suất cố định; trước tiên kiểm tra thứ tự và khoảng cách tương đối giữa các candidate.

## Dùng từ Python

Khi import `rerank.py` từ thư mục model:

```python
import sys

sys.path.insert(0, "models/jina-reranker-v3.5")
from rerank import GGUFReranker

reranker = GGUFReranker(
    model_path=(
        "models/jina-reranker-v3.5/"
        "jina-reranker-v3.5-Q4_K_M.gguf"
    ),
    projector_path="models/jina-reranker-v3.5/projector.safetensors",
    tokenizer_path="models/jina-reranker-v3.5/tokenizer.json",
    llama_embedding_path="./llama.cpp-jina/build/bin/llama-embedding",
    max_ctx_size=65536,
    n_gpu_layers=99,
)

query = "Viêm khớp dạng thấp do nguyên nhân nào?"
documents = [
    "Viêm khớp dạng thấp là bệnh tự miễn, trong đó hệ miễn dịch tấn công khớp.",
    "Thoái hóa khớp chủ yếu liên quan đến hao mòn sụn theo thời gian.",
    "Hà Nội là thủ đô của Việt Nam.",
]

results = reranker.rerank(query, documents, top_n=2)
for item in results:
    print(item["index"], item["relevance_score"], item["document"])
```

Mỗi kết quả gồm:

- `index`: vị trí trong mảng document đầu vào;
- `relevance_score`: cosine score, càng cao càng liên quan;
- `document`: document gốc;
- `embedding`: mặc định là `None`, hoặc vector projected khi dùng `return_embeddings=True`.

Có thể truyền instruction theo domain:

```python
results = reranker.rerank(
    query,
    documents,
    top_n=2,
    instruction="Prioritize passages that provide medically accurate causal explanations.",
)
```

Nên viết instruction rõ ràng và giữ nguyên instruction giữa lúc đánh giá và production.

## Context, block splitting và tài nguyên

Các mặc định đáng chú ý trong `rerank.py`:

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `block_size` | 125 | số document tối đa trong một block |
| `max_length` | 131072 | ngân sách logic theo model gốc |
| `max_query_length` | 2048 | giới hạn query |
| `max_doc_length` | 8192 | giới hạn mỗi document |
| `max_ctx_size` | 65536 | context tối đa thực tế cho mỗi lần gọi llama.cpp |
| `--ubatch-size` nội bộ | 512 | physical micro-batch |
| `n_gpu_layers` | 99 | yêu cầu offload toàn bộ 28 layer nếu GPU hỗ trợ |

Model card nói model hỗ trợ 131K, nhưng script mặc định chặn mỗi subprocess ở 65.536 token. Muốn thử context dài hơn phải đặt `max_ctx_size=131072` và bảo đảm backend/phần cứng đủ bộ nhớ. Nếu gặp lỗi block prompt vượt context, giảm `block_size`, `max_doc_length` hoặc tăng `max_ctx_size`.

`rerank.py` tự làm tròn context của mỗi block lên bội số 256, chạy từng block bằng một subprocess `llama-embedding`, rồi fusion kết quả. Cách này đúng với implementation phát hành nhưng việc nạp/chạy subprocess nhiều lần có thể làm latency tăng khi có nhiều block. Hãy benchmark bằng số lượng và độ dài document thực tế.

### Chạy CPU

Đặt `n_gpu_layers=0` trong `GGUFReranker` và dùng binary build CPU:

```python
reranker = GGUFReranker(
    model_path="models/jina-reranker-v3.5/jina-reranker-v3.5-Q4_K_M.gguf",
    projector_path="models/jina-reranker-v3.5/projector.safetensors",
    tokenizer_path="models/jina-reranker-v3.5/tokenizer.json",
    llama_embedding_path="./llama.cpp-jina/build/bin/llama-embedding",
    n_gpu_layers=0,
)
```

Với CUDA, `CUDA_VISIBLE_DEVICES` là cách đơn giản để giới hạn GPU vì wrapper hiện không expose option `--device`.

## Các option llama.cpp mà wrapper sử dụng

`rerank.py` tự gọi binary gần tương đương:

```text
llama-embedding -m MODEL -f PROMPT_FILE \
  --no-escape \
  --pooling none \
  --embd-separator <#JINA_SEP#> \
  --embd-normalize -1 \
  --embd-output-format json \
  --output-token-ids 151670,151671 \
  --ubatch-size 512 \
  --ctx-size CONTEXT \
  --flash-attn on \
  --fit off \
  -ngl 99
```

Ý nghĩa quan trọng:

- `--pooling none`: lấy hidden state theo token, không mean/last/rank pooling;
- `--embd-normalize -1`: không chuẩn hóa hidden state trước projector;
- `--output-token-ids 151670,151671`: chỉ xuất vị trí document/query đặc biệt;
- `--no-escape`: tránh biến đổi backslash làm lệch token count;
- `--fit off`: không để cơ chế auto-fit tự thay context;
- `--flash-attn on`: bật Flash Attention;
- `-ngl`: số layer offload sang GPU.

Không nên tự đổi pooling hoặc normalization nếu muốn giữ hành vi tương đương model gốc.

## Dùng checkpoint Transformers gốc

Nếu không cần GGUF, checkpoint gốc đơn giản hơn vì remote code đã bao gồm prompt, projector, block splitting và scoring:

```bash
pip install -U transformers torch
```

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "jinaai/jina-reranker-v3.5",
    dtype="auto",
    trust_remote_code=True,
)
model.eval()

results = model.rerank(
    "Trà xanh có lợi ích gì cho sức khỏe?",
    [
        "Trà xanh chứa catechin có tính chống oxy hóa.",
        "Bóng rổ là một môn thể thao phổ biến.",
        "Green tea may support metabolism and brain function.",
    ],
    top_n=2,
)
```

`trust_remote_code=True` là bắt buộc vì model dùng implementation tùy biến. Có thể thêm `return_embeddings=True` nếu cần document embedding do reranker tạo ra.

## Jina Reranker API

Dịch vụ do Jina vận hành dùng schema rerank thông thường:

```bash
curl -X POST https://api.jina.ai/v1/rerank \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $JINA_API_KEY" \
  -d '{
    "model": "jina-reranker-v3.5",
    "query": "slm markdown",
    "documents": [
      "Tài liệu về small language models",
      "Hướng dẫn cú pháp Markdown",
      "Một bài viết không liên quan"
    ],
    "top_n": 2,
    "return_documents": false
  }'
```

Response có `results[]` đã sắp theo score giảm dần, với `index` trỏ về document đầu vào. Nếu đang dùng API của `jina-reranker-v3`, chỉ cần đổi model thành `jina-reranker-v3.5`; schema request giữ nguyên.

## Lỗi thường gặp

### `unrecognized argument: --output-token-ids`

Đang dùng stock llama.cpp hoặc fork/build cũ. Build đúng `https://github.com/littlewine/llama.cpp` và trỏ `llama_embedding_path` tới binary mới.

### Thiếu `projector.safetensors`

GGUF chỉ chứa backbone. Tải projector cùng repository và truyền đúng `projector_path`.

### Không tìm thấy `tokenizer.json`

Đặt tokenizer cạnh file GGUF hoặc truyền `tokenizer_path` rõ ràng. Không thay bằng tokenizer tùy ý từ một checkpoint Qwen khác.

### `Expected ... output embeddings`

Thường do binary không hỗ trợ đúng `--output-token-ids`, tokenizer/prompt không đồng bộ, hoặc đã sửa `rerank.py`. Dùng nguyên bộ `rerank.py`, `tokenizer.json`, projector và GGUF từ cùng repository/revision.

### Block vượt `max_ctx_size`

Giảm `block_size`/`max_doc_length`, hoặc tăng `max_ctx_size` nếu phần cứng và build hỗ trợ. Context 131K là giới hạn model, không phải cam kết rằng cấu hình mặc định đủ RAM/VRAM.

### Score khác Transformers

Quantization có thể làm thay đổi score. Ngoài ra cần kiểm tra cùng query, instruction, giới hạn token, block fusion và revision. So sánh thứ tự/nDCG trên tập dữ liệu thật thay vì yêu cầu float giống tuyệt đối.

## Ghi chú triển khai

- Dùng reranker sau bước retrieval đầu tiên; chỉ rerank top-k candidate thay vì toàn bộ corpus.
- Giữ cố định model revision, quantization, tokenizer, projector và fork llama.cpp trong production.
- Không trộn `rerank.py` hoặc projector từ revision khác với GGUF.
- Đánh giá các quant thấp bằng Recall@k, MRR hoặc nDCG trên dữ liệu domain thật.
- Wrapper phát hành gọi một subprocess cho mỗi block và chưa cung cấp HTTP server. Nếu cần service, bọc một process Python dài hạn bằng FastAPI/Flask; không thay bằng endpoint rerank chuẩn của stock `llama-server`.
- Tuân thủ CC BY-NC 4.0 và liên hệ Jina AI nếu cần sử dụng thương mại.

## Nguồn

- [jinaai/jina-reranker-v3.5](https://huggingface.co/jinaai/jina-reranker-v3.5)
- [jinaai/jina-reranker-v3.5-GGUF](https://huggingface.co/jinaai/jina-reranker-v3.5-GGUF)
- [PR hỗ trợ llama.cpp #26286](https://github.com/ggml-org/llama.cpp/pull/26286)
- [Fork llama.cpp được model card chỉ định](https://github.com/littlewine/llama.cpp)
- [Help llama.cpp cục bộ](llama-cpp.md)
