# Chạy jina-embeddings-v5-omni bằng llama.cpp

## 1. Tổng quan

`jina-embeddings-v5-omni` là họ model embedding đa phương thức của Jina AI. Model nhận **text, ảnh, video và audio**, rồi ánh xạ tất cả vào cùng một không gian vector. Không gian này tương thích với model `jina-embeddings-v5-text` cùng kích thước, vì vậy có thể lập chỉ mục corpus bằng text và truy vấn bằng modality khác mà không cần re-index.

| Thuộc tính | `nano` | `small` |
|---|---:|---:|
| Tham số của model base | ~1.04B | ~1.74B |
| Tham số bản task-specific GGUF | ~0.95B | ~1.56B |
| Số chiều đầy đủ | 768 | 1024 |
| Context tối đa | 8.192 | 32.768 |
| Pooling | last token | last token |
| Matryoshka dimensions | 32, 64, 128, 256, 512, 768 | 32, 64, 128, 256, 512, 768, 1024 |
| Input | text, image, video, audio | text, image, video, audio |
| Giấy phép | CC BY-NC 4.0 | CC BY-NC 4.0 |

Model base hỗ trợ bốn task adapter:

- `retrieval`: semantic search và RAG;
- `classification`: tạo đặc trưng để phân loại;
- `clustering`: gom cụm;
- `text-matching`: similarity đối xứng, paraphrase và duplicate detection.

Với llama.cpp, dùng repository GGUF đã merge đúng adapter. Cả `nano` và `small` đều có đủ bốn task theo pattern:

```text
jinaai/jina-embeddings-v5-omni-{size}-{task}-GGUF
```

| Task | Repository `nano` | Repository `small` |
|---|---|---|
| Retrieval | `jinaai/jina-embeddings-v5-omni-nano-retrieval-GGUF` | `jinaai/jina-embeddings-v5-omni-small-retrieval-GGUF` |
| Text matching | `jinaai/jina-embeddings-v5-omni-nano-text-matching-GGUF` | `jinaai/jina-embeddings-v5-omni-small-text-matching-GGUF` |
| Clustering | `jinaai/jina-embeddings-v5-omni-nano-clustering-GGUF` | `jinaai/jina-embeddings-v5-omni-small-clustering-GGUF` |
| Classification | `jinaai/jina-embeddings-v5-omni-nano-classification-GGUF` | `jinaai/jina-embeddings-v5-omni-small-classification-GGUF` |

Mỗi repository là một model task-specific độc lập; llama.cpp không đổi adapter lúc runtime. Không dùng model của task này cho task khác. Model classification chỉ sinh vector đặc trưng, không trực tiếp trả về nhãn.

> **Giấy phép:** CC BY-NC 4.0 không cho phép sử dụng thương mại nếu chưa có thỏa thuận riêng với Jina AI.

## 2. Cấu trúc GGUF và chọn model

Mỗi bản task-specific gồm ba phần độc lập:

1. text-tower GGUF, chứa model, vocabulary và tokenizer;
2. vision `mmproj` F16 cho ảnh và video;
3. audio `mmproj` F16 cho audio.

Tên file tuân theo cùng một pattern:

| Model | Text GGUF khuyến nghị | Vision projector | Audio projector |
|---|---|---|---|
| nano | `jina-embeddings-v5-omni-nano-{task}-Q5_K_M.gguf` | `jina-embeddings-v5-omni-nano-{task}-vision-mmproj-F16.gguf` | `jina-embeddings-v5-omni-nano-{task}-audio-mmproj-F16.gguf` |
| small | `jina-embeddings-v5-omni-small-{task}-Q4_K_M.gguf` | `jina-embeddings-v5-omni-small-{task}-vision-mmproj-F16.gguf` | `jina-embeddings-v5-omni-small-{task}-audio-mmproj-F16.gguf` |

Thay `{task}` bằng `retrieval`, `text-matching`, `clustering` hoặc `classification`. Model và hai projector phải thuộc **cùng size và cùng task**; không ghép text GGUF retrieval với projector classification.

Khuyến nghị quantization từ model card:

- **nano:** bắt đầu bằng `Q5_K_M`; dùng `Q6_K` hoặc `Q8_0` nếu dữ liệu có nhiều tiêu đề, query rất ngắn hoặc đa ngôn ngữ. `Q4_K_M` giảm đáng kể độ tương đồng với torch trên input 2–4 token;
- **small:** `Q4_K_M` là mức mặc định thực dụng; dùng `Q5_K_M`, `Q6_K` hoặc `Q8_0` khi ưu tiên chất lượng;
- `Q2_K`, IQ1 và IQ2 chỉ phù hợp thử nghiệm khi rất thiếu bộ nhớ;
- vision/audio projector chỉ được phát hành ở F16. Không tự quantize projector vì model card ghi nhận suy giảm parity lớn.

Có thể tải cả ba file bằng `huggingface-cli`:

```bash
pip install -U huggingface_hub

huggingface-cli download \
  jinaai/jina-embeddings-v5-omni-small-retrieval-GGUF \
  jina-embeddings-v5-omni-small-retrieval-Q4_K_M.gguf \
  jina-embeddings-v5-omni-small-retrieval-vision-mmproj-F16.gguf \
  jina-embeddings-v5-omni-small-retrieval-audio-mmproj-F16.gguf \
  --local-dir models/jina-omni-small
```

Đổi `small` thành `nano` và `Q4_K_M` thành `Q5_K_M` để dùng bản nano. Để đổi task, thay toàn bộ `retrieval` trong repository và cả ba tên file bằng task tương ứng; các command còn lại trong tài liệu dùng retrieval làm ví dụ.

## 3. Bắt buộc dùng fork llama.cpp của Jina

Các patch cần cho audio chunked attention, video temporal-pair, combined encoder decode và Jina v5 omni **chưa có trong llama.cpp upstream** tại thời điểm model card được công bố. Không dùng image Docker hoặc binary upstream nếu chưa xác nhận các patch này đã được merge.

```bash
git clone https://github.com/jina-ai/llama.cpp.git
cd llama.cpp
git checkout feat-v5-omni

# CUDA
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j "$(nproc)"
```

Bỏ `-DGGML_CUDA=ON` nếu chỉ dùng CPU. Kiểm tra binary và backend:

```bash
./build/bin/llama-server --version
./build/bin/llama-server --list-devices
```

Các option liên quan trong [llama-cpp.md](llama-cpp.md):

- `--embedding`/`--embeddings`: giới hạn server cho embedding model;
- `--pooling last`: lấy hidden state token cuối;
- `--embd-normalize 2`: chuẩn hóa L2, cũng là mặc định;
- `-b` và `-ub`: logical batch và physical micro-batch;
- `-ngl all`: offload tối đa layer sang GPU;
- `--device`: chọn backend, kiểm tra tên bằng `--list-devices`;
- `--mmproj`: nạp multimodal projector;
- `--mmproj-offload`/`--no-mmproj-offload`: đặt projector trên GPU/CPU;
- `--parallel`: số server slot;
- `--api-key`, `--metrics`, `--slots`: xác thực và giám sát.

Riêng fork `feat-v5-omni` cho phép truyền `--mmproj` nhiều lần: tối đa một vision projector và một audio projector.

### H100/H200

Trên GPU Hopper, đặt biến sau trước khi chạy server:

```bash
export GGML_CUDA_DISABLE_GRAPHS=1
```

Nếu không, model card cảnh báo CUDA graph có thể crash với lỗi `cudaMemcpyAsync ... illegal instruction` khi trích xuất embedding. CPU, Metal, Vulkan, L4 và A100 không bị ảnh hưởng.

## 4. Sanity check chỉ với text

`-hf REPO:QUANT` tự tải và cache text GGUF. Nó thuận tiện cho kiểm tra một lần:

```bash
./build/bin/llama-embedding \
  -hf jinaai/jina-embeddings-v5-omni-small-retrieval-GGUF:Q4_K_M \
  --pooling last \
  --embd-normalize 2 \
  -p 'Query: Hành tinh nào được gọi là Hành tinh Đỏ?' \
  --embd-output-format json
```

Với nano, thay repository và quant bằng:

```text
jinaai/jina-embeddings-v5-omni-nano-retrieval-GGUF:Q5_K_M
```

## 5. Khởi chạy server

### 5.1. Text + ảnh + video

Ảnh và video dùng chung vision projector:

```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-server \
  -m models/jina-omni-small/jina-embeddings-v5-omni-small-retrieval-Q4_K_M.gguf \
  --mmproj models/jina-omni-small/jina-embeddings-v5-omni-small-retrieval-vision-mmproj-F16.gguf \
  --alias jina-embeddings-v5-omni-small-retrieval \
  --embedding --pooling last --embd-normalize 2 \
  -c 32768 -b 8192 -ub 1024 \
  -fa on -ngl all --device CUDA0 \
  --parallel 1 \
  --host 0.0.0.0 --port 8080 \
  --api-key llama-cpp-api-key \
  --metrics --slots
```

### 5.2. Text + audio

Audio 30 giây có thể nở thành khoảng 750 token. Model card khuyên tăng batch vật lý ít nhất lên 4096:

```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-server \
  -m models/jina-omni-small/jina-embeddings-v5-omni-small-retrieval-Q4_K_M.gguf \
  --mmproj models/jina-omni-small/jina-embeddings-v5-omni-small-retrieval-audio-mmproj-F16.gguf \
  --alias jina-embeddings-v5-omni-small-retrieval \
  --embedding --pooling last --embd-normalize 2 \
  -c 32768 -b 4096 -ub 4096 \
  -fa on -ngl all --device CUDA0 \
  --parallel 1 \
  --host 0.0.0.0 --port 8081 \
  --api-key llama-cpp-api-key
```

### 5.3. Nạp tất cả modality trong một process

```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-server \
  -m models/jina-omni-small/jina-embeddings-v5-omni-small-retrieval-Q4_K_M.gguf \
  --mmproj models/jina-omni-small/jina-embeddings-v5-omni-small-retrieval-vision-mmproj-F16.gguf \
  --mmproj models/jina-omni-small/jina-embeddings-v5-omni-small-retrieval-audio-mmproj-F16.gguf \
  --alias jina-embeddings-v5-omni-small-retrieval \
  --embedding --pooling last --embd-normalize 2 \
  -c 32768 -b 8192 -ub 8192 \
  -fa on -ngl all --device CUDA0 \
  --parallel 1 \
  --host 0.0.0.0 --port 8080 \
  --api-key llama-cpp-api-key
```

Nếu thiếu VRAM, ưu tiên chỉ nạp projector cần dùng, giảm `-c`, `-b`, `-ub`, dùng `--no-mmproj-offload`, hoặc chọn quant text nhỏ hơn. Với nano, context tối đa là `8192` và quant nên là `Q5_K_M`.

### 5.4. CPU

```bash
./build/bin/llama-server \
  -m models/jina-omni-small/jina-embeddings-v5-omni-small-retrieval-Q4_K_M.gguf \
  --mmproj models/jina-omni-small/jina-embeddings-v5-omni-small-retrieval-vision-mmproj-F16.gguf \
  --alias jina-embeddings-v5-omni-small-retrieval \
  --embedding --pooling last --embd-normalize 2 \
  -c 8192 -b 4096 -ub 512 \
  --device none --threads -1 \
  --host 0.0.0.0 --port 8080 \
  --api-key llama-cpp-api-key
```

## 6. Quy tắc prefix theo task

Quy tắc của các model card **omni GGUF** là:

| Task GGUF | Input gửi tới llama.cpp |
|---|---|
| Retrieval query | `Query: {input}` |
| Retrieval document/corpus | `Document: {input}` |
| Text matching | `{input}` — không prefix |
| Clustering | `{input}` — không prefix |
| Classification | `{input}` — không prefix |

Với retrieval, quy tắc query/document áp dụng cho cả text lẫn multimodal input. Ví dụ ảnh query là `Query: <media-marker>`, còn ảnh document là `Document: <media-marker>`. llama.cpp không gọi `encode_query()` hoặc `encode_document()`, nên client phải thêm prefix retrieval thủ công. Không dùng `Passage:`.

Các GGUF card của classification, clustering và text-matching lại yêu cầu embed input **nguyên văn**, không thêm `Query:` hay `Document:`. Đây là khác biệt cần chú ý so với đường raw Transformers/vLLM của model base hoặc task-specific PyTorch, nơi model card yêu cầu `Document:` cho task không phải retrieval. Hãy tuân theo model card của đúng artifact/backend đang chạy, không sao chép preprocessing giữa GGUF và Transformers.

## 7. API text tương thích OpenAI

Dùng `/v1/embeddings` cho text:

```bash
curl -fsS http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer llama-cpp-api-key' \
  -d '{
    "model": "jina-embeddings-v5-omni-small-retrieval",
    "input": [
      "Query: Hành tinh nào được gọi là Hành tinh Đỏ?",
      "Document: Sao Hỏa được gọi là Hành tinh Đỏ vì bề mặt có màu đỏ.",
      "Document: Sao Kim có kích thước gần giống Trái Đất."
    ],
    "encoding_format": "float"
  }'
```

Server trả một vector cho mỗi phần tử `input`. Không áp dụng chat template: retrieval chỉ cần prefix đúng; các GGUF task còn lại nhận input nguyên văn. Không cần system/user/assistant message như một số model Qwen embedding khác.

## 8. Media marker của llama.cpp

Endpoint OpenAI `/v1/embeddings` chuẩn chỉ dành cho text. Với multimedia, dùng endpoint low-level `/embedding` hoặc `/embeddings`, trường `prompt_string` và dữ liệu base64.

Model card GGUF dùng marker cố định `<__media__>`. Tuy nhiên các bản llama.cpp mới có thể sinh marker ngẫu nhiên lúc server khởi động để tránh xung đột với nội dung người dùng. Nếu `GET /props` có trường `media_marker`, luôn dùng giá trị đó:

```bash
MEDIA_MARKER="$(curl -fsS http://127.0.0.1:8080/props \
  -H 'Authorization: Bearer llama-cpp-api-key' | jq -r '.media_marker // "<__media__>"')"
printf '%s\n' "$MEDIA_MARKER"
```

Lấy lại marker sau mỗi lần restart. Nếu fork `feat-v5-omni` đang dùng không trả `media_marker`, dùng `<__media__>` đúng như model card. Số marker trong `prompt_string` phải bằng số phần tử media tương ứng.

## 9. Embedding ảnh

```bash
IMAGE_B64="$(base64 -w 0 photo.jpg)"
MEDIA_MARKER="$(curl -fsS http://127.0.0.1:8080/props \
  -H 'Authorization: Bearer llama-cpp-api-key' | jq -r '.media_marker // "<__media__>"')"

jq -n \
  --arg prompt "Query: ${MEDIA_MARKER}" \
  --arg image "$IMAGE_B64" \
  '{content: [{prompt_string: $prompt, multimodal_data: [$image]}]}' |
curl -fsS http://127.0.0.1:8080/embeddings \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer llama-cpp-api-key' \
  -d @-
```

Một ảnh document dùng `Document: ${MEDIA_MARKER}`. Text + ảnh tạo **một embedding hợp nhất** bằng cách đặt text và marker trong cùng `prompt_string`, ví dụ `Query: đôi giày mùa đông ${MEDIA_MARKER}`.

Trên macOS, thay `base64 -w 0 photo.jpg` bằng `base64 < photo.jpg | tr -d '\n'`.

## 10. Embedding audio

Server tự resample WAV, MP3 hoặc FLAC về mono 16 kHz. Ví dụ gửi audio như một query:

```bash
AUDIO_B64="$(base64 -w 0 speech.wav)"
MEDIA_MARKER="$(curl -fsS http://127.0.0.1:8081/props \
  -H 'Authorization: Bearer llama-cpp-api-key' | jq -r '.media_marker // "<__media__>"')"

jq -n \
  --arg prompt "Query: ${MEDIA_MARKER}" \
  --arg audio "$AUDIO_B64" \
  '{content: [{prompt_string: $prompt, multimodal_data: [$audio]}]}' |
curl -fsS http://127.0.0.1:8081/embeddings \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer llama-cpp-api-key' \
  -d @-
```

Theo model card, clip 11 giây sinh khoảng 275 audio token, clip 30 giây khoảng 750 token.

## 11. Embedding video

Vision encoder dùng temporal patch size 2. Client phải decode video thành frame, ghép các frame liên tiếp thành cặp, encode từng frame thành base64 và gửi qua `videopair_data`:

```python
import base64
import io

import imageio.v3 as iio
import numpy as np
import requests
from PIL import Image

URL = "http://127.0.0.1:8080"
HEADERS = {"Authorization": "Bearer llama-cpp-api-key"}

props = requests.get(f"{URL}/props", headers=HEADERS).json()
marker = props.get("media_marker", "<__media__>")
frames = list(iio.imiter("clip.mp4"))


def b64(frame):
    buf = io.BytesIO()
    Image.fromarray(np.asarray(frame)).convert("RGB").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


pairs = [
    (b64(frames[i]), b64(frames[i + 1]))
    for i in range(0, len(frames) - 1, 2)
]
prompt = "Query: " + marker * len(pairs)

response = requests.post(
    f"{URL}/embeddings",
    headers=HEADERS,
    json={"content": [{"prompt_string": prompt, "videopair_data": pairs}]},
)
response.raise_for_status()
embedding = response.json()[0]["embedding"]
```

Cài thư viện cho ví dụ:

```bash
pip install requests pillow numpy imageio imageio-ffmpeg
```

Không gửi trực tiếp file MP4 vào `multimodal_data` theo ví dụ ảnh. Luồng GGUF được model card xác nhận dùng `videopair_data`.

## 12. Batching

Endpoint `/embeddings` nhận danh sách trong `content`:

```bash
curl -fsS http://127.0.0.1:8080/embeddings \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer llama-cpp-api-key' \
  -d '{
    "content": [
      {"prompt_string": "Query: truy vấn thứ nhất"},
      {"prompt_string": "Query: truy vấn thứ hai"}
    ]
  }'
```

Mỗi phần tử trả về một embedding. Text hưởng lợi nhiều nhất từ batch lớn. Multimedia vẫn được forward theo từng sample. Tăng `-b`/`-ub` dần và benchmark với dữ liệu thật; giảm `-ub` nếu peak VRAM quá cao.

## 13. Matryoshka: giảm số chiều

llama.cpp trả vector đầy đủ. Để dùng vector ngắn hơn, lấy đúng số chiều đầu và chuẩn hóa L2 lại:

```python
import numpy as np

full = np.asarray(embedding, dtype=np.float32)
short = full[:256]
short /= np.linalg.norm(short)
```

Chỉ dùng các kích thước đã huấn luyện:

- nano: `32, 64, 128, 256, 512, 768`;
- small: `32, 64, 128, 256, 512, 768, 1024`.

Corpus và query phải dùng cùng model, task, prefix, quantization, số chiều và quy tắc chuẩn hóa. Không cắt tùy ý thành một kích thước như 300.

## 14. Dùng model base ngoài llama.cpp

Model base phù hợp khi cần classification, clustering, text-matching hoặc muốn chọn adapter lúc load. Cách dễ nhất cho cả bốn modality là `sentence-transformers`:

```bash
pip install torch transformers sentence-transformers pillow numpy
# Tùy modality:
pip install librosa soundfile av imageio imageio-ffmpeg
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "jinaai/jina-embeddings-v5-omni-small",
    trust_remote_code=True,
    model_kwargs={"default_task": "retrieval", "modality": "omni"},
)

query = model.encode_query("Hành tinh nào được gọi là Hành tinh Đỏ?")
document = model.encode_document("Sao Hỏa thường được gọi là Hành tinh Đỏ.")
image_query = model.encode_query("photo.jpg")
audio_document = model.encode_document("speech.wav")
```

Có thể đặt `modality` thành `text`, `vision`, `audio` hoặc `omni` để không nạp tower không cần thiết. Với task không phải retrieval, chọn `default_task` tương ứng và dùng `encode_document()`; các task này chỉ dùng prefix `Document: `, không có phân biệt query/document.

Yêu cầu model card: `torch>=2.5`; nano multimodal cần `transformers>=5.0` còn text-only chạy từ `>=4.57`; small cần `transformers>=4.57` và khuyến nghị `>=5.1`.

## 15. Lỗi thường gặp

- **Model hoặc audio/video không chạy trên upstream:** dùng đúng fork `jina-ai/llama.cpp`, branch `feat-v5-omni`.
- **Embedding retrieval kém:** kiểm tra `Query: `, `Document: `, `--pooling last` và L2 normalization.
- **Số media marker không khớp:** mỗi ảnh/audio cần đúng một marker; mỗi cặp frame video cần một marker.
- **Marker `<__media__>` không hoạt động:** lấy `.media_marker` từ `/props`; không thêm backslash vào marker.
- **Thiếu projector:** nạp vision projector cho ảnh/video hoặc audio projector cho audio. Text GGUF một mình chỉ xử lý text.
- **OOM:** chỉ nạp modality cần thiết, giảm context/batch/micro-batch, dùng quant text nhỏ hơn hoặc `--no-mmproj-offload`.
- **Audio dài lỗi batch:** tăng cả `-b` và `-ub`, bắt đầu từ 4096 cho clip tối đa khoảng 30 giây.
- **H100/H200 crash CUDA graph:** đặt `GGML_CUDA_DISABLE_GRAPHS=1`.
- **Nano cho kết quả kém trên query rất ngắn:** dùng ít nhất Q5_K_M; thử Q6_K/Q8_0 hoặc chuyển sang small.
- **Đổi model/pipeline trong production:** phải re-embed corpus nếu đổi model, task, prefix, pooling, normalization hoặc số chiều.

## Nguồn

- [Omni small retrieval GGUF](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small-retrieval-GGUF)
- [Omni small text-matching GGUF](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small-text-matching-GGUF)
- [Omni small clustering GGUF](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small-clustering-GGUF)
- [Omni small classification GGUF](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small-classification-GGUF)
- [Omni nano retrieval GGUF](https://huggingface.co/jinaai/jina-embeddings-v5-omni-nano-retrieval-GGUF)
- [Omni nano text-matching GGUF](https://huggingface.co/jinaai/jina-embeddings-v5-omni-nano-text-matching-GGUF)
- [Omni nano clustering GGUF](https://huggingface.co/jinaai/jina-embeddings-v5-omni-nano-clustering-GGUF)
- [Omni nano classification GGUF](https://huggingface.co/jinaai/jina-embeddings-v5-omni-nano-classification-GGUF)
- [jina-embeddings-v5-omni-small](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small)
- [jina-embeddings-v5-omni-nano](https://huggingface.co/jinaai/jina-embeddings-v5-omni-nano)
- [Jina v5 omni GGUF umbrella](https://github.com/jina-ai/jina-embeddings-v5-omni-gguf)
- [Fork llama.cpp của Jina](https://github.com/jina-ai/llama.cpp/tree/feat-v5-omni)
- Help của binary trong repository: [llama-cpp.md](llama-cpp.md)
