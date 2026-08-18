# Hướng dẫn chạy Qwen3.8-27B bằng `llama.cpp`

Tài liệu này được tổng hợp từ toàn bộ các tệp trong `Qwen3.8-27B/` và bản trợ giúp `llama-cpp.md` của **chính binary hiện tại**. Các số tốc độ từ diễn đàn chỉ nên xem là điểm tham khảo: phiên bản `llama.cpp`, quant GGUF, driver, CUDA, CPU/RAM, prompt và độ dài context thực tế đều có thể làm kết quả thay đổi.

## 1. Cấu hình nên dùng ngay

### 1.1. RTX 5090 32 GB — cân bằng chất lượng/tốc độ, text và coding

Ưu tiên `UD-Q6_K_XL` nếu cần chất lượng; đổi thành `UD-Q5_K_XL` nếu cần thêm khoảng trống VRAM hoặc vision. Cấu hình text-only dưới đây là điểm bắt đầu hợp lý:

```bash
./llama.cpp/llama-server \
  -hf unsloth/Qwen3.8-27B-GGUF:UD-Q6_K_XL \
  --no-mmproj \
  --host 0.0.0.0 --port 8080 \
  --alias qwen3.8-27b \
  --n-gpu-layers all --gpu-layers-draft all \
  --fit off \
  --ctx-size 131072 --parallel 1 --kv-unified \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-type-k-draft q8_0 --cache-type-v-draft q8_0 \
  --batch-size 1024 --ubatch-size 512 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.0 \
  --jinja --reasoning on --reasoning-effort medium \
  --reasoning-preserve --reasoning-budget 8192 \
  --reasoning-budget-message " -- Reasoning budget exceeded, proceed to final answer." \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --cache-prompt --cache-ram 8192 --ctx-checkpoints 32
```

Ghi chú cho 5090:

- Báo cáo thực tế với Q6 + MTP đạt khoảng **90–115 tok/s** sau khi ổn định; Q4/Q5 có thể đạt khoảng 100–120 tok/s. Đây không phải cam kết hiệu năng.
- `--spec-draft-n-max 2` thường là điểm tối ưu. Sweep Q4 cho thấy `n=2` nhanh nhất; tăng lên 3–5 có thể tốt với một số prompt nhưng cũng có thể chậm hơn. Hãy benchmark `2`, `3`, rồi `none`.
- Nếu OOM: hạ `--ctx-size` xuống `114688`, `98304` hoặc `65536`; sau đó mới giảm quant từ Q6 xuống Q5/Q4.
- `--no-mmproj` rất quan trọng nếu chỉ dùng text: vision projector và MTP tranh VRAM với KV cache. Nếu cần vision, bỏ cờ này, dùng `--image-min-tokens 1024`, bắt buộc giữ `--parallel 1`, và bắt đầu ở context 64K.
- Có thể dùng `--no-mmproj-offload` để chuyển projector sang CPU, lấy thêm context nhưng xử lý ảnh có thể chậm khoảng 5–10 lần; tốc độ text sau đó gần như không đổi.
- Chỉ đặt `GGML_CUDA_DISABLE_GRAPHS=1` khi gặp treo/stall CUDA graph trên Blackwell hoặc khi offload động. Tắt graph có thể làm mất hiệu năng, không nên dùng mặc định nếu hệ thống ổn định.
- Với server CUDA từng được báo lỗi khiến KV `q5_1` rơi về CPU và prefill chậm 33–45 lần. `q8_0` là lựa chọn an toàn trước tiên.

Nếu Q6 không đủ VRAM trên máy đang dùng GPU để xuất màn hình:

```bash
# Chỉ thay model; giữ các cờ còn lại
-hf unsloth/Qwen3.8-27B-GGUF:UD-Q5_K_XL
```

### 1.2. NVIDIA A100 40 GB — chất lượng cao, một GPU

BF16 của model khoảng 55 GB nên **không vừa** một A100 40 GB. Nên dùng:

- Mặc định: `UD-Q6_K_XL` + KV `q8_0`, context 128K, MTP 2.
- Ưu tiên chất lượng tối đa: GGUF `Q8_0` khoảng 29 GB, bắt đầu context 64K; tăng dần nếu còn VRAM.
- Ưu tiên context/throughput: `UD-Q5_K_XL`, context 128K–192K tùy mức dùng MTP/vision.

```bash
./llama.cpp/llama-server \
  -hf unsloth/Qwen3.8-27B-GGUF:UD-Q6_K_XL \
  --no-mmproj \
  --device CUDA0 \
  --host 0.0.0.0 --port 8080 \
  --alias qwen3.8-27b \
  --n-gpu-layers all --gpu-layers-draft all \
  --fit off \
  --ctx-size 131072 --parallel 1 --kv-unified \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-type-k-draft q8_0 --cache-type-v-draft q8_0 \
  --batch-size 2048 --ubatch-size 512 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.0 \
  --jinja --reasoning on --reasoning-effort medium \
  --reasoning-preserve --reasoning-budget 8192 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --cache-prompt --cache-ram 16384 --ctx-checkpoints 32
```

A100 có băng thông bộ nhớ lớn nhưng không có lợi thế kernel Blackwell. Không dùng biến môi trường workaround dành cho RTX 50 nếu không gặp lỗi. Tệp nguồn chỉ có số A100 80 GB chạy **vLLM**, vì vậy không nên lấy con số 18 tok/s trong đó làm dự đoán cho A100 40 GB chạy `llama.cpp` GGUF.

## 2. Cách chọn model, context và KV cache

### Quant model

| Nhu cầu | Quant gợi ý | Nhận xét |
|---|---|---|
| Chất lượng cao trên 32–40 GB | `UD-Q6_K_XL` | Điểm cân bằng tốt, khoảng 26 GB theo dữ liệu tham khảo |
| Thêm context/vision/MTP | `UD-Q5_K_XL` | Khoảng 20 GB, chất lượng vẫn cao |
| Tốc độ/context, còn muốn Q4 | `UD-Q4_K_XL`, `Q4_K_M` | Khoảng 17–18 GB |
| VRAM rất hạn chế | `UD-Q3_K_XL`, `IQ3_XXS` | Có suy giảm reasoning/coding; chỉ chọn khi cần |
| Chất lượng gần gốc trên A100 40 GB | `Q8_0` | Khoảng 29 GB; context và MTP bị giới hạn hơn |

Không nên mặc định chọn quant thấp nhất chỉ để quảng cáo context lớn. Với coding/agent, lỗi cú pháp, tool call và khả năng giữ chỉ dẫn dài thường suy giảm trước khi câu trả lời ngắn trông có vẻ sai.

### KV cache

- `f16`: chất lượng chuẩn, tốn VRAM nhất.
- `q8_0`: lựa chọn mặc định được khuyến nghị; gần lossless, kernel CUDA thường ổn định.
- `q5_1`/`q5_0`: tiết kiệm hơn nhưng phải kiểm tra có CPU fallback trong build đang dùng hay không.
- `q4_0`/`q4_1`/`iq4_nl`: dành cho context rất dài hoặc card ít VRAM; cần test dài hạn vì sai lệch cuối context/tool call có thể xuất hiện. Một số build phải biên dịch với `-DGGML_CUDA_FA_ALL_QUANTS=ON` để Flash Attention cho KV 4/5-bit không rơi về CPU.
- K và V có thể đặt khác nhau, nhưng nên bắt đầu cùng `q8_0` để dễ chẩn đoán.

Thứ tự xử lý OOM nên là: giảm context → giảm batch/ubatch → tắt vision → tắt MTP → đổi Q6 thành Q5/Q4 → cuối cùng mới quant KV xuống Q5/Q4 hoặc offload CPU.

## 3. MTP, n-gram và reasoning

### MTP tích hợp

Qwen3.8-27B GGUF phù hợp có head `nextn`, vì vậy không cần model draft rời:

```bash
--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.0
```

MTP dùng thêm VRAM cho draft context. Trên 5090 Q5, một báo cáo đo được MTP làm giảm trần context từ khoảng 190K xuống 144K nhưng gần gấp đôi tốc độ sinh token (66 → 121 tok/s). Nếu ưu tiên context tối đa hơn tốc độ, dùng `--spec-type none`.

Có thể thử n-gram cùng MTP cho agent sửa code hoặc lặp lại nội dung đã có trong prompt:

```bash
--spec-type ngram-mod,draft-mtp \
--spec-draft-n-max 2 \
--spec-ngram-mod-n-match 24 \
--spec-ngram-mod-n-min 48 \
--spec-ngram-mod-n-max 64
```

N-gram gần như không giúp nội dung hoàn toàn mới và đôi khi còn chậm hơn. Benchmark đúng workload của bạn.

### Reasoning effort

Template chính thức của Qwen3.8 hỗ trợ đúng ba mức: `low`, `medium`, `xhigh` (mặc định). Dù trợ giúp chung của `llama-server` liệt kê thêm `minimal`, `high`, `max`, template Qwen có thể từ chối các giá trị đó.

```bash
--reasoning on --reasoning-effort medium --reasoning-preserve
```

Hoặc cách tương đương qua Jinja:

```bash
--chat-template-kwargs '{"preserve_thinking":true,"reasoning_effort":"medium"}'
```

Khuyến nghị:

- `medium`: mặc định hằng ngày, coding/agent; cân bằng nhất.
- `low`: chat đơn giản, giảm chi phí token.
- `xhigh`: bài khó, planning/debug cuối; có thể dùng hàng chục nghìn reasoning token và lấp đầy context.
- Đặt `--reasoning-budget 8192` hoặc 12000–16000 để tránh model suy luận hết cửa sổ context. Budget quá thấp có thể cắt ngang lời giải.

Sampler chính thức được trích trong các nguồn:

```text
Thinking:     temp=1.0 top_p=0.95 top_k=20 min_p=0.0 presence=0.0 repeat=1.0
Non-thinking: temp=0.7 top_p=0.80 top_k=20 min_p=0.0 presence=1.5 repeat=1.0
```

Không gọi `temp=0.4 top_p=0.90 top_k=15 min_p=0.02` là “official”; đó chỉ là cấu hình cộng đồng và đã bị phản biện trong nguồn.

## 4. Những CLI argument quan trọng và cách dùng

Bảng dưới liệt kê các argument có ảnh hưởng trực tiếp khi triển khai Qwen3.8-27B. Tên viết tắt và tên dài là alias của nhau.

### Model, GPU và bộ nhớ

| Argument | Công dụng / cách chọn |
|---|---|
| `-m`, `--model FILE` | Nạp GGUF cục bộ. |
| `-hf`, `--hf-repo repo[:quant]` | Tải/nạp từ Hugging Face; tự lấy mmproj nếu có. |
| `-hff`, `--hf-file FILE`; `-hft`, `--hf-token TOKEN` | Chỉ định file HF và token. |
| `-mu`, `--model-url`; `-dr`, `--docker-repo` | Nguồn model qua URL/Docker Hub. |
| `-dev`, `--device`; `--list-devices` | Chọn GPU/backend, ví dụ `CUDA0`. |
| `-ngl`, `--n-gpu-layers all` | Đưa toàn bộ layer lên GPU. Nên dùng `all`, không cần `99/999`. |
| `-ot`, `--override-tensor PATTERN=BUFFER` | Offload chọn lọc tensor, thường FFN sang CPU nếu thiếu VRAM. Chỉ dùng sau khi đo đạc. |
| `-sm`, `--split-mode`; `-ts`, `--tensor-split`; `-mg`, `--main-gpu` | Multi-GPU. Một GPU không cần đặt. |
| `-fit`, `--fit on/off`; `--fit-target`; `--fit-ctx` | Tự điều chỉnh để vừa VRAM. Lần đầu dùng `on`; khi đã biết cấu hình vừa, dùng `off` để tránh tự đẩy layer sang CPU. |
| `-ctk`, `-ctv` | Kiểu KV K/V: `f32,f16,bf16,q8_0,q4_0,q4_1,iq4_nl,q5_0,q5_1`. |
| `--kv-offload` / `--no-kv-offload` | KV trên GPU/CPU; mặc định offload GPU bật. |
| `--op-offload`; `--repack`; `--no-host` | Điều khiển offload phép tính, repack weight và host buffer; giữ mặc định trừ khi debug. |
| `--load-mode auto|none|mmap|mlock|mmap+mlock|dio` | Cách nạp model. `auto` là mặc định; `mmap+mlock` hữu ích khi đủ RAM. |
| `--numa TYPE` | Tối ưu máy nhiều NUMA node; A100 server nhiều socket mới cần thử. |
| `--check-tensors`, `--override-kv` | Kiểm tra GGUF hoặc override metadata nâng cao. |

`--cpu-moe` và `--n-cpu-moe` có trong CLI nhưng Qwen3.8-27B là dense, không phải MoE, nên không có ích cho model này.

### Context, batch, CPU và cache

| Argument | Công dụng / khuyến nghị |
|---|---|
| `-c`, `--ctx-size N` | Tổng context. Dùng 65536/98304/131072 rồi tăng dần. `0` lấy metadata model. |
| `-n`, `--n-predict N` | Số token sinh; `-1` không giới hạn. |
| `-b`, `--batch-size N` | Logical batch, chủ yếu ảnh hưởng prefill; 1024–2048. |
| `-ub`, `--ubatch-size N` | Physical batch; 512 mặc định, giảm 256/128 để cứu VRAM. |
| `-t`, `--threads`; `-tb`, `--threads-batch` | Thread CPU cho decode/prefill. Full GPU thường để auto hoặc đặt theo core vật lý. |
| `-C/-Cr`, `-Cb/-Crb`, `--cpu-strict*`, `--prio*`, `--poll*` | CPU affinity, priority và polling; chỉ cần khi tối ưu latency/NUMA. |
| `-fa`, `--flash-attn on` | Nên bật cho NVIDIA và context dài. |
| `-np`, `--parallel N` | Số slot. Single-user và context lớn: `1`. |
| `--cont-batching` / `--no-cont-batching` | Batching liên tục; bật cho nhiều request, có thể tắt cho một người dùng. |
| `--kv-unified` | Một KV buffer dùng chung; hữu ích cho server/slot, nhưng vẫn dùng `parallel 1` khi tối đa context. |
| `--ctx-checkpoints N`; `--checkpoint-min-step N` | Checkpoint context/SWA. 32 là mặc định; giảm/tắt nếu build/model gặp vấn đề. |
| `--cache-ram MiB` | Prompt cache RAM, không phải VRAM KV. |
| `--cache-prompt`; `--cache-reuse N`; `--cache-idle-slots` | Tái sử dụng prefix cho hội thoại/agent. |
| `--context-shift` | Dịch context khi sinh vô hạn; cân nhắc cho chat dài, không thay thế context thật. |
| `--keep N`; `--swa-full` | Giữ token đầu và điều khiển SWA cache; thường giữ mặc định model. |
| `--warmup` / `--no-warmup` | Warmup lúc nạp. Tắt giúp khởi động nhanh nhưng request đầu chậm hơn. |
| `--perf` | Bật timing nội bộ để benchmark. |

Không tự override `--rope-*`/`--yarn-*` nếu GGUF đã có metadata đúng. Chỉ dùng khi thật sự kéo context ngoài thiết kế và đã kiểm tra chất lượng.

### MTP/speculative decoding

| Argument | Ý nghĩa |
|---|---|
| `--spec-type ...` | Chọn `none`, `draft-mtp`, các draft khác hoặc các loại n-gram; có thể ghép bằng dấu phẩy. |
| `--spec-draft-n-max N` | Số token draft tối đa; Qwen3.8 nên bắt đầu `2`. |
| `--spec-draft-n-min N` | Số draft tối thiểu. |
| `--spec-draft-p-min P` | Ngưỡng xác suất draft; bắt đầu `0.0`. Đừng tối ưu chỉ theo acceptance %. |
| `--spec-draft-p-split P` | Xác suất split, nâng cao. |
| `--gpu-layers-draft all`; `--device-draft` | Đưa phần draft lên GPU/chọn thiết bị. |
| `--cache-type-k-draft`, `--cache-type-v-draft` | KV cache của draft; bắt đầu `q8_0`. |
| `--model-draft`, `--hf-repo-draft` | Draft model rời; không cần cho MTP tích hợp. |
| `--threads-draft`, `--threads-batch-draft` và nhóm affinity/prio/poll draft | Tuning CPU cho draft rời. |
| `--override-tensor-draft`, `--cpu-moe-draft`, `--n-cpu-moe-draft` | Offload draft nâng cao; MoE draft không liên quan MTP tích hợp của model này. |
| `--spec-draft-backend-sampling` | Backend sampling cho draft; mặc định bật. |
| `--spec-ngram-mod-n-min/max/match` | Tuning `ngram-mod`. |
| `--spec-ngram-simple-*`, `--spec-ngram-map-k-*`, `--spec-ngram-map-k4v-*` | Tuning các engine n-gram tương ứng. |
| `--lookup-cache-static/dynamic` | File lookup cache cho lookup decoding. |
| `--spec-default` | Bật cấu hình speculative mặc định; cấu hình production nên ghi rõ `--spec-type` và `n-max`. |

Các cờ `--draft`, `--draft-min`, `--spec-ngram-size-*`, `--spec-ngram-min-hits` đã bị gỡ; dùng cờ `--spec-*` mới.

### Vision

| Argument | Ý nghĩa |
|---|---|
| `--mmproj FILE`, `--mmproj-url URL` | Projector vision. `-hf` có thể tự tải. |
| `--no-mmproj` | Text-only, tiết kiệm đáng kể VRAM. |
| `--mmproj-offload` / `--no-mmproj-offload` | Projector GPU/CPU. CPU tiết kiệm VRAM nhưng encode ảnh chậm. |
| `--image-min-tokens`, `--image-max-tokens` | Token ảnh cho dynamic resolution. |
| `--mtmd-batch-max-tokens` | Số image token tối đa mỗi batch encode. |
| `--media-path PATH` | Cho phép `file://` từ thư mục chỉ định. |

Vision + MTP từng có cảnh báo `non-consecutive token position`; đồng thời nhiều slot có thể OOM vì context cũ giữ VRAM. Với vision luôn bắt đầu `--parallel 1`.

### Chat, reasoning và structured output

| Argument | Ý nghĩa |
|---|---|
| `--jinja` | Dùng template Jinja trong GGUF; nên bật/giữ mặc định. |
| `--chat-template`, `--chat-template-file` | Override template; không làm nếu template GGUF đúng. |
| `--chat-template-kwargs JSON` | Truyền `reasoning_effort`, `preserve_thinking`, v.v. |
| `--reasoning on|off|auto` | Bật/tắt thinking. |
| `--reasoning-effort LEVEL` | Với Qwen dùng `low|medium|xhigh`. |
| `--reasoning-budget N`, `--reasoning-budget-message` | Giới hạn thinking. |
| `--reasoning-preserve` | Giữ trace reasoning trong lịch sử; hữu ích cho agent nhưng tốn context. |
| `--reasoning-format none|deepseek|deepseek-legacy` | Cách trả thought qua API. |
| `--skip-chat-parsing`, `--prefill-assistant` | Điều khiển parser/prefill nâng cao. |
| `--grammar`, `--grammar-file` | Ép output theo grammar. |
| `-j`, `--json-schema`; `-jf`, `--json-schema-file` | Ép JSON theo schema, rất hữu ích cho tool/agent. |

### Sampling

Tất cả sampler được binary hỗ trợ: `--samplers`, `--sampler-seq`, `--seed`, `--ignore-eos`, `--temp`, `--top-k`, `--top-p`, `--min-p`, `--top-nsigma`, `--xtc-probability`, `--xtc-threshold`, `--typical`, `--repeat-last-n`, `--repeat-penalty`, `--presence-penalty`, `--frequency-penalty`, `--dry-multiplier`, `--dry-base`, `--dry-allowed-length`, `--dry-penalty-last-n`, `--dry-sequence-breaker`, `--adaptive-target`, `--adaptive-decay`, `--dynatemp-range`, `--dynatemp-exp`, `--mirostat`, `--mirostat-lr`, `--mirostat-ent`, `--logit-bias`, `--backend-sampling`.

Với Qwen3.8, trước tiên dùng sampler chính thức ở mục 3; đừng bật đồng thời Mirostat/DRY/XTC/adaptive nếu chưa có benchmark chất lượng.

## 5. Danh mục đầy đủ các argument còn lại của `llama-server`

Các cờ dưới đây cũng được binary trong `llama-cpp.md` hỗ trợ. Chúng không riêng cho Qwen nhưng vẫn dùng được khi vận hành server.

### Kiểm tra, log và model phụ trợ

- `--help`, `--usage`, `--version`, `--cache-list`, `--completion-bash`, `--list-devices`.
- `--log-disable`, `--log-file`, `--log-colors`, `--verbose`, `--offline`, `--verbosity`, `--log-prefix`, `--log-timestamps`, `--log-prompts-dir`.
- `--escape/--no-escape`, `--special`, `--reverse-prompt`, `--spm-infill`, `--pooling`, `--embd-normalize`.
- LoRA/control: `--lora`, `--lora-scaled`, `--lora-init-without-apply`, `--control-vector`, `--control-vector-scaled`, `--control-vector-layer-range`.
- Legacy/deprecated: `--defrag-thold`, `--mlock`, `--mmap`, `--direct-io`; nên dùng `--load-mode` thay thế.

### HTTP, Web UI và bảo mật

- Mạng: `--host`, `--port`, `--reuse-port`, `--path`, `--api-prefix`, `--timeout`, `--sse-ping-interval`, `--threads-http`.
- CORS: `--cors-origins`, `--cors-methods`, `--cors-headers`, `--cors-credentials/--no-cors-credentials`.
- Xác thực/TLS: `--api-key`, `--api-key-file`, `--ssl-key-file`, `--ssl-cert-file`.
- UI: `--ui/--no-ui`, `--ui-config`, `--ui-config-file`, `--ui-mcp-proxy`.
- Quan sát/quản trị: `--metrics`, `--props`, `--slots/--no-slots`, `--slot-save-path`, `--slot-prompt-similarity`, `--sleep-idle-seconds`.
- `--embedding/--embeddings` và `--rerank/--reranking` chỉ phù hợp model embedding/reranker, không phải Qwen chat 27B.

Không bind `--host 0.0.0.0` mà thiếu firewall/API key. `--agent`, built-in tools và MCP có thể thực thi lệnh/đọc ghi file; không bật trên mạng không tin cậy.

### Tool, agent và MCP

- `--tools TOOL1,...`: `read_file`, `file_glob_search`, `grep_search`, `exec_shell_command`, `write_file`, `edit_file`, `get_info`, hoặc `all`.
- `--tools-runtime`: host mặc định, hoặc `docker:`, `podman:`, container có sẵn, `ssh:`.
- `--mcp-servers-config`, `--mcp-servers-json`.
- `--agent/--no-agent`: bật proxy và toàn bộ built-in tools; có rủi ro bảo mật cao.

### Router nhiều model

- `--models-dir`, `--models-preset`, `--models-max`, `--models-autoload/--no-models-autoload`, `--alias`, `--tags`.
- Preset tải model có sẵn: `--embd-gemma-default`, `--fim-qwen-1.5b-default`, `--fim-qwen-3b-default`, `--fim-qwen-7b-default`, `--fim-qwen-7b-spec`, `--fim-qwen-14b-spec`, `--fim-qwen-30b-default`, `--gpt-oss-20b-default`, `--gpt-oss-120b-default`, `--vision-gemma-4b-default`, `--vision-gemma-12b-default`. Chúng không phải cờ để tối ưu Qwen3.8-27B đang nạp.

### RoPE/YaRN nâng cao

Binary hỗ trợ `--rope-scaling`, `--rope-scale`, `--rope-freq-base`, `--rope-freq-scale`, `--yarn-orig-ctx`, `--yarn-ext-factor`, `--yarn-attn-factor`, `--yarn-beta-slow`, `--yarn-beta-fast`. Không thay các giá trị này chỉ để model “load được” context lớn; context load thành công không chứng minh model còn chính xác ở phần đuôi.

## 6. Quy trình tuning an toàn

1. Cập nhật `llama.cpp`, bảo đảm build CUDA và chạy `--list-devices`.
2. Khởi động với `--fit on`, context 32K/64K, `parallel 1`, Q8 KV và text-only.
3. Xác nhận log cho thấy toàn bộ layer, KV và draft nằm trên CUDA; theo dõi `nvidia-smi`.
4. Bật MTP `n=2`, so tốc độ với `--spec-type none` bằng cùng prompt/seed.
5. Tăng context từng bước; giữ tối thiểu khoảng 1–2 GiB headroom cho spike/batch/ảnh.
6. Khi cấu hình đã chắc chắn vừa VRAM, chuyển `--fit off` để tránh auto-offload ngoài ý muốn.
7. Chỉ sau đó mới thử ubatch 256/1024, MTP 3, n-gram hoặc KV Q5/Q4.
8. Đánh giá bằng workload thật: prefill dài, sinh code, nhiều turn, tool call và truy xuất thông tin ở cuối context; đừng chỉ nhìn tok/s hay acceptance rate.

Lệnh theo dõi:

```bash
watch -n 0.5 nvidia-smi
curl http://127.0.0.1:8080/health
```

Nếu prefill đột ngột chỉ còn vài chục tok/s trong khi GPU không bận, nghi ngờ KV Flash Attention hoặc tensor đã rơi về CPU. Nếu decode chỉ còn 1–10 tok/s, kiểm tra model có thực sự vừa hoàn toàn trong VRAM hay không trước khi chỉnh sampler.

## 7. Các biến thể Qwen3.8-27B đã giảm safety/refusal

Các model dưới đây đều bắt nguồn từ `Qwen/Qwen3.8-27B`, nhưng đã sửa trọng số để giảm hành vi từ chối. Chúng không chỉ là các bản quant khác nhau của cùng một checkpoint: phương pháp abliteration, chat template, MTP, vision và mức độ thay đổi so với base đều khác nhau.

> Các số refusal/KL dưới đây do từng repository tự công bố. Chúng dùng dataset, prompt, template, chế độ thinking và cách chấm khác nhau, vì vậy **không được so sánh trực tiếp như một leaderboard**. Việc giảm refusal cũng không bảo đảm giữ nguyên reasoning, coding, tool calling hoặc tính đúng đắn.

### 7.1. Bảng so sánh

| Repository | Đặc điểm chính | Refusal/KL được công bố | MTP | Vision | Quant có sẵn |
|---|---|---|---|---|---|
| [`JonathanColetti/Qwen3.8-27B-Uncensored-GGUF`](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF) | Heretic tối ưu refusal và KL; tài liệu kiểm chứng chi tiết | 12/100; KL 0.1191 | Có bản tích hợp và draft rời | Có projector | IQ2_M, IQ4_XS, Q4_K_M, Q5_K_M, Q6_K, Q8_0 |
| [`0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF`](https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF) | RVN, thêm hai lượt ARA trên checkpoint ARA có sẵn | 0–1/100; KL 0.0085 | Không, build với `--no-nextn` | Không có projector trong repo | IQ1 đến F16/BF16 |
| [`Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF`](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF) | Standard K-quant, multimodal, execution prompt nhúng trong template | 11/450 (2.4%), theo residual funnel | Có, nhúng trong model | Có projector F16/Q8_0 | Q2_K đến Q8_0, không có IQ/imatrix |
| [`0xKitkat/Qwen3.8-27B-Uncensored-Aggressive`](https://huggingface.co/0xKitkat/Qwen3.8-27B-Uncensored-Aggressive) | Rank-5 ablation, sửa cả `lm_head`; template ép đóng thinking | Không có benchmark harmful đầy đủ | Có, giữ nguyên | Có projector F16 | Q4_K_M, Q5_K_M, Q6_K hỗn hợp |

### 7.2. JonathanColetti Uncensored

Đây là lựa chọn bảo thủ hơn nếu cần giữ model gần base và muốn có số liệu kỹ thuật để kiểm tra:

- Heretic sửa chủ yếu `attn.o_proj` và `mlp.down_proj`; LoRA được merge vào BF16 rồi mới chuyển/quantize.
- MTP bị mất trong bước qua Transformers đã được chép lại từ base. Tác giả kiểm tra đủ 65/65 block sau quantization, thay vì chỉ dựa vào metadata.
- Có hai cách chạy: GGUF fused chứa MTP, hoặc file `noMTP` ghép với `draft-Q8_0` riêng.
- Benchmark BF16 công bố mức giảm trung bình khoảng 0.5 điểm trên MMLU, ARC-Challenge, HellaSwag và Winogrande. Perplexity cho thấy IQ2_M suy giảm rõ hơn các quant còn lại.

Với file fused, có thể giữ cấu hình MTP ở mục 1 và 3. Ví dụ model Q4:

```bash
-hf JonathanColetti/Qwen3.8-27B-Uncensored-GGUF:Q4_K_M \
--spec-type draft-mtp --spec-draft-n-max 2
```

Không giả định build `llama.cpp` cũ sẽ dùng MTP: repository yêu cầu phiên bản có hỗ trợ native MTP tương ứng. Để đánh giá chất lượng/refusal nên ưu tiên Q6_K hoặc Q8_0 trước khi kết luận từ IQ2_M.

### 7.3. 0bserverx RVN Heretic Abliterated

RVN bắt đầu từ `trohrbaugh/Qwen3.8-27B-heretic-ara`, sau đó áp dụng thêm hai lượt full-weight ARA. Repository công bố mức refusal thấp nhất trong bốn lựa chọn và có dải quant rộng nhất, phù hợp cả máy rất ít RAM/VRAM.

Khác biệt vận hành quan trọng là các file RVN **không chứa NextN/MTP**. Không sao chép nguyên cấu hình MTP ở mục 1 sang model này:

```bash
./llama.cpp/llama-server \
  -hf 0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF \
  --hf-file RVN-Q4_K_M.gguf \
  --spec-type none --no-mmproj \
  --jinja -c 65536 -ngl all
```

Repository còn giữ một file Q4 legacy có tên gần giống model cũ; khi triển khai mới nên chọn file bắt đầu bằng `RVN-`. Một bản `RVN-IQ3_M` từng có tensor hỏng và đã được upload lại, vì vậy cache/download cũ cần được xóa và kiểm tra checksum nếu gặp output bất thường.

### 7.4. Blackfrost Abliterated

Blackfrost phù hợp khi cần standard K-quant, vision và MTP trong một file:

- Chín main GGUF từ Q2_K đến Q8_0 đều chứa block MTP thứ 65; không dùng draft sidecar.
- Vision dùng một trong hai file `mmproj-Qwen3.8-27B-ABLITERATED-F16.gguf` hoặc `mmproj-Qwen3.8-27B-ABLITERATED-Q8_0.gguf`.
- Short execution prompt được nhúng trong Jinja template. Phải giữ `--jinja`; prompt này chỉ là hướng dẫn hành vi, không phải lớp bảo mật.
- Không có IQ/imatrix quant. Q4_K_M khoảng 16.8 GB là lựa chọn mặc định của repository.

Ví dụ text-only:

```bash
./llama.cpp/llama-server \
  -hf Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF:Q4_K_M \
  --no-mmproj --jinja -ngl all -c 65536 \
  --spec-type draft-mtp --spec-draft-n-max 2
```

Số 11/450 không phải một lần chạy mới toàn bộ 450 case trên từng GGUF cuối cùng. Đó là residual funnel nhiều vòng trên derivative W4A4 của cùng BF16 parent, có thay prompt giữa các vòng; chỉ nên xem là mô tả của quy trình tác giả.

### 7.5. 0xKitkat Uncensored Aggressive v4

Biến thể này ưu tiên hành vi trả lời ngay trong LM Studio/`llama-server`, thay vì giữ thinking mode chuẩn:

- Xây refusal basis rank-5 từ unembedding, sửa các residual writer, một phần input projection và cả `output.weight`/`lm_head`.
- Template nhúng sẵn luôn mở rồi đóng `<think>` rỗng, đồng thời chèn unrestricted system prompt nếu request không có system message.
- MTP và vision tower được giữ nguyên; projector `mmproj-F16.gguf` là tùy chọn.
- Chỉ có Q4_K_M, Q5_K_M và Q6_K. Các file là requant từ cùng một Q6 bake; bản Q6 lớn khoảng 27.5 GB vì các tensor đã ablate và `lm_head` được giữ Q8_0.

Không dùng cấu hình `--reasoning on --reasoning-effort medium` ở mục 1 nếu muốn đúng hành vi thiết kế của v4. Giữ template trong GGUF và dùng sampler non-thinking:

```bash
./llama.cpp/llama-server \
  -hf 0xKitkat/Qwen3.8-27B-Uncensored-Aggressive:Q4_K_M \
  --no-mmproj --jinja --reasoning off -ngl all -c 65536 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --temp 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5
```

Repository chỉ công bố smoke test chat, không có HarmBench đầy đủ. Do mức sửa `lm_head` mạnh hơn và thinking bị khóa đóng, cần tự benchmark reasoning, code, tiếng Việt, structured output và tool calling trước khi dùng production.

### 7.6. Chọn nhanh

- Ưu tiên gần base, số liệu và provenance chi tiết, MTP được kiểm tra: **JonathanColetti**.
- Ưu tiên mức compliance tự công bố cao và nhiều quant rất nhỏ, không cần MTP/vision: **0bserverx RVN**.
- Ưu tiên multimodal, standard quant và MTP một file: **Blackfrost**.
- Ưu tiên chat/roleplay trả lời trực tiếp, không policy preamble và chấp nhận tắt thinking: **0xKitkat v4**.

Để so sánh công bằng, tải cùng cấp quant (tốt nhất Q5_K_M hoặc Q6_K), dùng cùng phiên bản `llama.cpp`, sampler, context và một bộ prompt riêng gồm reasoning, coding, tiếng Việt, vision/tool call và các case refusal mong muốn. Không dùng template của model này để chấm model khác.
