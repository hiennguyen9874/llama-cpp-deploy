# 1) Nguyên tắc tối ưu trên A100 40GB

- **Offload tối đa lên GPU**: A100 40GB đủ để tải full trọng số **Q5_K_XL** của 36B + phần KV cache hợp lý ⇒ dùng `-ngl 999` (tức “đưa hết lên GPU”).
- **Flash Attention**: luôn bật `-fa` để giảm băng thông bộ nhớ và tăng throughput/độ trễ.
- **KV Cache nén (q8_0)**: với context 16k–32k và/hoặc nhiều phiên đồng thời, chọn `-ctk q8_0 -ctv q8_0`. Nếu bạn luôn ở 8k và ít phiên, có thể cân nhắc `f16` cho chút ít chất lượng/tính ổn định (đổi lại tốn VRAM).
- **Batch / micro-batch**: giữ `-b` (logical batch) cao để tận dụng continuous batching, và điều chỉnh `-ub` (micro/physical batch) vừa phải để không “xé” VRAM. Thực tế A100 40GB với 36B Q5_K_XL rất hợp với `-b 4096` và `-ub 1024–2048`.
- **Chat template**: dùng sẵn template `seed_oss` để format hội thoại chuẩn: `--chat-template seed_oss`.
- **Continuous batching**: mặc định **đã bật** trong server; giữ nguyên để phục vụ đa người dùng hiệu quả.
- **Theo dõi & tinh chỉnh**: bật `--metrics --slots` để theo dõi /metrics và /slots; điều chỉnh `-c` (ctx), `--parallel`, `-ub` theo nvidia-smi và độ trễ thực tế.

---

# 2) Hồ sơ cấu hình khuyến nghị (3 profile)

> Thay **đường dẫn model** cho đúng máy của bạn, ví dụ:
> `/models/Seed-OSS-36B-Instruct-Q5_K_XL.gguf`

## A) Low-latency (độ trễ thấp), hội thoại 8–16k (ít người dùng)

Phù hợp chatbot nội bộ, request ngắn–trung bình, ưu tiên thời gian token đầu tiên (TTFT).

```bash
./llama.cpp/build/bin/llama-server \
  -hf unsloth/Seed-OSS-36B-Instruct-GGUF:Q5_K_XL \
  --alias seed-oss-36b-q5kx \
  --chat-template seed_oss \
  --host 0.0.0.0 --port 8080 \
  -fa on -ngl 999 --device CUDA0 \
  -c 16384 \
  -b 4096 -ub 1024 \
  -ctk q8_0 -ctv q8_0 \
  --parallel 4 \
  --threads $(nproc) --threads-http -1 \
  --metrics --slots
```

**Giải thích nhanh**

- `-c 16384`: 16k context đủ rộng cho hội thoại thông thường; KV q8_0 giúp dư VRAM cho batch.
- `--parallel 4`: giải mã song song 4 phiên (tối ưu theo workload; tăng/giảm dựa vào độ trễ).
- `-b 4096 -ub 1024`: logical batch cao + micro-batch 1024 giúp tận dụng GPU, vẫn an toàn VRAM.

---

## B) Balanced (cân bằng latency/throughput), 16–32k (nhiều người dùng hơn)

Khi cần context dài hơn và/hoặc nhiều slot đồng thời.

```bash
./llama.cpp/build/bin/llama-server \
  -hf unsloth/Seed-OSS-36B-Instruct-GGUF:Q5_K_XL \
  --alias seed-oss-36b-q5kx \
  --chat-template seed_oss \
  --host 0.0.0.0 --port 8080 \
  -fa on -ngl 999 --device CUDA0 \
  -c 32768 \
  -b 4096 -ub 2048 \
  -ctk q8_0 -ctv q8_0 \
  --parallel 8 \
  --cache-reuse 1024 \
  --threads $(nproc) --threads-http -1 \
  --metrics --slots
```

**Ghi chú**

- `-c 32768`: 32k tăng nhu cầu KV → dùng `q8_0` là hợp lý; `-ub 2048` giúp throughput cao.
- `--parallel 8`: tăng số phiên. Theo dõi latency; nếu tăng cao, giảm `--parallel` hoặc `-ub`.

---

## C) Long-context “an toàn VRAM”, 32k, ưu tiên ổn định

Nếu bạn thấy OOM khi tải nhiều người dùng, giảm `-ub`, giữ `-b` cao để tận dụng batching.

```bash
./llama.cpp/build/bin/llama-server \
  -hf unsloth/Seed-OSS-36B-Instruct-GGUF:Q5_K_XL \
  --alias seed-oss-36b-q5kx \
  --chat-template seed_oss \
  --host 0.0.0.0 --port 8080 \
  -fa on -ngl 999 --device CUDA0 \
  -c 32768 \
  -b 4096 -ub 1024 \
  -ctk q8_0 -ctv q8_0 \
  --parallel 6 \
  --threads $(nproc) --threads-http -1 \
  --metrics --slots
```

---

# 3) Tham số “chốt hạ” & tại sao

- `-ngl 999` + `--device CUDA0`
  Đẩy tối đa layer lên GPU A100. Với Q5_K_XL 36B, A100 40GB đủ “gánh” trọng số + KV vừa phải.

- `-fa` (Flash Attention)
  Giảm áp lực băng thông bộ nhớ, cải thiện TTFT và tokens/s.

- `-ctk q8_0 -ctv q8_0` (KV q8_0)
  Nén KV cache rất đáng kể so với `f16`. Rất hữu ích khi `-c` ≥ 16k **hoặc** `--parallel` cao.

- `-b` vs `-ub`

  - `-b` (logical batch) cao giúp **continuous batching** gom yêu cầu hiệu quả.
  - `-ub` là micro-batch thực sự chạy trên GPU. Tăng `-ub` → throughput tăng, nhưng dễ chạm trần VRAM.
  - Điểm ngọt cho A100 40GB + 36B Q5_K_XL thường là `-ub` **1024–2048**.

- `--parallel N`
  Số chuỗi giải mã song song. Tăng để nâng thông lượng; giảm nếu độ trễ tăng hoặc VRAM căng.

- `-c` (ctx-size)
  8k–16k đủ cho phần lớn chat. 32k khi cần đoạn dài. Context càng dài, KV càng lớn → theo dõi VRAM.

- `--chat-template seed_oss`
  Đảm bảo định dạng nhắc/đáp đúng phong cách Seed-OSS Instruct.

- `--metrics --slots`
  Bật Prometheus `/metrics` và giám sát “slot” để tuning live.

- `--threads $(nproc)` / `--threads-http -1`
  Cho pipeline máy chủ HTTP/JSON và công đoạn tiền xử lý hoạt động mượt trên CPU host.

- **Sampling (tuỳ chọn, hợp với Instruct)**

  ```
  --temp 0.6 --top-p 0.95 --top-k 60 --min-p 0.05 \
  --repeat-last-n 64 --repeat-penalty 1.05
  ```

  Cân bằng tính “đúng mực” và đa dạng.

---

# 4) Mẹo vận hành & xử lý sự cố

- **OOM / VRAM cao**

  1. Hạ `-ub` (ví dụ 2048 → 1024)
  2. Giảm `--parallel` (8 → 6 → 4)
  3. Giảm `-c` (32k → 16k)
  4. Giữ `-ctk/-ctv q8_0` (đừng chuyển về f16 nếu đang thiếu VRAM)

- **TTFT cao (token đầu lâu ra)**

  - Bật `-fa` (nếu chưa)
  - Giảm `-ub` nhẹ (để hạn chế việc “đông người” trong một vi mẻ)
  - Giảm `--parallel` nếu đang quá cao

- **Thông lượng chưa đạt kỳ vọng**

  - Tăng dần `-ub` (1024 → 1536 → 2048) miễn còn VRAM
  - Tăng `--parallel` từng nấc (4 → 6 → 8), theo dõi latency
  - Đảm bảo build có `-DGGML_CUDA=ON` (và dùng CUDA cho A100)

- **Bảo mật & tích hợp**

  - Dùng `--api-key` hoặc `--api-key-file` nếu mở port ra ngoài
  - Bật `--metrics` để scrape qua Prometheus, giám sát QPS/latency/slots

---

# 5) Tùy chọn nâng cao (chỉ dùng khi thật sự cần)

- **Rope/YARN**: chỉ chạm tới nếu bạn định mở rộng context vượt training default. Hãy giữ nguyên mặc định model; nếu tăng context, ưu tiên `--rope-scaling linear` hoặc theo metadata model.
- **Speculative decoding**: hiệu quả nếu có draft model tương thích (1–3B). Với Seed-OSS 36B chưa có draft sẵn, thường **bỏ qua** để đơn giản.
- **NUMA / pin CPU**: trên single-socket phổ biến không cần; nếu multi-socket nhiều vCPU, có thể dùng `--numa isolate`/`--cpu-range` để cải thiện ổn định.

---

# 6) Mẫu câu lệnh “drop-in” (copy & chạy)

### Profile khuyến nghị (cân bằng) cho production

```bash
./llama.cpp/build/bin/llama-server \
  -hf unsloth/Seed-OSS-36B-Instruct-GGUF:Q5_K_XL \
  --alias seed-oss-36b-q5kx \
  --chat-template seed_oss \
  --host 0.0.0.0 --port 8080 \
  -fa on -ngl 999 --device CUDA0 \
  -c 32768 \
  -b 4096 -ub 2048 \
  -ctk q8_0 -ctv q8_0 \
  --parallel 8 \
  --threads $(nproc) --threads-http -1 \
  --metrics --slots
```

### Profile độ trễ thấp (8–16k), ít người dùng đồng thời

```bash
./llama.cpp/build/bin/llama-server \
  -hf unsloth/Seed-OSS-36B-Instruct-GGUF:Q5_K_XL \
  --alias seed-oss-36b-q5kx \
  --chat-template seed_oss \
  --host 0.0.0.0 --port 8080 \
  -fa on -ngl 999 --device CUDA0 \
  -c 16384 \
  -b 4096 -ub 1024 \
  -ctk q8_0 -ctv q8_0 \
  --parallel 4 \
  --threads $(nproc) --threads-http -1 \
  --metrics --slots
```

---

# 7) Checklist nhanh khi lên máy

- [ ] Build **CUDA ON**: `-DGGML_CUDA=ON`
- [ ] Driver/CUDA tương thích A100; bật **Persistence Mode** nếu cần
- [ ] Đảm bảo **hugepages** không bắt buộc; để mặc định `mmap` (trừ khi bạn có lý do tắt)
- [ ] Mở `--metrics` và giám sát `nvidia-smi` trong giờ đầu vận hành
- [ ] Tinh chỉnh `-ub`, `--parallel`, `-c` theo workload thực tế (độ trễ/VRAM)

---

Nếu bạn muốn, mình có thể viết thêm một **docker-compose** mẫu (kèm Prometheus scrape `/metrics`) và **dashboard** Grafana tối thiểu để bạn cắm vào ngay.
