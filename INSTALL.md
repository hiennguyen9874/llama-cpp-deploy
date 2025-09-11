# Install llama.cpp

## For cuda

- Link: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cuda

### Update apt sources.list

```bash
tee /etc/apt/sources.list > /dev/null <<'EOF'
deb http://mirror.viettelcloud.vn/ubuntu jammy main restricted universe multiverse
deb http://mirror.viettelcloud.vn/ubuntu jammy-updates main restricted universe multiverse
deb http://mirror.viettelcloud.vn/ubuntu jammy-backports main restricted universe multiverse
deb http://mirror.viettelcloud.vn/ubuntu jammy-security main restricted universe multiverse

deb http://mirror.clearsky.vn/ubuntu jammy main restricted universe multiverse
deb http://mirror.clearsky.vn/ubuntu jammy-updates main restricted universe multiverse
deb http://mirror.clearsky.vn/ubuntu jammy-backports main restricted universe multiverse
deb http://mirror.clearsky.vn/ubuntu jammy-security main restricted universe multiverse

deb http://mirror.bizflycloud.vn/ubuntu jammy main restricted universe multiverse
deb http://mirror.bizflycloud.vn/ubuntu jammy-updates main restricted universe multiverse
deb http://mirror.bizflycloud.vn/ubuntu jammy-backports main restricted universe multiverse
deb http://mirror.bizflycloud.vn/ubuntu jammy-security main restricted universe multiverse

deb http://archive.ubuntu.com/ubuntu jammy main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu jammy-updates main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu jammy-backports main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu jammy-security main restricted universe multiverse
EOF
```

### Step by step

```
apt update -y && apt install -y curl libssl-dev libcurl4-openssl-dev

git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80" -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=ON -DGGML_CUDA_F16=ON -DGGML_CUDA_MMV_Y=2 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=4096 -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_USE_GRAPHS=ON
cmake --build build --config Release -j 8
```
