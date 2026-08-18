# Llama cpp

## Install

- Download from [github.com/ggml-org/llama.cpp/releases](https://github.com/ggml-org/llama.cpp/releases)
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

```bash
apt update -y && apt install -y curl libssl-dev libcurl4-openssl-dev

git clone https://github.com/ggml-org/llama.cpp

rm -rf build

export CUDA_HOME=/usr/local/cuda-12.6
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
    -DCMAKE_INSTALL_RPATH="/usr/local/cuda-12.6/lib64;\$ORIGIN" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
```
