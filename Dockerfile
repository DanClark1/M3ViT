# Use PyTorch image with full CUDA toolkit (includes nvcc)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Specify CUDA architectures
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="$CUDA_HOME/bin:$PATH"

# Clone FastMoE, checkout specific commit, and install
RUN git clone https://github.com/laekov/fastmoe.git && \
    cd fastmoe && \
    python setup.py install

RUN apt-get update && apt-get install -y libgl1
RUN pip install -r requirements.txt