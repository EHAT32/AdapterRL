FROM nvcr.io/nvidia/pytorch:24.08-py3

# Понизить версию PyTorch до 2.6.0 с CUDA 12.4
RUN pip install --force-reinstall torch==2.6.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

RUN apt-get update && \
    apt-get install -y \
    wget \
    git \
    gnutls-bin \
    openssh-client \
    libghc-x11-dev \
    gcc-multilib \
    g++-multilib \
    libglew-dev \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    xvfb \
    mesa-utils \
    libegl1-mesa \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    unzip \
    openjdk-8-jdk && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install ipywidgets && \
    pip install MineStudio && \
    pip uninstall pyglet -y && \
    pip install pyglet==1.5.27 && \
    pip install datasets huggingface_hub && \
    python -m minestudio.simulator.entry -y

CMD ["python", "-m", "minestudio.simulator.entry"]
