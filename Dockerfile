FROM nvcr.io/nvidia/pytorch:24.03-py3
SHELL ["/bin/bash", "-lc"]

# --- OS dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      curl pciutils screen && \
    rm -rf /var/lib/apt/lists/*

# --- Upgrade pip
RUN python -m pip install --no-cache-dir -U pip setuptools wheel

RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# --- Install project requirements \
WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app