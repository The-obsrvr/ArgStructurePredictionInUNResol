FROM nvcr.io/nvidia/pytorch:24.03-py3
SHELL ["/bin/bash", "-lc"]

# --- OS deps ---
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      curl pciutils screen && \
    rm -rf /var/lib/apt/lists/*

# --- Upgrade pip toolchain ---
RUN python -m pip install --no-cache-dir -U pip setuptools wheel

# --- Install project requirements (includes transformers >= 4.55.0) ---
WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# --- User setup (as before) ---
ARG uid=1000
ARG gid=1000
ARG USER=dh
ARG USER_GROUP=dh

RUN addgroup --gid ${gid} ${USER_GROUP} && \
    adduser --gecos "" --disabled-password --uid ${uid} --gid ${gid} ${USER}

USER ${USER}
COPY . /app