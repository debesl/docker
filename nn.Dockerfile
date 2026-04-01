FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /train_nn

RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
      build-essential \
      software-properties-common \
      ca-certificates \
      git \
      htop \
      wget && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.9 \
      python3.9-distutils \
      python3.9-venv && \
    wget -q https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py && \
    python3.9 /tmp/get-pip.py && \
    rm -f /tmp/get-pip.py && \
    rm -rf /var/lib/apt/lists/*

COPY CAM4BWE ./CAM4BWE
COPY BWE_data ./BWE_data

RUN useradd -m -s /bin/bash audiolab && \
    chown -R audiolab:audiolab /train_nn

USER audiolab

RUN python3.9 -m pip install --upgrade pip
COPY requirements.txt .
RUN python3.9 -m pip install -r requirements.txt
