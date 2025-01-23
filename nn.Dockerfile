# Pull base image.
FROM nvidia/cuda:12.1.0-base-ubuntu22.04


# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Update package list and install dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Manually replace the default python3 with python3.9
RUN ln -sf /usr/bin/python3.9 /usr/bin/python3

# Ensure pip is installed for Python 3.9
RUN python3 -m ensurepip && python3 -m pip install --upgrade pip

#Audiobackend for soundfile and torchaudio
RUN apt-get update && \
      apt-get -y install sudo
RUN sudo apt-get -y install libsndfile1

#RUN apt-get install -y \
#    git \
#     htop \
#     wget \
#     curl 

RUN pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# Add files.
ADD src src
ADD data data 

RUN pip install -r src/requirements.txt
WORKDIR /train_nn
#Add User otherwise work on root user (not recommended)
RUN adduser --disabled-password --gecos "" audiolab
USER audiolab
