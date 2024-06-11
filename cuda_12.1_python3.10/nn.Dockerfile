# Pull base image.
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install.
RUN apt-get -y upgrade \
  apt-get install -y build-essential \
  sudo apt install Python3-pip \ 
  sudo apt-get install git-all \
  git \
  htop \
  wget 


# Add files.
ADD src src

WORKDIR /train_nn
#Add User otherwise work on root user (not recommended)
RUN adduser ---create-home audiolab
USER audiolab

#Install requirements 
RUN cd src && pip3 install -r requirements.txt 

RUN mkdir -p data

#RUN git clone https://github.com/debesl/SpeechEnhancement/tree/GAN





