# Pull base image.
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install.
RUN apt-get -y upgrade \
  apt-get install -y build-essential 

#unistall all otther python versions 
RUN apt remove -y python2 python2-minimal python2.7 python2.7-minimal \ 
  apt remove -y python3 python3-minimal python3.10 python3.10-minimal \ 
  apt purge -y `dpkg --list | grep python | awk '{ print $2 }'` \
  apt autoremove -y \
  apt autoclean

RUN  sudo apt install Python3-pip \ 
  sudo apt-get install git-all \
  git \
  htop \
  wget 

#install python 3.9
RUN apt-get install python3.9 \
    apt-get install python3-pip

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





