FROM ultralytics/ultralytics:latest-jetson-jetpack4

# Define o ambiente para evitar prompts interativos
ENV DEBIAN_FRONTEND=noninteractive

# Atualiza os repositórios e instala pacotes essenciais, incluindo:
# tzdata, OpenCV, ferramentas de compilação, nano e os plugins do GStreamer

RUN apt-get update && apt-get install -y \
    tzdata \
    python3-opencv \
    build-essential \
    g++ \
    pkg-config \
    libopencv-dev \
    cmake \
    git \
    llvm \
    libedit-dev \
    zlib1g-dev \
    libcurl4-openssl-dev \
    libboost-dev \
    libtinfo-dev \
    make \
    nano \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly && \
    ln -fs /usr/share/zoneinfo/Europe/Lisbon /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata


WORKDIR /ultralytics

COPY models/ /ultralytics/models_joe/

CMD ["/bin/bash"]

