FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel
# Install dependencies

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get install -y git cmake build-essential ninja-build libboost-program-options-dev libboost-graph-dev libboost-system-dev libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev curl ffmpeg pkg-config python3 python3-dev rsync software-properties-common unzip libopencv-dev imagemagick

WORKDIR /etri_workspace
RUN git clone https://github.com/colmap/colmap.git
WORKDIR /etri_workspace/colmap
RUN git checkout tags/3.11.0
WORKDIR /etri_workspace/colmap/build
RUN cmake .. -GNinja
RUN ninja
RUN ninja install

COPY ./_internal /etri_workspace/_internal
COPY ./checkpoints /etri_workspace/checkpoints
COPY ./make_layer /etri_workspace
COPY ./SCAPI.py /etri_workspace


ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV PATH /etri_workspace:$PATH

RUN echo 'ln /dev/null /dev/raw1394' >> ~/.bashrc

#RUN mkdir -p /workspace
WORKDIR /etri_workspace
RUN chmod -R a+w /etri_workspace