FROM ubuntu:20.04

ENV DEBIAN_FRONTEND='noninteractive'
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV CXX_VERSION 14

# Install minimum tools for run applications

WORKDIR /temp

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y dist-upgrade && \
    apt-get -y autoremove && \
    apt-get install -y build-essential gdb wget git libssl-dev clang-format cmake curl && \
    apt-get install -y libperlio-gzip-perl libjson-perl libpq-dev libsqlite3-dev unzip && \
    apt-get install -y zlib1g zlib1g-dev && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

COPY scripts/use_clang.sh scripts/use_gcc.sh /usr/bin/

RUN ln -s /usr/bin/python3 /usr/bin/python

# Install required libraries

RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-Linux-x86_64.sh \
    -q -O /tmp/cmake-install.sh \
    && chmod u+x /tmp/cmake-install.sh \
    && mkdir /opt/cmake-3.24.1 \
    && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.24.1 \
    && rm /tmp/cmake-install.sh \
    && ln -s /opt/cmake-3.24.1/bin/* /usr/local/bin

RUN wget https://sourceforge.net/projects/boost/files/boost/1.81.0/boost_1_81_0.tar.gz && \
    tar -zxvf boost_1_81_0.tar.gz && cd boost_1_81_0 && ./bootstrap.sh && ./b2 cxxflags="-std=c++$CXX_VERSION" \
    --reconfigure --with-fiber --with-context --with-atomic --with-date_time --with-filesystem --with-url install && \
    cd /temp/ && git clone -b v1.15 https://github.com/linux-test-project/lcov.git && cd lcov && make install

RUN cd /temp/ && \
    git clone https://github.com/okyfirmansyah/libasyik && \
    cd libasyik/ && git checkout tags/0.9.5 && \
    git submodule update --init --recursive && \
    mkdir build && \
    cd build && \
    cmake .. -DLIBASYIK_ENABLE_SOCI=OFF && \
    make -j$(nproc) && \
    make install

RUN cd /temp/ && \
    git clone https://github.com/Tencent/rapidjson.git && cd rapidjson/ && \
    git submodule update --init && mkdir build && cd build && \
    cmake  .. && make -j$(nproc) && make install

RUN cd /temp/ && \
    wget -O opencv.zip https://codeload.github.com/opencv/opencv/zip/refs/tags/4.6.0 && \
    unzip opencv.zip && cd opencv-4.6.0/ && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DBUILD_LIST=core,highgui,improc,imgcodecs .. && \
    make -j$(nproc) && \
    make install

RUN cd /temp/ && \
    git clone https://github.com/triton-inference-server/client.git && cd client/ && \
    git checkout r22.06 && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DTRITON_ENABLE_CC_HTTP=ON  -DTRITON_ENABLE_CC_GRPC=ON \
    -DTRITON_COMMON_REPO_TAG=r22.06 -DTRITON_THIRD_PARTY_REPO_TAG=r22.06 -DTRITON_CORE_REPO_TAG=r22.06 \
    -DTRITON_BACKEND_REPO_TAG=r22.06 .. && \
    make -j$(nproc) cc-clients

# Cleanup
RUN apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*
RUN rm -rf /temp/*

# App directory
WORKDIR /workspace/cpp_server/

COPY include include/
COPY src src/
COPY tests tests/
COPY main.cpp CMakeLists.txt ./

RUN mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc)

WORKDIR /workspace/cpp_server/build/

CMD ["/bin/bash"]