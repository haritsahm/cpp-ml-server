FROM haritsahm/cpp-ml-triton

ENV CXX_VERSION 14

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