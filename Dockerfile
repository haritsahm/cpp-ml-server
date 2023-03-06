ARG ENGINE_TYPE=all

FROM haritsahm/cpp-ml-${ENGINE_TYPE}

ARG ENGINE_TYPE
ENV CXX_VERSION 14

# App directory
WORKDIR /workspace/cpp_server/

COPY include include/
COPY src src/
COPY tests tests/
COPY examples examples/
COPY scripts scripts/
COPY CMakeLists.txt ./

RUN sh ./scripts/cmake_build.sh ${ENGINE_TYPE}

WORKDIR /workspace/cpp_server/build/

CMD ["/bin/bash"]