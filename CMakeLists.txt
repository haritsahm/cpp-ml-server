cmake_minimum_required(VERSION 3.15)
project(cpp-server)

if(DEFINED ENV{CXX_VERSION})
    set(CMAKE_CXX_STANDARD $ENV{CXX_VERSION}) # set to 14 or 17 when available
else()
    set(CMAKE_CXX_STANDARD 14)
endif()

option(ENABLE_TRITON "Enable Triton Engine" ON)
option(ENABLE_ONNXRT "Enable ONNXRT Engine" OFF)
option(RUN_TESTS "Wether to run tests" OFF)

if(RUN_TESTS)
    message("Building with lcov Code Coverage Tools")

    FIND_PROGRAM(GCOV_PATH gcov)
    FIND_PROGRAM(LCOV_PATH lcov)
    FIND_PROGRAM(GENHTML_PATH genhtml)

    # Warning/Error messages
    if(NOT(CMAKE_BUILD_TYPE STREQUAL "Debug"))
        message(WARNING "Code coverage results with an optimized (non-Debug) build may be misleading")
    endif()

    if(NOT LCOV_PATH)
        message(FATAL_ERROR "lcov not found! Aborting...")
    endif()

    if(NOT GENHTML_PATH)
        message(FATAL_ERROR "genhtml not found! Aborting...")
    endif()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
endif()

set(LIBASYIK_ENABLE_SOCI OFF) # if SOCI database support is desired

find_package(libasyik REQUIRED)
find_package(Boost COMPONENTS context fiber date_time url REQUIRED)
find_package(RapidJSON REQUIRED)
find_package(OpenCV 4 REQUIRED)

if(ENABLE_TRITON)
    find_package(Protobuf REQUIRED)
    find_package(TritonCommon REQUIRED PATHS /usr/local/lib/cmake/TritonCommon)
    find_package(TritonClient REQUIRED PATHS /usr/local/lib/cmake/TritonClient)
endif()

include_directories(
    include
)

add_library(common_utils
    src/utils/error.cpp
    src/utils/base64.cpp
)

add_library(image_processor
    src/image_processor.cpp
)

if(ENABLE_TRITON)
    add_library(triton_inference_engine
        src/triton_engine.cpp
        src/triton_helper.cpp
    )
endif()

if(ENABLE_ONNXRT)
    add_library(onnxrt_inference_engine
        src/onnxrt_helper.cpp
    )
endif()

target_link_libraries(image_processor common_utils)

if(RapidJSON_FOUND)
    target_include_directories(image_processor PUBLIC ${RapidJSON_INCLUDE_DIRS})
endif()

if(OpenCV_FOUND)
    target_include_directories(image_processor PUBLIC ${OpenCV_INCLUDE_DIR})
    target_link_libraries(image_processor ${OpenCV_LIBS})
endif()

if(ENABLE_TRITON)

    if(OpenCV_FOUND)
        target_include_directories(triton_inference_engine PUBLIC ${OpenCV_INCLUDE_DIR})
        target_link_libraries(triton_inference_engine ${OpenCV_LIBS})
    endif()

    if(Protobuf_FOUND)
        target_include_directories(triton_inference_engine PUBLIC ${Protobuf_INCLUDE_DIRS})
        target_link_libraries(triton_inference_engine ${Protobuf_LIBRARIES})
    endif()

    if(TritonClient_FOUND)
        target_include_directories(image_processor PUBLIC ${TritonClient_INCLUDE_DIR} ${TritonCommon_INCLUDE_DIR})
        target_include_directories(triton_inference_engine PUBLIC ${TritonClient_INCLUDE_DIR} ${TritonCommon_INCLUDE_DIR})
        target_link_libraries(image_processor TritonClient::httpclient TritonClient::grpcclient)
        target_link_libraries(triton_inference_engine TritonClient::httpclient TritonClient::grpcclient)
    endif()

    if(TritonCommon_FOUND)
        target_include_directories(triton_inference_engine PUBLIC ${TritonCommon_INCLUDE_DIR})
        target_link_libraries(triton_inference_engine TritonCommon::triton-common-model-config TritonCommon::triton-common-json)
    endif()
endif()

if(ENABLE_ONNXRT)
    target_include_directories(onnxrt_inference_engine PUBLIC /usr/local/include/onnxruntime)
    target_link_libraries(onnxrt_inference_engine /usr/local/lib/libonnxruntime.so)
endif()

add_subdirectory(examples)

if(RUN_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
