cmake_minimum_required(VERSION 3.15)
project(cpp-server)

if(DEFINED ENV{CXX_VERSION})
    set(CMAKE_CXX_STANDARD $ENV{CXX_VERSION}) # set to 14 or 17 when available
else()
    set(CMAKE_CXX_STANDARD 14)
endif()

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
find_package(Protobuf REQUIRED)
find_package(TritonCommon REQUIRED PATHS /usr/local/lib/cmake/TritonCommon)
find_package(TritonClient REQUIRED PATHS /usr/local/lib/cmake/TritonClient)
find_package(OpenCV 4 REQUIRED)

include_directories(
    include
)

add_library(common_utils
    src/utils/error.cpp
    src/utils/base64_decoder.cpp
)

add_library(image_processor
    src/image_processor.cpp
)

add_library(inference_engines
    src/triton_engine.cpp
    src/triton_helper.cpp
)

add_executable(${PROJECT_NAME}
    main.cpp
)

target_link_libraries(image_processor common_utils)
target_link_libraries(${PROJECT_NAME} common_utils image_processor inference_engines)

if(libasyik_FOUND)
    target_include_directories(${PROJECT_NAME} PUBLIC ${libasyik_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} libasyik)
endif()

if(RapidJSON_FOUND)
    target_include_directories(image_processor PUBLIC ${RapidJSON_INCLUDE_DIRS})
    target_include_directories(${PROJECT_NAME} PUBLIC ${RapidJSON_INCLUDE_DIRS})
    endif()

if(Protobuf_FOUND)
    target_include_directories(inference_engines PUBLIC ${Protobuf_INCLUDE_DIRS})
    target_link_libraries(inference_engines ${Protobuf_LIBRARIES})
endif()

if(TritonClient_FOUND)
    target_include_directories(image_processor PUBLIC ${TritonClient_INCLUDE_DIR} ${TritonCommon_INCLUDE_DIR})
    target_include_directories(inference_engines PUBLIC ${TritonClient_INCLUDE_DIR} ${TritonCommon_INCLUDE_DIR})
    target_link_libraries(image_processor TritonClient::httpclient TritonClient::grpcclient)
    target_link_libraries(inference_engines TritonClient::httpclient TritonClient::grpcclient)
endif()

if(TritonCommon_FOUND)
    target_include_directories(inference_engines PUBLIC ${TritonCommon_INCLUDE_DIR})
    target_link_libraries(inference_engines TritonCommon::triton-common-model-config TritonCommon::triton-common-json)
endif()

if(Boost_FOUND)
    target_include_directories(${PROJECT_NAME} PUBLIC ${Boost_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} Boost::fiber Boost::context Boost::date_time Boost::url)
endif()

if(OpenCV_FOUND)
    target_include_directories(image_processor PUBLIC ${OpenCV_INCLUDE_DIR})
    target_include_directories(inference_engines PUBLIC ${OpenCV_INCLUDE_DIR})
    target_link_libraries(image_processor ${OpenCV_LIBS})
    target_link_libraries(inference_engines ${OpenCV_LIBS})
endif()

find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} Threads::Threads)

find_package(OpenSSL REQUIRED)
target_link_libraries(${PROJECT_NAME} OpenSSL::SSL)

if(RUN_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
