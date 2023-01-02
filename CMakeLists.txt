cmake_minimum_required(VERSION 3.14)
project(cpp-server)
SET(LIBASYIK_ENABLE_SOCI ON) # if SOCI database support is desired

if(DEFINED ENV{CXX_VERSION})
    set(CMAKE_CXX_STANDARD $ENV{CXX_VERSION}) # set to 14 or 17 when available
else()
    set(CMAKE_CXX_STANDARD 11)
endif()

find_package(libasyik REQUIRED)
find_package(Boost COMPONENTS context fiber date_time REQUIRED)
find_package(TritonCommon REQUIRED PATHS /usr/local/lib/cmake/TritonCommon)
find_package(TritonClient REQUIRED PATHS /usr/local/lib/cmake/TritonClient)
find_package(OpenCV 4 REQUIRED)

include_directories(
    include
)

add_executable(${PROJECT_NAME} main.cpp) # add more source code here

if(libasyik_FOUND)
    target_include_directories(${PROJECT_NAME} PUBLIC ${libasyik_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} libasyik)
endif()

if(TritonClient_FOUND)
    target_include_directories(${PROJECT_NAME} PUBLIC ${TritonClient_INCLUDE_DIR} ${TritonCommon_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} TritonClient::httpclient TritonCommon::triton-common-json)
endif()

if(Boost_FOUND)
    target_include_directories(${PROJECT_NAME} PUBLIC ${Boost_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} Boost::fiber Boost::context Boost::date_time)
endif()

if(OpenCV_FOUND)
    target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
endif()

find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} Threads::Threads)

find_package(OpenSSL REQUIRED)
target_link_libraries(${PROJECT_NAME} OpenSSL::SSL)
