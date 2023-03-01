cmake_minimum_required (VERSION 3.18)

set(LIBASYIK_ENABLE_SOCI OFF) # if SOCI database support is desired

find_package(libasyik REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)

if(ENABLE_TRITON)
    add_executable(
        image_processing_triton
        image_processing_triton.cpp
    )

    target_link_libraries(image_processing_triton
        common_utils
        image_processor
        triton_inference_engine
        Threads::Threads
        OpenSSL::SSL)

    if(libasyik_FOUND)
        target_include_directories(image_processing_triton PUBLIC ${libasyik_INCLUDE_DIR})
        target_link_libraries(image_processing_triton libasyik)
    endif()

    if(RapidJSON_FOUND)
        target_include_directories(image_processing_triton PUBLIC ${RapidJSON_INCLUDE_DIRS})
    endif()

    if(Boost_FOUND)
        target_include_directories(image_processing_triton PUBLIC ${Boost_INCLUDE_DIR})
        target_link_libraries(image_processing_triton Boost::fiber Boost::context Boost::date_time Boost::url)
    endif()


endif()
