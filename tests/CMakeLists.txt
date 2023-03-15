include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0
)
FetchContent_MakeAvailable(googletest)
add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)

add_executable(test_error_func
    test_error.cpp
)
target_link_libraries(test_error_func
    PRIVATE
    GTest::GTest
    common_utils
)

add_executable(test_rapid_json
    test_rapidjson.cpp
)
target_link_libraries(test_rapid_json
    PRIVATE
    GTest::GTest
    common_utils
)

add_executable(test_base64
    test_base64.cpp
)
target_link_libraries(test_base64
    PRIVATE
    GTest::GTest
    common_utils
)

add_executable(test_common
    test_common.cpp
)
target_link_libraries(test_common
    PRIVATE
    GTest::GTest
    common_utils
)

if(ENABLE_ONNXRT)
    add_executable(test_orthelper
        test_orthelper.cpp
    )

    target_link_libraries(test_orthelper
        PRIVATE
        GTest::GTest
        common_utils
        onnxrt_inference_engine
    )

    add_test(NAME test_orthelper COMMAND $<TARGET_FILE:test_orthelper>)

endif(ENABLE_ONNXRT)

add_test(NAME test_error_func COMMAND $<TARGET_FILE:test_error_func>)
add_test(NAME test_rapid_json COMMAND $<TARGET_FILE:test_rapid_json>)
add_test(NAME test_base64 COMMAND $<TARGET_FILE:test_base64>)
add_test(NAME test_common COMMAND $<TARGET_FILE:test_common>)
