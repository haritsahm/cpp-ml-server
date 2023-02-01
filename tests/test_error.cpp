#include <gtest/gtest.h>
#include "cpp_server/utils/error.hpp"

namespace cps_utils = cpp_server::utils;

TEST(ERRORClass, error_class_handler){
    EXPECT_STREQ(cps_utils::Error::CodeString(cps_utils::Error::Code::SUCCESS), "OK");
    EXPECT_STREQ(cps_utils::Error::CodeString(cps_utils::Error::Code::INTERNAL), "Internal");
}
