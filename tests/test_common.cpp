#include <gtest/gtest.h>
#include <vector>
#include "cpp_server/utils/common.hpp"

namespace cps_utils = cpp_server::utils;

TEST(COMMONTools, basic_common_convertion)
{
    std::vector<int> test_data1{1, 50, 123, 45};
    std::vector<uint8_t> test_data_enc_1 = cps_utils::vectorT_to_blob<int>(test_data1);
    std::vector<int> test_data_res_1 = cps_utils::blob_to_vectorT<int>(test_data_enc_1);

    for (int i = 0; i < test_data1.size(); ++i) {
        EXPECT_EQ(test_data_res_1[i], test_data1[i]);
    }

    std::vector<float> test_data2{100.12, 50.1234, 125.2356, 45.0};
    std::vector<uint8_t> test_data_enc_2 = cps_utils::vectorT_to_blob<float>(test_data2);
    std::vector<float> test_data_res_2 = cps_utils::blob_to_vectorT<float>(test_data_enc_2);

    for (int i = 0; i < test_data1.size(); ++i) {
        EXPECT_EQ(test_data_res_2[i], test_data2[i]);
    }

}
