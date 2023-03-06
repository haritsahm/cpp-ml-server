#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "cpp_server/utils/error.hpp"
#include "cpp_server/utils/common.hpp"
#include "cpp_server/onnxrt_helper.hpp"

namespace cps_utils = cpp_server::utils;
namespace cps_inferencer = cpp_server::inferencer;

TEST(Runner, config_reading)
{
    std::string model_path = "/model-repository/imagenet_classification_static/1/model.onnx";

    cps_inferencer::ORTRunner runner(model_path);

    std::vector<cps_utils::ModelConfig> configs = runner.getModelConfigs();
    EXPECT_STREQ(configs[0].input_name_.c_str(), "input");
    EXPECT_STREQ(configs[0].input_datatype_.c_str(), "FP32");
    EXPECT_EQ(configs[0].input_byte_size_, 1769472);
    EXPECT_STREQ(configs[0].output_name_.c_str(), "output");
    EXPECT_STREQ(configs[0].output_datatype_.c_str(), "FP32");
    EXPECT_EQ(configs[0].output_byte_size_, 4000);
}