#ifndef ONNXRT_ENGINE_HPP
#define ONNXRT_ENGINE_HPP

#include <vector>
#include <memory>

#include <rapidjson/document.h>
#include "base/inference_engine.hpp"
#include "utils/common.hpp"
#include "utils/error.hpp"


namespace cps_utils = cpp_server::utils;

namespace cpp_server
{
    namespace inferencer
    {
        class ONNXRTEngine : public InferenceEngine
        {
        public:
            ONNXRTEngine() = default;
            ~ONNXRTEngine() {};

            ONNXRTEngine(const cpp_server::utils::ModelConfig &model_config, const int &batch_size);

            ONNXRTEngine(const ONNXRTEngine &engine) = delete;
            ONNXRTEngine &operator=(const ONNXRTEngine &engine);
            ONNXRTEngine(ONNXRTEngine &&engine) = delete;
            ONNXRTEngine &operator=(ONNXRTEngine &&engine);

            cps_utils::Error process(const std::vector<cps_utils::InferenceData<uint8_t>> &infer_data, std::vector<cps_utils::InferenceResult<uint8_t>> &infer_results);

        private:
            cps_utils::Error validate(const std::vector<cps_utils::InferenceData<uint8_t>> &infer_data);

        }

    }
}

#endif