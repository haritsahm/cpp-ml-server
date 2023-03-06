#ifndef ONNXRT_ENGINE_HPP
#define ONNXRT_ENGINE_HPP

#include <memory>
#include <numeric>
#include <vector>

#include <rapidjson/document.h>
#include "utils/common.hpp"
#include "utils/error.hpp"
#include "base/inference_engine.hpp"
#include "cpp_server/onnxrt_helper.hpp"


namespace cps_utils = cpp_server::utils;

namespace cpp_server
{
    namespace inferencer
    {
        template <typename T>
        class ONNXRTEngine : public InferenceEngine<T>
        {
        public:
            ONNXRTEngine() = default;
            ~ONNXRTEngine() {};

            ONNXRTEngine(const std::string &model_path, const int &batch_size);

            ONNXRTEngine(const ONNXRTEngine &engine) = delete;
            ONNXRTEngine &operator=(const ONNXRTEngine &engine);
            ONNXRTEngine(ONNXRTEngine &&engine) = delete;
            ONNXRTEngine &operator=(ONNXRTEngine &&engine);

            cps_utils::Error process(const std::vector<cps_utils::InferenceData<T>> &infer_data, std::vector<cps_utils::InferenceResult<T>> &infer_results);

        private:
            std::unique_ptr<ORTRunner> ort_runner;
            std::vector<cps_utils::ModelConfig> model_configs;
            std::vector<Ort::Value> input_tensors_, output_tensors_;

            cps_utils::Error validate(const std::vector<cps_utils::InferenceData<T>> &infer_data);

        };
    }
}

#endif