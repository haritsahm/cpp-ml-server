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
        /// @brief Inference engine using ONNXRuntime
        template <typename T>
        class ONNXRTEngine : public InferenceEngine<T>
        {
        public:
            ONNXRTEngine() = default;
            ~ONNXRTEngine() {};

            /// @brief Construct inference engine based on model path and batch size
            /// @param model_path path to onnx model.
            /// @param batch_size desired batch size.
            ONNXRTEngine(const std::string &model_path, const int &batch_size);

            ONNXRTEngine(const ONNXRTEngine &engine) = delete;
            ONNXRTEngine &operator=(const ONNXRTEngine &engine);
            ONNXRTEngine(ONNXRTEngine &&engine) = delete;
            ONNXRTEngine &operator=(ONNXRTEngine &&engine);

            /// @brief Process data using inference engine.
            /// @param infer_data vector of inference data.
            /// @param infer_results vector of inference results.
            /// @return Error code to validate process.
            cps_utils::Error process(const std::vector<cps_utils::InferenceData<T>> &infer_data, std::vector<cps_utils::InferenceResult<T>> &infer_results);

        private:
            /// @brief ONNXRuntime handler.
            std::unique_ptr<ORTRunner> ort_runner;
            /// @brief Model configuration
            std::vector<cps_utils::ModelConfig> model_configs;
            /// @brief vector to store inputs and outputs as ORT Values
            std::vector<Ort::Value> input_tensors_, output_tensors_;

            /// @brief Validate inference data with confugration and buffer allocations.
            /// @param data vector of input data.
            /// @return Error code to validate process.
            cps_utils::Error validate(const std::vector<cps_utils::InferenceData<T>> &infer_data);

        };
    }
}

#endif