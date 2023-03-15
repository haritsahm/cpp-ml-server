#ifndef ONNXRT_HELPER_HPP
#define ONNXRT_HELPER_HPP

#include <memory>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include "cpp_server/utils/common.hpp"
#include "cpp_server/utils/error.hpp"
#include <fstream>

namespace cps_utils = cpp_server::utils;

namespace cpp_server
{
    namespace inferencer
    {
        /// @brief Get ONNX Element Type as a string
        /// @param dtype int code from ONNXTensorElementDataType
        /// @return Element type as a string.
        static std::string getONNXStrElementType(const int &dtype)
        {
            switch(dtype)
            {
                // ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
                case 0:
                    return "UNDEFINED";
                // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                case 1:
                    return "FP32";
                // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
                case 2:
                    return "UINT8";
                // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
                case 3:
                    return "INT8";
                // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
                case 4:
                    return "UINT16";
                // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
                case 5:
                    return "INT16";
                // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
                case 6:
                    return "INT32";
                // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                case 7:
                    return "INT64";
                // ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
                case 8:
                    return "STRING";
                // ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
                case 9:
                    return "BOOL";
                // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
                case 10:
                    return "FP16";
                // ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
                case 11:
                    return "FP64";
                // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
                case 12:
                    return "UINT32";
                // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
                case 13:
                    return "UINT64";
                // ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64
                case 14:
                    return "UNDEFINED";
                // ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128
                case 15:
                    return "UNDEFINED";
                // ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
                case 16:
                    return "BF16";
                default:
                    break;
            }

            return "UNDEFINED";
        }

        class ORTRunner
        {
        public:
            ORTRunner() = default;
            /// @brief Construct ONNXRuntime Runner based on model path
            /// @param model_path path to onnx model.
            ORTRunner(const std::string &model_path);
            ~ORTRunner(){};

            /// @brief read model configurations from onnx file.
            cps_utils::Error readModelConfigs();
            /// @brief Get vector of model configurations.
            std::vector<cps_utils::ModelConfig> getModelConfigs() { return model_configs_;};
            /// @brief Check if the session is valid.
            bool isValid() {bool val = session_ == nullptr ? false : true; return val;};

            /// @brief Process data using inference engine.
            /// @param input_tensors vector of input data in Ort Value.
            /// @param output_tensors vector of output data in Ort Value.
            /// @return Error code to validate process.
            cps_utils::Error process(const std::vector<Ort::Value> &input_tensors, std::vector<Ort::Value> &output_tensors);

        private:
            /// @brief onnxruntime session handler.
            std::unique_ptr<Ort::Session> session_;
            /// @brief onnxruntime environment.
            Ort::Env env_;
            /// @brief onnxruntime run options.
            Ort::RunOptions run_options_;

            /// @brief vector to store input and output names.
            std::vector<const char*> input_node_names_, output_node_names_;
            /// @brief vector to store model configurations.
            std::vector<cps_utils::ModelConfig> model_configs_;

        };
    }
}


#endif