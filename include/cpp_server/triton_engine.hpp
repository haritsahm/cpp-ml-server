#ifndef TRITON_ENGINE_HPP
#define TRITON_ENGINE_HPP

#include <string>
#include <memory>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <iostream>
#include <http_client.h>
#include "json_utils.h"
#include <rapidjson/document.h>
#include "inference_engine.hpp"
#include "common.hpp"
#include "triton_helper.hpp"
#include "error.h"

namespace cpp_server
{
    /// @brief Inference engine using Triton Inference Client
    class TritonEngine : public InferenceEngine
    {
    public:
        TritonEngine() = default;

        /// @brief Construct inference engine based on client configuration and batch size
        /// @param client_config triton client configurations.
        /// @param batch_size desired batch size.
        TritonEngine(const ClientConfig &client_config, const int &batch_size);
        ~TritonEngine()
        {
            for (auto p : infer_inputs)
            {
                delete p;
            }
            infer_inputs.clear();

            for (auto p : infer_outputs)
            {
                delete p;
            }
            infer_outputs.clear();
        };
        TritonEngine(const TritonEngine &engine) = delete;
        TritonEngine &operator=(const TritonEngine &engine);
        TritonEngine(TritonEngine &&engine) = delete;
        TritonEngine &operator=(TritonEngine &&engine);

        /// @brief Process data using inference engine.
        /// @param infer_data vector of inference data.
        /// @param infer_results vector of inference results.
        /// @return Error code to validate process.
        Error process(const std::vector<InferenceData<uint8_t>> &infer_data, std::vector<InferenceResult<uint8_t>> &infer_results);

    private:
        /// @brief Model metadata in json format
        rapidjson::Document model_metadata_json;
        /// @brief Model configuration in json format
        rapidjson::Document model_config_json;

        /// @brief Client configuration.
        ClientConfig client_config{};
        /// @brief Model configuration
        ModelConfig model_config{};
        /// @brief Triton client handler.
        TritonClient triton_client;

        /// @brief Pointer to store Triton's InferInput
        std::shared_ptr<tc::InferInput> input_ptr;
        /// @brief Pointer to store Triton's output
        std::shared_ptr<tc::InferRequestedOutput> output_ptr;
        /// @brief Vector to store raw pointer Triton's input
        std::vector<tc::InferInput *> infer_inputs;
        /// @brief Vector to store raw pointer Triton's output
        std::vector<const tc::InferRequestedOutput *> infer_outputs;
        /// @brief Triton inference options
        tc::InferOptions infer_options{""};

        /// @brief Read triton model configuration from server
        /// @return Error code to validate process.
        Error readModelConfig();

        /// @brief Initialize and allocate buffer memory
        /// @return Error code to validate process.
        Error initializeMemory();

        /// @brief Validate inference data with confugration and buffer allocations.
        /// @param infer_data vector of input data.
        /// @return Error code to validate process.
        Error validate(const std::vector<InferenceData<uint8_t>> &infer_data);
        /// @brief Apply postprocessing to convert response from server to buffer outputs.
        /// @param result pointer to inference result.
        /// @param res Store buffer output.
        /// @param batch_size inference batch size.
        /// @param output_name output name.
        /// @return Error code to validate process.
        Error postprocess(const std::unique_ptr<tc::InferResult> &result, InferenceResult<uint8_t> &res,
                          const size_t &batch_size, const std::string &output_name);
    };
};

#endif
