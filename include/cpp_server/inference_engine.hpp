#ifndef INFERENCE_ENGINE_HPP
#define INFERENCE_ENGINE_HPP

#include <string>
#include <vector>
#include <cstdint>
#include "common.hpp"
#include "error.hpp"

namespace cpp_server
{
    /// @brief Abstract class for inference engine
    class InferenceEngine
    {
    public:
        InferenceEngine() = default;

        /// @brief Inference engine constructor from client config and desired batch size.
        /// @param batch_size Desired processing batch size.
        InferenceEngine(const int &batch_size)
            : batch_size(batch_size){};
        virtual ~InferenceEngine(){};
        InferenceEngine(const InferenceEngine &engine) = delete;
        InferenceEngine &operator=(InferenceEngine &engine);
        InferenceEngine(InferenceEngine &&engine) = delete;
        InferenceEngine &operator=(InferenceEngine &&engine);

        /// @brief Process data using inference engine.
        /// @param infer_data vector of inference data.
        /// @param infer_results vector of inference results.
        /// @return Error code to validate process.
        virtual Error process(const std::vector<InferenceData<uint8_t>> &infer_data, std::vector<InferenceResult<uint8_t>> &infer_results) = 0;

        /// @brief Check if the inference engine is valid.
        /// @return boolean status.
        bool isOk() { return status; }

        /// @brief Get model configuration data from inference engine.
        /// @return model configuration.
        ModelConfig modelConfig() { return model_config; }

    protected:
        /// @brief Model configuration data
        ModelConfig model_config{};

        /// @brief Desired batch size.
        int batch_size{1};

        /// @brief Inference engine status.
        bool status{false};
    };
};

#endif