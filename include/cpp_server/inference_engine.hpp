#ifndef INFERENCE_ENGINE_HPP
#define INFERENCE_ENGINE_HPP

#include <string>
#include <vector>
#include <cstdint>
#include "common.hpp"

class InferenceEngine
{
public:
    InferenceEngine() = default;
    InferenceEngine(const ModelConfig &config, const int &batch_size)
        : model_config(config), batch_size(batch_size){};
    virtual ~InferenceEngine(){};
    InferenceEngine(const InferenceEngine &engine) = delete;
    InferenceEngine &operator=(InferenceEngine &engine);
    InferenceEngine(InferenceEngine &&engine) = delete;
    InferenceEngine &operator=(InferenceEngine &&engine);

    virtual void process(const std::vector<InferenceData<uint8_t>> &infer_data, std::vector<InferenceResult<uint8_t>> &infer_results) = 0;

protected:
    ModelConfig model_config{};
    int batch_size{1};

    // TODO: Tuple of error code and explenation in case of error.
};

#endif