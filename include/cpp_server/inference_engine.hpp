#ifndef INFERENCE_ENGINE_HPP
#define INFERENCE_ENGINE_HPP

#include <string>

struct EngineConfig
{
    std::string output_name_{"output"};
    std::string input_name_{"input"};
    std::string input_datatype_;
    std::string input_format_{"FORMAT_NCHW"};
    int max_batch_size_{1};
};

class InferenceEngine
{
public:
    InferenceEngine() = default;
    InferenceEngine(const EngineConfig &config)
        : engine_config(config){};
    virtual ~InferenceEngine(){};
    InferenceEngine(const InferenceEngine &engine) = delete;
    InferenceEngine &operator=(InferenceEngine &engine);
    InferenceEngine(InferenceEngine &&engine) = delete;
    InferenceEngine &operator=(InferenceEngine &&engine);

    void process();
    virtual bool validate() = 0;

private:
    EngineConfig engine_config{};
};

#endif