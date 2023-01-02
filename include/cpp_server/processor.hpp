#ifndef PROCESSOR_HPP
#define PROCESSOR_HPP

#include <string>
#include "inference_engine.hpp"

struct InferenceResponse
{
    std::string name;
    float score;
};

class Processor :
{
public:
    Processor() = default;
    Processor(ClientConfig &config);
    virtual ~Processor(){};

    Processor(const Processor &processor) = delete;
    Processor &operator=(const Processor &processor);

    Processor(Processor &&processor) = delete;
    Processor &operator=(Processor &&processor);

    virtual InferenceResponse process(const std::string &ss) = 0;

private:
    ClientConfig client_config;
    std::unique_ptr<InferenceEngine> infer_engine;
};

#endif