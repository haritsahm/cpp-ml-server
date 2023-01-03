#ifndef PROCESSOR_HPP
#define PROCESSOR_HPP

#include <string>
#include <memory>
#include "inference_engine.hpp"
#include "common.hpp"

struct InferenceResponse
{
    std::string name;
    float score;
};

class Processor
{
public:
    Processor() = default;
    virtual ~Processor(){};

    Processor(const Processor &processor) = delete;
    Processor &operator=(const Processor &processor);

    Processor(Processor &&processor) = delete;
    Processor &operator=(Processor &&processor);

    virtual InferenceResponse process(const std::string &ss) = 0;

private:
    std::unique_ptr<InferenceEngine> infer_engine;
};

#endif