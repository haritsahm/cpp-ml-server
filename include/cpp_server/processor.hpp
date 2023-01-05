#ifndef PROCESSOR_HPP
#define PROCESSOR_HPP

#include <string>
#include <memory>
#include <rapidjson/document.h>
#include "error.hpp"
#include "inference_engine.hpp"

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

    virtual cpp_server::Error process(const rapidjson::Document &data, rapidjson::Document &result) = 0;

protected:
    std::unique_ptr<cpp_server::InferenceEngine> infer_engine;
};

#endif