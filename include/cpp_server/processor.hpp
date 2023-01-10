#ifndef PROCESSOR_HPP
#define PROCESSOR_HPP

#include <string>
#include <memory>
#include <rapidjson/document.h>
#include "error.hpp"
#include "inference_engine.hpp"

class Processor
{
public:
    Processor() = default;
    virtual ~Processor(){};

    Processor(const Processor &processor) = delete;
    Processor &operator=(const Processor &processor);

    Processor(Processor &&processor) = delete;
    Processor &operator=(Processor &&processor);

    virtual cpp_server::Error process(const rapidjson::Document &data_doc, rapidjson::Document &result_doc) = 0;

protected:
    std::unique_ptr<cpp_server::InferenceEngine> infer_engine;
};

#endif