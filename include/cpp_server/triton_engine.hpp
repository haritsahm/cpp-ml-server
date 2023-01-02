#ifndef TRITON_ENGINE_HPP
#define TRITON_ENGINE_HPP

#include <string>
#include <http_client.h>
#include <json_utils.h>
#include "inference_engine.hpp"
#include "common.hpp"

namespace tc = triton::client;

enum ProtocolType
{
    HTTP = 0,
    GRPC = 1
};

struct ClientConfig
{
    std::string model_name;
    std::string model_version{""};
    std::string url("localhost:8000");
    ProtocolType protocol = ProtocolType::HTTP;
    tc::Headers http_headers;
}

class TritonEngine : public InferenceEngine
{
public:
    TritonEngine() = default;
    TritonEngine(const ModelConfig &model_config, const ClientConfig &client_config, const int &batch_size);
    ~TritonEngine(){};
    TritonEngine(const TritonEngine &engine) = delete;
    TritonEngine &operator=(const TritonEngine &engine);
    TritonEngine(TritonEngine &&engine) = delete;
    TritonEngine &operator=(TritonEngine &&engine);

    void process(const InferenceData &data);
    bool validate(const InferenceData &data);
    bool isOk() {return status;}

private:
    bool status{false};
    rapidjson::Document model_metadata_json;
    rapidjson::Document model_config_json;

    ClientConfig client_config{};
    ModelConfig model_config{};
    TritonClient triton_client;

    void readModelConfig();
}

#endif
