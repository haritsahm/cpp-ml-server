#ifndef TRITON_ENGINE_HPP
#define TRITON_ENGINE_HPP

#include <string>
#include <http_client.h>
#include <json_utils.h>
#include "inference_engine.hpp"

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
    TritonEngine(const EngineConfig &engine_config, const ClientConfig &client_config)
        : InferenceEngine(engine_config), client_config(client_config){};
    ~TritonEngine(){};
    TritonEngine(const TritonEngine &engine) = delete;
    TritonEngine &operator=(const TritonEngine &engine);
    TritonEngine(TritonEngine &&engine) = delete;
    TritonEngine &operator=(TritonEngine &&engine);

private:
    ClientConfig client_config{};
}

#endif