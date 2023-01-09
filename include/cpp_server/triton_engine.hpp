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
    class TritonEngine : public InferenceEngine
    {
    public:
        TritonEngine() = default;
        TritonEngine(const ModelConfig &model_config, const ClientConfig &client_config, const int &batch_size);
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
            infer_inputs.clear();
        };
        TritonEngine(const TritonEngine &engine) = delete;
        TritonEngine &operator=(const TritonEngine &engine);
        TritonEngine(TritonEngine &&engine) = delete;
        TritonEngine &operator=(TritonEngine &&engine);

        Error process(const std::vector<InferenceData<uint8_t>> &infer_data, std::vector<InferenceResult<uint8_t>> &infer_results);

    private:
        rapidjson::Document model_metadata_json;
        rapidjson::Document model_config_json;

        ClientConfig client_config{};
        ModelConfig model_config{};
        TritonClient triton_client;

        std::shared_ptr<tc::InferInput> input_ptr;
        std::shared_ptr<tc::InferRequestedOutput> output_ptr;
        std::vector<tc::InferInput *> infer_inputs;                  // TODO: Destructor cleanup
        std::vector<const tc::InferRequestedOutput *> infer_outputs; // TODO: Destructor cleanup
        tc::InferOptions infer_options{""};

        std::mutex mtx;
        std::condition_variable cv;

        Error readModelConfig();
        Error initializeMemory();
        Error validate(const std::vector<InferenceData<uint8_t>> &infer_data);
        Error postprocess(const std::unique_ptr<tc::InferResult> &result, InferenceResult<uint8_t> &res,
                          const size_t &batch_size, const std::string &output_name);
    };
};

#endif
