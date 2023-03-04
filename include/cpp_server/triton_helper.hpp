#ifndef TRITON_HELPER_HPP
#define TRITON_HELPER_HPP

#include <string>
#include <vector>
#include <grpc_client.h>
#include <http_client.h>
#include <rapidjson/document.h>
#include <triton/common/model_config.h>
#include <opencv2/core.hpp>
#include "utils/common.hpp"

namespace tc = triton::client;
namespace cps_utils = cpp_server::utils;

namespace cpp_server
{
    namespace inferencer
    {
        /// @brief TritonClient inference pointer
        /// Source: https://github.com/triton-inference-server/client/blob/main/src/c%2B%2B/examples/image_client.cc#L741
        union TritonClient
        {
            TritonClient()
            {
                new (&http_client_) std::unique_ptr<tc::InferenceServerHttpClient>{};
            }
            ~TritonClient() {}

            std::unique_ptr<tc::InferenceServerHttpClient> http_client_;
            std::unique_ptr<tc::InferenceServerGrpcClient> grpc_client_;
        };

        /// @brief TritonClient protocol type
        /// Source: https://github.com/triton-inference-server/client/blob/main/src/c%2B%2B/examples/image_client.cc#L67
        enum ProtocolType
        {
            HTTP = 0,
            GRPC = 1
        };

        /// @brief Configuration to access triton server.
        struct ClientConfig
        {
            std::string model_name;
            std::string model_version{""};
            std::string url{"localhost:8000"};
            ProtocolType protocol = ProtocolType::HTTP;
            tc::Headers http_headers;
            bool verbose;
        };

        /// @brief Parse model configuration from grpc client.
        /// Source: https://github.com/triton-inference-server/client/blob/main/src/c%2B%2B/examples/image_client.cc#L410
        bool ParseModelGrpc(
            const inference::ModelMetadataResponse &model_metadata,
            const inference::ModelConfigResponse &model_config, const size_t &batch_size,
            cps_utils::ModelConfig *model_info);

        /// @brief Parse model configuration from http client.
        /// Source: https://github.com/triton-inference-server/client/blob/main/src/c%2B%2B/examples/image_client.cc#L536
        bool ParseModelHttp(
            const rapidjson::Document &model_metadata,
            const rapidjson::Document &model_config, const size_t &batch_size,
            cps_utils::ModelConfig *model_info);
    }
}

#endif