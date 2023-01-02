#include "cpp_server/processor.hpp"

Processor::Processor(ClientConfig &config)
    : client_config(config)
{
    err = tc::InferenceServerHttpClient::Create(&triton_client.http_client_, url, verbose);
    if (!err.IsOk())
    {
        // TODO: Should be raise
        std::cerr << "error: unable to create client for inference: " << err
                  << std::endl;
        exit(1);
    }
}
