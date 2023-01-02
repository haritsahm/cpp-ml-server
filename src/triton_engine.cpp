#include "cpp_server/triton_engine.hpp"

TritonEngine::TritonEngine(const ModelConfig &model_config, const ClientConfig &client_config, const int &batch_size)
    : InferenceEngine(model_config, batch_size), client_config(client_config){

    tc::Error err;
    if (client_config.protocol == ProtocolType::HTTP) {
        err = tc::InferenceServerHttpClient::Create(
            &triton_client.http_client_, client_config.url, client_config.verbose);
    } else {
        err = tc::InferenceServerGrpcClient::Create(
            &triton_client.grpc_client_, client_config.url, client_config.verbose);
    }
    // TODO: Move to error code or status
    if (!err.IsOk()) {
        std::cerr << "error: unable to create client for inference: " << err
                << std::endl;
        exit(1);
    }

    readModelConfig();

}

void TritonEngine::readModelConfig(){
    if (client_config.protocol == ProtocolType::HTTP){
        std::string model_metadata;
        err = triton_client.http_client_->ModelMetadata(
            &model_metadata, client_config.model_name, client_config.model_version, client_config.http_headers);
        if (!err.IsOk()) {
            std::cerr << "error: failed to get model metadata: " << err << std::endl;
            status = false;
        }

        err = tc::ParseJson(&model_metadata_json, model_metadata);
        if (!err.IsOk()) {
            std::cerr << "error: failed to parse model metadata: " << err
                        << std::endl;
            status = false;
        }
        std::string model_config;
        err = triton_client.http_client_->ModelConfig(
            &model_config, client_config.model_name, client_config.model_version, client_config.http_headers);
        if (!err.IsOk()) {
            std::cerr << "error: failed to get model config: " << err << std::endl;
            status = false;
        }

        err = tc::ParseJson(&model_config_json, model_config);
        if (!err.IsOk()) {
        std::cerr << "error: failed to parse model config: " << err << std::endl;
        }
        if (!ParseModelHttp(
            model_metadata_json, model_config_json, batch_size, &model_config)){
                status = false;
            }
    }
    else {
        inference::ModelMetadataResponse model_metadata;
        err = triton_client.grpc_client_->ModelMetadata(
            &model_metadata, client_config.model_name, client_config.model_version, client_config.http_headers);
        if (!err.IsOk()) {
            std::cerr << "error: failed to get model metadata: " << err << std::endl;
            status = false;
        }
        inference::ModelConfigResponse model_config;
        err = triton_client.grpc_client_->ModelConfig(
            &model_config, client_config.model_name, client_config.model_version, http_headers);
        if (!err.IsOk()) {
            std::cerr << "error: failed to get model config: " << err << std::endl;
            status = false;
        }
        if (!ParseModelGrpc(model_metadata, model_config, batch_size, &model_info)){
            status = false;
        }
    }
    status = true;
}

bool TritonEngine::validate(const InferenceData &data){
    /*
    TODO: Validate input data size is below reserved input size

    */
    return true;
}

void TritonEngine::process(const InferenceData &data){

    if (!validate(data))
    {
        exit(1);
    }

    // Copy to tc::InferInput

    // Request Output

    // Copy to tc::InferResult result

    // Return output data

}

