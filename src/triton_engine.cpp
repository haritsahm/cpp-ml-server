#include "cpp_server/triton_engine.hpp"

TritonEngine::TritonEngine(const ModelConfig &model_config, const ClientConfig &client_config, const int &batch_size)
    : InferenceEngine(model_config, batch_size), client_config(client_config)
{

    tc::Error err;
    if (client_config.protocol == ProtocolType::HTTP)
    {
        err = tc::InferenceServerHttpClient::Create(
            &triton_client.http_client_, client_config.url, client_config.verbose);
    }
    else
    {
        err = tc::InferenceServerGrpcClient::Create(
            &triton_client.grpc_client_, client_config.url, client_config.verbose);
    }
    // TODO: Move to error code or status
    if (!err.IsOk())
    {
        std::cerr << "error: unable to create client for inference: " << err
                  << std::endl;
        exit(1);
    }

    readModelConfig();
    initializeMemory();
}

void TritonEngine::readModelConfig()
{
    tc::Error err;
    if (client_config.protocol == ProtocolType::HTTP)
    {
        std::string model_metadata;
        err = triton_client.http_client_->ModelMetadata(
            &model_metadata, client_config.model_name, client_config.model_version, client_config.http_headers);
        if (!err.IsOk())
        {
            std::cerr << "error: failed to get model metadata: " << err << std::endl;
            status = false;
        }

        err = tc::ParseJson(&model_metadata_json, model_metadata);
        if (!err.IsOk())
        {
            std::cerr << "error: failed to parse model metadata: " << err
                      << std::endl;
            status = false;
        }
        std::string model_config_str;
        err = triton_client.http_client_->ModelConfig(
            &model_config_str, client_config.model_name, client_config.model_version, client_config.http_headers);
        if (!err.IsOk())
        {
            std::cerr << "error: failed to get model config: " << err << std::endl;
            status = false;
        }

        err = tc::ParseJson(&model_config_json, model_config_str);
        if (!err.IsOk())
        {
            std::cerr << "error: failed to parse model config: " << err << std::endl;
        }
        if (!ParseModelHttp(
                model_metadata_json, model_config_json, batch_size, &model_config))
        {
            status = false;
        }
    }
    else
    {
        inference::ModelMetadataResponse model_metadata_response;
        err = triton_client.grpc_client_->ModelMetadata(
            &model_metadata_response, client_config.model_name, client_config.model_version, client_config.http_headers);
        if (!err.IsOk())
        {
            std::cerr << "error: failed to get model metadata: " << err << std::endl;
            status = false;
        }
        inference::ModelConfigResponse model_config_response;
        err = triton_client.grpc_client_->ModelConfig(
            &model_config_response, client_config.model_name, client_config.model_version, client_config.http_headers);
        if (!err.IsOk())
        {
            std::cerr << "error: failed to get model config: " << err << std::endl;
            status = false;
        }
        if (!ParseModelGrpc(model_metadata_response, model_config_response, batch_size, &model_config))
        {
            status = false;
        }
    }
    status = true;
}

void TritonEngine::initializeMemory()
{
    // Initialize the inputs with the data.
    tc::InferInput *input;
    tc::Error err;
    err = tc::InferInput::Create(
        &input, model_config.input_name_, model_config.input_shape_, model_config.input_datatype_);
    if (!err.IsOk())
    {
        std::cerr << "unable to get input: " << err << std::endl;
        exit(1);
    }
    input_ptr.reset(input);

    tc::InferRequestedOutput *output;
    // Set the number of classification expected
    err =
        tc::InferRequestedOutput::Create(&output, model_config.output_name_);
    if (!err.IsOk())
    {
        std::cerr << "unable to get output: " << err << std::endl;
        exit(1);
    }
    output_ptr.reset(output);

    // The inference settings. Will be using default for now.
    infer_options.model_name_ = client_config.model_name;
    infer_options.model_version_ = client_config.model_version;

    infer_inputs.push_back(input_ptr.get());   //! Moving raw pointer
    infer_outputs.push_back(output_ptr.get()); //! Moving raw pointer
}

bool TritonEngine::validate(const std::vector<InferenceData<uint8_t>> &data)
{
    if (data.size() != batch_size)
    {
        // std::cerr << "error: failed to get model config: " << err << std::endl;
        return false;
    }
    size_t data_byte_size = 0;
    for (const InferenceData<uint8_t> &d : data)
    {
        data_byte_size += sizeof(uint8_t) * d.data.size();
    }
    if (data_byte_size > model_config.input_byte_size_)
    {
        // std::cerr << "error: failed to get model config: " << err << std::endl;
        return false;
    }

    return true;
}

InferenceResult<uint8_t> TritonEngine::postprocess(const std::unique_ptr<tc::InferResult> &result, const size_t &batch_size, const std::string &output_name)
{
    InferenceResult<uint8_t> res;
    if (!result->RequestStatus().IsOk())
    {
        std::cerr << "inference  failed with error: " << result->RequestStatus()
                  << std::endl;
        exit(1);
    }

    // Get and validate the shape and datatype
    tc::Error err = result->Shape(output_name, &res.shape);
    if (!err.IsOk())
    {
        std::cerr << "unable to get shape for " << output_name << std::endl;
        exit(1);
    }

    err = result->Datatype(output_name, &res.data_dtype);
    if (!err.IsOk())
    {
        std::cerr << "unable to get datatype for " << output_name << std::endl;
        exit(1);
    }

    size_t output_byte_size;
    err = result->RawData(output_name, (const uint8_t **)&res.data[0], &output_byte_size);
    if (!err.IsOk())
    {
        std::cerr << "unable to get data for " << output_name << std::endl;
        exit(1);
    }
    return res;
}

void TritonEngine::process(const std::vector<InferenceData<uint8_t>> &infer_data, std::vector<InferenceResult<uint8_t>> &infer_results)
{

    if (!validate(infer_data))
    {
        exit(1);
    }

    size_t data_idx = 0;
    size_t done_cnt = 0;
    size_t sent_count = 0;
    bool last_request = false;
    tc::Error err;
    std::vector<std::unique_ptr<tc::InferResult>> results;

    while (!last_request)
    {
        // Reset the input for new request.
        err = input_ptr->Reset();
        if (!err.IsOk())
        {
            std::cerr << "failed resetting input: " << err << std::endl;
            exit(1);
        }

        // Set input to be the next 'batch_size' images (preprocessed).
        for (int idx = 0; idx < batch_size; ++idx)
        {
            err = input_ptr->AppendRaw(infer_data[data_idx].data);
            if (!err.IsOk())
            {
                std::cerr << "failed setting input: " << err << std::endl;
                exit(1);
            }

            data_idx = (data_idx + 1) % infer_data.size();
            if (data_idx == 0)
            {
                last_request = true;
            }
        }

        infer_options.request_id_ = std::to_string(sent_count);

        tc::InferResult *result;
        if (client_config.protocol == ProtocolType::HTTP)
        {
            err = triton_client.http_client_->Infer(
                &result, infer_options, infer_inputs, infer_outputs, client_config.http_headers);
        }
        else
        {
            err = triton_client.grpc_client_->Infer(
                &result, infer_options, infer_inputs, infer_outputs, client_config.http_headers);
        }
        if (!err.IsOk())
        {
            std::cerr << "failed sending synchronous infer request: " << err
                      << std::endl;
            exit(1);
        }
        results.emplace_back(std::move(std::unique_ptr<tc::InferResult>(result)));
        sent_count++;
    }

    // Post-process the results to make prediction(s)
    for (size_t idx = 0; idx < results.size(); idx++)
    {
        std::cout << "Request " << idx << ", batch size " << batch_size
                  << std::endl;
        InferenceResult<uint8_t> output_data = postprocess(results[idx], batch_size, model_config.output_name_);
        infer_results.push_back(output_data);
    }
}
