#include "cpp_server/triton_engine.hpp"

namespace cpp_server
{
    namespace inferencer
    {
        template <typename T>
        TritonEngine<T>::TritonEngine(const cpp_server::inferencer::ClientConfig &client_config, const int &batch_size)
            : client_config(client_config)
        {
            this->batch_size = batch_size;

            tc::Error tc_err;
            if (client_config.protocol == ProtocolType::HTTP)
            {
                tc_err = tc::InferenceServerHttpClient::Create(
                    &triton_client.http_client_, client_config.url, client_config.verbose);
            }
            else
            {
                tc_err = tc::InferenceServerGrpcClient::Create(
                    &triton_client.grpc_client_, client_config.url, client_config.verbose);
            }
            // TODO: Move to error code or status
            if (!tc_err.IsOk())
            {
                std::cout << tc_err.Message() << std::endl;
                this->status = false;
            }
            cps_utils::Error p_err;
            p_err = readModelConfig();
            if (!tc_err.IsOk())
            {
                std::cout << tc_err.Message() << std::endl;
                this->status = false;
            }
            p_err = initializeMemory();
            if (!tc_err.IsOk())
            {
                std::cout << tc_err.Message() << std::endl;
                this->status = false;
            }
            this->status = true;
        }

        template <typename T>
        cps_utils::Error TritonEngine<T>::readModelConfig()
        {
            tc::Error tc_err;
            if (client_config.protocol == ProtocolType::HTTP)
            {
                std::string model_metadata;
                tc_err = triton_client.http_client_->ModelMetadata(
                    &model_metadata, client_config.model_name, client_config.model_version, client_config.http_headers);
                if (!tc_err.IsOk())
                {
                    return cps_utils::Error(cps_utils::Error::Code::INTERNAL, "Failed to get triton model metadata");
                }

                tc_err = tc::ParseJson(&model_metadata_json, model_metadata);
                if (!tc_err.IsOk())
                {
                    return cps_utils::Error(cps_utils::Error::Code::INTERNAL, "Failed to parse triton model metadata");
                }
                std::string model_config_str;
                tc_err = triton_client.http_client_->ModelConfig(
                    &model_config_str, client_config.model_name, client_config.model_version, client_config.http_headers);
                if (!tc_err.IsOk())
                {
                    return cps_utils::Error(cps_utils::Error::Code::INTERNAL, "Failed to get triton model config");
                }

                tc_err = tc::ParseJson(&model_config_json, model_config_str);
                if (!tc_err.IsOk())
                {
                    return cps_utils::Error(cps_utils::Error::Code::INTERNAL, "Failed to parse triton model config");
                }
                if (!ParseModelHttp(
                        model_metadata_json, model_config_json, this->batch_size, &model_config))
                {
                    return cps_utils::Error(cps_utils::Error::Code::INTERNAL, "Failed to parse model configuration and metadata");
                }
            }
            else
            {
                inference::ModelMetadataResponse model_metadata_response;
                tc_err = triton_client.grpc_client_->ModelMetadata(
                    &model_metadata_response, client_config.model_name, client_config.model_version, client_config.http_headers);
                if (!tc_err.IsOk())
                {
                    return cps_utils::Error(cps_utils::Error::Code::INTERNAL, "Failed to get triton model metadata");
                }
                inference::ModelConfigResponse model_config_response;
                tc_err = triton_client.grpc_client_->ModelConfig(
                    &model_config_response, client_config.model_name, client_config.model_version, client_config.http_headers);
                if (!tc_err.IsOk())
                {
                    return cps_utils::Error(cps_utils::Error::Code::INTERNAL, "Failed to get triton model config");
                }
                if (!ParseModelGrpc(model_metadata_response, model_config_response, this->batch_size, &model_config))
                {
                    return cps_utils::Error(cps_utils::Error::Code::INTERNAL, "Failed to parse model configuration and metadata");
                }
            }
            return cps_utils::Error::Success;
        }

        template <typename T>
        cps_utils::Error TritonEngine<T>::initializeMemory()
        {
            // Initialize the inputs with the data.
            tc::InferInput *input;
            tc::Error tc_err;
            tc_err = tc::InferInput::Create(
                &input, model_config.input_name_, model_config.input_shape_, model_config.input_datatype_);
            if (!tc_err.IsOk())
            {
                return cps_utils::Error(cps_utils::Error::Code::INTERNAL, "Unable to get TritonClient::Input input");
            }
            input_ptr.reset(input);

            tc::InferRequestedOutput *output;
            // Set the number of classification expected
            tc_err =
                tc::InferRequestedOutput::Create(&output, model_config.output_name_);
            if (!tc_err.IsOk())
            {
                return cps_utils::Error(cps_utils::Error::Code::INTERNAL, "Unable to get TritonClient::Output output");
            }
            output_ptr.reset(output);

            // The inference settings. Will be using default for now.
            infer_options.model_name_ = client_config.model_name;
            infer_options.model_version_ = client_config.model_version;

            infer_inputs.push_back(input_ptr.get());   //! Moving raw pointer
            infer_outputs.push_back(output_ptr.get()); //! Moving raw pointer
            return cps_utils::Error::Success;
        }

        template <typename T>
        cps_utils::Error TritonEngine<T>::validate(const std::vector<cps_utils::InferenceData<uint8_t>> &data)
        {
            if (data.size() != this->batch_size)
            {
                return cps_utils::Error(cps_utils::Error::Code::VALIDATION_ERROR, "Number of data is different from batch size.");
            }
            size_t data_byte_size = 0;
            for (const cps_utils::InferenceData<uint8_t> &d : data)
            {
                data_byte_size += sizeof(uint8_t) * d.data.size();
            }
            if (data_byte_size > model_config.input_byte_size_)
            {
                return cps_utils::Error(cps_utils::Error::Code::VALIDATION_ERROR, "Total data bytesize is different from allocated bytesize.");
            }

            return cps_utils::Error::Success;
        }

        template <typename T>
        cps_utils::Error TritonEngine<T>::postprocess(const std::unique_ptr<tc::InferResult> &result, cps_utils::InferenceResult<T> &res, const size_t &batch_size, const std::string &output_name)
        {
            if (!result->RequestStatus().IsOk())
            {
                return cps_utils::Error(cps_utils::Error::Code::INFERENCE_ERROR, result->RequestStatus().Message());
            }

            // Get and validate the shape and datatype
            tc::Error err = result->Shape(output_name, &res.shape);
            if (!err.IsOk())
            {
                return cps_utils::Error(cps_utils::Error::Code::INTERNAL, "Unable to get shape for output " + output_name);
            }

            err = result->Datatype(output_name, &res.data_dtype);
            if (!err.IsOk())
            {
                return cps_utils::Error(cps_utils::Error::Code::INTERNAL, "Unable to get datatype for output " + output_name);
            }

            std::shared_ptr<uint8_t> buf_ptr(new uint8_t);
            err = result->RawData(output_name, (const uint8_t **)&buf_ptr, &res.byte_size);
            if (!err.IsOk())
            {
                return cps_utils::Error(cps_utils::Error::Code::INTERNAL, "Unable to get data for output " + output_name);
            }
            std::vector<uint8_t> uint8_blob_= std::vector<uint8_t>(buf_ptr.get(), buf_ptr.get() + res.byte_size);
            res.data = cpp_server::utils::blob_to_vectorT<T>(uint8_blob_);

            return cps_utils::Error::Success;
        }

        template <typename T>
        cps_utils::Error TritonEngine<T>::process(const std::vector<cps_utils::InferenceData<T>> &infer_data, std::vector<cps_utils::InferenceResult<T>> &infer_results)
        {
            std::vector<cps_utils::InferenceData<uint8_t>> input_uint8_;
            for(auto &T_data: infer_data)
            {
                cpp_server::utils::InferenceData<uint8_t> tdata_;
                std::vector<uint8_t> array_uint8 = cpp_server::utils::vectorT_to_blob<T>(T_data.data);
                tdata_.data = array_uint8;

                tdata_.name = T_data.name;
                tdata_.data_dtype = T_data.data_dtype;
                tdata_.shape = T_data.shape;
                input_uint8_.push_back(tdata_);
            }

            cps_utils::Error p_err;
            p_err = validate(input_uint8_);
            if (!p_err.IsOk())
            {
                return p_err;
            }

            size_t data_idx = 0;
            size_t done_cnt = 0;
            size_t sent_count = 0;
            bool last_request = false;
            tc::Error tc_err;
            std::vector<std::unique_ptr<tc::InferResult>> results;

            while (!last_request)
            {
                // Reset the input for new request.
                tc_err = input_ptr->Reset();
                if (!tc_err.IsOk())
                {
                    return cps_utils::Error(cps_utils::Error::Code::INFERENCE_ERROR, "Failed resetting input ptr");
                }

                // Set input to be the next 'batch_size' images (preprocessed).
                for (int idx = 0; idx < this->batch_size; ++idx)
                {
                    tc_err = input_ptr->AppendRaw(input_uint8_[data_idx].data);
                    if (!tc_err.IsOk())
                    {
                        return cps_utils::Error(cps_utils::Error::Code::INFERENCE_ERROR, "Failed setting input");
                    }

                    data_idx = (data_idx + 1) % input_uint8_.size();
                    if (data_idx == 0)
                    {
                        last_request = true;
                    }
                }

                infer_options.request_id_ = std::to_string(sent_count);

                tc::InferResult *result;
                if (client_config.protocol == ProtocolType::HTTP)
                {
                    tc_err = triton_client.http_client_->Infer(
                        &result, infer_options, infer_inputs, infer_outputs, client_config.http_headers);
                }
                else
                {
                    tc_err = triton_client.grpc_client_->Infer(
                        &result, infer_options, infer_inputs, infer_outputs, client_config.http_headers);
                }
                if (!tc_err.IsOk())
                {
                    return cps_utils::Error(cps_utils::Error::Code::INFERENCE_ERROR, "Failed sending synchronous infer request");
                }
                results.emplace_back(std::move(std::unique_ptr<tc::InferResult>(result)));
                sent_count++;
            }

            // Post-process the results to make prediction(s)
            for (size_t idx = 0; idx < results.size(); idx++)
            {
                cps_utils::InferenceResult<T> output_data;
                cps_utils::Error p_err;
                try
                {
                    p_err = postprocess(results[idx], output_data, this->batch_size, model_config.output_name_);
                    if (!p_err.IsOk())
                    {
                        return p_err;
                    }
                    infer_results.push_back(output_data);
                }
                catch (std::exception &e)
                {
                    std::cout << e.what() << std::endl;
                    return cps_utils::Error(cps_utils::Error::Code::INTERNAL, "Unable to run postprocessing for " + model_config.output_name_);
                }
            }
            return cps_utils::Error::Success;
        }
    };
};

//  https://isocpp.org/wiki/faq/templates#separate-template-class-defn-from-decl
//  * NOTE: Solve template function linker problem

template class cpp_server::inferencer::TritonEngine<float>;
