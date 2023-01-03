#ifndef HELPER_HPP
#define HELPER_HPP

#include <string>
#include <vector>
#include <grpc_client.h>
#include <http_client.h>
#include "json_utils.h"
#include <rapidjson/document.h>
#include <triton/common/model_config.h>
#include <opencv2/core.hpp>
#include "common.hpp"

namespace tc = triton::client;

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

enum ProtocolType
{
    HTTP = 0,
    GRPC = 1
};

struct ClientConfig
{
    std::string model_name;
    std::string model_version{""};
    std::string url{"localhost:8000"};
    ProtocolType protocol = ProtocolType::HTTP;
    tc::Headers http_headers;
    bool verbose;
};

bool ParseModelGrpc(
    const inference::ModelMetadataResponse &model_metadata,
    const inference::ModelConfigResponse &model_config, const size_t &batch_size,
    cpp_server::ModelConfig *model_info)
{
    if (model_metadata.inputs().size() != 1)
    {
        std::cerr << "expecting 1 input, got " << model_metadata.inputs().size()
                  << std::endl;
        return false;
    }

    if (model_metadata.outputs().size() != 1)
    {
        std::cerr << "expecting 1 output, got " << model_metadata.outputs().size()
                  << std::endl;
        return false;
    }

    if (model_config.config().input().size() != 1)
    {
        std::cerr << "expecting 1 input in model configuration, got "
                  << model_config.config().input().size() << std::endl;
        return false;
    }

    auto input_metadata = model_metadata.inputs(0);
    auto input_config = model_config.config().input(0);
    auto output_cohnfig = model_config.config().output(0);
    auto output_metadata = model_metadata.outputs(0);

    model_info->max_batch_size_ = model_config.config().max_batch_size();

    // Model specifying maximum batch size of 0 indicates that batching
    // is not supported and so the input tensors do not expect a "N"
    // dimension (and 'batch_size' should be 1 so that only a single
    // image instance is inferred at a time).
    if (model_info->max_batch_size_ == 0)
    {
        if (batch_size != 1)
        {
            std::cerr << "batching not supported for model \""
                      << model_metadata.name() << "\"" << std::endl;
            return false;
        }
    }
    else
    {
        //  model_info->max_batch_size_ > 0
        if (batch_size > (size_t)model_info->max_batch_size_)
        {
            std::cerr << "expecting batch size <= " << model_info->max_batch_size_
                      << " for model '" << model_metadata.name() << "'" << std::endl;
            return false;
        }
    }

    // Output is expected to be a vector. But allow any number of
    // dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    // }, { 10, 1, 1 } are all ok).
    bool output_batch_dim = (model_info->max_batch_size_ > 0);
    size_t non_one_cnt = 0;
    for (const auto &dim : output_metadata.shape())
    {
        if (output_batch_dim)
        {
            output_batch_dim = false;
        }
        else if (dim == -1)
        {
            std::cerr << "variable-size dimension in model output not supported"
                      << std::endl;
            return false;
        }
        else if (dim > 1)
        {
            non_one_cnt += 1;
            if (non_one_cnt > 1)
            {
                std::cerr << "expecting model output to be a vector" << std::endl;
                return false;
            }
        }
    }

    if (model_info->max_batch_size_ > 0)
    {
        model_info->input_shape_.push_back(batch_size);
        model_info->output_shape_.push_back(batch_size);
    }
    model_info->output_shape_.insert(std::end(model_info->output_shape_), std::begin(output_metadata.shape()), std::end(output_metadata.shape()));
    model_info->input_shape_.insert(std::end(model_info->input_shape_), std::begin(input_metadata.shape()), std::end(input_metadata.shape()));

    model_info->input_format_ = inference::ModelInput_Format_Name(input_config.format());
    model_info->channel_first_ = inference::ModelInput::FORMAT_NCHW ? 1 : 0;
    model_info->input_name_ = input_metadata.name();
    model_info->input_datatype_ = input_metadata.datatype();
    model_info->output_name_ = output_metadata.name();
    model_info->output_datatype_ = output_metadata.datatype();
    auto input_datatype_ = triton::common::ProtocolStringToDataType(model_info->input_datatype_);
    auto output_datatype_ = triton::common::ProtocolStringToDataType(model_info->input_datatype_);
    std::vector<int64_t> input_shape_uint(model_info->input_shape_.begin(), model_info->input_shape_.end());
    std::vector<int64_t> output_shape_uint(model_info->output_shape_.begin(), model_info->output_shape_.end());
    model_info->input_byte_size_ = triton::common::GetByteSize(input_datatype_, input_shape_uint);
    model_info->output_byte_size_ = triton::common::GetByteSize(output_datatype_, output_shape_uint);
    return true;
}

bool ParseModelHttp(
    const rapidjson::Document &model_metadata,
    const rapidjson::Document &model_config, const size_t &batch_size,
    cpp_server::ModelConfig *model_info)
{
    const auto &input_itr = model_metadata.FindMember("inputs");
    size_t input_count = 0;
    if (input_itr != model_metadata.MemberEnd())
    {
        input_count = input_itr->value.Size();
    }
    if (input_count != 1)
    {
        std::cerr << "expecting 1 input, got " << input_count << std::endl;
        return false;
    }

    const auto &output_itr = model_metadata.FindMember("outputs");
    size_t output_count = 0;
    if (output_itr != model_metadata.MemberEnd())
    {
        output_count = output_itr->value.Size();
    }
    if (output_count != 1)
    {
        std::cerr << "expecting 1 output, got " << output_count << std::endl;
        return false;
    }

    const auto &input_config_itr = model_config.FindMember("input");
    input_count = 0;
    if (input_config_itr != model_config.MemberEnd())
    {
        input_count = input_config_itr->value.Size();
    }
    if (input_count != 1)
    {
        std::cerr << "expecting 1 input in model configuration, got " << input_count
                  << std::endl;
        return false;
    }

    const auto &output_config_itr = model_config.FindMember("output");
    size_t onput_count = 0;
    if (output_config_itr != model_config.MemberEnd())
    {
        onput_count = output_config_itr->value.Size();
    }
    if (onput_count != 1)
    {
        std::cerr << "expecting 1 output in model configuration, got " << onput_count
                  << std::endl;
        return false;
    }

    const auto &input_metadata = *input_itr->value.Begin();
    const auto &input_config = *input_config_itr->value.Begin();
    const auto &output_metadata = *output_itr->value.Begin();

    int max_batch_size = 0;
    const auto bs_itr = model_config.FindMember("max_batch_size");
    if (bs_itr != model_config.MemberEnd())
    {
        max_batch_size = bs_itr->value.GetUint();
    }
    model_info->max_batch_size_ = max_batch_size;

    // Model specifying maximum batch size of 0 indicates that batching
    // is not supported and so the input tensors do not expect a "N"
    // dimension (and 'batch_size' should be 1 so that only a single
    // image instance is inferred at a time).
    if (max_batch_size == 0)
    {
        if (batch_size != 1)
        {
            std::cerr << "batching not supported for model '"
                      << model_metadata["name"].GetString() << "'" << std::endl;
            return false;
        }
    }
    else
    {
        // max_batch_size > 0
        if (batch_size > (size_t)max_batch_size)
        {
            std::cerr << "expecting batch size <= " << max_batch_size
                      << " for model '" << model_metadata["name"].GetString() << "'"
                      << std::endl;
            return false;
        }
    }

    // Output is expected to be a vector. But allow any number of
    // dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    // }, { 10, 1, 1 } are all ok).
    bool output_batch_dim = (max_batch_size > 0);
    size_t non_one_cnt = 0;
    const auto output_shape_itr = output_metadata.FindMember("shape");
    if (output_shape_itr != output_metadata.MemberEnd())
    {
        const rapidjson::Value &shape_json = output_shape_itr->value;
        for (rapidjson::SizeType i = 0; i < shape_json.Size(); i++)
        {
            if (output_batch_dim)
            {
                output_batch_dim = false;
            }
            else if (shape_json[i].GetInt() == -1)
            {
                std::cerr << "variable-size dimension in model output not supported"
                          << std::endl;
                return false;
            }
            else if (shape_json[i].GetInt() > 1)
            {
                non_one_cnt += 1;
                if (non_one_cnt > 1)
                {
                    std::cerr << "expecting model output to be a vector" << std::endl;
                    return false;
                }
            }
        }
    }
    else
    {
        std::cerr << "output missing shape in the metadata for model'"
                  << model_metadata["name"].GetString() << "'" << std::endl;
        return false;
    }

    if (model_info->max_batch_size_ > 0)
    {
        model_info->input_shape_.push_back(batch_size);
        model_info->output_shape_.push_back(batch_size);
    }
    const auto input_shape_itr = input_metadata.FindMember("shape");
    if (input_shape_itr != input_metadata.MemberEnd())
    {
        const rapidjson::Value &shape_json = input_shape_itr->value;
        for (rapidjson::SizeType i = 0; i < shape_json.Size(); i++)
        {
            model_info->input_shape_.push_back(shape_json[i].GetInt());
        }
    }
    else
    {
        std::cerr << "input missing shape in the metadata for model'"
                  << model_metadata["shape"].GetString() << "'" << std::endl;
        return false;
    }

    if (output_shape_itr != output_metadata.MemberEnd())
    {
        const rapidjson::Value &shape_json = output_shape_itr->value;
        for (rapidjson::SizeType i = 0; i < shape_json.Size(); i++)
        {
            model_info->output_shape_.push_back(shape_json[i].GetInt());
        }
    }

    model_info->input_format_ = std::string(
        input_config["format"].GetString(),
        input_config["format"].GetStringLength());
    model_info->channel_first_ = model_info->input_format_.compare("FORMAT_NCHW") == 0 ? 1 : 0;
    model_info->output_name_ = std::string(
        output_metadata["name"].GetString(),
        output_metadata["name"].GetStringLength());
    model_info->input_name_ = std::string(
        input_metadata["name"].GetString(),
        input_metadata["name"].GetStringLength());
    model_info->input_datatype_ = std::string(
        input_metadata["datatype"].GetString(),
        input_metadata["datatype"].GetStringLength());
    model_info->output_datatype_ = std::string(
        output_metadata["datatype"].GetString(),
        output_metadata["datatype"].GetStringLength());
    auto input_datatype_ = triton::common::ProtocolStringToDataType(model_info->input_datatype_);
    auto output_datatype_ = triton::common::ProtocolStringToDataType(model_info->input_datatype_);
    std::vector<int64_t> input_shape_uint(model_info->input_shape_.begin(), model_info->input_shape_.end());
    std::vector<int64_t> output_shape_uint(model_info->output_shape_.begin(), model_info->output_shape_.end());
    model_info->input_byte_size_ = triton::common::GetByteSize(input_datatype_, input_shape_uint);
    model_info->output_byte_size_ = triton::common::GetByteSize(output_datatype_, output_shape_uint);

    return true;
}

#endif