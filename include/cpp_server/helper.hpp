#ifndef HELPER_HPP
#define HELPER_HPP

#include <string>
#include <opencv2/core.hpp>

static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

static inline bool is_base64(unsigned char c)
{
    return (isalnum(c) || (c == '+') || (c == '/'));
}

std::string base64_decode(std::string const &encoded_string)
{
    int in_len = encoded_string.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::string ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_]))
    {
        char_array_4[i++] = encoded_string[in_];
        in_++;
        if (i == 4)
        {
            for (i = 0; i < 4; i++)
                char_array_4[i] = base64_chars.find(char_array_4[i]);

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; (i < 3); i++)
                ret += char_array_3[i];
            i = 0;
        }
    }

    if (i)
    {
        for (j = i; j < 4; j++)
            char_array_4[j] = 0;

        for (j = 0; j < 4; j++)
            char_array_4[j] = base64_chars.find(char_array_4[j]);

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; (j < i - 1); j++)
            ret += char_array_3[j];
    }

    return ret;
}

bool ParseType(const std::string &dtype, int *type1, int *type3)
{
    if (dtype.compare("UINT8") == 0)
    {
        *type1 = cv::CV_8UC1;
        *type3 = cv::CV_8UC3;
    }
    else if (dtype.compare("INT8") == 0)
    {
        *type1 = cv::CV_8SC1;
        *type3 = cv::CV_8SC3;
    }
    else if (dtype.compare("UINT16") == 0)
    {
        *type1 = cv::CV_16UC1;
        *type3 = cv::CV_16UC3;
    }
    else if (dtype.compare("INT16") == 0)
    {
        *type1 = cv::CV_16SC1;
        *type3 = cv::CV_16SC3;
    }
    else if (dtype.compare("INT32") == 0)
    {
        *type1 = cv::CV_32SC1;
        *type3 = cv::CV_32SC3;
    }
    else if (dtype.compare("FP32") == 0)
    {
        *type1 = cv::CV_32FC1;
        *type3 = cv::CV_32FC3;
    }
    else if (dtype.compare("FP64") == 0)
    {
        *type1 = cv::CV_64FC1;
        *type3 = cv::CV_64FC3;
    }
    else
    {
        return false;
    }

    return true;
}

void ParseModelGrpc(
    const inference::ModelMetadataResponse &model_metadata,
    const inference::ModelConfigResponse &model_config, const size_t batch_size,
    ModelInfo *model_info)
{
    if (model_metadata.inputs().size() != 1)
    {
        std::cerr << "expecting 1 input, got " << model_metadata.inputs().size()
                  << std::endl;
        exit(1);
    }

    if (model_metadata.outputs().size() != 1)
    {
        std::cerr << "expecting 1 output, got " << model_metadata.outputs().size()
                  << std::endl;
        exit(1);
    }

    if (model_config.config().input().size() != 1)
    {
        std::cerr << "expecting 1 input in model configuration, got "
                  << model_config.config().input().size() << std::endl;
        exit(1);
    }

    auto input_metadata = model_metadata.inputs(0);
    auto input_config = model_config.config().input(0);
    auto output_metadata = model_metadata.outputs(0);

    if (output_metadata.datatype().compare("FP32") != 0)
    {
        std::cerr << "expecting output datatype to be FP32, model '"
                  << model_metadata.name() << "' output type is '"
                  << output_metadata.datatype() << "'" << std::endl;
        exit(1);
    }

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
            exit(1);
        }
    }
    else
    {
        //  model_info->max_batch_size_ > 0
        if (batch_size > (size_t)model_info->max_batch_size_)
        {
            std::cerr << "expecting batch size <= " << model_info->max_batch_size_
                      << " for model '" << model_metadata.name() << "'" << std::endl;
            exit(1);
        }
    }

    // Output is expected to be a vector. But allow any number of
    // dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    // }, { 10, 1, 1 } are all ok).
    bool output_batch_dim = (model_info->max_batch_size_ > 0);
    size_t non_one_cnt = 0;
    for (const auto dim : output_metadata.shape())
    {
        if (output_batch_dim)
        {
            output_batch_dim = false;
        }
        else if (dim == -1)
        {
            std::cerr << "variable-size dimension in model output not supported"
                      << std::endl;
            exit(1);
        }
        else if (dim > 1)
        {
            non_one_cnt += 1;
            if (non_one_cnt > 1)
            {
                std::cerr << "expecting model output to be a vector" << std::endl;
                exit(1);
            }
        }
    }

    // Model input must have 3 dims, either CHW or HWC (not counting the
    // batch dimension), either CHW or HWC
    const bool input_batch_dim = (model_info->max_batch_size_ > 0);
    const int expected_input_dims = 3 + (input_batch_dim ? 1 : 0);
    if (input_metadata.shape().size() != expected_input_dims)
    {
        std::cerr << "expecting input to have " << expected_input_dims
                  << " dimensions, model '" << model_metadata.name()
                  << "' input has " << input_metadata.shape().size() << std::endl;
        exit(1);
    }

    if ((input_config.format() != inference::ModelInput::FORMAT_NCHW) &&
        (input_config.format() != inference::ModelInput::FORMAT_NHWC))
    {
        std::cerr
            << "unexpected input format "
            << inference::ModelInput_Format_Name(input_config.format())
            << ", expecting "
            << inference::ModelInput_Format_Name(inference::ModelInput::FORMAT_NHWC)
            << " or "
            << inference::ModelInput_Format_Name(inference::ModelInput::FORMAT_NCHW)
            << std::endl;
        exit(1);
    }

    model_info->output_name_ = output_metadata.name();
    model_info->input_name_ = input_metadata.name();
    model_info->input_datatype_ = input_metadata.datatype();

    if (input_config.format() == inference::ModelInput::FORMAT_NHWC)
    {
        model_info->input_format_ = "FORMAT_NHWC";
        model_info->input_h_ = input_metadata.shape(input_batch_dim ? 1 : 0);
        model_info->input_w_ = input_metadata.shape(input_batch_dim ? 2 : 1);
        model_info->input_c_ = input_metadata.shape(input_batch_dim ? 3 : 2);
    }
    else
    {
        model_info->input_format_ = "FORMAT_NCHW";
        model_info->input_c_ = input_metadata.shape(input_batch_dim ? 1 : 0);
        model_info->input_h_ = input_metadata.shape(input_batch_dim ? 2 : 1);
        model_info->input_w_ = input_metadata.shape(input_batch_dim ? 3 : 2);
    }

    if (!ParseType(
            model_info->input_datatype_, &(model_info->type1_),
            &(model_info->type3_)))
    {
        std::cerr << "unexpected input datatype '" << model_info->input_datatype_
                  << "' for model \"" << model_metadata.name() << std::endl;
        exit(1);
    }
}

void ParseModelHttp(
    const rapidjson::Document &model_metadata,
    const rapidjson::Document &model_config, const size_t batch_size,
    ModelInfo *model_info)
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
        exit(1);
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
        exit(1);
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
        exit(1);
    }

    const auto &input_metadata = *input_itr->value.Begin();
    const auto &input_config = *input_config_itr->value.Begin();
    const auto &output_metadata = *output_itr->value.Begin();

    const auto &output_dtype_itr = output_metadata.FindMember("datatype");
    if (output_dtype_itr == output_metadata.MemberEnd())
    {
        std::cerr << "output missing datatype in the metadata for model'"
                  << model_metadata["name"].GetString() << "'" << std::endl;
        exit(1);
    }
    auto datatype = std::string(
        output_dtype_itr->value.GetString(),
        output_dtype_itr->value.GetStringLength());
    if (datatype.compare("FP32") != 0)
    {
        std::cerr << "expecting output datatype to be FP32, model '"
                  << model_metadata["name"].GetString() << "' output type is '"
                  << datatype << "'" << std::endl;
        exit(1);
    }

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
            exit(1);
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
            exit(1);
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
                exit(1);
            }
            else if (shape_json[i].GetInt() > 1)
            {
                non_one_cnt += 1;
                if (non_one_cnt > 1)
                {
                    std::cerr << "expecting model output to be a vector" << std::endl;
                    exit(1);
                }
            }
        }
    }
    else
    {
        std::cerr << "output missing shape in the metadata for model'"
                  << model_metadata["name"].GetString() << "'" << std::endl;
        exit(1);
    }

    // Model input must have 3 dims, either CHW or HWC (not counting the
    // batch dimension), either CHW or HWC
    const bool input_batch_dim = (max_batch_size > 0);
    const size_t expected_input_dims = 3 + (input_batch_dim ? 1 : 0);
    const auto input_shape_itr = input_metadata.FindMember("shape");
    if (input_shape_itr != input_metadata.MemberEnd())
    {
        if (input_shape_itr->value.Size() != expected_input_dims)
        {
            std::cerr << "expecting input to have " << expected_input_dims
                      << " dimensions, model '" << model_metadata["name"].GetString()
                      << "' input has " << input_shape_itr->value.Size() << std::endl;
            exit(1);
        }
    }
    else
    {
        std::cerr << "input missing shape in the metadata for model'"
                  << model_metadata["name"].GetString() << "'" << std::endl;
        exit(1);
    }

    model_info->input_format_ = std::string(
        input_config["format"].GetString(),
        input_config["format"].GetStringLength());
    if ((model_info->input_format_.compare("FORMAT_NCHW") != 0) &&
        (model_info->input_format_.compare("FORMAT_NHWC") != 0))
    {
        std::cerr << "unexpected input format " << model_info->input_format_
                  << ", expecting FORMAT_NCHW or FORMAT_NHWC" << std::endl;
        exit(1);
    }

    model_info->output_name_ = std::string(
        output_metadata["name"].GetString(),
        output_metadata["name"].GetStringLength());
    model_info->input_name_ = std::string(
        input_metadata["name"].GetString(),
        input_metadata["name"].GetStringLength());
    model_info->input_datatype_ = std::string(
        input_metadata["datatype"].GetString(),
        input_metadata["datatype"].GetStringLength());

    if (model_info->input_format_.compare("FORMAT_NHWC") == 0)
    {
        model_info->input_h_ =
            input_shape_itr->value[input_batch_dim ? 1 : 0].GetInt();
        model_info->input_w_ =
            input_shape_itr->value[input_batch_dim ? 2 : 1].GetInt();
        model_info->input_c_ =
            input_shape_itr->value[input_batch_dim ? 3 : 2].GetInt();
    }
    else
    {
        model_info->input_c_ =
            input_shape_itr->value[input_batch_dim ? 1 : 0].GetInt();
        model_info->input_h_ =
            input_shape_itr->value[input_batch_dim ? 2 : 1].GetInt();
        model_info->input_w_ =
            input_shape_itr->value[input_batch_dim ? 3 : 2].GetInt();
    }

    if (!ParseType(
            model_info->input_datatype_, &(model_info->type1_),
            &(model_info->type3_)))
    {
        std::cerr << "unexpected input datatype '" << model_info->input_datatype_
                  << "' for model \"" << model_metadata["name"].GetString()
                  << std::endl;
        exit(1);
    }
}

#endif