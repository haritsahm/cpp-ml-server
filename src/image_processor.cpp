#include "cpp_server/image_processor.hpp"

ImageProcessor::ImageProcessor(const cpp_server::ModelConfig &model_config, const ClientConfig &client_config, const int &batch_size)
{
    this->model_config = model_config;
    infer_engine.reset(new cpp_server::TritonEngine(model_config, client_config, batch_size));
}

void ImageProcessor::CHW2HWC(const cv::InputArray &src, cv::OutputArray &dst)
{
    const int src_h = src.rows();
    const int src_w = src.cols();
    const int src_c = src.channels();

    cv::Mat hw_c = src.getMat().reshape(1, src_h * src_w);

    const std::array<int, 3> dims = {src_c, src_h, src_w};
    dst.create(3, &dims[0], CV_MAKETYPE(src.depth(), 1));
    cv::Mat dst_1d = dst.getMat().reshape(1, {src_c, src_h, src_w});

    cv::transpose(hw_c, dst_1d);
}

cpp_server::Error ImageProcessor::preprocess_data(const std::string &ss, cv::Mat &output)
{
    std::string decoded_string = base64_decode(ss);
    std::vector<uchar> data(decoded_string.begin(), decoded_string.end());
    std::vector<int> network_shape;
    if (model_config.input_shape_.size() > 2)
        network_shape = std::vector<int>{model_config.input_shape_.end() - std::min<int>(model_config.input_shape_.size(), 2), model_config.input_shape_.end()};
    else
    {
        network_shape = std::vector<int>{384, 384};
    }
    cv::Mat image = cv::imdecode(data, cv::IMREAD_UNCHANGED);
    if (image.data == NULL)
        return cpp_server::Error(cpp_server::Error::Code::INVALID_DATA, "Invalid image data");
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    try
    {
        cv::resize(image, image, cv::Size(network_shape[0], network_shape[1]), cv::INTER_CUBIC);
    }
    catch (cv::Exception &e)
    {
        return cpp_server::Error(cpp_server::Error::Code::INVALID_DATA, "Input image is smaller than required output");
    }
    image.convertTo(image, CV_32FC3);
    cv::subtract(cv::Scalar(0.485, 0.456, 0.406), image, image);
    cv::divide(0.226, image, image); // divide by average std per channel
    CHW2HWC(image, output);

    return cpp_server::Error::Success;
}

cpp_server::Error ImageProcessor::postprocess_classifaction(const std::vector<cpp_server::InferenceResult<uint8_t>> &infer_results, std::vector<cpp_server::ClassificationResult> &output)
{
}

cpp_server::Error ImageProcessor::process(const rapidjson::Document &data, rapidjson::Document &result)
{
    cv::Mat preprocessed;
    cpp_server::Error p_err;
    p_err = preprocess_data(data["image"].GetString(), preprocessed);
    if (!p_err.IsOk())
    {
        return p_err;
    }
    std::vector<float> array_float;
    if (preprocessed.isContinuous())
    {
        // array.assign((float*)mat.datastart, (float*)mat.dataend); // <- has problems for sub-matrix like mat = big_mat.row(i)
        array_float.assign((float *)preprocessed.data, (float *)preprocessed.data + preprocessed.total() * preprocessed.channels());
    }
    else
    {
        for (int i = 0; i < preprocessed.rows; ++i)
        {
            array_float.insert(array_float.end(), preprocessed.ptr<float>(i), preprocessed.ptr<float>(i) + preprocessed.cols * preprocessed.channels());
        }
    }
    std::vector<uint8_t> array_uint8 = cpp_server::vectorT_to_blob<float>(array_float);

    if (!infer_engine || !infer_engine->isOk())
    {
        return cpp_server::Error(cpp_server::Error::Code::INTERNAL, "Can't intialize inference system");
    }

    cpp_server::InferenceData<uint8_t> input_data;
    input_data.data = array_uint8;
    input_data.name = "input";
    input_data.data_dtype = "FP32";
    input_data.shape = {preprocessed.size[0], preprocessed.size[1], preprocessed.size[2]};

    std::vector<cpp_server::InferenceData<uint8_t>> inference_datas;
    inference_datas.push_back(input_data);
    std::vector<cpp_server::InferenceResult<uint8_t>> inference_results;

    // TODO: inference process
    p_err = infer_engine->process(inference_datas, inference_results);
    if (!p_err.IsOk())
    {
        return p_err;
    }

    // TODO: postprocess
    std::vector<cpp_server::ClassificationResult> classification_output;

    // TODO: return inference response
    rapidjson::Pointer("/class").Set(result, "car");
    rapidjson::Pointer("/score").Set(result, 0.001);

    return cpp_server::Error::Success;
}
