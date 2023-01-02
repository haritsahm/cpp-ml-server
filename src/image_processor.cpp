#include "cpp_server/image_processor.hpp"

InferenceResponse ImageProcessor::process(const std::string &ss)
{
    cv::Mat processed_img = preprocess_data(ss);

    // TODO: inference process

    // TODO: postprocess

    // TODO: return inference response
}

cv::Mat ImageProcessor::preprocess_data(const std::string &ss)
{
    std::string decoded_string = base64_decode(ss);
    std::vector<uchar> data(decoded_string.begin(), decoded_string.end());

    cv::Mat image = cv::Imdecode(data, cv::IMREAD_UNCHANGED);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(384, 384), cv::INTER_CUBIC);
    Mat resized;
    image.convertTo(resized, cv::CV_32FC3);
    cv::substract(cv::Scalar(0.485, 0.456, 0.406), resized, resized);
    cv::divide(0.226, resized, resized); // divide by average std per channel

    return resized;
}