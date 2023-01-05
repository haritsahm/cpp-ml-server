#ifndef IMAGE_PRCOESSOR_HPP
#define IMAGE_PRCOESSOR_HPP

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rapidjson/document.h>
#include <rapidjson/pointer.h>
#include "common.hpp"
#include "base64_decoder.hpp"
#include "processor.hpp"
#include "error.hpp"

struct ImageModelInfo : cpp_server::ModelConfig
{
    // The shape of the input
    int input_c_{3};
    int input_h_{384};
    int input_w_{384};

};

class ImageProcessor : public Processor
{
public:
    ImageProcessor() = default;
    ~ImageProcessor(){};

    ImageProcessor(const ImageProcessor &base) = delete;
    ImageProcessor &operator=(const ImageProcessor &server);

    ImageProcessor(ImageProcessor &&server) = default;
    ImageProcessor &operator=(ImageProcessor &&server);

    cpp_server::Error process(const rapidjson::Document &data, rapidjson::Document &result);

private:
    cpp_server::Error preprocess_data(const std::string &ss, cv::Mat &output);
};

#endif