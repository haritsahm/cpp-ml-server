#ifndef IMAGE_PRCOESSOR_HPP
#define IMAGE_PRCOESSOR_HPP

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rapidjson/document.h>
#include <rapidjson/pointer.h>
#include "triton_engine.hpp"
#include "triton_helper.hpp"
#include "processor.hpp"
#include "error.hpp"
#include "common.hpp"
#include "base64_decoder.hpp"

class ImageProcessor : public Processor
{
public:
    ImageProcessor() = default;
    ImageProcessor(const cpp_server::ModelConfig &model_config, const ClientConfig &client_config, const int &batch_size);
    ~ImageProcessor()
    {
        infer_engine.reset(nullptr);
    };

    ImageProcessor(const ImageProcessor &base) = delete;
    ImageProcessor &operator=(const ImageProcessor &server);

    ImageProcessor(ImageProcessor &&server) = delete;
    ImageProcessor &operator=(ImageProcessor &&server);

    cpp_server::Error process(const rapidjson::Document &data, rapidjson::Document &result);

private:
    cpp_server::ModelConfig model_config;
    cpp_server::Error preprocess_data(const std::string &ss, cv::Mat &output);
};

#endif