#ifndef IMAGE_PRCOESSOR_HPP
#define IMAGE_PRCOESSOR_HPP

#include <string>
#include <vector>
#include <utility>
#include <exception>
#include <iostream>
#include <math.h>
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

    cpp_server::Error process(const rapidjson::Document &data_doc, rapidjson::Document &result_doc);

private:
    cpp_server::ModelConfig model_config;
    cpp_server::Error preprocess_data(const std::string &ss, std::vector<float> &output);
    cpp_server::Error postprocess_classifaction(const std::vector<cpp_server::InferenceResult<uint8_t>> &infer_results, std::vector<cpp_server::ClassificationResult> &output);
    void apply_softmax(std::vector<float> &input);
    void HWC2CHW(const cv::InputArray &src, cv::OutputArray &dst);
};

#endif