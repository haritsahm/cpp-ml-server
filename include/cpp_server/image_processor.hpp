#ifndef IMAGE_PRCOESSOR_HPP
#define IMAGE_PRCOESSOR_HPP

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "processor.hpp"
#include "helper.hpp"
#include "common.hpp"

struct ImageModelInfo : ModelConfig
{
    // The shape of the input
    int input_c_{3};
    int input_h_{384};
    int input_w_{384};

    // The format of the input
    int type1_{cv::8UC1};
    int type3_{cv::8UC3};

}

class ImageProcessor : public Processor
{
public:
    ImageProcessor() = default;
    ImageProcessor(ClientConfig &config)
        : Processor(config){};
    ~ImageProcessor(){};

    ImageProcessor(const ImageProcessor &base) = delete;
    ImageProcessor &operator=(const ImageProcessor &server);

    ImageProcessor(ImageProcessor &&server) = delete;
    ImageProcessor &operator=(ImageProcessor &&server);

    InferenceResponse process(const std::string &ss);

private:
    cv::Mat preprocess_data(const std::string &ss);
};

#endif