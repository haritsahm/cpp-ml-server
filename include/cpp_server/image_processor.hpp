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

/// @brief Image processing class.
class ImageProcessor : public Processor
{
public:
    ImageProcessor() = default;

    /// @brief Initialize class and use triton as its inference engine.
    /// @param client_config Triton client configuration.
    /// @param batch_size Batch size to process input data.
    ImageProcessor(const ClientConfig &client_config, const int &batch_size);
    ~ImageProcessor()
    {
        infer_engine.reset(nullptr);
    };

    ImageProcessor(const ImageProcessor &base) = delete;
    ImageProcessor &operator=(const ImageProcessor &server);

    ImageProcessor(ImageProcessor &&server) = delete;
    ImageProcessor &operator=(ImageProcessor &&server);

    /// @brief Function to process incoming data and update output data.
    /// @param data_doc Input data stored as JSON format.
    /// @param result_doc Output data stored as JSON format.
    /// @return Error code to validate process.
    cpp_server::Error process(const rapidjson::Document &data_doc, rapidjson::Document &result_doc);

private:
    /// @brief Store model configuration from inference engine
    cpp_server::ModelConfig model_config;

    /// @brief Preprocess incoming data by converting string to vector data.
    /// @param ss Input data as string, encoded as base64.
    /// @param output Processed output data as vector<float>.
    /// @return Error code to validate process.
    cpp_server::Error preprocess_data(const std::string &ss, std::vector<float> &output);

    /// @brief Postprocess raw inference result data into meaningful classification data.
    /// @param infer_results Vector of inference results, especially if processed in batches.
    /// @param output Vector to store output classification data.
    /// @return Error code to validate process.
    cpp_server::Error postprocess_classifaction(const std::vector<cpp_server::InferenceResult<uint8_t>> &infer_results, std::vector<cpp_server::ClassificationResult> &output);

    /// @brief Apply softmax to raw logits data and modify inplace.
    /// @param input vector of logits.
    void apply_softmax(std::vector<float> &input);
};

#endif