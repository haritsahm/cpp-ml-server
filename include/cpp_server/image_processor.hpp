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
#include "base/processor.hpp"
#include "base/inference_engine.hpp"
#include "utils/error.hpp"
#include "utils/common.hpp"
#include "utils/base64.hpp"

namespace cps_utils = cpp_server::utils;
namespace cps_inferencer = cpp_server::inferencer;

namespace cpp_server
{
    namespace processor
    {
        /// @brief Image processing class.
        class ImageProcessor : public Processor
        {
        public:
            ImageProcessor() = default;

            ImageProcessor(std::unique_ptr<cps_inferencer::InferenceEngine<float>> &engine);

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
            cps_utils::Error process(const rapidjson::Document &data_doc, rapidjson::Document &result_doc);

        private:
            /// @brief Pointer to inference engine.
            std::unique_ptr<cps_inferencer::InferenceEngine<float>> infer_engine;

            /// @brief Store model configuration from inference engine
            cps_utils::ModelConfig model_config;

            /// @brief Preprocess incoming data by converting string to vector data.
            /// @param ss Input data as string, encoded as base64.
            /// @param output Processed output data as vector<float>.
            /// @return Error code to validate process.
            cps_utils::Error preprocess_data(const std::string &ss, std::vector<float> &output);

            /// @brief Postprocess raw inference result data into meaningful classification data.
            /// @param infer_results Vector of inference results, especially if processed in batches.
            /// @param output Vector to store output classification data.
            /// @return Error code to validate process.
            cps_utils::Error postprocess_classifaction(const std::vector<cps_utils::InferenceResult<float>> &infer_results, std::vector<cps_utils::ClassificationResult> &output);

            /// @brief Apply softmax to raw logits data and modify inplace.
            /// @param input vector of logits.
            void apply_softmax(std::vector<float> &input);
        };
    }
}

#endif