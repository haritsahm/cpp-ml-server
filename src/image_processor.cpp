#include "cpp_server/image_processor.hpp"

namespace cpp_server
{
    namespace processor
    {
        ImageProcessor::ImageProcessor(std::unique_ptr<cps_inferencer::InferenceEngine> &engine)
        {
            assignEngine(engine);
        }

        void ImageProcessor::apply_softmax(std::vector<float> &input)
        {
            // Source: https://codereview.stackexchange.com/a/180506
            float m, sum;

            m = -INFINITY;
            for (int i = 0; i < input.size(); ++i)
            {
                if (m < input[i])
                {
                    m = input[i];
                }
            }

            sum = 0.0;
            for (int i = 0; i < input.size(); ++i)
            {
                sum += exp(input[i] - m);
            }

            const float constant = m + log(sum);
            for (int i = 0; i < input.size(); ++i)
            {
                input[i] = exp(input[i] - constant);
            }
        }

        cpp_server::utils::Error ImageProcessor::preprocess_data(const std::string &ss, std::vector<float> &output)
        {
            std::string decoded_string = cpp_server::utils::base64_decode(ss);
            std::vector<uchar> data(decoded_string.begin(), decoded_string.end());
            std::vector<int> network_shape;
            if (infer_engine->modelConfig().input_shape_.size() > 2)
                network_shape = std::vector<int>{
                    infer_engine->modelConfig().input_shape_.end() - std::min<int>(infer_engine->modelConfig().input_shape_.size(), 2),
                    infer_engine->modelConfig().input_shape_.end()};
            else
            {
                network_shape = std::vector<int>{384, 384};
            }
            cv::Mat image = cv::imdecode(data, cv::IMREAD_UNCHANGED);
            if (image.data == NULL)
                return cpp_server::utils::Error(cpp_server::utils::Error::Code::INVALID_DATA, "Invalid image data");
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
            try
            {
                cv::resize(image, image, cv::Size(network_shape[0], network_shape[1]), cv::INTER_CUBIC);
            }
            catch (cv::Exception &e)
            {
                return cpp_server::utils::Error(cpp_server::utils::Error::Code::INVALID_DATA, "Input image is smaller than required output");
            }
            // TODO: Fix Image normalization with Imagenet std and mean.
            image.convertTo(image, CV_32FC3, 1.f / 255);
            // cv::subtract(cv::Scalar(0.485, 0.456, 0.406), image, image);
            // cv::divide(0.226, image, image); // divide by average std per channel
            output.resize(image.channels() * image.rows * image.cols);

            // Store image to float as CHW
            for (int y = 0; y < image.rows; ++y)
            {
                for (int x = 0; x < image.cols; ++x)
                {
                    for (int c = 0; c < image.channels(); ++c)
                    {
                        output[c * (image.rows * image.cols) + y * image.cols + x] =
                            image.at<cv::Vec3f>(y, x)[c];
                    }
                }
            }

            return cpp_server::utils::Error::Success;
        }

        cpp_server::utils::Error ImageProcessor::postprocess_classifaction(const std::vector<cpp_server::utils::InferenceResult<uint8_t>> &infer_results, std::vector<cpp_server::utils::ClassificationResult> &output)
        {
            for (const cpp_server::utils::InferenceResult<uint8_t> &result : infer_results)
            {
                std::vector<float> data_float = cpp_server::utils::blob_to_vectorT<float>(result.data);
                int row_num = result.shape[0], col_num = result.shape[1];
                cpp_server::utils::ClassificationResult output_data;
                for (int i = 0; i < row_num; i++)
                {
                    int start_range = i * col_num, end_range = (i + 1) * col_num;
                    std::vector<float> batch_output((data_float.begin() + start_range), (data_float.begin() + end_range));
                    apply_softmax(batch_output);
                    int maxElementIndex = std::max_element(batch_output.begin(), batch_output.end()) - batch_output.begin();
                    float maxElement = *std::max_element(batch_output.begin(), batch_output.end());
                    output_data.class_idx = maxElementIndex + 1; // zero index
                    output_data.score = maxElement;
                    output_data.name = "temp";
                }
                output.push_back(output_data);
            }
            return cpp_server::utils::Error::Success;
        }

        cpp_server::utils::Error ImageProcessor::process(const rapidjson::Document &data_doc, rapidjson::Document &result_doc)
        {
            cv::Mat preprocessed;
            std::vector<float> array_float;
            cpp_server::utils::Error p_err;
            p_err = preprocess_data(data_doc["image"].GetString(), array_float);
            if (!p_err.IsOk())
            {
                return p_err;
            }
            std::vector<uint8_t> array_uint8 = cpp_server::utils::vectorT_to_blob<float>(array_float);

            if (!infer_engine || !infer_engine->isOk())
            {
                return cpp_server::utils::Error(cpp_server::utils::Error::Code::INTERNAL, "Can't intialize inference system");
            }

            cpp_server::utils::InferenceData<uint8_t> input_data;
            input_data.data = array_uint8;
            input_data.name = "input";
            input_data.data_dtype = "FP32";
            input_data.shape = {preprocessed.size[0], preprocessed.size[1], preprocessed.size[2]};

            std::vector<cpp_server::utils::InferenceData<uint8_t>> inference_datas;
            inference_datas.push_back(input_data);
            std::vector<cpp_server::utils::InferenceResult<uint8_t>> inference_results;

            p_err = infer_engine->process(inference_datas, inference_results);
            if (!p_err.IsOk())
            {
                return p_err;
            }

            std::vector<cpp_server::utils::ClassificationResult> classification_output;
            p_err = postprocess_classifaction(inference_results, classification_output);
            if (!p_err.IsOk())
            {
                return p_err;
            }

            result_doc.Parse("{\"results\":[]}");
            for (cpp_server::utils::ClassificationResult &output : classification_output)
            {
                rapidjson::Value obj(rapidjson::kObjectType);
                obj.AddMember("score", output.score, result_doc.GetAllocator());
                obj.AddMember("class", output.class_idx, result_doc.GetAllocator());
                rapidjson::SetValueByPointer(result_doc, "/results/-", obj);
            }

            return cpp_server::utils::Error::Success;
        }
    };
};
