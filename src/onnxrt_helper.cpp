#include "cpp_server/onnxrt_helper.hpp"

namespace cpp_server
{
    namespace inferencer
    {
        ORTRunner::ORTRunner(const std::string &model_path)
        {
            session_.reset(new Ort::Session(env_, model_path.c_str(), Ort::SessionOptions{nullptr}));

            cps_utils::Error p_err;
            p_err = readModelConfigs();
            if (!p_err.IsOk())
            {
                session_.reset(nullptr);
            }

        }

        cps_utils::Error ORTRunner::readModelConfigs()
        {
            Ort::AllocatorWithDefaultOptions allocator;

            try
            {
                for(size_t i = 0; i < session_->GetInputCount(); ++i)
                {
                    cps_utils::ModelConfig config_;
                    auto inputNodeName = session_->GetInputNameAllocated(0, allocator);
                    config_.input_name_ = std::string(inputNodeName.get());

                    Ort::TypeInfo inputTypeInfo = session_->GetInputTypeInfo(0);
                    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
                    config_.input_shape_ = inputTensorInfo.GetShape();
                    config_.max_batch_size_ = config_.input_shape_[0];
                    config_.channel_first_ = config_.input_shape_[1] == 3 ? true : false;

                    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
                    config_.input_datatype_ = getONNXStrElementType(inputType);
                    config_.input_byte_size_ = cps_utils::vectorProduct(config_.input_shape_) * cps_utils::ElementStrTypeSize[config_.input_datatype_];

                    model_configs_.push_back(config_);
                    input_node_names_.push_back(model_configs_[i].input_name_.c_str());

                }
            }
            catch (std::exception &ex)
            {
                return cps_utils::Error(cps_utils::Error::Code::INTERNAL, "Unable to read model input metadata, " + std::string(ex.what()));
            }

            try
            {
                for(size_t i = 0; i < session_->GetOutputCount(); ++i)
                {
                    auto outputNodeName = session_->GetOutputNameAllocated(0, allocator);
                    model_configs_[i].output_name_ = std::string(outputNodeName.get());
                    output_node_names_.push_back(model_configs_[i].output_name_.c_str());

                    Ort::TypeInfo outputTypeInfo = session_->GetOutputTypeInfo(0);
                    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

                    model_configs_[i].output_shape_ = outputTensorInfo.GetShape();

                    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
                    model_configs_[i].output_datatype_ = getONNXStrElementType(outputType);
                    model_configs_[i].output_byte_size_ = cps_utils::vectorProduct(model_configs_[i].output_shape_) * cps_utils::ElementStrTypeSize[model_configs_[i].input_datatype_];
                }
            }
            catch (std::exception &ex)
            {
                return cps_utils::Error(cps_utils::Error::Code::INTERNAL, "Unable to read model output metadata, " + std::string(ex.what()));
            }

            return cps_utils::Error::Success;
        }

        cps_utils::Error ORTRunner::process(const std::vector<Ort::Value> &input_tensors, std::vector<Ort::Value> &output_tensors)
        {
            if (!session_)
            {
                return cps_utils::Error(cps_utils::Error::Code::INTERNAL, "ONNXRT session is not initialized");
            }

            try
            {
                session_->Run(run_options_, input_node_names_.data(), input_tensors.data(), input_node_names_.size(), output_node_names_.data(), output_tensors.data(), output_node_names_.size());
            }
            catch (const std::exception& ex)
            {
                return cps_utils::Error(cps_utils::Error::Code::INFERENCE_ERROR, std::string(ex.what()));
            }

            return cps_utils::Error::Success;
        }
    }
} // namespace cpp_server
