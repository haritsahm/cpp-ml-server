#include "cpp_server/onnxrt_engine.hpp"

namespace cpp_server
{
    namespace inferencer
    {
        template <typename T>
        ONNXRTEngine<T>::ONNXRTEngine(const std::string &model_path, const int &batch_size)
        {
            ort_runner.reset(new ORTRunner(model_path));
            if (ort_runner->isValid())
            {
                model_configs = ort_runner->getModelConfigs();
                this->model_config = model_configs[0];
                this->status = true;
            }
            else
            {
                this->status = false;
            }
        }

        template <typename T>
        cps_utils::Error ONNXRTEngine<T>::validate(const std::vector<cps_utils::InferenceData<T>> &infer_data)
        {
            if (infer_data.size() != this->batch_size)
            {
                return cps_utils::Error(cps_utils::Error::Code::VALIDATION_ERROR, "Number of data is different from batch size.");
            }

            for(size_t i = 0; i < infer_data.size(); ++i)
            {
                size_t data_byte_size = sizeof(T) * infer_data[i].data.size();

                if (data_byte_size > model_configs[i].input_byte_size_)
                {
                    return cps_utils::Error(cps_utils::Error::Code::VALIDATION_ERROR, "Total data bytesize is different from allocated bytesize.");
                }
            }
            return cps_utils::Error::Success;
        }

        template <typename T>
        cps_utils::Error ONNXRTEngine<T>::process(const std::vector<cps_utils::InferenceData<T>> &infer_data, std::vector<cps_utils::InferenceResult<T>> &infer_results)
        {
            cps_utils::Error p_err;
            p_err = validate(infer_data);
            if (!p_err.IsOk())
            {
                return p_err;
            }

            std::vector<Ort::Value> inputTensors_;
            std::vector<Ort::Value> outputTensors_;
            try
            {
                Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
                    OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

                for(size_t i = 0; i < infer_data.size(); ++i)
                {
                    inputTensors_.push_back(
                        Ort::Value::CreateTensor<T>(
                            memoryInfo,
                            const_cast<T*>(infer_data[i].data.data()),
                            cps_utils::vectorProduct(infer_data[i].shape)  * cps_utils::ElementStrTypeSize[infer_data[i].data_dtype],
                            infer_data[i].shape.data(),
                            infer_data[i].shape.size()
                        )
                    );
                }

                {
                    std::vector<T> output_data(cps_utils::vectorProduct(model_configs[0].output_shape_));
                    infer_results.push_back(
                        cps_utils::InferenceResult<T>{
                            output_data,
                            "FP32",
                            model_configs[0].output_shape_,
                            model_configs[0].output_name_,
                            static_cast<uint64_t>(model_configs[0].output_byte_size_),
                            false
                        }
                    );

                    // TODO: Find a way to initialize with multiple inputs/outputs
                    outputTensors_.push_back(
                        Ort::Value::CreateTensor<T>(
                        memoryInfo,
                        infer_results[0].data.data(),
                        infer_results[0].byte_size,
                        infer_results[0].shape.data(),
                        infer_results[0].shape.size()
                        )
                    );
                }
            }
            catch (const std::exception &ex)
            {
                return cpp_server::utils::Error(
                    cpp_server::utils::Error::Code::INTERNAL,
                    std::string("Failed to initialize OrtTensor: ") + std::string(ex.what())
                );
            }

            try
            {
                p_err = ort_runner->process(inputTensors_, outputTensors_);
                if (!p_err.IsOk())
                {
                    return p_err;
                }
                infer_results[0].status = true;
            }
            catch (const std::exception &ex)
            {
                return cpp_server::utils::Error(
                    cpp_server::utils::Error::Code::INTERNAL,
                    std::string("Failed to inference data: ") + std::string(ex.what())
                );
            }

            return cps_utils::Error::Success;

        }
    }
}

//  https://isocpp.org/wiki/faq/templates#separate-template-class-defn-from-decl
//  * NOTE: Solve template function linker problem

template class cpp_server::inferencer::ONNXRTEngine<float>;
