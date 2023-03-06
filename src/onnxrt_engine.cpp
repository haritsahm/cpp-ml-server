#include "cpp_server/onnxrt_engine.hpp"

namespace cpp_server
{
    namespace inferencer
    {
        template <typename T>
        ONNXRTEngine<T>::ONNXRTEngine(const std::string &model_path, const int &batch_size)
        {
            ort_runner.reset(new ORTRunner(model_path, model_config));
        }

        template <typename T>
        cps_utils::Error ONNXRTEngine<T>::validate(const std::vector<cps_utils::InferenceData<T>> &infer_data)
        {
            return cps_utils::Error::Success;
        }

        template <typename T>
        cps_utils::Error ONNXRTEngine<T>::process(const std::vector<cps_utils::InferenceData<T>> &infer_data, std::vector<cps_utils::InferenceResult<T>> &infer_results)
        {
            return cps_utils::Error::Success;

        }
    }
}

//  https://isocpp.org/wiki/faq/templates#separate-template-class-defn-from-decl
//  * NOTE: Solve template function linker problem

template class cpp_server::inferencer::ONNXRTEngine<float>;
