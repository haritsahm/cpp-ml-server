#include "cpp_server/onnxrt_helper.hpp"

namespace cpp_server
{
    namespace inferencer
    {
        ORTRunner::ORTRunner(const std::string &model_path)
        {
            session_.reset(new Ort::Session(env_, model_path.c_str(), Ort::SessionOptions{nullptr}));
            model_configs = configs;

            for(cps_utils::ModelConfig &conf: model_configs)
            {
                input_node_names.push_back(conf.input_name_.c_str());
                output_node_names.push_back(conf.output_name_.c_str());
            }
        }

        cps_utils::Error ORTRunner::process(const std::vector<Ort::Value> &input_tensors, std::vector<Ort::Value> &output_tensors)
        {
            if (!session_)
            {
                return cps_utils::Error(cps_utils::Error::Code::INTERNAL, "ONNXRT session is not initialized");
            }

            try
            {
                session_->Run(run_options_, input_node_names.data(), input_tensors.data(), input_node_names.size(), output_node_names.data(), output_tensors.data(), output_node_names.size());
                return cps_utils::Error::Success;
            }
            catch (const std::exception& ex)
            {
                return cps_utils::Error(cps_utils::Error::Code::INFERENCE_ERROR, ex.what());
            }
        }
    }
} // namespace cpp_server
