#ifndef ONNXRT_HELPER_HPP
#define ONNXRT_HELPER_HPP

#include <memory>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include "cpp_server/utils/common.hpp"
#include "cpp_server/utils/error.hpp"
#include <fstream>

namespace cps_utils = cpp_server::utils;

namespace cpp_server
{
    namespace inferencer
    {
        class ORTRunner
        {
        public:
            ORTRunner() = default;
            ORTRunner(const std::string &model_path, const std::vector<cps_utils::ModelConfig> &configs);
            ~ORTRunner(){};

            cps_utils::Error process(const std::vector<Ort::Value> &input_tensors, std::vector<Ort::Value> &output_tensors);

        private:
            std::unique_ptr<Ort::Session> session_;
            Ort::Env env_;
            Ort::RunOptions run_options_;

            std::vector<const char*> input_node_names, output_node_names;
            std::vector<cps_utils::ModelConfig> model_configs;

        };
    }
}


#endif