#ifndef ONNXRT_HELPER_HPP
#define ONNXRT_HELPER_HPP

#include <memory>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <fstream>


namespace cpp_server
{
    namespace inferencer
    {
        class ORTRunner
        {
        public:
            ORTRunner() = default;
            ~ORTRunner(){};

        private:
            std::unique_ptr<Ort::Session> session_;


        bool readModelConfigJSON(const std::string &path, rapidjson::Document doc);

        };
    }
}


#endif