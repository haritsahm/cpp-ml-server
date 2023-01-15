#ifndef PROCESSOR_HPP
#define PROCESSOR_HPP

#include <string>
#include <memory>
#include <rapidjson/document.h>
#include "utils/error.hpp"
#include "inference_engine.hpp"

namespace cps_utils = cpp_server::utils;
namespace cps_inferencer = cpp_server::inferencer;

namespace cpp_server
{
    namespace processor
    {
        /// @brief Abstract base class for processing function.
        class Processor
        {
        public:
            Processor() = default;
            virtual ~Processor(){};

            Processor(const Processor &processor) = delete;
            Processor &operator=(const Processor &processor);

            Processor(Processor &&processor) = delete;
            Processor &operator=(Processor &&processor);

            /// @brief Abstract function to process incoming data and update output data.
            /// @param data_doc Input data stored as JSON format.
            /// @param result_doc Output data stored as JSON format.
            /// @return Error code to validate process.
            virtual cps_utils::Error process(const rapidjson::Document &data_doc, rapidjson::Document &result_doc) = 0;

        protected:
            /// @brief Pointer to inference engine.
            std::unique_ptr<cps_inferencer::InferenceEngine> infer_engine;
        };
    };
};

#endif