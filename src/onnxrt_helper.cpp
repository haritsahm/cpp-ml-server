#include "cpp_server/onnxrt_helper.hpp"

namespace cpp_server
{
    namespace inferencer
    {
        bool ORTRunner::readModelConfigJSON(const std::string &path, rapidjson::Document doc)
        {
            std::ifstream ifs(path);
            if (!ifs)
            {
                return false;
            }
            rapidjson::IStreamWrapper isw(ifs);

            return true;

        }
    }
} // namespace cpp_server
