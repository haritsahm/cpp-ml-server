#ifndef COMMON_HPP
#define COMMON_HPP

#include <vector>

namespace cpp_server
{
    struct ModelConfig
    {
        std::string input_name_{"input"};
        std::string output_name_{"output"};
        std::string input_datatype_{"FP32"};
        std::string output_datatype_{"FP32"};
        std::string input_format_{"FORMAT_NCHW"};
        std::vector<int64_t> input_shape_;
        std::vector<int64_t> output_shape_;
        int input_byte_size_{};
        int output_byte_size_{};
        int max_batch_size_{0};
        bool channel_first_{true};
    };

    template <typename T>
    struct InferenceData
    {
        std::vector<T> data;
        std::string name;
        std::string data_dtype;
        std::vector<int> shape;
    };

    template <typename T>
    struct InferenceResult
    {
        std::vector<T> data;
        std::string data_dtype;
        std::vector<int64_t> shape;
        std::string name;
        size_t byte_size;
        bool status;
    };

    template <typename T>
    static std::vector<unsigned char> vectorT_to_blob(std::vector<T> &dataT)
    {
        const unsigned char *bytes = reinterpret_cast<const unsigned char *>(&dataT[0]);
        std::vector<unsigned char> result(bytes, bytes + sizeof(T) * dataT.size());
        return result;
    }

};

#endif