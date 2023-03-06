#ifndef COMMON_HELPER_HPP
#define COMMON_HELPER_HPP

#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

namespace cpp_server
{
    namespace utils
    {
        static std::map<std::string, size_t> ElementStrTypeSize {
            {"FP32", sizeof(float)},
            {"UINT8", sizeof(uint8_t)},
            {"INT8", sizeof(int8_t)},
            {"UINT16", sizeof(uint16_t)},
            {"INT16", sizeof(int16_t)},
            {"INT32", sizeof(int32_t)},
            {"INT64", sizeof(int64_t)},
            {"BOOL", sizeof(bool)},
            {"FP16", 2},
            {"FP64", sizeof(double)},
            {"UINT32", sizeof(uint32_t)},
            {"UINT64", sizeof(uint64_t)},
        };

        template <typename T>
        T vectorProduct(const std::vector<T>& v)
        {
            return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
        }

        // TODO: Replace with singular form node type
        /// @brief Struct to store model configuration from inference engine.
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

        /// @brief Struct to store inference input data to inference process.
        /// @tparam T Type of inferece data.
        template <typename T>
        struct InferenceData
        {
            std::vector<T> data;
            std::string name;
            std::string data_dtype;
            std::vector<int64_t> shape;
        };

        /// @brief Struct to store inference result data from inference process.
        /// @tparam T Type of inferece data.
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

        /// @brief Struct to store classification result.
        struct ClassificationResult
        {
            std::string name;
            int class_idx;
            float score;
        };

        /// @brief Convert vector<T> to vector<uint8_t>
        /// @tparam T Type of input data.
        /// @param dataT vector of input data with type T.
        /// @return vector of uint8_t
        template <typename T>
        static std::vector<unsigned char> vectorT_to_blob(const std::vector<T> &dataT)
        {
            const unsigned char *bytes = reinterpret_cast<const unsigned char *>(&dataT[0]);
            std::vector<unsigned char> result(bytes, bytes + sizeof(T) * dataT.size());
            return result;
        }

        /// @brief Convert vector<uint8_t> to vector<T>
        /// @tparam T Type of output data.
        /// @param buffer input vector of input data.
        /// @return vector of T.
        template <typename T>
        static std::vector<T> blob_to_vectorT(const std::vector<unsigned char> &buffer)
        {
            std::vector<T> result(buffer.size() / sizeof(T));
            memcpy(result.data(), buffer.data(), buffer.size());
            return result;
        }
    } // namespace utils
} // namespace cpp_server

static std::ostream& operator<<(std::ostream& os, const cpp_server::utils::ModelConfig& m)
{
    os << "Input name: " <<  m.input_name_ << "\n";
    os << "Input shape: " <<  m.input_shape_ << "\n";
    os << "Input dtype: " <<  m.input_datatype_ << "\n";
    os << "Input bytesize: " <<  m.input_byte_size_ << "\n";
    os << "Output name: " << m.output_name_ << "\n";
    os << "Output shape: " <<  m.output_shape_ << "\n";
    os << "Output dtype: " <<  m.output_datatype_ << "\n";
    os << "Output bytesize: " <<  m.output_byte_size_ << "\n";
    os << "Max batch size: " << m.max_batch_size_;
    return os;
}

#endif
