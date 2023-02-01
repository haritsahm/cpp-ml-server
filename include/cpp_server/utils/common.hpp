#ifndef COMMON_HELPER_HPP
#define COMMON_HELPER_HPP

#include <vector>
#include <string>

namespace cpp_server
{
    namespace utils
    {
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
            std::vector<int> shape;
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
        static std::vector<unsigned char> vectorT_to_blob(std::vector<T> &dataT)
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

#endif
