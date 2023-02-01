// Base64 string decoder
// Source: https://stackoverflow.com/a/13935718

#ifndef BASE64_DECODER_HPP
#define BASE64_DECODER_HPP

#include <string>
#include <stdexcept>

namespace cpp_server
{
    namespace utils
    {
        /// @brief Decode encoded base64 string
        /// @param encoded_string string to decode
        /// @return Decoded string data
        std::string base64_decode(std::string const &encoded_string);

        /// @brief Encode raw string
        /// @param encoded_string string to encode
        /// @return Encoded string data
        std::string base64_encode(std::string const &string_input);
    } // namespace utils
} // namespace cpp_server

#endif
