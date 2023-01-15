// Base64 string decoder
// Source: https://stackoverflow.com/a/13935718

#ifndef BASE64_DECODER_HPP
#define BASE64_DECODER_HPP

#include <string>

namespace cpp_server
{
    namespace utils
    {
        static const std::string base64_chars =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789+/";

        static inline bool is_base64(unsigned char c)
        {
            return (isalnum(c) || (c == '+') || (c == '/'));
        }

        std::string base64_decode(std::string const &encoded_string);
    };
};

#endif