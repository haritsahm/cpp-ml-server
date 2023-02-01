#include "cpp_server/utils/base64.hpp"
#include <stdexcept>

static const std::string base64_chars = {
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/"};

static unsigned char pos_char_table(const unsigned char chr)
{

    if (chr >= 'A' && chr <= 'Z')
        return chr - 'A';
    else if (chr >= 'a' && chr <= 'z')
        return chr - 'a' + ('Z' - 'A') + 1;
    else if (chr >= '0' && chr <= '9')
        return chr - '0' + ('Z' - 'A') + ('z' - 'a') + 2;
    else if (chr == '+')
        return 62;
    else if (chr == '/')
        return 63;
    else
        throw std::runtime_error("Input is not valid base64-encoded data.");
}

namespace cpp_server
{
    namespace utils
    {
        std::string base64_encode(std::string const &bytes_string)
        {
            if (bytes_string.empty())
                return std::string();

            int str_length = bytes_string.length();
            int enc_length = (str_length + 2) / 3 * 4;

            std::string enc;
            enc.reserve(enc_length);

            for (int i = 0; i < str_length; i += 3)
            {
                unsigned char b0 = base64_chars[(bytes_string[i] & 0xfc) >> 2];
                enc.push_back(b0);

                if (i + 1 < str_length)
                {
                    unsigned char b1 = base64_chars[((bytes_string[i] & 0x03) << 4) + ((bytes_string[i + 1] & 0xf0) >> 4)];
                    enc.push_back(b1);

                    if (i + 2 < str_length)
                    {
                        unsigned char b2 = base64_chars[((bytes_string[i + 1] & 0x0f) << 2) + ((bytes_string[i + 2] & 0xc0) >> 6)];
                        unsigned char b3 = base64_chars[bytes_string[i + 2] & 0x3f];
                        enc.push_back(b2);
                        enc.push_back(b3);
                    }
                    else
                    {
                        // Pad last with trailing
                        unsigned char b2 = base64_chars[(bytes_string[i + 1] & 0x0f) << 2];
                        enc.push_back(b2);
                        enc.push_back('=');
                    }
                }
                else
                {
                    // Pad with double trailling
                    unsigned char b1 = base64_chars[(bytes_string[i] & 0x03) << 4];
                    enc.push_back(b1);
                    enc.push_back('=');
                    enc.push_back('=');
                }
            }

            return enc;
        }

        std::string base64_decode(std::string const &encoded_string)
        {
            if (encoded_string.empty())
                return std::string();

            int str_length = encoded_string.length();
            int dec_length = str_length / 4 * 3;
            std::string dec;
            dec.reserve(dec_length);

            for (int i = 0; i < str_length; i += 4)
            {
                unsigned char b0 = pos_char_table(encoded_string[i]);
                unsigned char b1 = pos_char_table(encoded_string[i + 1]);

                dec.push_back(static_cast<std::string::value_type>((b0 << 2 | ((b1 & 0xF0) >> 4))));

                if ((i + 2 < str_length) && (encoded_string[i + 2] != '='))
                {
                    unsigned char b2 = pos_char_table(encoded_string[i + 2]);
                    dec.push_back(static_cast<std::string::value_type>(((b1 & 0x0f) << 4) + ((b2 & 0x3c) >> 2)));

                    if ((i + 3 < str_length) && (encoded_string[i + 3] != '='))
                    {
                        dec.push_back(static_cast<std::string::value_type>(((b2 & 0x03) << 6) + pos_char_table(encoded_string[i + 3])));
                    }
                }
            }

            return dec;
        }
    } // namespace utils
} // namespace cpp_server
