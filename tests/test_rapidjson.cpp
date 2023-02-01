#include <iostream>
#include <gtest/gtest.h>
#include <string>
#include <tuple>
#include <rapidjson/document.h>

typedef std::tuple<uint16_t, std::string> HTTP_CODE;

HTTP_CODE validate_requests(rapidjson::Document &doc)
{

    if (!doc.HasMember("input_jpeg"))
    {
        return std::make_tuple(400, "input_jpeg is not available in data.");
    }
    else if (!doc.HasMember("desired_width"))
    {
        return std::make_tuple(400, "desired_width is not available in data.");
    }
    else if (!doc.HasMember("desired_height"))
    {
        return std::make_tuple(400, "desired_height is not available in data.");
    }

    return std::make_tuple(200, "");
}

TEST(RapidJSONDoc, data_validation)
{
    rapidjson::Document ss_test1;
    ss_test1.Parse("{\"desired_height\":[]}");
    HTTP_CODE code1 = validate_requests(ss_test1);
    EXPECT_EQ(std::get<0>(code1), 400);
    EXPECT_EQ(std::get<1>(code1), "input_jpeg is not available in data.");

    rapidjson::Document ss_test2;
    ss_test2.Parse("{\"input_jpeg\":[], \"desired_height\": 53}");
    HTTP_CODE code2 = validate_requests(ss_test2);
    EXPECT_EQ(std::get<0>(code2), 400);
    EXPECT_EQ(std::get<1>(code2), "desired_width is not available in data.");

    rapidjson::Document ss_test3;
    ss_test3.Parse("{\"input_jpeg\":[], \"desired_width\": 24}");
    HTTP_CODE code3 = validate_requests(ss_test3);
    EXPECT_EQ(std::get<0>(code3), 400);
    EXPECT_EQ(std::get<1>(code3), "desired_height is not available in data.");

    rapidjson::Document ss_test4;
    ss_test4.Parse("{\"input_jpeg\":[], \"desired_width\": 123, \"desired_height\": 123}");
    HTTP_CODE code4 = validate_requests(ss_test4);
    EXPECT_EQ(std::get<0>(code4), 200);
    EXPECT_EQ(std::get<1>(code4), "");
}
