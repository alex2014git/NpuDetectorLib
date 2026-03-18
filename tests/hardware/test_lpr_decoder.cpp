#include <gtest/gtest.h>
#include "pipeline/result_decoder.hpp"

#include <cmath>

// Handle platforms where std::nanf may not be available
#ifndef NAN
#define NAN (0.0f / 0.0f)
#endif

using namespace npu;

class LprDecoderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default charset for testing (simplified)
        _default_charset = {"#", "A", "B", "C", "D", "E", "F", "G", "H", "I"};
    }

    std::vector<std::string> _default_charset;
};

TEST_F(LprDecoderTest, BasicCtcDecoding) {
    LprDecoder decoder(_default_charset);

    // Input: [1, 1, 2, 2, 3, 0, 0] should produce "ABC"
    // Index 0 = "#" (blank), 1 = "A", 2 = "B", 3 = "C"
    std::vector<std::vector<float>> raw_outputs = {
        {1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 0.0f, 0.0f}
    };

    auto results = decoder.decode(raw_outputs, {});

    ASSERT_EQ(results.size(), 1);
    auto* lpr_result = std::get_if<LprResult>(&results[0]);
    ASSERT_NE(lpr_result, nullptr);
    EXPECT_EQ(lpr_result->text, "ABC");
}

TEST_F(LprDecoderTest, SkipDuplicates) {
    LprDecoder decoder(_default_charset);

    // Consecutive same indices should produce single char
    // [1, 1, 1, 2, 2, 3] should produce "ABC" (not "AAABBC")
    std::vector<std::vector<float>> raw_outputs = {
        {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 3.0f}
    };

    auto results = decoder.decode(raw_outputs, {});

    ASSERT_EQ(results.size(), 1);
    auto* lpr_result = std::get_if<LprResult>(&results[0]);
    ASSERT_NE(lpr_result, nullptr);
    EXPECT_EQ(lpr_result->text, "ABC");
}

TEST_F(LprDecoderTest, SkipBlanks) {
    LprDecoder decoder(_default_charset);

    // Index 0 = "#" (blank) should be skipped
    // [0, 1, 0, 2, 0, 3, 0] should produce "ABC"
    std::vector<std::vector<float>> raw_outputs = {
        {0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f}
    };

    auto results = decoder.decode(raw_outputs, {});

    ASSERT_EQ(results.size(), 1);
    auto* lpr_result = std::get_if<LprResult>(&results[0]);
    ASSERT_NE(lpr_result, nullptr);
    EXPECT_EQ(lpr_result->text, "ABC");
}

TEST_F(LprDecoderTest, EmptyOutput) {
    LprDecoder decoder(_default_charset);

    std::vector<std::vector<float>> raw_outputs = {
        {0.0f, 0.0f, 0.0f}
    };

    auto results = decoder.decode(raw_outputs, {});

    ASSERT_EQ(results.size(), 1);
    auto* lpr_result = std::get_if<LprResult>(&results[0]);
    ASSERT_NE(lpr_result, nullptr);
    EXPECT_EQ(lpr_result->text, "");
}

TEST_F(LprDecoderTest, NoRawOutput) {
    LprDecoder decoder(_default_charset);

    std::vector<std::vector<float>> raw_outputs;

    auto results = decoder.decode(raw_outputs, {});

    EXPECT_TRUE(results.empty());
}

TEST_F(LprDecoderTest, CustomCharset) {
    // Custom charset: index 1 = "京", 2 = "A", 3 = "1"
    std::vector<std::string> custom_charset = {"#", "京", "A", "1"};
    LprDecoder decoder(custom_charset);

    std::vector<std::vector<float>> raw_outputs = {
        {1.0f, 2.0f, 3.0f}
    };

    auto results = decoder.decode(raw_outputs, {});

    ASSERT_EQ(results.size(), 1);
    auto* lpr_result = std::get_if<LprResult>(&results[0]);
    ASSERT_NE(lpr_result, nullptr);
    EXPECT_EQ(lpr_result->text, "京A1");
}

TEST_F(LprDecoderTest, DefaultCharsetFallback) {
    // When empty charset is passed, should use default Chinese charset
    LprDecoder decoder({});

    // Default charset has 80+ characters
    std::vector<std::vector<float>> raw_outputs = {
        {42.0f, 52.0f, 53.0f, 54.0f}  // "0", "A", "B", "C"
    };

    auto results = decoder.decode(raw_outputs, {});

    ASSERT_EQ(results.size(), 1);
    auto* lpr_result = std::get_if<LprResult>(&results[0]);
    ASSERT_NE(lpr_result, nullptr);
    // Verify that some text was decoded (content depends on default charset)
    EXPECT_FALSE(lpr_result->text.empty());
}

TEST_F(LprDecoderTest, InvalidIndices) {
    LprDecoder decoder(_default_charset);

    // Indices outside charset range should be skipped
    std::vector<std::vector<float>> raw_outputs = {
        {1.0f, 100.0f, 2.0f, -1.0f, 3.0f}
    };

    auto results = decoder.decode(raw_outputs, {});

    ASSERT_EQ(results.size(), 1);
    auto* lpr_result = std::get_if<LprResult>(&results[0]);
    ASSERT_NE(lpr_result, nullptr);
    EXPECT_EQ(lpr_result->text, "ABC");
}

TEST_F(LprDecoderTest, HandlesNaNAndInf) {
    LprDecoder decoder(_default_charset);

    // NaN and Inf values should be skipped
    std::vector<std::vector<float>> raw_outputs = {
        {1.0f, NAN, 2.0f, std::numeric_limits<float>::infinity(), 3.0f}
    };

    auto results = decoder.decode(raw_outputs, {});

    ASSERT_EQ(results.size(), 1);
    auto* lpr_result = std::get_if<LprResult>(&results[0]);
    ASSERT_NE(lpr_result, nullptr);
    EXPECT_EQ(lpr_result->text, "ABC");
}

// Main function for the test
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
