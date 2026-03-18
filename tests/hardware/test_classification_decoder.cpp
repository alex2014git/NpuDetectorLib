#include <gtest/gtest.h>
#include "pipeline/result_decoder.hpp"

using namespace npu;

class ClassificationDecoderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default labels for testing
        _labels = {"cat", "dog", "bird", "fish", "horse", "car", "truck", "bike"};
    }

    std::vector<std::string> _labels;
};

TEST_F(ClassificationDecoderTest, ArgmaxSelection) {
    ClassificationDecoder decoder;

    // Highest value at index 1 ("dog")
    std::vector<std::vector<float>> raw_outputs = {
        {0.1f, 0.9f, 0.2f, 0.3f, 0.1f, 0.05f, 0.02f, 0.01f}
    };

    auto results = decoder.decode(raw_outputs, _labels);

    ASSERT_EQ(results.size(), 1);
    auto* cls_result = std::get_if<ClassificationResult>(&results[0]);
    ASSERT_NE(cls_result, nullptr);
    EXPECT_EQ(cls_result->class_id, 1);
    EXPECT_EQ(cls_result->label, "dog");
    EXPECT_FLOAT_EQ(cls_result->confidence, 0.9f);
}

TEST_F(ClassificationDecoderTest, TopKOrdering) {
    ClassificationDecoder decoder;

    // Values: [0.1, 0.9, 0.2, 0.3, 0.1, 0.05, 0.02, 0.01]
    // Top-5 should be: dog(1, 0.9), fish(3, 0.3), bird(2, 0.2), cat(0, 0.1), horse(4, 0.1)
    std::vector<std::vector<float>> raw_outputs = {
        {0.1f, 0.9f, 0.2f, 0.3f, 0.1f, 0.05f, 0.02f, 0.01f}
    };

    auto results = decoder.decode(raw_outputs, _labels);

    ASSERT_EQ(results.size(), 1);
    auto* cls_result = std::get_if<ClassificationResult>(&results[0]);
    ASSERT_NE(cls_result, nullptr);

    EXPECT_EQ(cls_result->top_k.size(), 5);

    // Top result should be dog
    EXPECT_EQ(cls_result->top_k[0].first, 1);
    EXPECT_FLOAT_EQ(cls_result->top_k[0].second, 0.9f);

    // Verify descending order
    for (size_t i = 1; i < cls_result->top_k.size(); ++i) {
        EXPECT_GE(cls_result->top_k[i - 1].second, cls_result->top_k[i].second);
    }
}

TEST_F(ClassificationDecoderTest, LabelMapping) {
    ClassificationDecoder decoder;

    // Test with 3 classes
    std::vector<std::string> labels = {"class_a", "class_b", "class_c"};
    std::vector<std::vector<float>> raw_outputs = {
        {0.1f, 0.5f, 0.2f}
    };

    auto results = decoder.decode(raw_outputs, labels);

    ASSERT_EQ(results.size(), 1);
    auto* cls_result = std::get_if<ClassificationResult>(&results[0]);
    ASSERT_NE(cls_result, nullptr);
    EXPECT_EQ(cls_result->class_id, 1);
    EXPECT_EQ(cls_result->label, "class_b");
}

TEST_F(ClassificationDecoderTest, MissingLabels) {
    ClassificationDecoder decoder;

    // Test with more classes than labels
    std::vector<std::string> labels = {"cat", "dog"};  // Only 2 labels
    std::vector<std::vector<float>> raw_outputs = {
        {0.1f, 0.5f, 0.2f, 0.3f}  // 4 classes
    };

    auto results = decoder.decode(raw_outputs, labels);

    ASSERT_EQ(results.size(), 1);
    auto* cls_result = std::get_if<ClassificationResult>(&results[0]);
    ASSERT_NE(cls_result, nullptr);
    EXPECT_EQ(cls_result->class_id, 1);
    EXPECT_EQ(cls_result->label, "dog");

    // Class 2 has no label, should get generated name
    EXPECT_EQ(cls_result->top_k[1].first, 3);
}

TEST_F(ClassificationDecoderTest, EmptyLabels) {
    ClassificationDecoder decoder;

    std::vector<std::vector<float>> raw_outputs = {
        {0.1f, 0.5f, 0.2f}
    };

    auto results = decoder.decode(raw_outputs, {});

    ASSERT_EQ(results.size(), 1);
    auto* cls_result = std::get_if<ClassificationResult>(&results[0]);
    ASSERT_NE(cls_result, nullptr);
    EXPECT_EQ(cls_result->class_id, 1);
    EXPECT_EQ(cls_result->label, "class_1");
}

TEST_F(ClassificationDecoderTest, NoRawOutput) {
    ClassificationDecoder decoder;

    std::vector<std::vector<float>> raw_outputs;

    auto results = decoder.decode(raw_outputs, _labels);

    EXPECT_TRUE(results.empty());
}

TEST_F(ClassificationDecoderTest, EmptyOutputTensor) {
    ClassificationDecoder decoder;

    std::vector<std::vector<float>> raw_outputs = {{}};

    auto results = decoder.decode(raw_outputs, _labels);

    EXPECT_TRUE(results.empty());
}

TEST_F(ClassificationDecoderTest, SingleClass) {
    ClassificationDecoder decoder;

    // Edge case: single class
    std::vector<std::vector<float>> raw_outputs = {
        {0.5f}
    };

    auto results = decoder.decode(raw_outputs, {"only_class"});

    ASSERT_EQ(results.size(), 1);
    auto* cls_result = std::get_if<ClassificationResult>(&results[0]);
    ASSERT_NE(cls_result, nullptr);
    EXPECT_EQ(cls_result->class_id, 0);
    EXPECT_EQ(cls_result->label, "only_class");
    EXPECT_EQ(cls_result->top_k.size(), 1);
}

TEST_F(ClassificationDecoderTest, SmallNumberOfClasses) {
    ClassificationDecoder decoder;

    // Less than 5 classes
    std::vector<std::vector<float>> raw_outputs = {
        {0.1f, 0.3f, 0.2f}
    };

    auto results = decoder.decode(raw_outputs, _labels);

    ASSERT_EQ(results.size(), 1);
    auto* cls_result = std::get_if<ClassificationResult>(&results[0]);
    ASSERT_NE(cls_result, nullptr);

    // Should only have 3 top-k entries
    EXPECT_EQ(cls_result->top_k.size(), 3);
}

TEST_F(ClassificationDecoderTest, TieBreaking) {
    ClassificationDecoder decoder;

    // Two equal values - first one should win (stable sort behavior)
    std::vector<std::vector<float>> raw_outputs = {
        {0.5f, 0.5f, 0.1f}
    };

    auto results = decoder.decode(raw_outputs, _labels);

    ASSERT_EQ(results.size(), 1);
    auto* cls_result = std::get_if<ClassificationResult>(&results[0]);
    ASSERT_NE(cls_result, nullptr);
    EXPECT_EQ(cls_result->class_id, 0);
}

TEST_F(ClassificationDecoderTest, ConfidenceBounds) {
    ClassificationDecoder decoder;

    // Test with confidence values at bounds
    std::vector<std::vector<float>> raw_outputs = {
        {0.0f, 1.0f, 0.5f}
    };

    auto results = decoder.decode(raw_outputs, _labels);

    ASSERT_EQ(results.size(), 1);
    auto* cls_result = std::get_if<ClassificationResult>(&results[0]);
    ASSERT_NE(cls_result, nullptr);
    EXPECT_FLOAT_EQ(cls_result->confidence, 1.0f);
    EXPECT_EQ(cls_result->class_id, 1);
}

// Main function for the test
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
