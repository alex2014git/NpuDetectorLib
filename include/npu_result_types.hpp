#ifndef _NPU_RESULT_TYPES_HPP_
#define _NPU_RESULT_TYPES_HPP_

#include <string>
#include <vector>
#include <variant>

// Forward declaration - object_roi_t defined in npu.hpp
struct _object_roi;
typedef struct _object_roi object_roi_t;

namespace npu {

// LPR (License Plate Recognition) result
struct LprResult {
    std::string text;
    float confidence = 1.0f;
    std::vector<float> char_confidences;
};

// Classification result
struct ClassificationResult {
    int class_id = -1;
    std::string label;
    float confidence = 0.0f;
    std::vector<std::pair<int, float>> top_k;  // Top-K predictions
};

// Detection result bounding box
struct Bbox {
    float x_min = 0.0f;
    float y_min = 0.0f;
    float x_max = 0.0f;
    float y_max = 0.0f;

    float width() const { return x_max - x_min; }
    float height() const { return y_max - y_min; }
    float area() const { return width() * height(); }
    float center_x() const { return (x_min + x_max) / 2.0f; }
    float center_y() const { return (y_min + y_max) / 2.0f; }
};

// Detection result from object detection models
struct DetectionResult {
    int class_id = -1;
    std::string class_name;
    float confidence = 0.0f;
    Bbox bbox;
};

// Pose result with keypoints
struct PoseResult {
    DetectionResult detection;
    std::vector<std::pair<float, float>> keypoints;  // (x, y) coordinates
    std::vector<float> keypoint_scores;
    int num_keypoints = 0;
};

// Segmentation result with mask
struct SegmentationResult {
    DetectionResult detection;
    std::vector<uint8_t> mask;  // Binary or multi-class mask
    int mask_width = 0;
    int mask_height = 0;
};

// Feature embedding result (for tracking/re-identification)
struct FeatureResult {
    std::vector<float> embedding;
    int feature_dim = 0;
};

// Unified result type - std::variant for type-safe polymorphism
using NpuResult = std::variant<
    LprResult,
    ClassificationResult,
    DetectionResult,
    PoseResult,
    SegmentationResult,
    FeatureResult
>;

} // namespace npu

#endif // _NPU_RESULT_TYPES_HPP_
