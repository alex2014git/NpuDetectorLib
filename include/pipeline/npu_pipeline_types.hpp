#ifndef _NPU_PIPELINE_TYPES_HPP_
#define _NPU_PIPELINE_TYPES_HPP_

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <variant>
#include <memory>
#include <chrono>

// Include npu.hpp for image_share_t definition
#include "npu.hpp"

namespace npu_pipeline {

// Object ROI (Region of Interest)
struct ObjectRoi {
    float y_min = 0.0f;
    float x_min = 0.0f;
    float y_max = 0.0f;
    float x_max = 0.0f;
    float confidence = 0.0f;
    int class_id = -1;
    std::string class_name;

    float width() const { return x_max - x_min; }
    float height() const { return y_max - y_min; }
    float area() const { return width() * height(); }
    float center_x() const { return (x_min + x_max) / 2.0f; }
    float center_y() const { return (y_min + y_max) / 2.0f; }
};

// Detection result from object detection models
struct DetectionResult {
    int class_id = -1;
    float confidence = 0.0f;
    ObjectRoi bbox;
    std::vector<float> keypoints;  // For pose detection
    std::vector<float> mask;       // For segmentation
};

// LPR (License Plate Recognition) result
struct LprResult {
    std::string text;
    float confidence = 0.0f;
    std::vector<float> char_confidences;
};

// Classification result
struct ClassificationResult {
    int class_id = -1;
    std::string label;
    float confidence = 0.0f;
    std::vector<std::pair<int, float>> top_k;  // Top-K predictions
};

// Segmentation result
struct SegmentationResult {
    int class_id = -1;
    float confidence = 0.0f;
    std::vector<uint8_t> mask;  // Binary or multi-class mask
    int mask_width = 0;
    int mask_height = 0;
};

// Feature embedding result (for tracking/re-identification)
struct FeatureResult {
    std::vector<float> embedding;
    int feature_dim = 0;
};

// Variant for stage results - no std::any overhead
using StageResult = std::variant<
    DetectionResult,
    LprResult,
    ClassificationResult,
    SegmentationResult,
    FeatureResult
>;

// Generic track state (persisted across frames)
struct TrackState {
    uint64_t id = 0;
    ObjectRoi last_roi;
    int frames_tracked = 0;
    int frames_since_update = 0;
    std::chrono::high_resolution_clock::time_point first_seen;
    std::chrono::high_resolution_clock::time_point last_seen;

    // Generic processing flags (node_name -> processed)
    std::unordered_map<std::string, bool> processed_by;

    // Mutex for thread-safe access to processed_by - use shared_ptr since mutex is not copyable
    mutable std::shared_ptr<std::mutex> processed_by_mutex;

    // Constructor initializes mutex
    TrackState() : processed_by_mutex(std::make_shared<std::mutex>()) {}

    // Copy constructor - don't copy mutex, create new one
    TrackState(const TrackState& other)
        : id(other.id), last_roi(other.last_roi),
          frames_tracked(other.frames_tracked),
          frames_since_update(other.frames_since_update),
          first_seen(other.first_seen), last_seen(other.last_seen),
          processed_by(other.processed_by),
          processed_by_mutex(std::make_shared<std::mutex>()) {}

    // Copy assignment
    TrackState& operator=(const TrackState& other) {
        if (this != &other) {
            id = other.id;
            last_roi = other.last_roi;
            frames_tracked = other.frames_tracked;
            frames_since_update = other.frames_since_update;
            first_seen = other.first_seen;
            last_seen = other.last_seen;
            processed_by = other.processed_by;
            // Don't copy mutex, keep existing one
        }
        return *this;
    }

    bool wasProcessed(const std::string& node) const {
        std::lock_guard<std::mutex> lock(*processed_by_mutex);
        auto it = processed_by.find(node);
        return it != processed_by.end() && it->second;
    }

    void markProcessed(const std::string& node) {
        std::lock_guard<std::mutex> lock(*processed_by_mutex);
        processed_by[node] = true;
    }

    bool isNew() const { return frames_tracked <= 1; }
    bool isLost() const { return frames_since_update > 30; }
    bool isActive() const { return !isLost(); }
};

// Pipeline object - flows through the pipeline
struct PipelineObject {
    uint64_t frame_id = 0;       // Which frame this object belongs to
    uint64_t track_id = 0;       // 0 if not tracked
    uint64_t object_id = 0;      // Unique within frame

    ObjectRoi roi;               // Current ROI (normalized coordinates)
    std::unordered_map<std::string, StageResult> stage_results;

    // Cropped image data (set by CROP_ROI transform)
    std::shared_ptr<image_share_t> cropped_image;

    // Metadata for pipeline routing
    std::unordered_map<std::string, std::string> metadata;

    template<typename T>
    const T* getResult(const std::string& stage) const {
        auto it = stage_results.find(stage);
        if (it == stage_results.end()) return nullptr;
        return std::get_if<T>(&it->second);
    }

    template<typename T>
    T* getResult(const std::string& stage) {
        auto it = stage_results.find(stage);
        if (it == stage_results.end()) return nullptr;
        return std::get_if<T>(&it->second);
    }

    bool hasResult(const std::string& stage) const {
        return stage_results.find(stage) != stage_results.end();
    }

    template<typename T>
    void setResult(const std::string& stage, T result) {
        stage_results[stage] = std::move(result);
    }
};

// Frame metadata
struct FrameMetadata {
    uint64_t frame_id = 0;
    std::chrono::high_resolution_clock::time_point timestamp;
    std::chrono::high_resolution_clock::time_point pipeline_start;
    int source_width = 0;
    int source_height = 0;
    int source_channels = 0;
    float fps = 0.0f;
};

// Batch item for accumulated inference
struct BatchItem {
    uint64_t frame_id = 0;
    uint64_t object_id = 0;
    PipelineObject object;
    std::shared_ptr<image_share_t> cropped_roi;
};

// Pipeline statistics
struct PipelineStats {
    uint64_t frames_processed = 0;
    uint64_t objects_detected = 0;
    uint64_t objects_tracked = 0;
    double avg_latency_ms = 0.0;
    double max_latency_ms = 0.0;
    double min_latency_ms = 999999.0;
    double throughput_fps = 0.0;
    std::unordered_map<std::string, double> node_latency_ms;
};

// Frame output (copyable result, returned to callers)
struct FrameOutput {
    FrameMetadata metadata;
    std::unordered_map<std::string, std::vector<PipelineObject>> node_outputs;
    std::unordered_map<uint64_t, TrackState> track_states;
};

} // namespace npu_pipeline

#endif // _NPU_PIPELINE_TYPES_HPP_
