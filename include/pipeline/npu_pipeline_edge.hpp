#ifndef _NPU_PIPELINE_EDGE_HPP_
#define _NPU_PIPELINE_EDGE_HPP_

#include "npu_pipeline_types.hpp"
#include "npu_pipeline_context.hpp"
#include <functional>
#include <string>
#include <vector>
#include <memory>

namespace npu_pipeline {

// Forward declarations
struct FrameResults;
class PipelineContext;

// Transform function type
using TransformFunc = std::function<std::vector<PipelineObject>(
    const std::vector<PipelineObject>& input,
    const FrameResults& frame,
    PipelineContext& ctx)>;

// Pipeline edge with transform logic
struct PipelineEdge {
    std::string from_node;
    std::string to_node;

    enum TransformType {
        PASS_THROUGH,              // Pass all objects unchanged
        FILTER_CLASS,              // Filter by class ID
        FILTER_CONFIDENCE,         // Filter by confidence threshold
        CROP_ROI,                  // Crop ROI from source frame
        CROP_ROI_PADDED,           // Crop with padding (context)
        FILTER_TRACK_NEW,          // Only new tracks (for tracking+LPR)
        FILTER_TRACK_UNPROCESSED,  // Only tracks not processed by to_node
        FILTER_TRACK_ACTIVE,       // Only active (not lost) tracks
        BATCH_ACCUMULATE,          // Accumulate for batch inference
        CUSTOM                     // User-defined transform
    } transform_type;

    // Transform-specific parameters
    struct Params {
        std::vector<int> target_classes;           // For FILTER_CLASS
        float confidence_threshold = 0.5f;         // For FILTER_CONFIDENCE
        float padding_ratio = 0.0f;                // For CROP_ROI_PADDED
        size_t batch_size = 1;                     // For BATCH_ACCUMULATE
        std::chrono::milliseconds batch_timeout{5}; // For BATCH_ACCUMULATE
        int min_crop_width = 0;                    // Minimum crop width
        int min_crop_height = 0;                   // Minimum crop height
        int max_crop_width = 0;                    // Maximum crop width (0 = no limit)
        int max_crop_height = 0;                   // Maximum crop height (0 = no limit)
    } params;

    // Custom transform function (for CUSTOM type)
    TransformFunc custom_transform;

    // Constructor
    PipelineEdge(const std::string& from, const std::string& to,
                 TransformType type = PASS_THROUGH)
        : from_node(from), to_node(to), transform_type(type) {}

    // Built-in transforms factory
    static TransformFunc makeTransform(TransformType type, const Params& p);

    // Execute transform
    std::vector<PipelineObject> execute(const std::vector<PipelineObject>& input,
                                        const FrameResults& frame,
                                        PipelineContext& ctx) const;
};

// Edge factory namespace
namespace Edge {

    // Pass-through edge (no transform)
    inline PipelineEdge passThrough(const std::string& from, const std::string& to) {
        return PipelineEdge(from, to, PipelineEdge::PASS_THROUGH);
    }

    // Filter by class ID
    inline PipelineEdge filterClass(const std::string& from, const std::string& to,
                                     int target_class) {
        PipelineEdge edge(from, to, PipelineEdge::FILTER_CLASS);
        edge.params.target_classes = {target_class};
        return edge;
    }

    // Filter by multiple class IDs
    inline PipelineEdge filterClasses(const std::string& from, const std::string& to,
                                       const std::vector<int>& classes) {
        PipelineEdge edge(from, to, PipelineEdge::FILTER_CLASS);
        edge.params.target_classes = classes;
        return edge;
    }

    // Filter by confidence
    inline PipelineEdge filterConfidence(const std::string& from, const std::string& to,
                                          float threshold) {
        PipelineEdge edge(from, to, PipelineEdge::FILTER_CONFIDENCE);
        edge.params.confidence_threshold = threshold;
        return edge;
    }

    // Crop ROI from source frame
    inline PipelineEdge cropRoi(const std::string& from, const std::string& to,
                                 int target_class = -1) {
        PipelineEdge edge(from, to, PipelineEdge::CROP_ROI);
        if (target_class >= 0) {
            edge.params.target_classes = {target_class};
        }
        return edge;
    }

    // Crop ROI with padding
    inline PipelineEdge cropRoiPadded(const std::string& from, const std::string& to,
                                       float padding_ratio = 0.2f) {
        PipelineEdge edge(from, to, PipelineEdge::CROP_ROI_PADDED);
        edge.params.padding_ratio = padding_ratio;
        return edge;
    }

    // Filter only new tracks
    inline PipelineEdge filterNewTracks(const std::string& from, const std::string& to) {
        return PipelineEdge(from, to, PipelineEdge::FILTER_TRACK_NEW);
    }

    // Filter unprocessed tracks (tracks not yet processed by destination node)
    inline PipelineEdge filterUnprocessed(const std::string& from, const std::string& to) {
        return PipelineEdge(from, to, PipelineEdge::FILTER_TRACK_UNPROCESSED);
    }

    // Filter active tracks (not lost)
    inline PipelineEdge filterActiveTracks(const std::string& from, const std::string& to) {
        return PipelineEdge(from, to, PipelineEdge::FILTER_TRACK_ACTIVE);
    }

    // Batch accumulation edge
    inline PipelineEdge batch(const std::string& from, const std::string& to,
                               size_t batch_size,
                               std::chrono::milliseconds timeout = std::chrono::milliseconds(5)) {
        PipelineEdge edge(from, to, PipelineEdge::BATCH_ACCUMULATE);
        edge.params.batch_size = batch_size;
        edge.params.batch_timeout = timeout;
        return edge;
    }

    // LPR trigger edge (unprocessed + active tracks)
    inline PipelineEdge lprTrigger(const std::string& from, const std::string& to) {
        return PipelineEdge(from, to, PipelineEdge::FILTER_TRACK_UNPROCESSED);
    }

    // Custom transform edge
    inline PipelineEdge custom(const std::string& from, const std::string& to,
                                TransformFunc func) {
        PipelineEdge edge(from, to, PipelineEdge::CUSTOM);
        edge.custom_transform = std::move(func);
        return edge;
    }

} // namespace Edge

// Built-in transform implementations
namespace Transforms {

    // Pass all objects unchanged
    std::vector<PipelineObject> passThrough(
        const std::vector<PipelineObject>& input,
        const FrameResults& frame,
        PipelineContext& ctx);

    // Filter by class ID
    std::vector<PipelineObject> filterByClass(
        const std::vector<PipelineObject>& input,
        const FrameResults& frame,
        PipelineContext& ctx,
        const std::vector<int>& target_classes);

    // Filter by confidence
    std::vector<PipelineObject> filterByConfidence(
        const std::vector<PipelineObject>& input,
        const FrameResults& frame,
        PipelineContext& ctx,
        float threshold);

    // Crop ROI from source frame
    std::vector<PipelineObject> cropRoi(
        const std::vector<PipelineObject>& input,
        const FrameResults& frame,
        PipelineContext& ctx,
        const PipelineEdge::Params& params);

    // Crop ROI with padding
    std::vector<PipelineObject> cropRoiPadded(
        const std::vector<PipelineObject>& input,
        const FrameResults& frame,
        PipelineContext& ctx,
        float padding_ratio);

    // Filter new tracks only
    std::vector<PipelineObject> filterNewTracks(
        const std::vector<PipelineObject>& input,
        const FrameResults& frame,
        PipelineContext& ctx);

    // Filter unprocessed tracks
    std::vector<PipelineObject> filterUnprocessed(
        const std::vector<PipelineObject>& input,
        const FrameResults& frame,
        PipelineContext& ctx,
        const std::string& node_id);

    // Filter active tracks
    std::vector<PipelineObject> filterActiveTracks(
        const std::vector<PipelineObject>& input,
        const FrameResults& frame,
        PipelineContext& ctx);

    // Batch accumulation (returns empty - handled by scheduler)
    std::vector<PipelineObject> batchAccumulate(
        const std::vector<PipelineObject>& input,
        const FrameResults& frame,
        PipelineContext& ctx,
        const std::string& node_id,
        const PipelineEdge::Params& params);

} // namespace Transforms

} // namespace npu_pipeline

#endif // _NPU_PIPELINE_EDGE_HPP_
