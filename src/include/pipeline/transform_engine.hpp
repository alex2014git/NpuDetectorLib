#ifndef TRANSFORM_ENGINE_HPP
#define TRANSFORM_ENGINE_HPP

#include "npu_pipeline_types.hpp"
#include "npu_pipeline_edge.hpp"
#include "npu_pipeline_context.hpp"
#include <vector>

namespace npu_pipeline {

// Encapsulates edge transform application logic
class TransformEngine {
public:
    // Apply transforms to pipeline objects
    static std::vector<PipelineObject> apply(
        const std::vector<PipelineObject>& inputs,
        const std::vector<PipelineEdge>& edges,
        const FrameResults& frame,
        PipelineContext& ctx);

    // Apply single transform
    static std::vector<PipelineObject> applySingle(
        const std::vector<PipelineObject>& inputs,
        const PipelineEdge& edge,
        const FrameResults& frame,
        PipelineContext& ctx);

    // Filter out BATCH_ACCUMULATE edges
    static std::vector<PipelineEdge> filterBatchEdges(
        const std::vector<PipelineEdge>& edges);
};

} // namespace npu_pipeline

#endif // TRANSFORM_ENGINE_HPP
