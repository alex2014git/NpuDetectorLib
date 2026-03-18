#include "pipeline/transform_engine.hpp"
#include "common/debug_logger.hpp"

namespace npu_pipeline {

std::vector<PipelineObject> TransformEngine::apply(
    const std::vector<PipelineObject>& inputs,
    const std::vector<PipelineEdge>& edges,
    const FrameResults& frame,
    PipelineContext& ctx) {

    if (inputs.empty()) return {};

    NPU_DEBUG_PREFIX("APPLY TRANSFORMS", inputs.size() << " inputs, " << edges.size() << " edges");

    std::vector<PipelineObject> result = inputs;

    for (const auto& edge : edges) {
        NPU_DEBUG_PREFIX("APPLY TRANSFORMS", "  Applying edge " << edge.from_node << "->" << edge.to_node
                         << " type=" << edge.transform_type);
        result = edge.execute(result, frame, ctx);
        NPU_DEBUG_PREFIX("APPLY TRANSFORMS", "  Result: " << result.size() << " objects");
    }

    return result;
}

std::vector<PipelineObject> TransformEngine::applySingle(
    const std::vector<PipelineObject>& inputs,
    const PipelineEdge& edge,
    const FrameResults& frame,
    PipelineContext& ctx) {

    return edge.execute(inputs, frame, ctx);
}

std::vector<PipelineEdge> TransformEngine::filterBatchEdges(
    const std::vector<PipelineEdge>& edges) {

    std::vector<PipelineEdge> result;
    result.reserve(edges.size());

    for (const auto& edge : edges) {
        if (edge.transform_type != PipelineEdge::BATCH_ACCUMULATE) {
            result.push_back(edge);
        }
    }

    return result;
}

} // namespace npu_pipeline
