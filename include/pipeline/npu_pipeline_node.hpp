#ifndef _NPU_PIPELINE_NODE_HPP_
#define _NPU_PIPELINE_NODE_HPP_

#include "npu_pipeline_types.hpp"
#include "npu_pipeline_context.hpp"
#include "npu.hpp"
#include <string>
#include <vector>
#include <memory>

namespace npu_pipeline {

// Forward declarations
class PipelineContext;
struct FrameResults;

// Base class for all pipeline nodes
class PipelineNode {
public:
    virtual ~PipelineNode() = default;

    // Node identification
    virtual const std::string& getName() const = 0;
    virtual const std::string& getType() const = 0;

    // Initialization
    virtual int initialize(const std::string& configJson, int streamId) = 0;

    // Release resources
    virtual void release() {}

    // Single object processing (for simple transforms)
    // Override this for nodes that process objects one at a time
    virtual PipelineObject processObject(const PipelineObject& input,
                                          const FrameResults& frame,
                                          PipelineContext& ctx) {
        // Default: no-op pass-through
        (void)frame;
        (void)ctx;
        return input;
    }

    // Batch processing - OVERRIDE THIS for NPU inference nodes
    // Default implementation calls processObject for each (inefficient but safe)
    virtual std::vector<PipelineObject> processBatch(
        const std::vector<PipelineObject>& inputs,
        const std::vector<uint64_t>& frame_ids,
        PipelineContext& ctx) {

        std::vector<PipelineObject> outputs;
        outputs.reserve(inputs.size());

        for (size_t i = 0; i < inputs.size(); ++i) {
            auto& frame = ctx.getFrame(frame_ids[i]);
            outputs.push_back(processObject(inputs[i], frame, ctx));
        }
        return outputs;
    }

    // Process batch from accumulator
    virtual std::vector<PipelineObject> processBatch(
        const BatchAccumulator& batch,
        PipelineContext& ctx) {

        std::vector<PipelineObject> inputs;
        std::vector<uint64_t> frame_ids;
        inputs.reserve(batch.size());
        frame_ids.reserve(batch.size());

        for (const auto& item : batch.items) {
            inputs.push_back(item.object);
            frame_ids.push_back(item.frame_id);
        }

        return processBatch(inputs, frame_ids, ctx);
    }

    // Return true if node supports efficient batch processing
    virtual bool supportsBatching() const { return false; }

    // Return expected input batch size (for optimization)
    virtual size_t getPreferredBatchSize() const { return 1; }

    // Required upstream nodes
    virtual std::vector<std::string> getInputNodes() const = 0;

    // Set input nodes (for graph construction)
    virtual void setInputNodes(const std::vector<std::string>& nodes) = 0;

    // Check if node is ready to execute (all inputs available)
    virtual bool isReady(const FrameResults& frame) const;

    // Get node latency (for scheduling optimization)
    virtual double getAverageLatencyMs() const { return 0.0; }

    // Reset node state (for new stream)
    virtual void reset() {}
};

// NPU inference node with batch support
class NpuInferenceNode : public PipelineNode {
public:
    NpuInferenceNode(const std::string& name, int algorithm_type);
    ~NpuInferenceNode() override = default;

    const std::string& getName() const override { return _name; }
    const std::string& getType() const override { return _type; }

    int initialize(const std::string& configJson, int streamId) override;
    void release() override;

    // Enable batching
    bool supportsBatching() const override { return true; }

    // Efficient batch inference
    std::vector<PipelineObject> processBatch(
        const std::vector<PipelineObject>& inputs,
        const std::vector<uint64_t>& frame_ids,
        PipelineContext& ctx) override;

    // Get/set preferred batch size
    size_t getPreferredBatchSize() const override { return _preferred_batch_size; }
    void setPreferredBatchSize(size_t size) { _preferred_batch_size = size; }

    // Required upstream nodes
    std::vector<std::string> getInputNodes() const override { return _input_nodes; }
    void setInputNodes(const std::vector<std::string>& nodes) override { _input_nodes = nodes; }

    // Get NPU handle
    std::shared_ptr<Npu> getNpu() const { return _npu; }

    // Set model configuration
    void setModelConfig(const std::string& config) { _model_config = config; }
    const std::string& getModelConfig() const { return _model_config; }

protected:
    // Preprocess input objects (crop, resize, normalize)
    virtual std::vector<std::shared_ptr<image_share_t>> preprocess(
        const std::vector<PipelineObject>& inputs,
        const std::vector<uint64_t>& frame_ids,
        PipelineContext& ctx);

    // Post-process NPU outputs
    virtual std::vector<PipelineObject> postprocess(
        const std::vector<PipelineObject>& inputs,
        const std::vector<std::vector<float>>& npu_outputs);

    // Single item inference (fallback)
    virtual std::vector<PipelineObject> inferSingle(
        const PipelineObject& input,
        const FrameResults& frame);

private:
    std::string _name;
    std::string _type = "npu_inference";
    int _algorithm_type = 0;
    std::string _model_config;
    int _stream_id = 0;
    size_t _preferred_batch_size = 8;

    std::shared_ptr<Npu> _npu;
    std::vector<std::string> _input_nodes;

    // Statistics
    mutable std::mutex _stats_mutex;
    double _avg_latency_ms = 0.0;
    uint64_t _inference_count = 0;
};

// Transform node for image processing (crop, resize, etc.)
class TransformNode : public PipelineNode {
public:
    enum TransformOp {
        RESIZE,
        CROP,
        NORMALIZE,
        LETTERBOX,
        FLIP,
        ROTATE
    };

    TransformNode(const std::string& name, TransformOp op);
    ~TransformNode() override = default;

    const std::string& getName() const override { return _name; }
    const std::string& getType() const override { return _type; }

    int initialize(const std::string& configJson, int streamId) override;

    // Single object processing
    PipelineObject processObject(const PipelineObject& input,
                                  const FrameResults& frame,
                                  PipelineContext& ctx) override;

    // Required upstream nodes
    std::vector<std::string> getInputNodes() const override { return _input_nodes; }
    void setInputNodes(const std::vector<std::string>& nodes) override { _input_nodes = nodes; }

    // Set transform parameters
    void setParams(const std::vector<float>& params) { _params = params; }

private:
    std::string _name;
    std::string _type = "transform";
    TransformOp _op;
    std::vector<float> _params;
    std::vector<std::string> _input_nodes;
};

// Result aggregation node (collects results from multiple branches)
class AggregateNode : public PipelineNode {
public:
    AggregateNode(const std::string& name);
    ~AggregateNode() override = default;

    const std::string& getName() const override { return _name; }
    const std::string& getType() const override { return _type; }

    int initialize(const std::string& configJson, int streamId) override;

    // Process all inputs from multiple upstream nodes
    std::vector<PipelineObject> processBatch(
        const std::vector<PipelineObject>& inputs,
        const std::vector<uint64_t>& frame_ids,
        PipelineContext& ctx) override;

    // Required upstream nodes
    std::vector<std::string> getInputNodes() const override { return _input_nodes; }
    void setInputNodes(const std::vector<std::string>& nodes) override { _input_nodes = nodes; }

private:
    std::string _name;
    std::string _type = "aggregate";
    std::vector<std::string> _input_nodes;
};

} // namespace npu_pipeline

#endif // _NPU_PIPELINE_NODE_HPP_
