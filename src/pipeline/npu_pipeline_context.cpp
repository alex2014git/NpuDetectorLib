#include "pipeline/npu_pipeline_context.hpp"
#include "pipeline/npu_pipeline_types.hpp"
#include <algorithm>

namespace npu_pipeline {

// Get or create frame results
FrameResults& PipelineContext::getFrame(uint64_t frame_id) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _frames.find(frame_id);
    if (it == _frames.end()) {
        auto frame = std::make_unique<FrameResults>();
        frame->metadata.frame_id = frame_id;
        auto& ref = *frame;
        _frames[frame_id] = std::move(frame);
        return ref;
    }
    return *it->second;
}

// Get or create frame results (const version)
const FrameResults& PipelineContext::getFrame(uint64_t frame_id) const {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _frames.find(frame_id);
    if (it != _frames.end()) {
        return *it->second;
    }
    static const FrameResults empty;
    return empty;
}

// Check if frame exists
bool PipelineContext::hasFrame(uint64_t frame_id) const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _frames.find(frame_id) != _frames.end();
}

// Remove a frame from context
void PipelineContext::removeFrame(uint64_t frame_id) {
    std::lock_guard<std::mutex> lock(_mutex);
    _frames.erase(frame_id);
    _completed_nodes.erase(frame_id);
}

// Called by nodes to write output
void PipelineContext::writeNodeOutput(uint64_t frame_id,
                                     const std::string& node_id,
                                     std::vector<PipelineObject> objects) {
    auto& frame = getFrame(frame_id);
    {
        std::lock_guard<std::mutex> lock(frame.results_mutex);
        frame.node_outputs[node_id] = std::move(objects);
    }
    frame.completed_nodes.fetch_add(1, std::memory_order_release);
}

// Called by nodes to read input (from upstream node)
const std::vector<PipelineObject>& PipelineContext::readNodeOutput(
    uint64_t frame_id,
    const std::string& upstream_node_id) const {
    auto& frame = getFrame(frame_id);
    return frame.getNodeOutput(upstream_node_id);
}

// Get track state (creates if not exists)
TrackState& PipelineContext::getTrackState(uint64_t frame_id, uint64_t track_id) {
    auto& frame = getFrame(frame_id);
    std::lock_guard<std::mutex> lock(frame.results_mutex);
    return frame.track_states[track_id];
}

// Update track state
void PipelineContext::updateTrackState(uint64_t frame_id, uint64_t track_id,
                                      const TrackState& state) {
    auto& frame = getFrame(frame_id);
    std::lock_guard<std::mutex> lock(frame.results_mutex);
    frame.track_states[track_id] = state;
}

// Batch accumulation
bool PipelineContext::accumulateForBatch(const std::string& node_id,
                                         BatchItem item,
                                         size_t max_batch_size,
                                         std::chrono::milliseconds timeout) {
    std::lock_guard<std::mutex> lock(_mutex);

    auto& accumulator = _batch_accumulators[node_id];
    accumulator.max_batch_size = max_batch_size;
    accumulator.timeout = timeout;

    if (accumulator.items.empty()) {
        accumulator.first_add = std::chrono::steady_clock::now();
    }

    accumulator.add(std::move(item));

    // Check if batch is ready
    if (accumulator.items.size() >= max_batch_size) {
        return true;
    }

    // Check timeout
    if (!accumulator.has_timeout) {
        auto elapsed = std::chrono::steady_clock::now() - accumulator.first_add;
        if (elapsed >= timeout) {
            accumulator.has_timeout = true;
            return true;
        }
    }

    return false;
}

// Get accumulated batch and clear
BatchAccumulator PipelineContext::getBatch(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _batch_accumulators.find(node_id);
    if (it != _batch_accumulators.end()) {
        BatchAccumulator result = std::move(it->second);
        it->second.clear();
        return result;
    }
    return BatchAccumulator{};
}

// Peek at batch without clearing
const BatchAccumulator& PipelineContext::peekBatch(const std::string& node_id) const {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _batch_accumulators.find(node_id);
    if (it != _batch_accumulators.end()) {
        return it->second;
    }
    static const BatchAccumulator empty;
    return empty;
}

// Check if batch is ready
bool PipelineContext::isBatchReady(const std::string& node_id) const {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _batch_accumulators.find(node_id);
    if (it != _batch_accumulators.end()) {
        return it->second.isReady();
    }
    return false;
}

// Clear batch accumulator
void PipelineContext::clearBatch(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(_mutex);
    _batch_accumulators.erase(node_id);
}

// Initialize frame with source image
void PipelineContext::initializeFrame(uint64_t frame_id,
                                      std::shared_ptr<image_share_t> source_frame,
                                      std::function<void(const FrameResults&)> callback) {
    auto& frame = getFrame(frame_id);
    frame.source_frame = source_frame;
    frame.metadata.timestamp = std::chrono::high_resolution_clock::now();
    frame.completion_callback = std::move(callback);
}

// Set number of expected nodes for completion tracking
void PipelineContext::setExpectedNodes(uint64_t frame_id, int num_nodes) {
    auto& frame = getFrame(frame_id);
    frame.pending_inputs.store(num_nodes, std::memory_order_release);
}

// Mark node as completed for a frame
void PipelineContext::markNodeCompleted(uint64_t frame_id, const std::string& node_id) {
    auto& frame = getFrame(frame_id);
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _completed_nodes[frame_id].insert(node_id);
    }

    int remaining = frame.pending_inputs.fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (remaining == 0) {
        frame.markComplete();
    }
}

// Wait for specific frame
void PipelineContext::waitForFrame(uint64_t frame_id) {
    if (!hasFrame(frame_id)) {
        return;
    }
    auto& frame = getFrame(frame_id);
    frame.waitForCompletion();
}

// Get frame count
size_t PipelineContext::frameCount() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _frames.size();
}

// Cleanup old frames
void PipelineContext::cleanupFramesBefore(uint64_t frame_id) {
    std::lock_guard<std::mutex> lock(_mutex);
    for (auto it = _frames.begin(); it != _frames.end();) {
        if (it->first < frame_id) {
            it = _frames.erase(it);
        } else {
            ++it;
        }
    }
}

// Get statistics
PipelineStats PipelineContext::getStats() const {
    std::lock_guard<std::mutex> lock(_stats_mutex);
    return _stats;
}

} // namespace npu_pipeline
