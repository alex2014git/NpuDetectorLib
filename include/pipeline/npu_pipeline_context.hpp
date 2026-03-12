#ifndef _NPU_PIPELINE_CONTEXT_HPP_
#define _NPU_PIPELINE_CONTEXT_HPP_

#include "npu_pipeline_types.hpp"
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <chrono>
#include <functional>

namespace npu_pipeline {

// Forward declarations
class PipelineContext;

// Results for a single frame - nodes append their outputs here
struct FrameResults {
    FrameMetadata metadata;

    // Source frame (shared_ptr = zero copy across all nodes)
    std::shared_ptr<image_share_t> source_frame;

    // Node outputs indexed by node_id
    // Each node writes: frame.node_outputs[node_id] = my_results
    std::unordered_map<std::string, std::vector<PipelineObject>> node_outputs;

    // Track state (persisted across frames)
    std::unordered_map<uint64_t, TrackState> track_states;

    // Execution tracking
    std::atomic<int> pending_inputs{0};     // Nodes still producing input
    std::atomic<int> completed_nodes{0};    // Nodes finished
    std::atomic<bool> execution_complete{false};

    // Completion callback
    std::function<void(const FrameResults&)> completion_callback;

    // Synchronization
    std::mutex results_mutex;
    std::condition_variable completion_cv;

    FrameResults() = default;
    FrameResults(const FrameResults&) = delete;
    FrameResults& operator=(const FrameResults&) = delete;
    FrameResults(FrameResults&&) = default;
    FrameResults& operator=(FrameResults&&) = default;

    // Clone for returning results
    std::unique_ptr<FrameResults> clone() const {
        auto copy = std::make_unique<FrameResults>();
        copy->metadata = metadata;
        copy->source_frame = source_frame;
        copy->node_outputs = node_outputs;
        copy->track_states = track_states;
        return copy;
    }

    // Get outputs from a specific node
    const std::vector<PipelineObject>& getNodeOutput(const std::string& node_id) const {
        auto it = node_outputs.find(node_id);
        if (it != node_outputs.end()) {
            return it->second;
        }
        static const std::vector<PipelineObject> empty;
        return empty;
    }

    // Check if all nodes have completed
    bool isComplete() const {
        return execution_complete.load(std::memory_order_acquire);
    }

    // Wait for frame completion
    void waitForCompletion() {
        if (execution_complete.load(std::memory_order_acquire)) {
            return;
        }
        std::unique_lock<std::mutex> lock(results_mutex);
        completion_cv.wait(lock, [this] {
            return execution_complete.load(std::memory_order_acquire);
        });
    }

    // Mark frame as complete
    void markComplete() {
        execution_complete.store(true, std::memory_order_release);
        completion_cv.notify_all();
        if (completion_callback) {
            completion_callback(*this);
        }
    }
};

// Batch accumulator for NPU efficiency
struct BatchAccumulator {
    std::vector<BatchItem> items;
    std::chrono::steady_clock::time_point first_add;
    bool has_timeout = false;
    std::chrono::milliseconds timeout{5};
    size_t max_batch_size = 1;

    bool isReady() const {
        if (items.empty()) return false;
        if (items.size() >= max_batch_size) return true;
        if (has_timeout) {
            auto elapsed = std::chrono::steady_clock::now() - first_add;
            return elapsed >= timeout;
        }
        return false;
    }

    bool isEmpty() const { return items.empty(); }

    size_t size() const { return items.size(); }

    void clear() {
        items.clear();
        has_timeout = false;
    }

    void add(const BatchItem& item) {
        if (items.empty()) {
            first_add = std::chrono::steady_clock::now();
        }
        items.push_back(item);
    }

    void add(BatchItem&& item) {
        if (items.empty()) {
            first_add = std::chrono::steady_clock::now();
        }
        items.push_back(std::move(item));
    }
};

// Shared context - single instance per pipeline
class PipelineContext {
public:
    PipelineContext() = default;
    ~PipelineContext() = default;

    PipelineContext(const PipelineContext&) = delete;
    PipelineContext& operator=(const PipelineContext&) = delete;

    // Get or create frame results
    FrameResults& getFrame(uint64_t frame_id);
    const FrameResults& getFrame(uint64_t frame_id) const;

    // Check if frame exists
    bool hasFrame(uint64_t frame_id) const;

    // Remove a frame from context
    void removeFrame(uint64_t frame_id);

    // Called by nodes to write output
    void writeNodeOutput(uint64_t frame_id,
                         const std::string& node_id,
                         std::vector<PipelineObject> objects);

    // Called by nodes to read input (from upstream node)
    const std::vector<PipelineObject>& readNodeOutput(
        uint64_t frame_id,
        const std::string& upstream_node_id) const;

    // Get track state (creates if not exists)
    TrackState& getTrackState(uint64_t frame_id, uint64_t track_id);

    // Update track state
    void updateTrackState(uint64_t frame_id, uint64_t track_id,
                          const TrackState& state);

    // Batch accumulation - critical for NPU efficiency
    // Add to batch, returns true if batch is ready
    bool accumulateForBatch(const std::string& node_id,
                            BatchItem item,
                            size_t max_batch_size,
                            std::chrono::milliseconds timeout);

    // Get accumulated batch (moves items out)
    BatchAccumulator getBatch(const std::string& node_id);

    // Peek at batch without clearing
    const BatchAccumulator& peekBatch(const std::string& node_id) const;

    // Check if batch is ready
    bool isBatchReady(const std::string& node_id) const;

    // Clear batch accumulator
    void clearBatch(const std::string& node_id);

    // Initialize frame with source image
    void initializeFrame(uint64_t frame_id,
                         std::shared_ptr<image_share_t> source_frame,
                         std::function<void(const FrameResults&)> callback = nullptr);

    // Set number of expected nodes for completion tracking
    void setExpectedNodes(uint64_t frame_id, int num_nodes);

    // Mark node as completed for a frame
    void markNodeCompleted(uint64_t frame_id, const std::string& node_id);

    // Wait for specific frame
    void waitForFrame(uint64_t frame_id);

    // Get frame count
    size_t frameCount() const;

    // Cleanup old frames (frames older than given frame_id)
    void cleanupFramesBefore(uint64_t frame_id);

    // Get statistics
    PipelineStats getStats() const;

private:
    mutable std::mutex _mutex;
    std::unordered_map<uint64_t, std::unique_ptr<FrameResults>> _frames;
    std::unordered_map<std::string, BatchAccumulator> _batch_accumulators;
    std::unordered_map<uint64_t, std::unordered_set<std::string>> _completed_nodes;

    // Statistics
    mutable std::mutex _stats_mutex;
    PipelineStats _stats;
};

} // namespace npu_pipeline

#endif // _NPU_PIPELINE_CONTEXT_HPP_
