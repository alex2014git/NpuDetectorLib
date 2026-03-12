#ifndef _NPU_PIPELINE_HPP_
#define _NPU_PIPELINE_HPP_

#include "pipeline/npu_pipeline_types.hpp"
#include "pipeline/npu_pipeline_context.hpp"
#include "pipeline/npu_pipeline_edge.hpp"
#include "pipeline/npu_pipeline_node.hpp"
#include "pipeline/npu_pipeline_scheduler.hpp"
#include "pipeline/npu_pipeline_graph.hpp"
#include "npu.hpp"

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <atomic>
#include <mutex>
#include <optional>
#include <numeric>

namespace npu_pipeline {

// Pipeline configuration
struct PipelineConfig {
    SchedulerConfig scheduler;
    std::string pipeline_name = "default";
    bool enable_profiling = false;
    bool auto_cleanup = true;           // Cleanup old frames automatically
    size_t max_pending_frames = 16;     // Max frames in pipeline
};

// Main pipeline orchestrator
class NpuPipeline {
public:
    NpuPipeline();
    ~NpuPipeline();

    // Disable copy
    NpuPipeline(const NpuPipeline&) = delete;
    NpuPipeline& operator=(const NpuPipeline&) = delete;

    // Enable move
    NpuPipeline(NpuPipeline&&) = default;
    NpuPipeline& operator=(NpuPipeline&&) = default;

    // Initialize pipeline
    int initialize(const PipelineConfig& config);

    // Build pipeline from graph
    int build(std::unique_ptr<PipelineGraph> graph);

    // Build with configuration
    int build(const PipelineConfig& config);

    // Add node to pipeline (before build)
    template<typename T, typename... Args>
    int addNode(const std::string& node_id, Args&&... args);

    // Add NPU inference node
    int addNpuNode(const std::string& node_id,
                   int algorithm_type,
                   const std::string& model_config);

    // Add edge
    int addEdge(const PipelineEdge& edge);

    // Convenience edge methods
    int addEdge(const std::string& from, const std::string& to);

    // Submit frame for processing (non-blocking, returns frame_id)
    uint64_t submit(std::shared_ptr<image_share_t> frame,
                    std::function<void(const FrameResults&)> callback = nullptr);

    // Process frame synchronously (blocking)
    FrameOutput process(std::shared_ptr<image_share_t> frame);

    // Process frame with explicit frame_id
    FrameOutput process(uint64_t frame_id, std::shared_ptr<image_share_t> frame);

    // Wait for specific frame
    void waitForFrame(uint64_t frame_id);

    // Wait for all pending frames
    void waitForAll();

    // Get results for frame (non-blocking)
    std::optional<FrameOutput> getResults(uint64_t frame_id);

    // Check if frame is complete
    bool isFrameComplete(uint64_t frame_id) const;

    // Get pipeline statistics
    PipelineStats getStats() const;

    // Reset statistics
    void resetStats();

    // Release all resources
    void release();

    // Stop pipeline
    void stop();

    // Resume pipeline
    void resume();

    // Check if pipeline is running
    bool isRunning() const { return _running.load(); }

    // Check if pipeline is built
    bool isBuilt() const { return _built.load(); }

    // Set profiling enabled/disabled
    void setProfiling(bool enabled);

    // Get profiling data
    std::string getProfilingReport() const;

    // Pipeline info
    size_t getNodeCount() const;
    size_t getEdgeCount() const;
    std::vector<std::string> getNodeNames() const;

    // Context access (for advanced use)
    PipelineContext& getContext() { return _context; }
    const PipelineContext& getContext() const { return _context; }

private:
    // Internal members
    PipelineConfig _config;
    std::unique_ptr<PipelineGraph> _graph;
    std::unique_ptr<PipelineScheduler> _scheduler;
    PipelineContext _context;

    // State
    std::atomic<bool> _initialized{false};
    std::atomic<bool> _built{false};
    std::atomic<bool> _running{true};
    std::atomic<uint64_t> _frame_counter{0};

    // Synchronization
    mutable std::mutex _mutex;
    std::condition_variable _cv;

    // Builder state (pre-build)
    std::unique_ptr<PipelineGraphBuilder> _builder;

    // Profiling
    bool _profiling_enabled = false;
    std::unordered_map<std::string, std::vector<double>> _profiling_data;
    mutable std::mutex _profiling_mutex;

    // Cleanup old frames
    void cleanupOldFrames();

    // Generate next frame ID
    uint64_t nextFrameId();
};

// Template implementation for addNode
template<typename T, typename... Args>
int NpuPipeline::addNode(const std::string& node_id, Args&&... args) {
    if (_built.load()) {
        return -1;  // Cannot add nodes after build
    }

    if (!_builder) {
        _builder = std::make_unique<PipelineGraphBuilder>();
    }

    _builder->addNode<T>(node_id, std::forward<Args>(args)...);
    return 0;
}

// Factory function for creating pipelines
std::unique_ptr<NpuPipeline> CreatePipeline(const PipelineConfig& config = PipelineConfig{});

// Predefined pipeline configurations
namespace Presets {

    // Detection + LPR pipeline configuration
    PipelineConfig DetectionLpr(size_t batch_size = 8);

    // Detection + Tracking + LPR pipeline configuration
    PipelineConfig DetectionTrackingLpr(size_t batch_size = 4);

    // Detection + Classification pipeline configuration
    PipelineConfig DetectionClassification(size_t batch_size = 16);

    // Single detection pipeline
    PipelineConfig SingleDetection();

} // namespace Presets

} // namespace npu_pipeline

// C-compatible API (optional)
extern "C" {

    // Opaque handle
    typedef struct NpuPipelineHandle* npu_pipeline_t;

    // Create/destroy
    npu_pipeline_t npu_pipeline_create(const char* config_json);
    void npu_pipeline_destroy(npu_pipeline_t pipeline);

    // Build
    int npu_pipeline_build(npu_pipeline_t pipeline);

    // Add nodes
    int npu_pipeline_add_npu_node(npu_pipeline_t pipeline,
                                   const char* node_id,
                                   int algorithm_type,
                                   const char* model_config);

    // Add edges
    int npu_pipeline_add_edge(npu_pipeline_t pipeline,
                               const char* from_node,
                               const char* to_node,
                               int transform_type);

    // Process
    uint64_t npu_pipeline_submit(npu_pipeline_t pipeline,
                                  image_share_t* frame);

    // Wait
    void npu_pipeline_wait(npu_pipeline_t pipeline, uint64_t frame_id);
    void npu_pipeline_wait_all(npu_pipeline_t pipeline);

} // extern "C"

#endif // _NPU_PIPELINE_HPP_
