#include "npu_pipeline.hpp"
#include "pipeline/npu_pipeline_graph.hpp"
#include "pipeline/npu_pipeline_scheduler.hpp"
#include <opencv2/opencv.hpp>
#include <optional>
#include <numeric>
#include <algorithm>

namespace npu_pipeline {

// Constructor
NpuPipeline::NpuPipeline()
    : _graph(nullptr)
    , _scheduler(nullptr)
    , _builder(nullptr) {
}

// Destructor
NpuPipeline::~NpuPipeline() {
    release();
}

// Initialize pipeline
int NpuPipeline::initialize(const PipelineConfig& config) {
    _config = config;
    _profiling_enabled = config.enable_profiling;
    _initialized.store(true);
    return 0;
}

// Build pipeline from graph
int NpuPipeline::build(std::unique_ptr<PipelineGraph> graph) {
    if (!graph || graph->empty()) {
        return -1;
    }

    // Validate graph
    std::string error_msg;
    if (!graph->validate(error_msg)) {
        return -1;
    }

    _graph = std::move(graph);

    // Initialize scheduler
    _scheduler = std::make_unique<PipelineScheduler>();
    _scheduler->initialize(*_graph, _config.scheduler);

    _built.store(true);
    return 0;
}

// Build with configuration
int NpuPipeline::build(const PipelineConfig& config) {
    _config = config;

    if (!_builder) {
        return -1;
    }

    // Build graph from builder
    std::string error_msg;
    if (!_builder->validate(error_msg)) {
        return -1;
    }

    _graph = _builder->build();
    _builder.reset();

    if (!_graph || _graph->empty()) {
        return -1;
    }

    // Initialize scheduler
    _scheduler = std::make_unique<PipelineScheduler>();
    _scheduler->initialize(*_graph, _config.scheduler);

    _built.store(true);
    return 0;
}

// Add NPU inference node
int NpuPipeline::addNpuNode(const std::string& node_id,
                            int algorithm_type,
                            const std::string& model_config) {
    if (_built.load()) {
        return -1;
    }

    if (!_builder) {
        _builder = std::make_unique<PipelineGraphBuilder>();
    }

    _builder->addNpuNode(node_id, algorithm_type, model_config);
    return 0;
}

// Add edge
int NpuPipeline::addEdge(const PipelineEdge& edge) {
    if (_built.load()) {
        return -1;
    }

    if (!_builder) {
        _builder = std::make_unique<PipelineGraphBuilder>();
    }

    _builder->addEdge(edge);
    return 0;
}

// Add edge (simple)
int NpuPipeline::addEdge(const std::string& from, const std::string& to) {
    return addEdge(Edge::passThrough(from, to));
}

// Submit frame for processing
uint64_t NpuPipeline::submit(std::shared_ptr<image_share_t> frame,
                             std::function<void(const FrameResults&)> callback) {
    if (!_built.load() || !_running.load()) {
        return 0;
    }

    uint64_t frame_id = nextFrameId();

    // Initialize frame in context
    _context.initializeFrame(frame_id, frame, callback);

    // Submit to scheduler
    _scheduler->submit(frame_id, _context);

    // Cleanup old frames if auto-cleanup enabled
    if (_config.auto_cleanup) {
        cleanupOldFrames();
    }

    return frame_id;
}

// Process frame synchronously
FrameOutput NpuPipeline::process(std::shared_ptr<image_share_t> frame) {
    uint64_t frame_id = nextFrameId();
    return process(frame_id, frame);
}

// Process frame with explicit frame_id
FrameOutput NpuPipeline::process(uint64_t frame_id, std::shared_ptr<image_share_t> frame) {
    if (!_built.load()) {
        return FrameOutput{};
    }

    // Initialize frame
    _context.initializeFrame(frame_id, frame);

    // Submit and wait
    std::promise<void> completion;
    std::future<void> future = completion.get_future();

    auto& frame_results = _context.getFrame(frame_id);
    frame_results.completion_callback = [&completion](const FrameResults&) {
        completion.set_value();
    };

    _scheduler->submit(frame_id, _context);

    // Wait for completion
    future.wait();

    // Return results by copying to FrameOutput
    FrameOutput results;
    results.metadata = frame_results.metadata;
    results.node_outputs = std::move(frame_results.node_outputs);
    results.track_states = std::move(frame_results.track_states);

    return results;
}

// Wait for specific frame
void NpuPipeline::waitForFrame(uint64_t frame_id) {
    if (!_built.load()) return;
    _context.waitForFrame(frame_id);
}

// Wait for all pending frames
void NpuPipeline::waitForAll() {
    if (!_scheduler) return;
    _scheduler->waitForAll();
}

// Get results for frame
std::optional<FrameOutput> NpuPipeline::getResults(uint64_t frame_id) {
    if (!_context.hasFrame(frame_id)) {
        return std::nullopt;
    }

    auto& frame = _context.getFrame(frame_id);
    if (!frame.isComplete()) {
        return std::nullopt;
    }

    FrameOutput copy;
    copy.metadata = frame.metadata;
    copy.node_outputs = frame.node_outputs;
    copy.track_states = frame.track_states;
    return copy;
}

// Check if frame is complete
bool NpuPipeline::isFrameComplete(uint64_t frame_id) const {
    return _context.hasFrame(frame_id) &&
           const_cast<PipelineContext&>(_context).getFrame(frame_id).isComplete();
}

// Get pipeline statistics
PipelineStats NpuPipeline::getStats() const {
    if (_scheduler) {
        return _scheduler->getStats();
    }
    return PipelineStats{};
}

// Reset statistics
void NpuPipeline::resetStats() {
    if (_scheduler) {
        _scheduler->resetStats();
    }
}

// Release all resources
void NpuPipeline::release() {
    stop();

    if (_scheduler) {
        _scheduler->waitForAll();
        _scheduler.reset();
    }

    _graph.reset();
    _builder.reset();
    _built.store(false);
    _initialized.store(false);
}

// Stop pipeline
void NpuPipeline::stop() {
    _running.store(false);
    if (_scheduler) {
        _scheduler->stop();
    }
}

// Resume pipeline
void NpuPipeline::resume() {
    _running.store(true);
    if (_scheduler) {
        // Scheduler restart logic if needed
    }
}

// Set profiling enabled/disabled
void NpuPipeline::setProfiling(bool enabled) {
    _profiling_enabled = enabled;
}

// Get profiling data
std::string NpuPipeline::getProfilingReport() const {
    if (!_profiling_enabled) {
        return "Profiling disabled";
    }

    std::lock_guard<std::mutex> lock(_profiling_mutex);
    std::string report = "Pipeline Profiling Report\n";
    report += "========================\n";

    for (const auto& [node, latencies] : _profiling_data) {
        if (latencies.empty()) continue;

        double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        double max_lat = *std::max_element(latencies.begin(), latencies.end());
        double min_lat = *std::min_element(latencies.begin(), latencies.end());

        report += node + ":\n";
        report += "  Avg: " + std::to_string(avg) + " ms\n";
        report += "  Min: " + std::to_string(min_lat) + " ms\n";
        report += "  Max: " + std::to_string(max_lat) + " ms\n";
    }

    return report;
}

// Pipeline info
size_t NpuPipeline::getNodeCount() const {
    if (_graph) {
        return _graph->size();
    }
    return _builder ? _builder->nodeCount() : 0;
}

size_t NpuPipeline::getEdgeCount() const {
    if (_builder) {
        return _builder->edgeCount();
    }
    return 0;
}

std::vector<std::string> NpuPipeline::getNodeNames() const {
    std::vector<std::string> names;
    if (_graph) {
        auto nodes = _graph->getAllNodes();
        names.reserve(nodes.size());
        for (const auto& node : nodes) {
            names.push_back(node->getName());
        }
    }
    return names;
}

// Cleanup old frames
void NpuPipeline::cleanupOldFrames() {
    if (_frame_counter.load() > _config.max_pending_frames) {
        uint64_t cleanup_before = _frame_counter.load() - _config.max_pending_frames;
        _context.cleanupFramesBefore(cleanup_before);
    }
}

// Generate next frame ID
uint64_t NpuPipeline::nextFrameId() {
    return ++_frame_counter;
}

// Factory function
std::unique_ptr<NpuPipeline> CreatePipeline(const PipelineConfig& config) {
    auto pipeline = std::make_unique<NpuPipeline>();
    if (pipeline->initialize(config) != 0) {
        return nullptr;
    }
    return pipeline;
}

// Preset configurations
namespace Presets {

PipelineConfig DetectionLpr(size_t batch_size) {
    PipelineConfig config;
    config.scheduler.strategy = SchedulerConfig::BATCHED;
    config.scheduler.batch_timeout = std::chrono::milliseconds(5);
    config.pipeline_name = "detection_lpr";
    return config;
}

PipelineConfig DetectionTrackingLpr(size_t batch_size) {
    PipelineConfig config;
    config.scheduler.strategy = SchedulerConfig::PARALLEL;
    config.scheduler.thread_pool_size = 4;
    config.pipeline_name = "detection_tracking_lpr";
    return config;
}

PipelineConfig DetectionClassification(size_t batch_size) {
    PipelineConfig config;
    config.scheduler.strategy = SchedulerConfig::BATCHED;
    config.scheduler.batch_timeout = std::chrono::milliseconds(5);
    config.pipeline_name = "detection_classification";
    return config;
}

PipelineConfig SingleDetection() {
    PipelineConfig config;
    config.scheduler.strategy = SchedulerConfig::SEQUENTIAL;
    config.pipeline_name = "single_detection";
    return config;
}

} // namespace Presets

} // namespace npu_pipeline
