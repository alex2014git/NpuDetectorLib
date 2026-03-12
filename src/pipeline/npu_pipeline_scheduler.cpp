#include "pipeline/npu_pipeline_scheduler.hpp"
#include "pipeline/npu_pipeline_graph.hpp"
#include "pipeline/npu_pipeline_node.hpp"
#include <algorithm>

namespace npu_pipeline {

// Constructor
PipelineScheduler::PipelineScheduler()
    : _thread_pool(nullptr)
    , _config{} {
}

// Destructor
PipelineScheduler::~PipelineScheduler() {
    stop();
}

// Initialize scheduler with graph
void PipelineScheduler::initialize(const PipelineGraph& graph, const SchedulerConfig& config) {
    _config = config;

    // Create thread pool for parallel/batched execution
    if (_config.strategy == SchedulerConfig::PARALLEL ||
        _config.strategy == SchedulerConfig::BATCHED) {
        _thread_pool = std::make_unique<ThreadPool>(_config.thread_pool_size);
    }

    // Build execution order
    buildExecutionOrder(graph);

    _initialized.store(true);
    _running.store(true);
}

// Build execution order from graph
void PipelineScheduler::buildExecutionOrder(const PipelineGraph& graph) {
    _execution_order = topologicalSort(graph);

    // Build node index
    for (size_t i = 0; i < _execution_order.size(); ++i) {
        _node_index[_execution_order[i].node_id] = i;
    }
}

// Topological sort for node ordering
std::vector<NodeTask> PipelineScheduler::topologicalSort(const PipelineGraph& graph) {
    std::vector<NodeTask> result;

    auto sorted_ids = graph.topologicalSort();
    result.reserve(sorted_ids.size());

    for (const auto& node_id : sorted_ids) {
        NodeTask task;
        task.node_id = node_id;
        task.node = graph.getNode(node_id);

        if (!task.node) continue;

        task.input_edges = graph.getInputEdges(node_id);
        task.output_edges = graph.getOutputEdges(node_id);

        for (const auto& edge : task.input_edges) {
            task.input_nodes.push_back(edge.from_node);
        }

        for (const auto& edge : task.output_edges) {
            task.output_nodes.push_back(edge.to_node);
        }

        task.supports_batching = task.node->supportsBatching();
        task.preferred_batch_size = task.node->getPreferredBatchSize();

        result.push_back(std::move(task));
    }

    return result;
}

// Submit frame for processing
void PipelineScheduler::submit(uint64_t frame_id, PipelineContext& ctx) {
    submit(frame_id, ctx, nullptr);
}

// Submit with callback
void PipelineScheduler::submit(uint64_t frame_id,
                               PipelineContext& ctx,
                               std::function<void(const FrameResults&)> callback) {
    if (!_running.load()) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(_frames_mutex);
        _pending_frames.insert(frame_id);
    }

    // Set expected node count
    ctx.setExpectedNodes(frame_id, static_cast<int>(_execution_order.size()));

    // Set completion callback
    if (callback) {
        auto& frame = ctx.getFrame(frame_id);
        frame.completion_callback = std::move(callback);
    }

    // Execute based on strategy
    switch (_config.strategy) {
        case SchedulerConfig::SEQUENTIAL:
            runSequential(frame_id, ctx);
            break;
        case SchedulerConfig::PARALLEL:
            runParallel(frame_id, ctx);
            break;
        case SchedulerConfig::BATCHED:
            runBatched(frame_id, ctx);
            break;
    }
}

// Sequential execution
void PipelineScheduler::runSequential(uint64_t frame_id, PipelineContext& ctx) {
    for (const auto& task : _execution_order) {
        executeNode(task, frame_id, ctx);
    }

    {
        std::lock_guard<std::mutex> lock(_frames_mutex);
        _pending_frames.erase(frame_id);
        _completed_frames.insert(frame_id);
    }

    // Mark completion
    auto& frame = ctx.getFrame(frame_id);
    frame.markComplete();

    _total_frames.fetch_add(1);
}

// Parallel execution
void PipelineScheduler::runParallel(uint64_t frame_id, PipelineContext& ctx) {
    std::vector<std::future<void>> futures;
    std::unordered_map<std::string, std::future<void>> node_futures;

    for (const auto& task : _execution_order) {
        // Wait for input nodes to complete
        auto future = _thread_pool->enqueue([this, &task, frame_id, &ctx, &node_futures]() {
            // Wait for dependencies
            for (const auto& input_node : task.input_nodes) {
                auto it = node_futures.find(input_node);
                if (it != node_futures.end()) {
                    it->second.wait();
                }
            }

            // Execute this node
            executeNode(task, frame_id, ctx);
        });

        node_futures[task.node_id] = std::move(future);
    }

    // Wait for all nodes to complete
    for (auto& [id, future] : node_futures) {
        future.wait();
    }

    {
        std::lock_guard<std::mutex> lock(_frames_mutex);
        _pending_frames.erase(frame_id);
        _completed_frames.insert(frame_id);
    }

    _total_frames.fetch_add(1);
}

// Batched execution
void PipelineScheduler::runBatched(uint64_t frame_id, PipelineContext& ctx) {
    // Similar to parallel but with batch accumulation
    std::vector<std::future<void>> futures;
    std::unordered_map<std::string, std::future<void>> node_futures;

    for (const auto& task : _execution_order) {
        auto future = _thread_pool->enqueue([this, &task, frame_id, &ctx, &node_futures]() {
            // Wait for dependencies
            for (const auto& input_node : task.input_nodes) {
                auto it = node_futures.find(input_node);
                if (it != node_futures.end()) {
                    it->second.wait();
                }
            }

            // Execute with batching support
            if (task.supports_batching) {
                executeNodeBatched(task, frame_id, ctx);
            } else {
                executeNode(task, frame_id, ctx);
            }
        });

        node_futures[task.node_id] = std::move(future);
    }

    // Wait for all nodes
    for (auto& [id, future] : node_futures) {
        future.wait();
    }

    {
        std::lock_guard<std::mutex> lock(_frames_mutex);
        _pending_frames.erase(frame_id);
        _completed_frames.insert(frame_id);
    }

    _total_frames.fetch_add(1);
}

// Execute single node
void PipelineScheduler::executeNode(const NodeTask& task,
                                    uint64_t frame_id,
                                    PipelineContext& ctx) {
    if (!task.node) return;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Gather inputs from upstream nodes
    auto inputs = gatherInputs(task, frame_id, ctx);

    // Apply input edge transforms
    auto& frame = ctx.getFrame(frame_id);
    inputs = applyTransforms(inputs, task.input_edges, frame, ctx);

    // Execute node
    std::vector<PipelineObject> outputs;
    if (task.supports_batching && inputs.size() > 1) {
        std::vector<uint64_t> frame_ids(inputs.size(), frame_id);
        outputs = task.node->processBatch(inputs, frame_ids, ctx);
    } else {
        // Process one by one
        outputs.reserve(inputs.size());
        for (const auto& input : inputs) {
            outputs.push_back(task.node->processObject(input, frame, ctx));
        }
    }

    // Write outputs
    ctx.writeNodeOutput(frame_id, task.node_id, std::move(outputs));
    ctx.markNodeCompleted(frame_id, task.node_id);

    // Update stats
    if (_config.enable_profiling) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto latency_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        std::lock_guard<std::mutex> lock(_stats_mutex);
        _stats.node_latency_ms[task.node_id] = latency_ms;
    }
}

// Execute node with batching
void PipelineScheduler::executeNodeBatched(const NodeTask& task,
                                           uint64_t frame_id,
                                           PipelineContext& ctx) {
    if (!task.node || !task.supports_batching) {
        executeNode(task, frame_id, ctx);
        return;
    }

    // Gather inputs
    auto inputs = gatherInputs(task, frame_id, ctx);
    auto& frame = ctx.getFrame(frame_id);
    inputs = applyTransforms(inputs, task.input_edges, frame, ctx);

    // Add to batch accumulator
    for (auto& obj : inputs) {
        BatchItem item;
        item.frame_id = frame_id;
        item.object_id = obj.object_id;
        item.object = std::move(obj);

        bool batch_ready = ctx.accumulateForBatch(
            task.node_id, std::move(item),
            task.preferred_batch_size, _config.batch_timeout);

        if (batch_ready) {
            processBatchAccumulator(task, ctx);
        }
    }

    // Process remaining items in batch (for the last frame)
    processBatchAccumulator(task, ctx);

    ctx.markNodeCompleted(frame_id, task.node_id);
}

// Process batch accumulator for a node
void PipelineScheduler::processBatchAccumulator(const NodeTask& task,
                                                PipelineContext& ctx) {
    if (!ctx.isBatchReady(task.node_id)) {
        return;
    }

    auto batch = ctx.getBatch(task.node_id);
    if (batch.isEmpty()) {
        return;
    }

    // Extract frame_ids and objects
    std::vector<PipelineObject> objects;
    std::vector<uint64_t> frame_ids;
    objects.reserve(batch.size());
    frame_ids.reserve(batch.size());

    for (const auto& item : batch.items) {
        objects.push_back(item.object);
        frame_ids.push_back(item.frame_id);
    }

    // Execute batch inference
    auto outputs = task.node->processBatch(objects, frame_ids, ctx);

    // Write outputs back to respective frames
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::vector<PipelineObject> single_output = {outputs[i]};
        ctx.writeNodeOutput(frame_ids[i], task.node_id, std::move(single_output));
    }
}

// Check if all inputs are ready
bool PipelineScheduler::areInputsReady(const NodeTask& task,
                                       uint64_t frame_id,
                                       const PipelineContext& ctx) {
    for (const auto& node_id : task.input_nodes) {
        const auto& outputs = ctx.readNodeOutput(frame_id, node_id);
        // This is a simple check - might need refinement
    }
    return true;
}

// Get inputs for a node
std::vector<PipelineObject> PipelineScheduler::gatherInputs(const NodeTask& task,
                                                            uint64_t frame_id,
                                                            PipelineContext& ctx) {
    std::vector<PipelineObject> all_inputs;

    for (const auto& node_id : task.input_nodes) {
        const auto& inputs = ctx.readNodeOutput(frame_id, node_id);
        all_inputs.insert(all_inputs.end(), inputs.begin(), inputs.end());
    }

    return all_inputs;
}

// Apply edge transforms
std::vector<PipelineObject> PipelineScheduler::applyTransforms(
    const std::vector<PipelineObject>& inputs,
    const std::vector<PipelineEdge>& edges,
    const FrameResults& frame,
    PipelineContext& ctx) {

    if (inputs.empty()) return {};

    std::vector<PipelineObject> result = inputs;

    for (const auto& edge : edges) {
        result = edge.execute(result, frame, ctx);
    }

    return result;
}

// Wait for specific frame
void PipelineScheduler::waitForFrame(uint64_t frame_id) {
    std::unique_lock<std::mutex> lock(_frames_mutex);
    // Wait until frame is no longer pending
    // This is simplified - in practice would use condition variable
}

// Wait for all pending frames
void PipelineScheduler::waitForAll() {
    if (_thread_pool) {
        // Wait for thread pool to drain
    }

    std::unique_lock<std::mutex> lock(_frames_mutex);
    // Wait until no pending frames
}

// Stop scheduler
void PipelineScheduler::stop() {
    _running.store(false);
    if (_thread_pool) {
        _thread_pool->shutdown();
    }
}

// Get statistics
PipelineStats PipelineScheduler::getStats() const {
    std::lock_guard<std::mutex> lock(_stats_mutex);
    PipelineStats stats = _stats;
    stats.frames_processed = _total_frames.load();
    return stats;
}

// Reset statistics
void PipelineScheduler::resetStats() {
    std::lock_guard<std::mutex> lock(_stats_mutex);
    _stats = PipelineStats{};
    _total_frames.store(0);
}

} // namespace npu_pipeline
