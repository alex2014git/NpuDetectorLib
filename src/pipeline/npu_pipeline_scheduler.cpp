#include "pipeline/npu_pipeline_scheduler.hpp"
#include "pipeline/npu_pipeline_graph.hpp"
#include "pipeline/npu_pipeline_node.hpp"
#include "pipeline/transform_engine.hpp"
#include "common/debug_logger.hpp"
#include <algorithm>

namespace npu_pipeline {

PipelineScheduler::PipelineScheduler() : _thread_pool(nullptr), _config{} {}
PipelineScheduler::~PipelineScheduler() { stop(); }

void PipelineScheduler::initialize(const PipelineGraph& graph, const SchedulerConfig& config) {
    if ((config.strategy == SchedulerConfig::PARALLEL || config.strategy == SchedulerConfig::BATCHED)
        && (config.thread_pool_size == 0 || config.max_concurrent_frames == 0)) {
        throw std::invalid_argument("PARALLEL/BATCHED strategy requires thread_pool_size > 0 and max_concurrent_frames > 0");
    }
    if (config.strategy == SchedulerConfig::BATCHED && config.batch_timeout.count() <= 0) {
        throw std::invalid_argument("BATCHED strategy requires batch_timeout > 0");
    }

    _config = config;
    if (_config.strategy == SchedulerConfig::PARALLEL || _config.strategy == SchedulerConfig::BATCHED) {
        _thread_pool = std::make_unique<ThreadPool>(_config.thread_pool_size);
    }

    buildExecutionOrder(graph);
    if (_execution_order.empty() && !graph.empty()) {
        throw std::runtime_error("Failed to build execution order - graph may have cycles");
    }
    _initialized.store(true);
    _running.store(true);
}

void PipelineScheduler::buildExecutionOrder(const PipelineGraph& graph) {
    _execution_order = topologicalSort(graph);
    for (size_t i = 0; i < _execution_order.size(); ++i) {
        _node_index[_execution_order[i].node_id] = i;
    }
}

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
        for (const auto& edge : task.input_edges) task.input_nodes.push_back(edge.from_node);
        for (const auto& edge : task.output_edges) task.output_nodes.push_back(edge.to_node);
        task.supports_batching = task.node->supportsBatching();
        task.preferred_batch_size = task.node->getPreferredBatchSize();
        result.push_back(std::move(task));
    }
    return result;
}

void PipelineScheduler::submit(uint64_t frame_id, PipelineContext& ctx) {
    submit(frame_id, ctx, nullptr);
}

void PipelineScheduler::submit(uint64_t frame_id, PipelineContext& ctx,
                               std::function<void(const FrameResults&)> callback) {
    if (!_running.load()) return;
    {
        std::lock_guard<std::mutex> lock(_frames_mutex);
        _pending_frames.insert(frame_id);
    }
    ctx.setExpectedNodes(frame_id, static_cast<int>(_execution_order.size()));
    if (callback) {
        auto& frame = ctx.getFrame(frame_id);
        frame.completion_callback = std::move(callback);
    }
    switch (_config.strategy) {
        case SchedulerConfig::SEQUENTIAL: runSequential(frame_id, ctx); break;
        case SchedulerConfig::PARALLEL: runParallel(frame_id, ctx); break;
        case SchedulerConfig::BATCHED: runBatched(frame_id, ctx); break;
    }
}

void PipelineScheduler::runSequential(uint64_t frame_id, PipelineContext& ctx) {
    for (const auto& task : _execution_order) executeNode(task, frame_id, ctx);
    {
        std::lock_guard<std::mutex> lock(_frames_mutex);
        _pending_frames.erase(frame_id);
        _completed_frames.insert(frame_id);
    }
    ctx.getFrame(frame_id).markComplete();
    _total_frames.fetch_add(1);
}

void PipelineScheduler::runParallel(uint64_t frame_id, PipelineContext& ctx) {
    std::unordered_map<std::string, std::shared_ptr<std::promise<void>>> node_promises;
    std::unordered_map<std::string, std::future<void>> node_futures;
    for (const auto& task : _execution_order) {
        auto promise = std::make_shared<std::promise<void>>();
        node_promises[task.node_id] = promise;
        node_futures[task.node_id] = promise->get_future();
    }
    for (const auto& task : _execution_order) {
        _thread_pool->enqueue([this, &task, frame_id, &ctx, &node_promises, &node_futures]() {
            for (const auto& input_node : task.input_nodes) {
                auto it = node_futures.find(input_node);
                if (it != node_futures.end()) it->second.wait();
            }
            executeNode(task, frame_id, ctx);
            auto it = node_promises.find(task.node_id);
            if (it != node_promises.end()) it->second->set_value();
        });
    }
    for (auto& [id, future] : node_futures) future.wait();
    ctx.getFrame(frame_id).markComplete();
    {
        std::lock_guard<std::mutex> lock(_frames_mutex);
        _pending_frames.erase(frame_id);
        _completed_frames.insert(frame_id);
    }
    _frame_completion_cv.notify_all();
    _total_frames.fetch_add(1);
}

void PipelineScheduler::runBatched(uint64_t frame_id, PipelineContext& ctx) {
    std::unordered_map<std::string, std::shared_ptr<std::promise<void>>> node_promises;
    std::unordered_map<std::string, std::future<void>> node_futures;
    for (const auto& task : _execution_order) {
        auto promise = std::make_shared<std::promise<void>>();
        node_promises[task.node_id] = promise;
        node_futures[task.node_id] = promise->get_future();
    }
    for (const auto& task : _execution_order) {
        _thread_pool->enqueue([this, &task, frame_id, &ctx, &node_promises, &node_futures]() {
            for (const auto& input_node : task.input_nodes) {
                auto it = node_futures.find(input_node);
                if (it != node_futures.end()) it->second.wait();
            }
            if (task.supports_batching) executeNodeBatched(task, frame_id, ctx);
            else executeNode(task, frame_id, ctx);
            auto it = node_promises.find(task.node_id);
            if (it != node_promises.end()) it->second->set_value();
        });
    }
    for (auto& [id, future] : node_futures) future.wait();
    ctx.getFrame(frame_id).markComplete();
    {
        std::lock_guard<std::mutex> lock(_frames_mutex);
        _pending_frames.erase(frame_id);
        _completed_frames.insert(frame_id);
    }
    _frame_completion_cv.notify_all();
    _total_frames.fetch_add(1);
}

void PipelineScheduler::executeNode(const NodeTask& task, uint64_t frame_id, PipelineContext& ctx) {
    if (!task.node) return;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto inputs = gatherInputs(task, frame_id, ctx);
    auto& frame = ctx.getFrame(frame_id);
    if (inputs.empty() && task.input_nodes.empty() && frame.source_frame) {
        PipelineObject source_obj;
        source_obj.frame_id = frame_id;
        source_obj.object_id = 0;
        source_obj.roi = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, -1};
        source_obj.cropped_image = frame.source_frame;
        inputs.push_back(std::move(source_obj));
    }
    auto non_batch_edges = TransformEngine::filterBatchEdges(task.input_edges);
    inputs = TransformEngine::apply(inputs, non_batch_edges, frame, ctx);
    std::vector<PipelineObject> outputs;
    if (task.supports_batching) {
        outputs = task.node->processBatch(inputs, std::vector<uint64_t>(inputs.size(), frame_id), ctx);
    } else if (inputs.size() > 1) {
        outputs.reserve(inputs.size());
        for (const auto& input : inputs) outputs.push_back(task.node->processObject(input, frame, ctx));
    } else if (!inputs.empty()) {
        outputs.push_back(task.node->processObject(inputs[0], frame, ctx));
    }
    NPU_DEBUG_PREFIX("BATCHED DEBUG", "Node " << task.node_id << " writing " << outputs.size() << " outputs");
    for (size_t i = 0; i < outputs.size(); ++i) {
        NPU_DEBUG_PREFIX("BATCHED DEBUG", "  Output[" << i << "]: class_id=" << outputs[i].roi.class_id
                      << ", conf=" << outputs[i].roi.confidence);
    }
    ctx.writeNodeOutput(frame_id, task.node_id, std::move(outputs));
    ctx.markNodeCompleted(frame_id, task.node_id);
    if (_config.enable_profiling) {
        auto latency_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start_time).count();
        std::lock_guard<std::mutex> lock(_stats_mutex);
        _stats.node_latency_ms[task.node_id] = latency_ms;
    }
}

void PipelineScheduler::executeNodeBatched(const NodeTask& task, uint64_t frame_id, PipelineContext& ctx) {
    if (!task.node || !task.supports_batching) {
        executeNode(task, frame_id, ctx);
        return;
    }
    auto inputs = gatherInputs(task, frame_id, ctx);
    auto& frame = ctx.getFrame(frame_id);
    if (inputs.empty() && task.input_nodes.empty() && frame.source_frame) {
        PipelineObject source_obj;
        source_obj.frame_id = frame_id;
        source_obj.object_id = 0;
        source_obj.roi = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, -1};
        source_obj.cropped_image = frame.source_frame;
        inputs.push_back(std::move(source_obj));
    }
    auto non_batch_edges = TransformEngine::filterBatchEdges(task.input_edges);
    inputs = TransformEngine::apply(inputs, non_batch_edges, frame, ctx);
    NPU_DEBUG_PREFIX("BATCHED DEBUG", "Node " << task.node_id << " has " << inputs.size() << " inputs after transforms");
    for (size_t i = 0; i < inputs.size(); ++i) {
        NPU_DEBUG_PREFIX("BATCHED DEBUG", "  Input[" << i << "]: class_id=" << inputs[i].roi.class_id
                      << ", conf=" << inputs[i].roi.confidence);
    }
    for (const auto& edge : non_batch_edges) {
        NPU_DEBUG_PREFIX("BATCHED DEBUG", "  Edge from " << edge.from_node << " type=" << edge.transform_type);
    }
    for (auto& obj : inputs) {
        BatchItem item{frame_id, obj.object_id, std::move(obj), nullptr};
        if (ctx.accumulateForBatch(task.node_id, std::move(item), task.preferred_batch_size, _config.batch_timeout)) {
            processBatchAccumulator(task, ctx);
        }
    }
    processBatchAccumulator(task, ctx);
    ctx.markNodeCompleted(frame_id, task.node_id);
}

void PipelineScheduler::processBatchAccumulator(const NodeTask& task, PipelineContext& ctx) {
    if (ctx.peekBatch(task.node_id).isEmpty()) return;
    auto batch = ctx.getBatch(task.node_id);
    if (batch.isEmpty()) return;
    NPU_DEBUG_PREFIX("BATCH DEBUG", "Processing batch for node " << task.node_id << " with " << batch.size() << " items");
    for (const auto& edge : task.input_edges) {
        NPU_DEBUG_PREFIX("BATCH DEBUG", "  Edge: " << edge.from_node << " -> " << edge.to_node << " type=" << edge.transform_type);
    }
    std::vector<PipelineObject> objects;
    std::vector<uint64_t> frame_ids;
    objects.reserve(batch.size());
    frame_ids.reserve(batch.size());
    for (const auto& item : batch.items) {
        objects.push_back(item.object);
        frame_ids.push_back(item.frame_id);
    }
    auto outputs = task.node->processBatch(objects, frame_ids, ctx);
    std::unordered_map<uint64_t, std::vector<PipelineObject>> outputs_by_frame;
    for (auto& output : outputs) outputs_by_frame[output.frame_id].push_back(std::move(output));
    for (auto& [fid, frame_outputs] : outputs_by_frame) {
        ctx.writeNodeOutput(fid, task.node_id, std::move(frame_outputs));
    }
}

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

void PipelineScheduler::waitForFrame(uint64_t frame_id) {
    std::unique_lock<std::mutex> lock(_frames_mutex);
    _frame_completion_cv.wait(lock, [this, frame_id] {
        return _completed_frames.find(frame_id) != _completed_frames.end();
    });
}

void PipelineScheduler::waitForAll() {
    if (_thread_pool) {}
    std::unique_lock<std::mutex> lock(_frames_mutex);
}

void PipelineScheduler::stop() {
    _running.store(false);
    if (_thread_pool) _thread_pool->shutdown();
}

PipelineStats PipelineScheduler::getStats() const {
    std::lock_guard<std::mutex> lock(_stats_mutex);
    PipelineStats stats = _stats;
    stats.frames_processed = _total_frames.load();
    return stats;
}

void PipelineScheduler::resetStats() {
    std::lock_guard<std::mutex> lock(_stats_mutex);
    _stats = PipelineStats{};
    _total_frames.store(0);
}

} // namespace npu_pipeline
