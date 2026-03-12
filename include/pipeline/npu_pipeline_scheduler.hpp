#ifndef _NPU_PIPELINE_SCHEDULER_HPP_
#define _NPU_PIPELINE_SCHEDULER_HPP_

#include "npu_pipeline_types.hpp"
#include "npu_pipeline_context.hpp"
#include "npu_pipeline_edge.hpp"
#include <memory>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <future>

namespace npu_pipeline {

// Forward declarations
class PipelineNode;
class PipelineGraph;

// Thread pool for parallel execution
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Enqueue a task
    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<decltype(f(args...))>;

    // Get number of threads
    size_t size() const { return _threads.size(); }

    // Shutdown
    void shutdown();

private:
    std::vector<std::thread> _threads;
    std::queue<std::function<void()>> _tasks;
    std::mutex _mutex;
    std::condition_variable _cv;
    std::atomic<bool> _stop{false};
};

// Scheduler configuration
struct SchedulerConfig {
    enum Strategy {
        SEQUENTIAL,     // One node at a time, deterministic
        PARALLEL,       // Independent nodes in parallel (thread pool)
        BATCHED         // Accumulate ROIs, batch inference
    };

    Strategy strategy = PARALLEL;
    size_t thread_pool_size = 4;
    size_t max_concurrent_frames = 16;  // Pipeline depth
    std::chrono::milliseconds batch_timeout{5};
    bool enable_profiling = false;
    bool dynamic_batching = true;       // Adjust batch size based on load
};

// Node task for execution
struct NodeTask {
    std::string node_id;
    std::vector<std::string> input_nodes;
    std::vector<std::string> output_nodes;
    std::vector<PipelineEdge> input_edges;
    std::vector<PipelineEdge> output_edges;
    std::shared_ptr<PipelineNode> node;
    int priority = 0;  // Lower = higher priority
    bool supports_batching = false;
    size_t preferred_batch_size = 1;
};

// Pipeline scheduler
class PipelineScheduler {
public:
    PipelineScheduler();
    ~PipelineScheduler();

    PipelineScheduler(const PipelineScheduler&) = delete;
    PipelineScheduler& operator=(const PipelineScheduler&) = delete;

    // Initialize scheduler with graph
    void initialize(const PipelineGraph& graph, const SchedulerConfig& config);

    // Submit frame for processing (non-blocking)
    void submit(uint64_t frame_id, PipelineContext& ctx);

    // Submit with callback
    void submit(uint64_t frame_id,
                PipelineContext& ctx,
                std::function<void(const FrameResults&)> callback);

    // Wait for specific frame
    void waitForFrame(uint64_t frame_id);

    // Wait for all pending frames
    void waitForAll();

    // Stop scheduler
    void stop();

    // Get statistics
    PipelineStats getStats() const;

    // Reset statistics
    void resetStats();

private:
    // Build execution order from graph
    void buildExecutionOrder(const PipelineGraph& graph);

    // Topological sort for node ordering
    std::vector<NodeTask> topologicalSort(const PipelineGraph& graph);

    // Strategy implementations
    void runSequential(uint64_t frame_id, PipelineContext& ctx);
    void runParallel(uint64_t frame_id, PipelineContext& ctx);
    void runBatched(uint64_t frame_id, PipelineContext& ctx);

    // Execute single node
    void executeNode(const NodeTask& task,
                     uint64_t frame_id,
                     PipelineContext& ctx);

    // Execute node with batching
    void executeNodeBatched(const NodeTask& task,
                            uint64_t frame_id,
                            PipelineContext& ctx);

    // Process batch accumulator for a node
    void processBatchAccumulator(const NodeTask& task,
                                  PipelineContext& ctx);

    // Check if all inputs are ready for a node
    bool areInputsReady(const NodeTask& task,
                        uint64_t frame_id,
                        const PipelineContext& ctx);

    // Get inputs for a node
    std::vector<PipelineObject> gatherInputs(const NodeTask& task,
                                              uint64_t frame_id,
                                              PipelineContext& ctx);

    // Apply edge transforms
    std::vector<PipelineObject> applyTransforms(
        const std::vector<PipelineObject>& inputs,
        const std::vector<PipelineEdge>& edges,
        const FrameResults& frame,
        PipelineContext& ctx);

    // Member variables
    std::vector<NodeTask> _execution_order;
    std::unordered_map<std::string, size_t> _node_index;
    std::unique_ptr<ThreadPool> _thread_pool;
    SchedulerConfig _config;

    // Frame tracking
    mutable std::mutex _frames_mutex;
    std::unordered_set<uint64_t> _pending_frames;
    std::unordered_set<uint64_t> _completed_frames;

    // Statistics
    mutable std::mutex _stats_mutex;
    PipelineStats _stats;
    std::atomic<uint64_t> _total_frames{0};

    // Control
    std::atomic<bool> _running{true};
    std::atomic<bool> _initialized{false};
};

// Thread pool template implementation
inline ThreadPool::ThreadPool(size_t num_threads) {
    for (size_t i = 0; i < num_threads; ++i) {
        _threads.emplace_back([this] {
            while (!_stop.load()) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(_mutex);
                    _cv.wait(lock, [this] {
                        return _stop.load() || !_tasks.empty();
                    });
                    if (_stop.load() && _tasks.empty()) {
                        return;
                    }
                    task = std::move(_tasks.front());
                    _tasks.pop();
                }
                task();
            }
        });
    }
}

inline ThreadPool::~ThreadPool() {
    shutdown();
}

inline void ThreadPool::shutdown() {
    _stop.store(true);
    _cv.notify_all();
    for (auto& thread : _threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

template<typename F, typename... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
    using return_type = decltype(f(args...));

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> result = task->get_future();

    {
        std::unique_lock<std::mutex> lock(_mutex);
        if (_stop.load()) {
            throw std::runtime_error("ThreadPool is stopped");
        }
        _tasks.emplace([task]() { (*task)(); });
    }
    _cv.notify_one();
    return result;
}

} // namespace npu_pipeline

#endif // _NPU_PIPELINE_SCHEDULER_HPP_
