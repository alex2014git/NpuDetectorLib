#ifndef BATCH_ACCUMULATOR_HPP
#define BATCH_ACCUMULATOR_HPP

#include "npu_pipeline_types.hpp"
#include <vector>
#include <string>
#include <chrono>

namespace npu_pipeline {

// Queue of batch items for accumulation
struct BatchQueue {
    std::vector<BatchItem> items;

    bool isEmpty() const { return items.empty(); }
    size_t size() const { return items.size(); }
    void clear() { items.clear(); }
};

// Configuration for batch accumulation
struct BatchAccumulatorConfig {
    size_t preferred_batch_size = 8;
    std::chrono::milliseconds timeout{5};
};

// Manages batch accumulation for a single node
class BatchAccumulator {
public:
    explicit BatchAccumulator(const BatchAccumulatorConfig& config = BatchAccumulatorConfig{});

    // Add item to batch, returns true if batch is ready for processing
    bool accumulate(const BatchItem& item);
    bool accumulate(BatchItem&& item);

    // Get current batch and clear accumulator
    BatchQueue flush();

    // Peek at current batch without removing
    const BatchQueue& peek() const { return _current_batch; }

    // Check if batch is empty
    bool isEmpty() const { return _current_batch.isEmpty(); }

    // Check if batch is ready based on size or timeout
    bool isReady() const;

    // Get current batch size
    size_t size() const { return _current_batch.size(); }

    // Reset/clear the accumulator
    void reset();

    // Get statistics
    struct Stats {
        uint64_t total_items = 0;
        uint64_t total_batches = 0;
        uint64_t timeout_flushes = 0;
    };

    const Stats& stats() const { return _stats; }

private:
    BatchAccumulatorConfig _config;
    BatchQueue _current_batch;
    std::chrono::steady_clock::time_point _first_item_time;
    Stats _stats;
};

} // namespace npu_pipeline

#endif // BATCH_ACCUMULATOR_HPP
