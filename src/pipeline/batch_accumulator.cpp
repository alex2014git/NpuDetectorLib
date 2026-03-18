#include "pipeline/batch_accumulator.hpp"

namespace npu_pipeline {

BatchAccumulator::BatchAccumulator(const BatchAccumulatorConfig& config)
    : _config(config) {}

bool BatchAccumulator::accumulate(const BatchItem& item) {
    if (_current_batch.isEmpty()) {
        _first_item_time = std::chrono::steady_clock::now();
    }

    _current_batch.items.push_back(item);
    _stats.total_items++;

    return isReady();
}

bool BatchAccumulator::accumulate(BatchItem&& item) {
    if (_current_batch.isEmpty()) {
        _first_item_time = std::chrono::steady_clock::now();
    }

    _current_batch.items.push_back(std::move(item));
    _stats.total_items++;

    return isReady();
}

BatchQueue BatchAccumulator::flush() {
    if (_current_batch.isEmpty()) {
        return BatchQueue{};
    }

    BatchQueue result = std::move(_current_batch);
    _current_batch = BatchQueue{};
    _stats.total_batches++;
    return result;
}

bool BatchAccumulator::isReady() const {
    if (_current_batch.isEmpty()) {
        return false;
    }

    // Check if batch size reached
    if (_current_batch.size() >= _config.preferred_batch_size) {
        return true;
    }

    // Check if timeout elapsed
    auto elapsed = std::chrono::steady_clock::now() - _first_item_time;
    if (elapsed >= _config.timeout) {
        return true;
    }

    return false;
}

void BatchAccumulator::reset() {
    _current_batch = BatchQueue{};
    _stats = Stats{};
}

} // namespace npu_pipeline
