/**
 * Classification Post-Processing
 *
 * Simple argmax-based classification result extraction.
 */

#ifndef CLASSIFICATION_POSTPROCESS_HPP
#define CLASSIFICATION_POSTPROCESS_HPP

#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

namespace npu_pipeline {

/**
 * @brief Classification result structure
 */
struct ClassificationOutput {
    int class_id = -1;
    float confidence = 0.0f;
    std::vector<std::pair<int, float>> top_k;
};

/**
 * @brief Simple argmax classification - find class with highest probability
 *
 * @param output Raw model output (float array)
 * @param output_size Number of output elements (number of classes)
 * @param top_k Number of top predictions to return
 * @return ClassificationOutput Result with class_id and confidence
 */
inline ClassificationOutput argmax_classification(const float* output, int output_size, int top_k = 5) {
    ClassificationOutput result;

    // Find all class probabilities
    std::vector<std::pair<int, float>> class_probs;
    class_probs.reserve(output_size);

    for (int i = 0; i < output_size; ++i) {
        float prob = output[i];
        // Skip invalid values
        if (!std::isnan(prob) && !std::isinf(prob)) {
            class_probs.emplace_back(i, prob);
        }
    }

    // Sort by probability (descending)
    std::sort(class_probs.begin(), class_probs.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Get top k
    int count = std::min(top_k, static_cast<int>(class_probs.size()));
    for (int i = 0; i < count; ++i) {
        result.top_k.emplace_back(class_probs[i].first, class_probs[i].second);
    }

    // Set primary result
    if (!class_probs.empty()) {
        result.class_id = class_probs[0].first;
        result.confidence = class_probs[0].second;
    }

    return result;
}

/**
 * @brief Get softmax probabilities from raw logits
 *
 * @param output Raw model output (logits)
 * @param output_size Number of output elements
 * @return std::vector<float> Softmax probabilities (sum to 1.0)
 */
inline std::vector<float> softmax(const float* output, int output_size) {
    std::vector<float> exp_values(output_size);
    float sum_exp = 0.0f;

    // Compute exp for each value
    for (int i = 0; i < output_size; ++i) {
        if (std::isnan(output[i]) || std::isinf(output[i])) {
            exp_values[i] = 0.0f;
        } else {
            exp_values[i] = std::exp(output[i]);
            sum_exp += exp_values[i];
        }
    }

    // Normalize
    if (sum_exp > 0.0f) {
        for (int i = 0; i < output_size; ++i) {
            exp_values[i] /= sum_exp;
        }
    }

    return exp_values;
}

} // namespace npu_pipeline

#endif // CLASSIFICATION_POSTPROCESS_HPP