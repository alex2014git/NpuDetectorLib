#include "pipeline/result_decoder.hpp"
#include <algorithm>
#include <cmath>

namespace npu {

// Default LPR charset for backward compatibility (Chinese license plates)
// This is used if the JSON config doesn't specify a character_set
static const char* g_default_lpr_charset[] = {
    "#","京","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","皖","闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川",
    "贵","云","藏","陕","甘","青","宁","新","学","警","港","澳","挂","使","领","民","航","危",
    "0","1","2","3","4","5","6","7","8","9",
    "A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z","险","品","I","O","-"
};

// LprDecoder implementation

LprDecoder::LprDecoder(const std::vector<std::string>& charset) {
    if (charset.empty()) {
        // Use default charset for backward compatibility
        size_t default_size = sizeof(g_default_lpr_charset) / sizeof(g_default_lpr_charset[0]);
        _charset.reserve(default_size);
        for (size_t i = 0; i < default_size; ++i) {
            _charset.push_back(g_default_lpr_charset[i]);
        }
    } else {
        _charset = charset;
    }
}

std::string LprDecoder::decodeCtc(const float* output, int output_size) {
    std::string plate;
    std::string prev = "#";

    for (int i = 0; i < output_size; ++i) {
        if (std::isnan(output[i]) || std::isinf(output[i])) {
            continue;
        }
        int idx = static_cast<int>(std::round(output[i]));
        if (idx < 0 || idx >= static_cast<int>(_charset.size())) {
            continue;
        }
        const std::string& c = _charset[idx];
        if (c != "#" && c != prev) {
            plate += c;
        }
        prev = c;
    }

    return plate;
}

std::vector<NpuResult> LprDecoder::decode(
    const std::vector<std::vector<float>>& raw_outputs,
    const std::vector<std::string>& /*labels*/) {

    std::vector<NpuResult> results;

    if (raw_outputs.empty()) {
        return results;
    }

    // LPR typically has a single output tensor with class indices
    const auto& output_buffer = raw_outputs[0];
    if (output_buffer.empty()) {
        return results;
    }

    LprResult result;
    result.text = decodeCtc(output_buffer.data(), static_cast<int>(output_buffer.size()));
    result.confidence = 1.0f;  // Could calculate from output if model provides probabilities

    results.push_back(std::move(result));
    return results;
}

// ClassificationDecoder implementation

ClassificationResult ClassificationDecoder::decodeArgmax(
    const float* output, int output_size,
    const std::vector<std::string>& labels) {

    ClassificationResult result;

    // Find argmax
    auto max_it = std::max_element(output, output + output_size);
    int max_idx = std::distance(output, max_it);

    result.class_id = max_idx;
    result.confidence = *max_it;
    if (max_idx >= 0 && max_idx < static_cast<int>(labels.size())) {
        result.label = labels[max_idx];
    } else {
        result.label = "class_" + std::to_string(max_idx);
    }

    // Build top-5
    std::vector<std::pair<float, int>> scored;
    scored.reserve(output_size);
    for (int i = 0; i < output_size; ++i) {
        scored.push_back({output[i], i});
    }
    std::partial_sort(scored.begin(), scored.begin() + std::min(5, output_size), scored.end(), std::greater<>());

    for (int i = 0; i < std::min(5, output_size); ++i) {
        result.top_k.push_back({scored[i].second, scored[i].first});
    }

    return result;
}

std::vector<NpuResult> ClassificationDecoder::decode(
    const std::vector<std::vector<float>>& raw_outputs,
    const std::vector<std::string>& labels) {

    std::vector<NpuResult> results;

    if (raw_outputs.empty()) {
        return results;
    }

    // Classification typically has a single output tensor with class probabilities
    const auto& output_buffer = raw_outputs[0];
    if (output_buffer.empty()) {
        return results;
    }

    ClassificationResult result = decodeArgmax(
        output_buffer.data(),
        static_cast<int>(output_buffer.size()),
        labels);

    results.push_back(std::move(result));
    return results;
}

} // namespace npu
