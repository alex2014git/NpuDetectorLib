#ifndef RESULT_DECODER_HPP
#define RESULT_DECODER_HPP

#include "npu_result_types.hpp"
#include <vector>
#include <string>
#include <memory>

namespace npu {

// Abstract interface for algorithm-specific result decoding
class ResultDecoder {
public:
    virtual ~ResultDecoder() = default;

    // Decode raw NPU outputs into structured results
    virtual std::vector<NpuResult> decode(
        const std::vector<std::vector<float>>& raw_outputs,
        const std::vector<std::string>& labels) = 0;
};

// LPR decoder with CTC-style decoding
class LprDecoder : public ResultDecoder {
public:
    // Charset for decoding - passed from model config character_set field
    explicit LprDecoder(const std::vector<std::string>& charset);

    std::vector<NpuResult> decode(
        const std::vector<std::vector<float>>& raw_outputs,
        const std::vector<std::string>& labels) override;

private:
    std::vector<std::string> _charset;
    std::string decodeCtc(const float* output, int output_size);
};

// Classification decoder with argmax + top-k
class ClassificationDecoder : public ResultDecoder {
public:
    std::vector<NpuResult> decode(
        const std::vector<std::vector<float>>& raw_outputs,
        const std::vector<std::string>& labels) override;

private:
    ClassificationResult decodeArgmax(
        const float* output, int output_size,
        const std::vector<std::string>& labels);
};

} // namespace npu

#endif // RESULT_DECODER_HPP
