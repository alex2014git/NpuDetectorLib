#include "core/npu_base_alg_impl.hpp"
#include <iostream>

NpuBaseAlgImpl::NpuBaseAlgImpl() : NpuBaseImpl() {
    // Simple base implementation for non-detection models
}

int NpuBaseAlgImpl::Initialize(std::string configJsonFile, int streamId) {
    // Do base initialization (config + NPU)
    int result = NpuBaseImpl::Initialize(configJsonFile, streamId);
    if (result < 0) {
        return result;
    }

    // Parse algorithm identity from JSON "name" field and create appropriate decoder
    // LPR config has: "name": "lpr"
    // Classification config has: "name": "classification"
    if (_idName == "lpr") {
        // Parse character_set from JSON config
        std::vector<std::string> charset;
        if (_dom.HasMember("character_set") && _dom["character_set"].IsArray()) {
            const auto& char_array = _dom["character_set"];
            for (size_t i = 0; i < char_array.Size(); ++i) {
                if (char_array[i].IsString()) {
                    charset.push_back(char_array[i].GetString());
                }
            }
        }
        _decoder = std::make_unique<npu::LprDecoder>(charset);
    } else if (_idName == "classification") {
        _decoder = std::make_unique<npu::ClassificationDecoder>();
    }

    return 0;
}

// Phase 1: Run inference
int NpuBaseAlgImpl::Detect(image_share_t imgData, bool needPreProcess) {
    // Run inference through base class Infer()
    int ret = Infer(imgData, needPreProcess);
    if (ret < 0) {
        return ret;
    }
    // No post-processing for ALG_BASE - return raw output count
    return PostProcess(imgData);
}

// Phase 2: Post-process (no-op for ALG_BASE)
int NpuBaseAlgImpl::PostProcess(image_share_t imgData) {
    (void)imgData;
    // ALG_BASE returns raw outputs without decoding
    // Users should call GetRawOutputFloat() or GetRawOutputUint8() instead
    return 0;
}

// Get results - uses decoder for LPR/Classification models
std::vector<npu::NpuResult> NpuBaseAlgImpl::GetResults() {
    if (!_decoder) {
        // No decoder configured - return empty (ALG_BASE behavior)
        return {};
    }
    // Use decoder to convert raw outputs to structured results
    return _decoder->decode(_output_buffer_float, _labels);
}

// Clear results (no-op for ALG_BASE)
void NpuBaseAlgImpl::ClearResults() {
    // ALG_BASE doesn't store parsed results - nothing to clear
    // Raw output buffers are managed by base class
}

void NpuBaseAlgImpl::DrawResult(image_share_t imgData, bool needFormat) {
    (void)imgData;
    (void)needFormat;

    // Base implementation - no drawing for generic models
    // LPR and Classification results are typically not drawn as bounding boxes
    // Subclasses can override if they need visualization
}
