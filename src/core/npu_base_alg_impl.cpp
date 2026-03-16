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

    // Base algorithm doesn't use NMS
    // Subclasses can override if needed
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

// Get results (empty for ALG_BASE - use GetRawOutputFloat/Uint8 instead)
std::vector<npu::NpuResult> NpuBaseAlgImpl::GetResults() const {
    // ALG_BASE doesn't produce parsed results - returns empty vector
    // Users should access raw outputs via GetRawOutputFloat() or GetRawOutputUint8()
    return {};
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
