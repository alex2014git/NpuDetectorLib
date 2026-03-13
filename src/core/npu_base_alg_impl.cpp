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

int NpuBaseAlgImpl::Detect(image_share_t imgData, bool needPreProcess) {
    // Run inference through base class template method
    MnpReturnCode ReadOutRet = NpuPorcessing<uint8_t>(imgData, needPreProcess);

    if (ReadOutRet != MnpReturnCode::SUCCESS) {
        std::cerr << "NpuBaseAlgImpl: Inference failed" << std::endl;
        return -1;
    }

    // Call post-processing (subclasses can override)
    return PostProcess(imgData);
}

int NpuBaseAlgImpl::PostProcess(image_share_t imgData) {
    (void)imgData;

    // Base implementation just returns success (0 detections)
    // Subclasses (LPR, Classification) will override this to parse outputs
    // For now, we just verify that output buffers are populated

    if (_out_format == HAILO_FORMAT_TYPE_FLOAT32) {
        // Check that we have output in float buffer
        if (_output_buffer_float.empty()) {
            return 0;  // No output
        }
        // Return number of output tensors (not detections)
        return static_cast<int>(_output_buffer_float.size());
    } else {
        // Check uint8 buffer
        if (_output_buffer_uint8.empty()) {
            return 0;  // No output
        }
        // Return number of output tensors
        return static_cast<int>(_output_buffer_uint8.size());
    }
}

void NpuBaseAlgImpl::DrawResult(image_share_t imgData, bool needFormat) {
    (void)imgData;
    (void)needFormat;

    // Base implementation - no drawing for generic models
    // LPR and Classification results are typically not drawn as bounding boxes
    // Subclasses can override if they need visualization
}
