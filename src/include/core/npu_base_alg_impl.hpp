#ifndef _NPU_BASE_ALG_IMPL_H
#define _NPU_BASE_ALG_IMPL_H

#include "core/npu_base_impl.hpp"
#include <vector>
#include <string>

// Forward declaration - results stored as raw output vectors
struct _object_roi;

// Simple base implementation for generic models (LPR, Classification)
// Does NOT include NMS logic - just runs inference and stores raw outputs
// Used by ALG_BASE, ALG_LPR, and ALG_CLASSIFICATION
class NpuBaseAlgImpl : public NpuBaseImpl {
public:
    NpuBaseAlgImpl();
    ~NpuBaseAlgImpl() = default;

    // Initialize from JSON config
    int Initialize(std::string configJsonFile, int streamId) override;

    // Run inference - just does inference, no NMS processing
    // Returns 0 on success, negative on error
    int Detect(image_share_t imgData, bool needPreProcess) override;

    // Draw results (basic or empty for models without visualizable results)
    void DrawResult(image_share_t imgData, bool needFormat) override;

    // Get raw output tensors for custom post-processing
    // Used by tests and custom post-processing logic
    const std::vector<std::vector<float>>& GetRawOutputFloat() const { return _output_buffer_float; }
    const std::vector<std::vector<uint8_t>>& GetRawOutputUint8() const { return _output_buffer_uint8; }

    // Get output tensor info
    const std::vector<hailo_vstream_info_t>& GetVstreamInfo() const { return _vstream_infos; }

protected:
    // Raw outputs are already stored in base class buffers
    // Subclasses can override PostProcess() for custom processing
    virtual int PostProcess(image_share_t imgData);

    // For storing simple detection results if needed
    std::vector<object_roi_t> _results;
};

#endif // _NPU_BASE_ALG_IMPL_H
