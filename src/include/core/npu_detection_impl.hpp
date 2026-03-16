#ifndef _NPU_DETECTION_IMPL_H
#define _NPU_DETECTION_IMPL_H

#include "core/npu_base_impl.hpp"
#include <vector>

// Forward declaration - object_roi_t is typedef'd in npu.hpp
struct _object_roi;

// Abstract base class for detection models
// Contains detection-specific functionality: object storage, NMS handling, drawing
class NpuDetectionImpl : public NpuBaseImpl {
public:
    NpuDetectionImpl();
    ~NpuDetectionImpl();  // Not override - base has no virtual destructor

    // Detection interface - subclasses must implement post-processing
    // Legacy API: Detect() = Infer() + PostProcess()
    int Detect(image_share_t imgData, bool needPreProcess) override = 0;

    // Phase 2: Post-process - subclasses must implement
    virtual int PostProcess(image_share_t imgData) = 0;

    // Get unified results (converts _objects to NpuResult)
    std::vector<npu::NpuResult> GetResults() const override;

    // Clear results for next inference
    void ClearResults() override;

    // Common detection functionality
    void DrawResult(image_share_t imgData, bool needFormat) override;

    // Legacy detection-specific getters
    const std::vector<object_roi_t>& GetDetectionResults() const { return _objects; }
    void ClearDetectionResults() { _objects.clear(); _objects.shrink_to_fit(); }

protected:
    // Detection-specific members
    bool _nms_core = false;  // Use hardware NMS
    std::vector<object_roi_t> _objects;  // Detection results

    // Detection helpers
    void DrawObject(image_share_t imgData, cv::Mat &showFrame, int new_width, int new_height,
                    int w_compen, int h_compen);
    std::string GetFinalLabel(float conf, std::string label);

    // Hardware NMS parsing - used by subclasses with hardware NMS support
    int ParseHardwareNmsResults(image_share_t imgData);
};

#endif // _NPU_DETECTION_IMPL_H