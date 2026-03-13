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
    virtual int Detect(image_share_t imgData, bool needPreProcess) = 0;

    // Common detection functionality
    void DrawResult(image_share_t imgData, bool needFormat) override;

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

    // Optional post-processing hook - subclasses can override if they use the base Detect()
    virtual int PostProcess(image_share_t imgData) { (void)imgData; return 0; }
};

#endif // _NPU_DETECTION_IMPL_H