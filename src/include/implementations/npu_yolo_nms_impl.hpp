#ifndef _NPU_YOLO_NMS_IMPL_H
#define _NPU_YOLO_NMS_IMPL_H

#include "core/npu_detection_impl.hpp"

// YOLO with hardware NMS support
// Used for ALG_BASE - models with "yolo_nms_core": true in JSON config
// The NPU performs NMS internally, output format is pre-filtered bounding boxes
class NpuYoloNmsImpl : public NpuDetectionImpl {
public:
    NpuYoloNmsImpl();
    ~NpuYoloNmsImpl() = default;

    int Initialize(std::string configJsonFile, int streamId) override;
    int Detect(image_share_t imgData, bool needPreProcess) override;

protected:
    // Post-processing for hardware NMS output
    int PostProcess(image_share_t imgData) override;
};

#endif // _NPU_YOLO_NMS_IMPL_H
