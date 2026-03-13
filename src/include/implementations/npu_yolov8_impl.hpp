#ifndef _NPU_YOLOV8_API_IMPL_H
#define _NPU_YOLOV8_API_IMPL_H

#include "npu.hpp"
#include "core/npu_detection_impl.hpp"
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include "hailo/hailort.hpp"
#include "algorithms/detection/yolov8_postprocess.hpp"

// YOLOv8 detection with software NMS
class NpuYolov8Impl : public NpuDetectionImpl {
public:
    NpuYolov8Impl();

    int Initialize(std::string configJsonFile, int streamId) override;
    int Detect(image_share_t imgData, bool needPreProcess) override;

protected:
    int PostProcess(image_share_t imgData) override { (void)imgData; return 0; }

private:
    bool _out_sigmoid = true;
    std::vector<int32_t> _feature_map_sizes;
};

#endif // #ifndef _NPU_YOLOV8_API_IMPL_H

