#ifndef _NPU_YOLOV8_SEG_API_IMPL_H
#define _NPU_YOLOV8_SEG_API_IMPL_H

#include "npu.hpp"
#include "core/npu_detection_impl.hpp"
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include "hailo/hailort.hpp"
#include "algorithms/detection/yolov8seg_postprocess.hpp"

// YOLOv8 instance segmentation with software NMS
class NpuYolov8SegImpl : public NpuDetectionImpl {
public:
    NpuYolov8SegImpl();

    int Initialize(std::string configJsonFile, int streamId) override;
    int Detect(image_share_t imgData, bool needPreProcess) override;
    void DrawResult(image_share_t imgData, bool needFormat) override;

protected:
    // PostProcess with needPreProcess for segmentation mask sizing
    int PostProcess(image_share_t imgData) override { return PostProcess(imgData, _last_need_preprocess); }
    int PostProcess(image_share_t imgData, bool needPreProcess);

private:
    std::vector<int32_t> _feature_map_sizes;
    std::vector<int32_t> _mask_sizes;
    std::vector<cv::Mat> _filtered_masks;
    bool _last_need_preprocess = true;  // Store needPreProcess from Detect()

    cv::Mat crop_mask(cv::Mat& mask, int x_min, int y_min, int x_max, int y_max);
};

#endif // #ifndef _NPU_YOLOV8_SEG_API_IMPL_H

