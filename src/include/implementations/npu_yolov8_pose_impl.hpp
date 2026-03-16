#ifndef _NPU_YOLOV8_POSE_API_IMPL_H
#define _NPU_YOLOV8_POSE_API_IMPL_H

#include "npu.hpp"
#include "core/npu_detection_impl.hpp"
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include "hailo/hailort.hpp"
#include "algorithms/pose/yolov8pose_postprocess.hpp"

// YOLOv8 pose estimation with software NMS
class NpuYolov8PoseImpl : public NpuDetectionImpl {
public:
    NpuYolov8PoseImpl();

    int Initialize(std::string configJsonFile, int streamId) override;
    int Detect(image_share_t imgData, bool needPreProcess) override;
    void DrawResult(image_share_t imgData, bool needFormat) override;

protected:
    // Phase 2: Post-process results
    int PostProcess(image_share_t imgData) override;

private:
    std::pair<std::vector<KeyPt>, std::vector<PairPairs>> _pose_extra;
    std::vector<int32_t> _feature_map_sizes;

    size_t post_processing_all(std::vector<std::vector<uint8_t>>& output_buffer_uint8,
                               std::vector<hailo_vstream_info_t>& vstream_infos,
                               std::pair<std::vector<KeyPt>, std::vector<PairPairs>>& keypoints_and_pairs);
};

#endif // #ifndef _NPU_YOLOV8_POSE_API_IMPL_H

