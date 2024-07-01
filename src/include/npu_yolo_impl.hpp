#ifndef _NPU_YOLO_API_IMPL_H
#define _NPU_YOLO_API_IMPL_H
#include "npu.hpp"
#include "npu_base_impl.hpp"
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include "hailo/hailort.hpp"

class NpuYoloImpl : public NpuBaseImpl {
public:
    NpuYoloImpl();

    int Initialize(std::string configJsonFile, int streamId) override;
    int Detect(image_share_t imgData, bool needPreProcess) override;

protected:
    bool _out_sigmoid                           = true;
    std::vector<int32_t> _feature_map_sizes;
    std::vector<std::vector<int32_t>> _anchors;
private:
    size_t GetDetectionsYolo(std::vector<uint8_t> &fm1, std::vector<uint8_t> &fm2,
                                  std::vector<uint8_t> &fm3, std::vector<qp_zp_scale_t> &quantizationInfo,
                                  float32_t thr, std::vector<float32_t> &results);
    size_t YoloPostProcessing(std::vector<std::vector<uint8_t>>& inferOutResult, std::vector<qp_zp_scale_t> &quantizationInfo,
                                  std::vector<float32_t>& detectionsResult);

};


#endif // #ifndef _NPU_YOLO_API_IMPL_H

