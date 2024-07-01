#ifndef _NPU_YOLOV8_API_IMPL_H
#define _NPU_YOLOV8_API_IMPL_H
#include "npu.hpp"
#include "npu_base_impl.hpp"
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include "hailo/hailort.hpp"
#include "yolov8_postprocess.hpp"

class NpuYolov8Impl : public NpuBaseImpl {
public:
    NpuYolov8Impl();

    int Initialize(std::string configJsonFile, int streamId) override;
    int Detect(image_share_t imgData, bool needPreProcess) override;

protected:
    bool _out_sigmoid = true;
    std::vector<int32_t> _feature_map_sizes;
};


#endif // #ifndef _NPU_YOLOV8_API_IMPL_H

