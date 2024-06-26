#include "npu_factory.hpp"
#include "npu_base_impl.hpp"
#include "npu_yolo_impl.hpp"
#include "npu_yolov8_impl.hpp"
#include "npu_yolov8_pose_impl.hpp"
#include "npu_yolov8_seg_impl.hpp"

std::shared_ptr<Npu> NpuFactory::CreateNpu(algorithm algType) {
    switch (algType) {
        case ALG_BASE:
            return std::make_shared<NpuBaseImpl>();
        case ALG_YOLO_V5:
            return std::make_shared<NpuYoloImpl>();
        case ALG_YOLO_V8:
            return std::make_shared<NpuYolov8Impl>();
        case ALG_POSE:
            return std::make_shared<NpuYolov8PoseImpl>();
        case ALG_YOLO_V8_SEG:
            return std::make_shared<NpuYolov8SegImpl>();
        // Add cases for other algorithms
        default:
            throw std::invalid_argument("Unsupported algorithm type");
    }
}
