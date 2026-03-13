#include "implementations/npu_yolo_nms_impl.hpp"
#include <iostream>

NpuYoloNmsImpl::NpuYoloNmsImpl() : NpuDetectionImpl() {
    // Hardware NMS is enabled by default for this class
}

int NpuYoloNmsImpl::Initialize(std::string configJsonFile, int streamId) {
    // First do base initialization
    int result = NpuDetectionImpl::Initialize(configJsonFile, streamId);
    if (result < 0) {
        return result;
    }

    // Check if hardware NMS is configured
    if (_dom.HasMember("yolo_nms_core") && _dom["yolo_nms_core"].IsBool()) {
        _nms_core = _dom["yolo_nms_core"].GetBool();
    } else {
        // Default to hardware NMS for this class
        _nms_core = true;
    }

    // Hardware NMS requires float32 output format
    if (_nms_core && _out_format != HAILO_FORMAT_TYPE_FLOAT32) {
        std::cerr << "Warning: Hardware NMS requires FLOAT32 output format" << std::endl;
    }

    return 0;
}

int NpuYoloNmsImpl::Detect(image_share_t imgData, bool needPreProcess) {
    // Run inference through base class template method
    MnpReturnCode ReadOutRet = NpuPorcessing<uint8_t>(imgData, needPreProcess);

    if (ReadOutRet != MnpReturnCode::SUCCESS) {
        return 0;
    }

    // Parse results based on NMS mode
    return PostProcess(imgData);
}

int NpuYoloNmsImpl::PostProcess(image_share_t imgData) {
    (void)imgData;  // Not needed for hardware NMS

    // Parse hardware NMS results
    int num_detections = ParseHardwareNmsResults(imgData);

#ifdef TIME_TRACE_DEBUG
    std::cout << "Hardware NMS detected " << num_detections << " objects" << std::endl;
#endif

    return num_detections;
}
