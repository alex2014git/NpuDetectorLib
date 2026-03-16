#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iostream>
#ifndef _WIN32
#include <dlfcn.h>
#include <sys/time.h>
#endif
#include "implementations/npu_yolo_impl.hpp"
#include "yolo_nms_decoder.hpp"

//#define TIME_TRACE_DEBUG

NpuYoloImpl::NpuYoloImpl(void)
{
    //
}

size_t NpuYoloImpl::GetDetectionsYolo(std::vector<uint8_t> &fm1, std::vector<uint8_t> &fm2, std::vector<uint8_t> &fm3, std::vector<qp_zp_scale_t> &quantizationInfo, float32_t thr, std::vector<float32_t> &results) {

    size_t totalPrediction = 0;
    static Yolov5NmsDecoder<uint8_t> yolov5Decoder(true);
    static bool decoder_init = false;

    if (decoder_init == false) {
        QunatizationInfo qInfo;
        yolov5Decoder.set_sigmoid(_out_sigmoid);
        yolov5Decoder.YoloConfig(_model_width, _model_height, _nclasses, thr);

        for(int i = 0; i < _feature_map_sizes.size(); i ++) {
            qInfo.qp_scale = quantizationInfo[i].qp_scale;
            qInfo.qp_zp = quantizationInfo[i].qp_zp;
            yolov5Decoder.YoloAddOutput(_feature_map_sizes[i], _feature_map_sizes[i], _anchors[i], &qInfo);
        }
        decoder_init = true;
    }

    results = yolov5Decoder.decode(fm1, fm2, fm3);    
    totalPrediction = results.size() / 6;

    return totalPrediction;
}

size_t NpuYoloImpl::YoloPostProcessing(std::vector<std::vector<uint8_t>>& inferOutResult, std::vector<qp_zp_scale_t> &quantizationInfo, std::vector<float32_t>& detectionsResult)
{
    // Validate that we have the expected 3 output buffers for YOLO feature maps
    // YOLOv5 expects 3 feature map outputs at different scales (e.g., 20x20, 40x40, 80x80)
    if (inferOutResult.size() < 3) {
        std::cerr << "ERROR: YoloPostProcessing expects 3 output buffers but got "
                  << inferOutResult.size() << std::endl;
        return 0;
    }
    return GetDetectionsYolo(inferOutResult[0], inferOutResult[1], inferOutResult[2], quantizationInfo, _conf_threshold, detectionsResult);
}

int NpuYoloImpl::Initialize(std::string configJsonFile, int streamId)
{
    int result = InitConfig(configJsonFile, streamId);
    if (result < 0) {
        return result;  // Propagate config loading failure
    }
    //get my extra json parameter from the _dom.
    if (_dom.HasMember("feature_map_size") && _dom["feature_map_size"].IsArray()) {
        const rapidjson::Value& arr = _dom["feature_map_size"];
        for (int i = 0; i < arr.Size(); ++i) {
        if (arr[i].IsInt())
            _feature_map_sizes.push_back(arr[i].GetInt());
        }
    }
    if (_dom.HasMember("anchors") && _dom["anchors"].IsArray()) {
        const rapidjson::Value& arr = _dom["anchors"];
        for (int i = 0; i < arr.Size(); ++i) {
            const rapidjson::Value& tmp = arr[i];
                std::vector<int32_t> anc_tmp;
        for (int j = 0; j < tmp.Size(); ++j) {
            if (tmp[j].IsInt())
                anc_tmp.push_back(tmp[j].GetInt());
        }
            _anchors.push_back(anc_tmp);
        }
    }
    if (_dom.HasMember("yolo_nms_core") && _dom["yolo_nms_core"].IsBool()) {
        _nms_core = _dom["yolo_nms_core"].GetBool();
    }
    if (_dom.HasMember("out_sigmoid") && _dom["out_sigmoid"].IsBool()) {
        _out_sigmoid = _dom["out_sigmoid"].GetBool();
    }
    return InitNPU();
}

int NpuYoloImpl::Detect(image_share_t imgData, bool needPreProcess)
{
    // Two-phase API: Infer() + PostProcess()
    int ret = Infer(imgData, needPreProcess);
    if (ret < 0) {
        return ret;
    }
    return PostProcess(imgData);
}

int NpuYoloImpl::PostProcess(image_share_t imgData)
{
    (void)imgData;  // Unused but kept for API compatibility

    // Clear previous detection results before processing
    _objects.clear();
    _objects.shrink_to_fit();

    // Process output buffers from inference
    if (_output_buffer_uint8.empty()) {
        return 0;  // No output to process
    }

    std::vector<float32_t> detectionsResult;
    size_t num_dets = YoloPostProcessing(_output_buffer_uint8, _quantization_info, detectionsResult);

    for (size_t k = 0; k < num_dets; k++) {
        object_roi_t cobj;
        float conf = detectionsResult[k*6+5];
        if (conf < _conf_threshold)
            continue;
        cobj.y_min = detectionsResult[k*6+0];
        cobj.x_min = detectionsResult[k*6+1];
        cobj.y_max = detectionsResult[k*6+2];
        cobj.x_max = detectionsResult[k*6+3];
        int category = static_cast<int>(detectionsResult[k*6+4]);
        cobj.confidence = conf;
        cobj.category = category;
        cobj.name = GetFinalLabel(conf, _labels[category]);
        _objects.push_back(cobj);
    }

    return static_cast<int>(_objects.size());
}


