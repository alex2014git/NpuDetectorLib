#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#ifndef _WIN32
#include <dlfcn.h>
#include <sys/time.h>
#endif
#include "npu_yolo_impl.hpp"
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
    return GetDetectionsYolo(inferOutResult[0], inferOutResult[1], inferOutResult[2], quantizationInfo, 0.4, detectionsResult);
}

int NpuYoloImpl::Initialize(std::string configJsonFile, int streamId)
{
    InitConfig(configJsonFile, streamId);
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
    MnpReturnCode ReadOutRet = MnpReturnCode::NO_DATA_AVAILABLE;
    std::vector<float32_t> detectionsResult;
    int class_id = 1;

    ReadOutRet = NpuPorcessing<uint8_t>(imgData, needPreProcess);
    if (ReadOutRet == MnpReturnCode::SUCCESS)
    {
#ifdef TIME_TRACE_DEBUG
      std::chrono::duration<double> total_time;
      std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::high_resolution_clock::now();
#endif
      size_t num_dets = 0;
      detectionsResult.clear();
      num_dets = YoloPostProcessing(_output_buffer_uint8, _quantization_info, detectionsResult);
#ifdef TIME_TRACE_DEBUG
      std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();
      total_time = t_end - t_start;
      std::cout << "-I- postprocessing run time: " << (double)total_time.count() << " sec" << std::endl;
#endif
      for (int k = 0; k < num_dets; k++) {
        object_roi_t cobj;
        float conf = detectionsResult[k*6+5];
        if (conf < _conf_threshold)
            continue;
        cobj.y_min = detectionsResult[k*6+0];
        cobj.x_min = detectionsResult[k*6+1];
        cobj.y_max = detectionsResult[k*6+2];
        cobj.x_max = detectionsResult[k*6+3];
        int category = detectionsResult[k*6+4];
        cobj.confidence = conf;
        cobj.category = category;
        cobj.name = cobj.name = GetFinalLabel(conf, _labels[category]);
      #ifdef TIME_TRACE_DEBUG
        printf("T result cobj.category %d, conf %f\n", cobj.category, cobj.confidence);
      #endif
        _objects.push_back(cobj);
      }
    }
    return _objects.size();
}


