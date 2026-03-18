#include "core/npu_base_impl.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <fstream>
#include <unordered_map>

#ifndef _WIN32
#include <dlfcn.h>
#include <sys/time.h>
#endif

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

NpuBaseImpl::NpuBaseImpl() {
    // Base initialization
}

NpuBaseImpl::~NpuBaseImpl() {
    Release();
}

static hailo_format_type_t enumStr2Enum(const char* enumStr) {
    static const std::unordered_map<std::string, hailo_format_type_t> strToEnumMap = {
        {"HAILO_FORMAT_TYPE_AUTO", HAILO_FORMAT_TYPE_AUTO},
        {"HAILO_FORMAT_TYPE_UINT8", HAILO_FORMAT_TYPE_UINT8},
        {"HAILO_FORMAT_TYPE_UINT16", HAILO_FORMAT_TYPE_UINT16},
        {"HAILO_FORMAT_TYPE_FLOAT32", HAILO_FORMAT_TYPE_FLOAT32},
        {"HAILO_FORMAT_TYPE_MAX_ENUM", HAILO_FORMAT_TYPE_MAX_ENUM}
    };
    auto it = strToEnumMap.find(enumStr);
    if (it != strToEnumMap.end()) {
        return it->second;
    }
    return HAILO_FORMAT_TYPE_MAX_ENUM;
}

int NpuBaseImpl::InitConfig(std::string configJsonFile, int streamId) {
    _stream_id = std::to_string(streamId);
    std::ifstream in(configJsonFile, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Open config file failed: " << configJsonFile << std::endl;
        return -1;
    }
    std::string json_content((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
    in.close();

    if (_dom.Parse(json_content.c_str()).HasParseError()) {
        std::cerr << "Can not parse JSON file!" << std::endl;
        return -1;
    }

    // Parse common configuration
    if (_dom.HasMember("name") && _dom["name"].IsString()) {
        _idName = _dom["name"].GetString();
    }
    if (_dom.HasMember("model_path") && _dom["model_path"].IsString()) {
        _model_path = _dom["model_path"].GetString();
    }
    if (_dom.HasMember("classes") && _dom["classes"].IsInt()) {
        _nclasses = _dom["classes"].GetInt();
    }
    if (_dom.HasMember("labels") && _dom["labels"].IsArray()) {
        const rapidjson::Value& arr = _dom["labels"];
        for (size_t i = 0; i < arr.Size(); ++i) {
            if (arr[i].IsString()) {
                _labels.push_back(arr[i].GetString());
            }
        }
    }
    if (_dom.HasMember("size") && _dom["size"].IsArray()) {
        const rapidjson::Value& arr = _dom["size"];
        if (arr[0].IsInt()) {
            _model_width = arr[0].GetInt();
        }
        if (arr[1].IsInt()) {
            _model_height = arr[1].GetInt();
        }
        if (arr[2].IsInt()) {
            _model_channel = static_cast<float>(arr[2].GetInt());
        } else if (arr[2].IsFloat()) {
            _model_channel = arr[2].GetFloat();
        }
        if (_model_channel == 1.5f) {
            _img_nv12 = true;  // NV12 format
        }
        _network_input_size = static_cast<size_t>(_model_height * _model_width * _model_channel);
    }
    if (_dom.HasMember("threshold") && _dom["threshold"].IsNumber()) {
        if (_dom["threshold"].IsInt()) {
            _conf_threshold = static_cast<float>(_dom["threshold"].GetInt());
        } else {
            _conf_threshold = _dom["threshold"].GetFloat();
        }
    }
    if (_dom.HasMember("output_order_by_name") && _dom["output_order_by_name"].IsArray()) {
        const rapidjson::Value& arr = _dom["output_order_by_name"];
        for (size_t i = 0; i < arr.Size(); ++i) {
            if (arr[i].IsString()) {
                _output_order_by_name.push_back(arr[i].GetString());
            }
        }
    }
    if (_dom.HasMember("out_format") && _dom["out_format"].IsString()) {
        _out_format = enumStr2Enum(_dom["out_format"].GetString());
    }
    if (_dom.HasMember("in_format") && _dom["in_format"].IsString()) {
        _in_format = enumStr2Enum(_dom["in_format"].GetString());
    }

    return 0;
}

int NpuBaseImpl::InitNPU() {
    if (!_backend) {
        std::cerr << "-E- Backend not injected! Call SetBackend() before Initialize()." << std::endl;
        return -1;
    }
    if (!_backend->Initialize()) {
        std::cerr << "-W- Hailo device/module not found!" << std::endl;
        return -1;
    }

    NetworkConfig Network;
    Network.hef_path = _model_path;
    Network.output_order_by_name = _output_order_by_name;
    Network.batch_size = _batch_size;
    Network.out_format = _out_format;
    Network.out_quantized = ((_out_format == HAILO_FORMAT_TYPE_FLOAT32) ? false : true);
    Network.id_name = _idName + _stream_id;

    if (_backend->AddNetwork(Network) != MnpReturnCode::SUCCESS) {
        std::cerr << "AddNetwork error on " << _stream_id << std::endl;
        return -1;
    }

    _backend->GetNetworkQuantizationInfo(Network.id_name, _quantization_info);
    for (const auto& info : _quantization_info) {
        _out_zps.push_back(info.qp_zp);
        _out_scales.push_back(info.qp_scale);
    }
    _backend->GetNetworkVstreamInfo(Network.id_name, _vstream_infos);
    _backend->GetNetworkInputSize(Network.id_name, _network_input_size);

    if (_out_format == HAILO_FORMAT_TYPE_FLOAT32) {
        _backend->InitializeOutputBuffer(Network.id_name, _output_buffer_float);
    } else {
        _backend->InitializeOutputBuffer(Network.id_name, _output_buffer_uint8);
    }

    _initialized = true;
    return 0;
}

cv::Mat NpuBaseImpl::Letterbox(const cv::Mat& img, int target_width, int target_height,
                               float& ratio, int color) {
    int width = img.cols;
    int height = img.rows;

    cv::Mat letterbox(target_height, target_width, img.type(), cv::Scalar(color, color, color));

    float scale = std::min(float(target_width) / width, float(target_height) / height);
    int new_width = static_cast<int>(std::round(width * scale));
    int new_height = static_cast<int>(std::round(height * scale));

    int x_offset = (target_width - new_width) / 2;
    int y_offset = (target_height - new_height) / 2;
    cv::Rect roi(x_offset, y_offset, new_width, new_height);

    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(new_width, new_height));
    resized_img.copyTo(letterbox(roi));

    ratio = 1.0f / scale;
    return letterbox;
}

void NpuBaseImpl::PreProcessing(cv::Mat& org_frame, bool needPreProcess,
                                cv::Mat& out_frame, float& ratio) {
    if (!needPreProcess) {
        out_frame = org_frame;
        return;
    }

    cv::Mat tmp;
#ifdef LETTER_BOX
    tmp = Letterbox(org_frame, _model_width, _model_height, ratio, _letterbox_color);
#else
    cv::resize(org_frame, tmp, cv::Size(_model_width, _model_height));
#endif

    // Convert to RGB or NV12 based on model requirements
    if (!_img_nv12) {
        cv::cvtColor(tmp, out_frame, cv::COLOR_BGR2RGB);
    } else {
        // NV12 conversion would go here
        // For now, just copy
        out_frame = tmp;
    }
}

template <typename T>
MnpReturnCode NpuBaseImpl::NpuPorcessing(image_share_t imgData, bool needPreProcess) {
    cv::Mat inferFrame;
    std::vector<T> inferData;
    MnpReturnCode ReadOutRet = MnpReturnCode::NO_DATA_AVAILABLE;
    std::string idName = _idName + _stream_id;

    inferData.resize(_network_input_size);

    if (!needPreProcess) {
        inferData.assign(static_cast<T*>(imgData.data),
                         static_cast<T*>(imgData.data) + _network_input_size);
    } else {
        float ratio = 1.0f;
        cv::Mat oriFrame(cv::Size(imgData.width, imgData.height), CV_8UC3, imgData.data);
        PreProcessing(oriFrame, needPreProcess, inferFrame, ratio);

        if (!inferFrame.isContinuous()) {
            std::cerr << "Image is not continuous after preprocessing!" << std::endl;
            return MnpReturnCode::FAILED;
        }

        size_t totalsz = inferFrame.total() * inferFrame.elemSize();
        if (totalsz != _network_input_size) {
            std::cerr << "Size mismatch: image " << totalsz << " vs network "
                      << _network_input_size << std::endl;
            return MnpReturnCode::FAILED;
        }

        inferData.assign(inferFrame.datastart, inferFrame.datastart + totalsz);
    }

    _backend->Infer(idName, inferData);

    if (_out_format == HAILO_FORMAT_TYPE_FLOAT32) {
        ReadOutRet = _backend->ReadOutput(idName, _output_buffer_float);
    } else {
        ReadOutRet = _backend->ReadOutput(idName, _output_buffer_uint8);
    }

    return ReadOutRet;
}

// Explicit template instantiation for NpuPorcessing
template MnpReturnCode NpuBaseImpl::NpuPorcessing<uint8_t>(image_share_t imgData, bool needPreProcess);
// template MnpReturnCode NpuBaseImpl::NpuPorcessing<float>(image_share_t imgData, bool needPreProcess);

// Two-Phase API Implementation
// Phase 1: Run inference (NPU execution)
int NpuBaseImpl::Infer(image_share_t imgData, bool needPreProcess) {
    // NPU input is always uint8_t, output format (_out_format) can be float or uint8
    MnpReturnCode ret = NpuPorcessing<uint8_t>(imgData, needPreProcess);
    return (ret == MnpReturnCode::SUCCESS) ? 0 : -1;
}

// Legacy API - calls Infer() + PostProcess()
int NpuBaseImpl::Detect(image_share_t imgData, bool needPreProcess) {
    int ret = Infer(imgData, needPreProcess);
    if (ret < 0) return ret;
    return PostProcess(imgData);
}

int NpuBaseImpl::Initialize(std::string configJsonFile, int streamId) {
    int result = InitConfig(configJsonFile, streamId);
    if (result < 0) {
        return result;
    }
    return InitNPU();
}

std::string NpuBaseImpl::GetVersion() {
    return _idName + "-" + _verString;
}

void NpuBaseImpl::Release() {
    if (_initialized && _backend) {
        std::string network_id = _idName + _stream_id;
        _backend->RemoveNetwork(network_id);
    }
    _initialized = false;
}
