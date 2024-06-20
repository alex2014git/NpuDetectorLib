#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <sys/time.h>

#include <iostream>
#include <fstream>
#include <unordered_map>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "npu_base_impl.hpp"
#include "yolo_nms_decoder.hpp"
#include "common/hailo_objects.hpp"
#include "common.hpp"

//#define TIME_TRACE_DEBUG
//#define LETTER_BOX

hailo_format_type_t enumStr2Enum(const char* enumStr) {
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

static void swapYUV_I420toNV12(unsigned char* i420bytes, unsigned char* nv12bytes, int width, int height)
{
    int nLenY = width * height;
    int nLenU = nLenY / 4;

    memcpy(nv12bytes, i420bytes, width * height);

    for (int i = 0; i < nLenU; i++) {
        nv12bytes[nLenY + 2 * i] = i420bytes[nLenY + i];                    // U
        nv12bytes[nLenY + 2 * i + 1] = i420bytes[nLenY + nLenU + i];        // V
    }
}

static void BGR2YUV_nv12(cv::Mat src, cv::Mat &dst)
{
    int w_img = src.cols;
    int h_img = src.rows;
    dst = cv::Mat(h_img*3/2, w_img, CV_8UC1, cv::Scalar(0));
    cv::Mat src_YUV_I420(h_img*3/2, w_img, CV_8UC1, cv::Scalar(0));  //YUV_I420
    cvtColor(src, src_YUV_I420, cv::COLOR_BGR2YUV_I420);
    swapYUV_I420toNV12(src_YUV_I420.data, dst.data, w_img, h_img);
}

NpuBaseImpl::NpuBaseImpl(void)
{
    //
}

int NpuBaseImpl::InitConfig(std::string configJsonFile, int streamId)
{
    _stream_id = std::to_string(streamId);
    std::ifstream in(configJsonFile, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "Open config file failed!" << std::endl;
        return -1;
    }
    std::string json_content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    if (!_dom.Parse(json_content.c_str()).HasParseError()) {
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
            for (int i = 0; i < arr.Size(); ++i) {
                if (arr[i].IsString())
                    _labels.push_back(arr[i].GetString());
            }
        }
        if (_dom.HasMember("size") && _dom["size"].IsArray()) {
            const rapidjson::Value& arr = _dom["size"];
            if (arr[0].IsInt())
                _model_height = arr[0].GetInt();
            if (arr[1].IsInt())
                _model_width = arr[1].GetInt();
            if (arr[2].IsInt())
                _model_channel = (float)arr[2].GetInt();
            else
                _model_channel = (float)arr[2].GetFloat();
            if (_model_channel == 1.5f) {
                _img_nv12 = true; //if we find the channel is 1.5f, we think it is NV12
            }
            _network_input_size = (size_t)(_model_height * _model_width * _model_channel);
          #ifdef TIME_TRACE_DEBUG
            printf("format %d, size %ld wxh %dx%d c %f\n", _img_nv12, _network_input_size, _model_height, _model_width, _model_channel);
          #endif
        }
        if (_dom.HasMember("threshold") && _dom["threshold"].IsInt()) {
            _conf_threshold = _dom["threshold"].GetInt();
        }
        if (_dom.HasMember("output_order_by_name") && _dom["output_order_by_name"].IsArray()) {
            const rapidjson::Value& arr = _dom["output_order_by_name"];
            for (int i = 0; i < arr.Size(); ++i) {
                if (arr[i].IsString())
                    _output_order_by_name.push_back(arr[i].GetString());
            }
        }
        if (_dom.HasMember("out_format") && _dom["out_format"].IsString()) {
            _out_format = enumStr2Enum(_dom["out_format"].GetString());
        }
        if (_dom.HasMember("in_format") && _dom["in_format"].IsString()) {
            _in_format = enumStr2Enum(_dom["in_format"].GetString());
        }
    } else {
        std::cout << "Can not parse Json file!" << std::endl;
    }

  return 0;
}

int NpuBaseImpl::InitNPU()
{
    /* Create the neural network */
  #ifdef TIME_TRACE_DEBUG
    std::cout << "Loading model..." << _model_path << std::endl;
  #endif
    pHailoPipeline = MultiNetworkPipeline::GetInstance();
    if (pHailoPipeline->InitializeHailo() <= 0) {
        std::cout << "-W- Hailo device/module not found!" << std::endl;  
        exit(0);        
    }

    stNetworkModelInfo Network;
    Network.hef_path = _model_path;
    Network.output_order_by_name.clear();
    Network.output_order_by_name = _output_order_by_name;
    Network.batch_size = _batch_size;    
    Network.out_format = _out_format;
    Network.out_quantized = ((_out_format == HAILO_FORMAT_TYPE_FLOAT32) ? false : true); 
    Network.in_quantized = ((_in_format == HAILO_FORMAT_TYPE_FLOAT32) ? false : true);;
    Network.in_format = _in_format;
    Network.id_name = _idName + _stream_id;

    if (pHailoPipeline->AddNetwork(0, Network, _stream_id) != MnpReturnCode::SUCCESS) {
        std::cout << "AddNetwork error on " << _stream_id << std::endl;  
        return -1;
    }
  #ifdef TIME_TRACE_DEBUG
    std::cout << "AddNetwork " << _stream_id << std::endl;
  #endif

    pHailoPipeline->GetNetworkQuantizationInfo(Network.id_name, _quantization_info);
    for (int i = 0; i < _quantization_info.size(); i++) {
        _out_zps.push_back(_quantization_info[i].qp_zp);
        _out_scales.push_back(_quantization_info[i].qp_scale);
    }
    pHailoPipeline->GetNetworkVstream_Info(Network.id_name, _vstream_infos);
    pHailoPipeline->GetNetworkInputSize(Network.id_name, _network_input_size);
    if(_out_format == HAILO_FORMAT_TYPE_FLOAT32) {
        pHailoPipeline->InitializeOutputBuffer(Network.id_name, _output_buffer_float, _stream_id);
    } else {
        pHailoPipeline->InitializeOutputBuffer(Network.id_name, _output_buffer_uint8, _stream_id);
    }
    _initialized = true;
    return 0;
}

cv::Mat NpuBaseImpl::Letterbox( const cv::Mat& img, int target_width, float &ratio, int color = 114 )
{
    int width = img.cols;
    int height = img.rows;

    cv::Mat square( target_width, target_width, img.type(), cv::Scalar(color, color, color) );
    int max_dim = ( width >= height ) ? width : height;
    float32_t scale = ( ( float32_t ) target_width ) / max_dim;
    cv::Rect roi;
    if (width >= height) {
        roi.width = target_width;
        roi.x = 0;
        roi.height = height * scale;
        roi.y = ( target_width - roi.height ) / 2;
    } else {
        roi.y = 0;
        roi.height = target_width;
        roi.width = width * scale;
        roi.x = ( target_width - roi.width ) / 2;
    }
    ratio = 1.0 / scale;
    cv::resize( img, square( roi ), roi.size() );
    return square;
}

void NpuBaseImpl::PreProcessing(cv::Mat &oriFrame, bool needPreProcess, cv::Mat &outFrame, float &ratio)
{
    cv::Mat tmp;
    if ( !needPreProcess ) {
        outFrame = oriFrame;
        return;
    }

#ifdef LETTER_BOX
    tmp = Letterbox(oriFrame, _model_height, ratio, 144);
#else
    cv::resize( oriFrame, tmp, cv::Size(_model_width, _model_height));
#endif
    //we assume all the input is BGR 3 chanel data, if you want to convert it NV12
    if (!_img_nv12) {
        cv::cvtColor(tmp, outFrame, cv::COLOR_BGR2RGB);
      #ifdef TIME_TRACE_DEBUG
        printf("BGR to RGB\n");
      #endif
    } else {
        BGR2YUV_nv12(tmp, outFrame);
      #ifdef TIME_TRACE_DEBUG
        printf("BGR to NV12\n");
      #endif
    }
}

void NpuBaseImpl::DrawObject(image_share_t imgData, cv::Mat &showFrame, int max_dim, int w_compen, int h_compen)
{
    int x1, y1, x2, y2;
    for (const auto& object : _objects) {
      #ifdef LETTER_BOX
        x1 = object.x_min * float(max_dim) - w_compen;
        y1 = object.y_min * float(max_dim) - h_compen;
        x2 = object.x_max * float(max_dim) - w_compen;
        y2 = object.y_max * float(max_dim) - h_compen;
      #else
        x1 = object.x_min * showFrame.cols;
        y1 = object.y_min * showFrame.rows;
        x2 = object.x_max * showFrame.cols;
        y2 = object.y_max * showFrame.rows;
      #endif
        rectangle(showFrame, cv::Point(x1, y1),
                    cv::Point(x2, y2), cv::Scalar(0, 255, 0, 255), 2);
        putText(showFrame, object.name, cv::Point(x1, y1 - 12), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0, 0, 255, 255), 1 , 0);
    }
    _objects.clear();
    _objects.shrink_to_fit();
}

template <typename T>
MnpReturnCode NpuBaseImpl::NpuPorcessing(image_share_t imgData, bool needPreProcess)
{
    cv::Mat inferFrame;
    std::vector<T> inferData;
    MnpReturnCode ReadOutRet = MnpReturnCode::NO_DATA_AVAILABLE;
    std::string idName;
    int class_id = 1;

    idName = _idName + _stream_id;
  #ifdef TIME_TRACE_DEBUG
    std::cout << "NpuBaseImpl::NpuPorcessing " << idName << std::endl;
  #endif

#ifdef TIME_TRACE_DEBUG
    std::chrono::duration<double> total_time;
    std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::high_resolution_clock::now();
#endif
    inferData.resize(_network_input_size);
    //printf("_stream_id %d size %d, img_size %dx%d add %p\n", _stream_id,  _network_input_size, imgData.width, imgData.height, imgData.data);
    if (needPreProcess == false) {
        inferData.assign((T*)imgData.data, (T*)imgData.data + _network_input_size);
    } else {
        cv::Mat oriFrame(cv::Size(imgData.width, imgData.height), CV_8UC3, (void *)imgData.data);
        PreProcessing(oriFrame, needPreProcess, inferFrame, _input_scale);
        int totalsz = inferFrame.dataend - inferFrame.datastart;
        if (inferFrame.isContinuous() && (totalsz == _network_input_size)) {
            inferData.assign(inferFrame.datastart, inferFrame.datastart + totalsz);
        } else {
            std::cout << "img is not continuous, or the size error! img size:" << totalsz << "netsize:" << _network_input_size << std::endl;
            return ReadOutRet;
        }
    }
#ifdef TIME_TRACE_DEBUG
    std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();
    total_time = t_end - t_start;
    std::cout << BOLDBLUE << "-I- preprocess run time: " << (double)total_time.count() << " sec" << RESET << std::endl;
    t_start = std::chrono::high_resolution_clock::now();
#endif
    pHailoPipeline->Infer(idName, inferData, _stream_id);
    //printf("do infer\n");
    if(_out_format == HAILO_FORMAT_TYPE_FLOAT32) {
        ReadOutRet = pHailoPipeline->ReadOutputById(idName, _output_buffer_float, _stream_id);
    } else {
        ReadOutRet = pHailoPipeline->ReadOutputById(idName, _output_buffer_uint8, _stream_id);
    }
    //printf("get  infer out\n");
    if (ReadOutRet == MnpReturnCode::SUCCESS)
    {
#ifdef TIME_TRACE_DEBUG
      std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();
      total_time = t_end - t_start;
      std::cout << BOLDBLUE << "-I- inference run time: " << (double)total_time.count() << " sec" << RESET << std::endl;
#endif
    }

    return ReadOutRet;
}

int NpuBaseImpl::Initialize(std::string configJsonFile, int streamId)
{
    InitConfig(configJsonFile, streamId);
    //get my extra json parameter from the _dom.
    if(_dom.HasMember("yolo_nms_core") && _dom["yolo_nms_core"].IsBool()) {
        _nms_core = _dom["yolo_nms_core"].GetBool();
    }
    return InitNPU();
}

std::string NpuBaseImpl::GetVersion()
{
    return _idName + "-" + _verString;
}

std::string NpuBaseImpl::GetFinalLabel(float conf, std::string label)
{
  #ifdef SHOW_LABEL
    float rounded_provability = floorf(conf*10000) / 100;
    std::ostringstream os_label;
    os_label << label;
    os_label << "(" << rounded_provability << "%)";
    std::string labelout = os_label.str();
  #else
    std::string labelout = "";
  #endif
    return labelout;
}

int NpuBaseImpl::Detect(image_share_t imgData, bool needPreProcess)
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
      if(_nms_core && (_out_format == HAILO_FORMAT_TYPE_FLOAT32)) {
        /*
         *
         decodes the nms buffer received from the output tensor of the network.
        returns a vector of DetectonObject filtered by the detection threshold.

        The data is sorted by the number of the classes.
        for each class - first comes the number of boxes in the class, then the boxes one after the other,
        each box contains x_min, y_min, x_max, y_max and score (uint16_t\float32 each) and can be casted to common::hailo_bbox_t struct (5*uint16_t).
        means that a frame size of one class is sizeof(bbox_count) + bbox_count * sizeof(common::hailo_bbox_t).
        and the actual size of the data is (frame size of one class)*number of classes.

        If the data comes after quantization - so dequantization to float32 is needed.

        As an example - quantized data buffer of a frame that contains a person and two dogs:
        (person class id = 1, dog class id = 18)

        1 107 96 143 119 172 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 2 123 124 140 150 92 112 125 138 147 91 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

        taking the dogs as example - 2 123 124 140 150 92 112 125 138 147 91
        can be splitted to two different boxes
        common::hailo_bbox_t st_1 = 123 124 140 150 92
        common::hailo_bbox_t st_2 = 112 125 138 147 91
        now after dequntization of st_1 - we get common::hailo_bbox_float32_t:
        ymin = 0.551805 xmin = 0.389635 ymax = 0.741805 xmax = 0.561974 score = 0.95
         */
        //std::cout << _output_buffer_float[0].size() << std::endl;
        for (int i = 0, class_id = 0; i < _output_buffer_float[0].size(); i++) {
           int obj_num = _output_buffer_float[0][i];
           if(obj_num != 0) {
               //std::cout << "index " << i << " object num: " << obj_num << " for class id " << class_id << std::endl;
               for(int j = 0; j < obj_num; j++) {
                       detectionsResult.push_back(_output_buffer_float[0][i+1]);
                       detectionsResult.push_back(_output_buffer_float[0][i+2]);
                       detectionsResult.push_back(_output_buffer_float[0][i+3]);
                       detectionsResult.push_back(_output_buffer_float[0][i+4]);
                       detectionsResult.push_back(class_id);
                       detectionsResult.push_back(_output_buffer_float[0][i+5]);
                       i = i+5;
                       num_dets++;
                       //std::cout << i << std::endl;
               }
           }
           class_id ++;
           if(class_id > _nclasses)
              break;
        }
        //std::cout << class_id << std::endl;
      }
#ifdef TIME_TRACE_DEBUG
      std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::high_resolution_clock::now();
      total_time = t_end - t_start;
      std::cout << BOLDBLUE << "-I- postprocessing run time: " << (double)total_time.count() << " sec" << std::endl;
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
        cobj.name = GetFinalLabel(conf, _labels[category+1]);
      #ifdef TIME_TRACE_DEBUG
        printf("T result cobj.category %d, conf %f\n", cobj.category, cobj.confidence);
      #endif
        _objects.push_back(cobj);
      }
    }
    return _objects.size();
}

void NpuBaseImpl::DrawResult(image_share_t imgData, bool needFormat)
{
    //the image data is RGB
    int width = imgData.width;
    int height = imgData.height;
    int max_dim = ( width >= height ) ? width : height;
    int w_compen = ( width >= height ) ? 0 : ((height - width) / 2);
    int h_compen = ( width >= height ) ? ((width - height) / 2) : 0;
    cv::Mat showFrame(cv::Size(width, height), CV_8UC3, (void *)imgData.data);
    if(needFormat) {
        memset(showFrame.data, 0, width * height * 3);
    }
    DrawObject(imgData, showFrame, max_dim, w_compen, h_compen);
}

void NpuBaseImpl::Release(void)
{
    pHailoPipeline->ReleaseStreamChannel(0, _stream_id);
    _initialized = false;
}

NpuBaseImpl::~NpuBaseImpl()
{
    Release();
}
