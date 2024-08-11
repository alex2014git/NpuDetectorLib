#ifndef _NPU_BASE_API_IMPL_H
#define _NPU_BASE_API_IMPL_H
#include "npu.hpp"
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include "hailo/hailort.hpp"
#include "opencv2/opencv.hpp"
#include "MultiNetworkPipeline/MultiNetworkPipeline.hpp"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"

class NpuBaseImpl : public Npu {
public:
    NpuBaseImpl();
    ~NpuBaseImpl(); // Destructor declaration

    virtual int Initialize(std::string configJsonFile, int streamId);
public:
    /// @brief 获取算法版本号
    /// @return 算法版本号
    virtual std::string GetVersion();

    virtual int Detect(image_share_t imgData, bool needPreProcess);

    virtual void DrawResult(image_share_t imgData, bool needFormat);

    /// @brief 释放npu
    virtual void Release();

protected:
    bool _nms_core             = false;
    bool _initialized          = false;
    bool _img_nv12             = false;
    int _nclasses              = 0;
    int _batch_size            = 1;
    int _letterbox_color       = 144;
    float _model_channel       = 3;  //RGB is 3, NV12 is 3/2;
    int _model_width           = 640;
    int _model_height          = 640;
    size_t _network_input_size = 1228800; //3*640*640
    float _conf_threshold      = 0.25;
    float _input_scale         = 1.0;
    std::string _stream_id     = "0";
    std::string _idName        = "";
    std::string _model_path    = "";
    std::string _verString     = "v2.0";

    hailo_format_type_t _out_format = HAILO_FORMAT_TYPE_UINT8;
    hailo_format_type_t _in_format = HAILO_FORMAT_TYPE_UINT8;
    std::vector<float> _out_scales;
    std::vector<int32_t> _out_zps;
    std::vector<std::string> _labels;
    std::vector<std::string> _output_order_by_name;
    std::vector<qp_zp_scale_t> _quantization_info;
    std::vector<std::vector<uint8_t>> _output_buffer_uint8;
    std::vector<std::vector<float>> _output_buffer_float;
    std::vector<hailo_vstream_info_t> _vstream_infos;
    std::vector<_object_roi> _objects;
    MultiNetworkPipeline *pHailoPipeline;
    rapidjson::Document _dom;
    struct timeval _start_time, _stop_time;

private:
    cv::Mat Letterbox( const cv::Mat& img, int target_width, int target_height, float &ratio, int color);
    void PreProcessing(cv::Mat &org_frame, bool needPreProcess, cv::Mat &out_frame, float &ratio);

protected:
    int InitNPU();
    int InitConfig(std::string configJsonFile, int streamId);
    void DrawObject(image_share_t imgData, cv::Mat &showFrame, int new_width, int new_height, int w_compen, int h_compen);
    template <typename T>
    MnpReturnCode NpuPorcessing(image_share_t imgData, bool needPreProcess);
    std::string GetFinalLabel(float conf, std::string label);
};

#endif // #ifndef _NPU_BASE_API_IMPL_H

