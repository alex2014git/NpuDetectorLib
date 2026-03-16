#ifndef _NPU_BASE_API_IMPL_H
#define _NPU_BASE_API_IMPL_H

#include "npu.hpp"
#include "npu_result_types.hpp"
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include "hailo/hailort.hpp"
#include "opencv2/opencv.hpp"
#include "MultiNetworkPipeline/MultiNetworkPipeline.hpp"
#include "async_backend.hpp"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"

// Pure abstract base class for all NPU inference implementations
// Provides common functionality: NPU initialization, preprocessing, buffer management
// Subclasses must implement Detect() and DrawResult()
class NpuBaseImpl : public Npu {
public:
    NpuBaseImpl();
    ~NpuBaseImpl();  // Destructor - base class doesn't declare virtual destructor

    // Initialize from JSON config file
    int Initialize(std::string configJsonFile, int streamId) override;

    /// @brief Get algorithm version
    std::string GetVersion() override;

    // Two-Phase Inference API
    // Phase 1: Run inference (NPU execution) - implemented in base
    int Infer(image_share_t imgData, bool needPreProcess) override;

    // Phase 2: Post-process results (CPU decoding) - subclasses must implement
    virtual int PostProcess(image_share_t imgData) = 0;

    // Get parsed results - subclasses must implement
    virtual std::vector<npu::NpuResult> GetResults() const = 0;

    // Clear results for next inference - subclasses must implement
    virtual void ClearResults() = 0;

    // Legacy API - implemented in base (calls Infer() + PostProcess())
    int Detect(image_share_t imgData, bool needPreProcess) override;

    // Pure virtual - subclasses must implement result visualization
    virtual void DrawResult(image_share_t imgData, bool needFormat) = 0;

    /// @brief Release NPU resources
    void Release() override;

    /// @brief Get model input width
    int GetModelWidth() const override { return _model_width; }

    /// @brief Get model input height
    int GetModelHeight() const override { return _model_height; }

protected:
    // Configuration and state
    bool _initialized = false;
    bool _img_nv12 = false;
    int _nclasses = 0;
    int _batch_size = 1;
    int _letterbox_color = 144;
    float _model_channel = 3;  // RGB is 3, NV12 is 3/2
    int _model_width = 640;
    int _model_height = 640;
    size_t _network_input_size = 1228800; // 3*640*640
    float _conf_threshold = 0.25f;
    float _input_scale = 1.0f;
    std::string _stream_id = "0";
    std::string _idName = "";
    std::string _model_path = "";
    std::string _verString = "v2.0";

    // HailoRT configuration
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

    // Backend and JSON config
    AsyncBackend* pAsyncBackend;
    rapidjson::Document _dom;
    struct timeval _start_time, _stop_time;

    // Protected methods for subclasses
    int InitNPU();
    int InitConfig(std::string configJsonFile, int streamId);

    // Preprocessing
    void PreProcessing(cv::Mat &org_frame, bool needPreProcess, cv::Mat &out_frame, float &ratio);
    cv::Mat Letterbox(const cv::Mat& img, int target_width, int target_height, float &ratio, int color);

    // Core NPU inference - template for uint8_t/float
    template <typename T>
    MnpReturnCode NpuPorcessing(image_share_t imgData, bool needPreProcess);
};

#endif // #ifndef _NPU_BASE_API_IMPL_H

