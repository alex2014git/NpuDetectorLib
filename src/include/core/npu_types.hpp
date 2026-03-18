#ifndef _NPU_TYPES_H_
#define _NPU_TYPES_H_

#include <cstdint>
#include <vector>
#include <string>
#include "hailo/hailort.h"

// Legacy return codes from MultiNetworkPipeline (extracted for compatibility)
enum class MnpReturnCode {
    SUCCESS                 = 0,
    DUPLICATED              = 1,
    NO_DATA_AVAILABLE       = 2,
    FAILED                  = -1,
    NOT_FOUND               = -2,
    RUNNING_INFERENCE       = -3,
    HAILO_NOT_INITIALIZED   = -4,
    INVALID_PARAMETER       = -5,
};

// Legacy quantization info from MultiNetworkPipeline (extracted for compatibility)
typedef struct qp_zp_scale_t {
    float32_t qp_zp;
    float32_t qp_scale;
} qp_zp_scale_t;

// Network configuration (extracted from AsyncBackend)
struct NetworkConfig {
    std::string id_name;
    std::string hef_path;
    size_t batch_size = 1;
    hailo_format_type_t out_format = HAILO_FORMAT_TYPE_FLOAT32;
    bool out_quantized = false;
    std::vector<std::string> output_order_by_name;
};

#endif // _NPU_TYPES_H_
