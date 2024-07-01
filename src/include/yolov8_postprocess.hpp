/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#pragma once
#include "common/hailo_objects.hpp"
#include "common/hailo_common.hpp"

__BEGIN_DECLS
void filter_yolov8(HailoROIPtr roi, int class_nums, std::vector<std::string> labels, bool out_sigmoid, float conf_thr,
                           std::vector<int> strides,
                           std::vector<int> network_dims);
__END_DECLS
