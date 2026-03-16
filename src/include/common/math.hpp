/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#pragma once

#include "xtensor/xarray.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xio.hpp"


namespace common
{

    //-------------------------------
    // COMMON FILTERS
    //-------------------------------
    void softmax_1D(float *data, const int size);

    void softmax_2D(float *data, const int num_rows, const int num_cols);

    void sigmoid(float *data, const int size);

}
