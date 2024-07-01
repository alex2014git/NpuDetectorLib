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
    xt::xarray<int> top_k(xt::xarray<uint8_t> &data, const int k);

    xt::xarray<float> vector_normalization(xt::xarray<float> &data);

    xt::xarray<float> softmax_xtensor(xt::xarray<float> &scores);

    void softmax_1D(float *data, const int size);

    void softmax_2D(float *data, const int num_rows, const int num_cols);

    void softmax_3D(float *data, const int dim1_size, const int dim2_size, const int dim3_size);

    void sigmoid(float *data, const int size);

}
