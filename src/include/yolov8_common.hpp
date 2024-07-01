#include "common/math.hpp"
#include "common/hailo_objects.hpp"
#include "common/tensors.hpp"
#include "common/nms.hpp"

float dequantize_value(uint8_t val, float32_t qp_scale, float32_t qp_zp);

                   
void dequantize_box_values(xt::xarray<float>& dequantized_outputs, int index, 
                        xt::xarray<uint8_t>& quantized_outputs,
                        size_t dim1, size_t dim2, float32_t qp_scale, float32_t qp_zp);

float iou_calc(const HailoBBox &box_1, const HailoBBox &box_2);

std::vector<xt::xarray<double>> get_centers(std::vector<int>& strides, std::vector<int>& network_dims,
                                        std::size_t boxes_num, int strided_width, int strided_height);