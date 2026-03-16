#include "common/math.hpp"

namespace common
{
  // Implementations of raw array softmax functions

  void softmax_1D(float *data, const int size)
  {
    float sum = 0;
    for (int i = 0; i < size; i++)
      sum += std::exp(data[i]);
    for (int i = 0; i < size; i++)
      data[i] = std::exp(data[i]) / sum;
  }

  void softmax_2D(float *data, const int num_rows, const int num_cols)
  {
    int size = num_rows * num_cols;
    for (int i = 0; i < size; i += num_cols)
      softmax_1D(&data[i], num_cols);
  }

  void sigmoid(float *data, const int size)
  {
    for (int i = 0; i < size; i++)
      data[i] = 1.0f / (1.0f + std::exp(-1.0 * data[i]));
  }
}