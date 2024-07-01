#ifndef _NPU_FACTORY_H
#define _NPU_FACTORY_H

#include "npu.hpp"
#include <memory>
//#include <stdexcept>

class NpuFactory {
public:
    static std::shared_ptr<Npu> CreateNpu(algorithm algType);
};

#endif // #ifndef _NPU_FACTORY_H
