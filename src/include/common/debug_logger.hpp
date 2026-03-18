#ifndef DEBUG_LOGGER_HPP
#define DEBUG_LOGGER_HPP

#include <iostream>

#ifdef TIME_TRACE_DEBUG
    #define NPU_DEBUG(msg) \
        do { std::cout << "[NPU] " << msg << std::endl; } while(0)
    #define NPU_DEBUG_PREFIX(prefix, msg) \
        do { std::cout << "[" << prefix << "] " << msg << std::endl; } while(0)
#else
    #define NPU_DEBUG(msg) ((void)0)
    #define NPU_DEBUG_PREFIX(prefix, msg) ((void)0)
#endif

#endif // DEBUG_LOGGER_HPP
