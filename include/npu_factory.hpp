#ifndef _NPU_FACTORY_H
#define _NPU_FACTORY_H

#include "npu.hpp"
#include <memory>
#include <mutex>
#include <unordered_map>

class NpuFactory {
public:
    using CreatorFunc = std::shared_ptr<Npu>(*)();

    /// @brief Register an algorithm type with its creator function (thread-safe)
    /// @param algType The algorithm type to register
    /// @param creator Function pointer that creates the Npu instance
    static void Register(algorithm algType, CreatorFunc creator);

    /// @brief Create an Npu instance for the given algorithm type (thread-safe)
    /// @param algType The algorithm type to create
    /// @return Shared pointer to the created Npu instance
    /// @throws std::invalid_argument if the algorithm type is not registered
    static std::shared_ptr<Npu> CreateNpu(algorithm algType);

    /// @brief Check if an algorithm type is registered
    /// @param algType The algorithm type to check
    /// @return true if registered, false otherwise
    static bool IsRegistered(algorithm algType);

private:
    static std::mutex& GetMutex() {
        static std::mutex mutex;
        return mutex;
    }

    static std::unordered_map<algorithm, CreatorFunc>& GetRegistry() {
        static std::unordered_map<algorithm, CreatorFunc> registry;
        return registry;
    }
};

// Macro to auto-register implementations
#define REGISTER_NPU_IMPL(algType, ImplClass) \
    namespace { \
        struct ImplClass##Registrar { \
            ImplClass##Registrar() { \
                NpuFactory::Register(algType, []() -> std::shared_ptr<Npu> { \
                    return std::make_shared<ImplClass>(); \
                }); \
            } \
        }; \
        static ImplClass##Registrar g_##ImplClass##Registrar; \
    }

#endif // #ifndef _NPU_FACTORY_H
