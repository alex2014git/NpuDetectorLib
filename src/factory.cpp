#include "npu_factory.hpp"
#include "core/npu_base_impl.hpp"
#include "core/npu_detection_impl.hpp"
#include "core/npu_base_alg_impl.hpp"
#include "implementations/npu_yolo_nms_impl.hpp"
#include "implementations/npu_yolo_impl.hpp"
#include "implementations/npu_yolov8_impl.hpp"
#include "implementations/npu_yolov8_pose_impl.hpp"
#include "implementations/npu_yolov8_seg_impl.hpp"
#include "backend/async_npu_backend.hpp"

// Register built-in implementations
// These static initializers will register each algorithm type before main() runs
namespace {
    struct BaseRegistrar {
        BaseRegistrar() {
            // ALG_BASE uses simple base implementation (no NMS)
            NpuFactory::Register(ALG_BASE, []() -> std::shared_ptr<Npu> {
                auto npu = std::make_shared<NpuBaseAlgImpl>();
                // Create and inject the backend
                auto backend = std::make_shared<AsyncNpuBackend>();
                npu->SetBackend(backend);
                return npu;
            });
        }
    };
    static BaseRegistrar g_baseRegistrar;

    struct YoloNmsRegistrar {
        YoloNmsRegistrar() {
            // ALG_YOLO_NMS uses YOLO with hardware NMS
            NpuFactory::Register(ALG_YOLO_NMS, []() -> std::shared_ptr<Npu> {
                auto npu = std::make_shared<NpuYoloNmsImpl>();
                // Create and inject the backend
                auto backend = std::make_shared<AsyncNpuBackend>();
                npu->SetBackend(backend);
                return npu;
            });
        }
    };
    static YoloNmsRegistrar g_yoloNmsRegistrar;

    struct YoloRegistrar {
        YoloRegistrar() {
            NpuFactory::Register(ALG_YOLO_V5, []() -> std::shared_ptr<Npu> {
                auto npu = std::make_shared<NpuYoloImpl>();
                // Create and inject the backend
                auto backend = std::make_shared<AsyncNpuBackend>();
                npu->SetBackend(backend);
                return npu;
            });
        }
    };
    static YoloRegistrar g_yoloRegistrar;

    struct Yolov8Registrar {
        Yolov8Registrar() {
            NpuFactory::Register(ALG_YOLO_V8, []() -> std::shared_ptr<Npu> {
                auto npu = std::make_shared<NpuYolov8Impl>();
                // Create and inject the backend
                auto backend = std::make_shared<AsyncNpuBackend>();
                npu->SetBackend(backend);
                return npu;
            });
        }
    };
    static Yolov8Registrar g_yolov8Registrar;

    struct PoseRegistrar {
        PoseRegistrar() {
            NpuFactory::Register(ALG_POSE, []() -> std::shared_ptr<Npu> {
                auto npu = std::make_shared<NpuYolov8PoseImpl>();
                // Create and inject the backend
                auto backend = std::make_shared<AsyncNpuBackend>();
                npu->SetBackend(backend);
                return npu;
            });
        }
    };
    static PoseRegistrar g_poseRegistrar;

    struct SegRegistrar {
        SegRegistrar() {
            NpuFactory::Register(ALG_YOLO_V8_SEG, []() -> std::shared_ptr<Npu> {
                auto npu = std::make_shared<NpuYolov8SegImpl>();
                // Create and inject the backend
                auto backend = std::make_shared<AsyncNpuBackend>();
                npu->SetBackend(backend);
                return npu;
            });
        }
    };
    static SegRegistrar g_segRegistrar;

    struct LprRegistrar {
        LprRegistrar() {
            // ALG_LPR uses simple base implementation (no NMS needed)
            NpuFactory::Register(ALG_LPR, []() -> std::shared_ptr<Npu> {
                auto npu = std::make_shared<NpuBaseAlgImpl>();
                // Create and inject the backend
                auto backend = std::make_shared<AsyncNpuBackend>();
                npu->SetBackend(backend);
                return npu;
            });
        }
    };
    static LprRegistrar g_lprRegistrar;

    struct ClassificationRegistrar {
        ClassificationRegistrar() {
            // ALG_CLASSIFICATION uses simple base implementation (no NMS needed)
            NpuFactory::Register(ALG_CLASSIFICATION, []() -> std::shared_ptr<Npu> {
                auto npu = std::make_shared<NpuBaseAlgImpl>();
                // Create and inject the backend
                auto backend = std::make_shared<AsyncNpuBackend>();
                npu->SetBackend(backend);
                return npu;
            });
        }
    };
    static ClassificationRegistrar g_classificationRegistrar;
}

void NpuFactory::Register(algorithm algType, CreatorFunc creator) {
    std::lock_guard<std::mutex> lock(GetMutex());
    GetRegistry()[algType] = creator;
}

std::shared_ptr<Npu> NpuFactory::CreateNpu(algorithm algType) {
    std::lock_guard<std::mutex> lock(GetMutex());
    auto& registry = GetRegistry();
    auto it = registry.find(algType);
    if (it != registry.end()) {
        return it->second();
    }
    throw std::invalid_argument("Unsupported algorithm type");
}

bool NpuFactory::IsRegistered(algorithm algType) {
    std::lock_guard<std::mutex> lock(GetMutex());
    return GetRegistry().find(algType) != GetRegistry().end();
}
