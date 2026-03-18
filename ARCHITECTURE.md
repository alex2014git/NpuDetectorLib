# NpuDetectorLib Architecture

This document describes the architecture of NpuDetectorLib, a C++17 library for Hailo NPU inference supporting YOLO object detection, LPR, and classification models.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Class Hierarchy](#class-hierarchy)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Backend Abstraction](#backend-abstraction)
5. [Result Decoding](#result-decoding)
6. [Test Structure](#test-structure)
7. [Build System](#build-system)
8. [File Organization](#file-organization)

---

## Architecture Overview

NpuDetectorLib follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│         (TestExecutable, validation tests)                  │
├─────────────────────────────────────────────────────────────┤
│                     Pipeline Layer                          │
│    (NpuPipeline, DAG execution, multi-model chaining)       │
├─────────────────────────────────────────────────────────────┤
│                    Algorithm Layer                          │
│  (NpuYoloImpl, NpuYolov8Impl, NpuBaseAlgImpl, etc.)         │
├─────────────────────────────────────────────────────────────┤
│                     Backend Layer                           │
│       (NpuBackend interface, AsyncNpuBackend adapter)       │
├─────────────────────────────────────────────────────────────┤
│                    Hardware Layer                           │
│              (HailoRT, NPU hardware)                        │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Separation of Concerns**: Each layer has a single responsibility
2. **Dependency Injection**: Backend abstraction allows testable code
3. **RAII**: Resource management through scope guards and smart pointers
4. **Type Safety**: Strong typing with variants for result handling
5. **Thread Safety**: Minimal locking, lock-free where possible

---

## Class Hierarchy

### NpuBackend Abstraction

```
                    ┌──────────────────┐
                    │   NpuBackend     │  (interface)
                    │  (core/npu_      │
                    │   backend.hpp)   │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │                             │
    ┌─────────▼──────────┐      ┌──────────▼──────────┐
    │  AsyncNpuBackend   │      │   MockNpuBackend    │
    │  (backend/async_   │      │   (tests/mock/      │
    │   npu_backend.hpp) │      │    mock_npu_        │
    │                    │      │    backend.hpp)     │
    │ Wraps AsyncBackend │      │   For unit testing  │
    │ (HailoRT wrapper)  │      │                     │
    └────────────────────┘      └─────────────────────┘
```

### Npu Algorithm Implementations

```
                    ┌──────────────────┐
                    │     NpuBase      │  (interface)
                    │   (include/npu   │
                    │     .hpp)        │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │                             │
    ┌─────────▼──────────┐      ┌──────────▼──────────┐
    │    NpuBaseImpl     │      │  NpuBaseAlgImpl     │
    │  (src/include/core/│      │ (src/include/core/  │
    │   npu_base_impl    │      │  npu_base_alg_      │
    │   .hpp)            │      │  impl.hpp)          │
    │                    │      │                     │
    │ Detection models   │      │ Simple models (LPR, │
    │ (YOLO variants)    │      │ Classification)     │
    └────────┬───────────┘      └─────────────────────┘
             │
    ┌────────┼────────┬──────────┬──────────┐
    │        │        │          │          │
    ▼        ▼        ▼          ▼          ▼
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────────┐
│NpuYolo│ │NpuYolo│ │NpuYolo│ │NpuYolo│ │NpuYoloNms │
│ Impl  │ │v8Impl │ │v8Pose │ │v8Seg  │ │   Impl    │
└───────┘ │ Impl  │ │ Impl  │ │ Impl  │ └───────────┘
          └───────┘ └───────┘ └───────┘
```

---

## Pipeline Architecture

The pipeline implements a DAG (Directed Acyclic Graph) execution model for multi-model inference workflows.

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      NpuPipeline                                │
│                   (include/npu_pipeline.hpp)                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │PipelineGraph│  │PipelineSched│  │    PipelineContext      │  │
│  │   (DAG)     │  │   (executor)│  │    (shared state)       │  │
│  └──────┬──────┘  └──────┬──────┘  └─────────────────────────┘  │
│         │                │                                       │
│         ▼                ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │              PipelineNode (base class)                  │     │
│  │  ┌──────────┐ ┌──────────┐ ┌─────────────────────┐      │     │
│  │  │NpuInferen│ │TransformN│ │ TrackingNode        │      │     │
│  │  │ceNode    │ │ode       │ │ (optional)          │      │     │
│  │  └──────────┘ └──────────┘ └─────────────────────┘      │     │
│  └─────────────────────────────────────────────────────────┘     │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │              PipelineEdge (transforms)                  │     │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │     │
│  │  │PASS_THROU│ │CROP_ROI  │ │ BATCH    │ │FILTER_CLA│   │     │
│  │  │GH        │ │          │ │          │ │SS        │   │     │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │     │
│  └─────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Input Frame
    │
    ▼
┌─────────────────┐
│  submit(frame)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Topological    │────▶│  Execute Nodes  │
│     Sort        │     │  (thread pool)  │
└─────────────────┘     └────────┬────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌────────┐    ┌────────┐   ┌────────┐
              │ Node 1 │───▶│ Node 2 │──▶│ Node 3 │
              │(detect)│    │(crop)  │   │(LPR)   │
              └────────┘    └────────┘   └────────┘
                    │            │            │
                    └────────────┴────────────┘
                                 │
                         ┌───────▼────────┐
                         │  FrameResults  │
                         │  (aggregated)  │
                         └────────────────┘
```

### Scheduler Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `SEQUENTIAL` | One node at a time, deterministic | Debugging, simple pipelines |
| `PARALLEL` | Independent nodes in parallel | Multi-branch pipelines |
| `BATCHED` | Accumulate ROIs, batch inference | Detection + secondary models |

---

## Backend Abstraction

The backend abstraction layer decouples algorithm implementations from the HailoRT hardware interface.

### NpuBackend Interface

```cpp
// src/include/core/npu_backend.hpp
class NpuBackend {
public:
    virtual ~NpuBackend() = default;

    // Initialization
    virtual bool Initialize() = 0;
    virtual bool AddNetwork(const NetworkConfig& config) = 0;

    // Inference
    virtual bool Infer(uint32_t network_id, const void* input_data,
                      size_t input_size) = 0;
    virtual bool ReadOutput(uint32_t network_id, uint32_t output_idx,
                           void* output_buffer, size_t buffer_size) = 0;

    // Utility
    virtual std::vector<qp_zp_scale_t> GetQuantizationParams(
        uint32_t network_id, uint32_t output_idx) = 0;
    virtual void Release() = 0;
};
```

### AsyncNpuBackend Adapter

The `AsyncNpuBackend` class wraps the existing `AsyncBackend` singleton to provide the `NpuBackend` interface:

```cpp
// src/include/backend/async_npu_backend.hpp
class AsyncNpuBackend : public NpuBackend {
public:
    bool Initialize() override;
    bool AddNetwork(const NetworkConfig& config) override;
    bool Infer(uint32_t network_id, const void* input_data,
              size_t input_size) override;
    // ... other methods

private:
    std::shared_ptr<AsyncBackend> backend_;
    std::unordered_map<uint32_t, NetworkInfo> networks_;
};
```

### Dependency Injection

Algorithm implementations receive the backend via constructor injection:

```cpp
// In NpuBaseImpl or factory
template<typename T>
std::shared_ptr<NpuBase> CreateNpuWithBackend(
    std::shared_ptr<NpuBackend> backend = nullptr) {

    if (!backend) {
        backend = std::make_shared<AsyncNpuBackend>();
    }
    return std::make_shared<T>(backend);
}
```

This enables:
- **Unit testing** with `MockNpuBackend`
- **Hardware-in-the-loop** testing with real HailoRT
- **Future backends** (simulation, different NPU vendors)

---

## Result Decoding

Result decoding has been abstracted to support different model types (LPR, Classification) without algorithm-specific checks in the pipeline.

### ResultDecoder Interface

```cpp
// src/include/pipeline/result_decoder.hpp
class ResultDecoder {
public:
    virtual ~ResultDecoder() = default;
    virtual std::vector<NpuResult> decode(
        const std::vector<std::vector<float>>& outputs,
        const NetworkConfig& config) = 0;
};
```

### Concrete Implementations

| Decoder | Model Type | Decoding Logic |
|---------|-----------|----------------|
| `LprDecoder` | LPR | CTC greedy decoding with blank handling |
| `ClassificationDecoder` | Classification | Argmax or top-k selection |
| `YoloDecoder` | Detection | NMS + bbox extraction (in algorithm impls) |

### Usage Pattern

```cpp
// In NpuBaseAlgImpl::PostProcess()
auto decoder = ResultDecoderFactory::Create(algorithm_type_);
auto results = decoder->decode(raw_outputs_, config_);
```

This eliminates:
- `dynamic_cast` in pipeline code
- Algorithm-specific `if/else` chains
- Hard-coded post-processing logic

---

## Test Structure

Tests are organized in a three-tier hierarchy:

```
tests/
├── common/
│   └── test_macros.hpp          # Shared test utilities
├── mock/
│   └── mock_npu_backend.hpp     # Mock backend for unit tests
├── unit/                        # Fast, isolated tests
│   ├── README.md
│   └── (unit tests using mocks)
├── integration/                 # Component integration tests
│   └── README.md
└── hardware/                    # Hardware-dependent tests
    ├── test_lpr_decoder.cpp
    ├── test_classification_decoder.cpp
    ├── test_async_backend.cpp
    ├── test_factory_mapping.cpp
    ├── test_lpr_classification.cpp
    ├── test_model_loading.cpp
    ├── test_pipeline_lpr.cpp
    ├── test_single_inference.cpp
    ├── test_thread_safety.cpp
    └── main.cpp                 # TestExecutable entry point
```

### Test Categories

| Category | Build Option | Description | Execution Time |
|----------|--------------|-------------|----------------|
| Unit | `BUILD_UNIT_TESTS` | Mock-based, no hardware | < 1 second |
| Integration | `BUILD_INTEGRATION_TESTS` | Component interactions | < 10 seconds |
| Hardware | `BUILD_HARDWARE_TESTS` | Requires Hailo NPU | Variable |

---

## Build System

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TESTER` | OFF | Build TestExecutable (hardware tests entry point) |
| `BUILD_UNIT_TESTS` | OFF | Build unit tests (mock-based) |
| `BUILD_INTEGRATION_TESTS` | OFF | Build integration tests |
| `BUILD_HARDWARE_TESTS` | ON | Build hardware-dependent tests |
| `BUILD_VALIDATION_TESTS` | OFF | Alias for hardware tests |
| `LETTER_BOX` | ON | Enable letterboxing preprocessing |
| `SHOW_LABEL` | OFF | Enable label display on output |
| `TIME_TRACE_DEBUG` | OFF | Enable debug timing output |

### Build Commands

```bash
# Standard build
cmake -H. -Bbuild
cmake --build build

# With tests
cmake -H. -Bbuild -DBUILD_TESTER=ON -DBUILD_HARDWARE_TESTS=ON
cmake --build build

# With all test types
cmake -H. -Bbuild -DBUILD_TESTER=ON -DBUILD_UNIT_TESTS=ON \
    -DBUILD_INTEGRATION_TESTS=ON -DBUILD_HARDWARE_TESTS=ON
cmake --build build
```

### Running Tests

```bash
# Run all registered tests
ctest

# Run specific test
./build/tests/TestLprDecoder
./build/tests/TestClassificationDecoder

# Run with verbose output
ctest -V
```

---

## File Organization

### Public Headers

| File | Description |
|------|-------------|
| `include/npu.hpp` | Main NPU interface abstract base class |
| `include/npu_factory.hpp` | Factory for creating NPU instances |
| `include/npu_pipeline.hpp` | Pipeline API for multi-model inference |
| `include/npu_result_types.hpp` | Result variant types (DetectionResult, LprResult, etc.) |
| `include/pipeline/npu_pipeline_types.hpp` | Pipeline-specific types |
| `include/pipeline/npu_pipeline_context.hpp` | Shared pipeline state |
| `include/pipeline/npu_pipeline_graph.hpp` | DAG structure |
| `include/pipeline/npu_pipeline_scheduler.hpp` | Execution scheduler |
| `include/pipeline/npu_pipeline_node.hpp` | Pipeline node base class |
| `include/pipeline/npu_pipeline_edge.hpp` | Edge transforms |

### Core Implementation Headers

| File | Description |
|------|-------------|
| `src/include/core/npu_backend.hpp` | Backend abstraction interface |
| `src/include/core/npu_types.hpp` | Shared types (MnpReturnCode, NetworkConfig) |
| `src/include/core/npu_base_impl.hpp` | Base implementation for detection models |
| `src/include/core/npu_base_alg_impl.hpp` | Base implementation for simple models |
| `src/include/core/npu_detection_impl.hpp` | Detection-specific base class |
| `src/include/backend/async_npu_backend.hpp` | AsyncBackend adapter |

### Pipeline Components

| File | Description |
|------|-------------|
| `src/include/pipeline/result_decoder.hpp` | Result decoding interface |
| `src/include/pipeline/transform_engine.hpp` | Transform application engine |
| `src/include/pipeline/batch_accumulator.hpp` | Batch accumulation for batched scheduler |
| `src/pipeline/npu_pipeline.cpp` | Pipeline implementation |
| `src/pipeline/npu_pipeline_scheduler.cpp` | Scheduler implementation (296 lines) |
| `src/pipeline/npu_pipeline_node.cpp` | Node implementation (190 lines) |
| `src/pipeline/npu_pipeline_edge.cpp` | Edge transforms implementation |
| `src/pipeline/npu_pipeline_context.cpp` | Context implementation |
| `src/pipeline/result_decoder.cpp` | Decoder implementations |

### Common Utilities

| File | Description |
|------|-------------|
| `src/include/common/debug_logger.hpp` | NPU_DEBUG logging macros |
| `src/include/common/scope_guard.hpp` | RAII cleanup utilities |
| `src/include/common/nms.hpp` | Non-maximum suppression |
| `src/include/common/math.hpp` | Math utilities |

### Algorithm Implementations

| File | Description |
|------|-------------|
| `src/include/implementations/npu_yolo_impl.hpp` | YOLOv5 implementation |
| `src/include/implementations/npu_yolov8_impl.hpp` | YOLOv8 implementation |
| `src/include/implementations/npu_yolov8_pose_impl.hpp` | YOLOv8 pose implementation |
| `src/include/implementations/npu_yolov8_seg_impl.hpp` | YOLOv8 segmentation implementation |
| `src/include/implementations/npu_yolo_nms_impl.hpp` | Hardware NMS implementation |
| `src/core/npu_yolo_impl.cpp` | YOLOv5 implementation |
| `src/core/npu_yolov8_impl.cpp` | YOLOv8 implementation |
| `src/core/npu_yolov8_pose_impl.cpp` | YOLOv8 pose implementation |
| `src/core/npu_yolov8_seg_impl.cpp` | YOLOv8 segmentation implementation |
| `src/core/npu_yolo_nms_impl.cpp` | Hardware NMS implementation |
| `src/core/npu_base_impl.cpp` | Base implementation |
| `src/core/npu_base_alg_impl.cpp` | Simple model base implementation |
| `src/core/npu_detection_impl.cpp` | Detection base implementation |

### Test Files

| File | Description |
|------|-------------|
| `tests/hardware/test_lpr_decoder.cpp` | LPR CTC decoding tests |
| `tests/hardware/test_classification_decoder.cpp` | Classification argmax/top-k tests |
| `tests/hardware/test_async_backend.cpp` | Async backend stress tests |
| `tests/hardware/test_factory_mapping.cpp` | Factory registration validation |
| `tests/hardware/test_lpr_classification.cpp` | LPR + Classification integration |
| `tests/hardware/test_model_loading.cpp` | Model config parsing tests |
| `tests/hardware/test_pipeline_lpr.cpp` | Detection + LPR pipeline tests |
| `tests/hardware/test_single_inference.cpp` | Single image inference tests |
| `tests/hardware/test_thread_safety.cpp` | Concurrent access tests |
| `tests/mock/mock_npu_backend.hpp` | Mock backend for unit testing |
| `tests/common/test_macros.hpp` | Shared test macros |

---

## Architecture Decisions

### ADR-001: Backend Abstraction Interface

**Decision**: Create `NpuBackend` interface to abstract HailoRT-specific code.

**Rationale**:
- Enables unit testing without hardware
- Allows future backend implementations (simulation, other NPU vendors)
- Removes singleton pattern from algorithm code

**Consequences**:
- Additional indirection layer
- Requires adapter pattern for AsyncBackend
- Simplifies testing significantly

### ADR-002: Result Decoder Pattern

**Decision**: Abstract result decoding with `ResultDecoder` interface.

**Rationale**:
- Eliminates algorithm-specific checks in pipeline
- Supports different decoding strategies (CTC, argmax, etc.)
- Enables decoder unit testing

**Consequences**:
- Small overhead of virtual dispatch
- Cleaner separation of concerns
- Easier to add new model types

### ADR-003: Scheduler Component Extraction

**Decision**: Extract `TransformEngine` and `BatchAccumulator` from scheduler.

**Rationale**:
- Reduces scheduler complexity (555 -> 296 lines)
- Separates concerns: scheduling vs transform application
- Enables independent testing of components

**Consequences**:
- More files to manage
- Clearer component boundaries
- Better testability

### ADR-004: Test Organization

**Decision**: Organize tests into unit/integration/hardware tiers.

**Rationale**:
- Fast feedback loop with unit tests
- Clear separation of hardware dependencies
- Scalable test organization

**Consequences**:
- More build options to manage
- Requires mock implementation
- Better CI/CD integration

---

## Performance Considerations

### Expected Throughput

| Configuration | Expected FPS | Notes |
|---------------|--------------|-------|
| YOLOv5s, 1 thread | ~75-80 | Single stream baseline |
| YOLOv5s, 4 threads | ~280-300 | Good scaling |
| YOLOv5s, 8 threads | ~400-450 | NPU saturation |
| YOLOv5s, 16 threads | ~500 | Hardware limited |

### Optimization Guidelines

1. **Batch Size**: Start with 8 for secondary models, adjust based on NPU utilization
2. **Thread Pool**: 4 threads is good default for H8L (8 cores)
3. **Pipeline Depth**: 16 concurrent frames for high throughput, 4 for low latency
4. **Batch Timeout**: 5ms default, decrease for lower latency

---

## Migration Notes

### From Legacy AsyncBackend

The `AsyncBackend` singleton is now wrapped by `AsyncNpuBackend`. Algorithm code should:

1. Accept `NpuBackend` via constructor injection
2. Use backend methods instead of `AsyncBackend::GetInstance()`
3. Remove direct AsyncBackend dependencies

### From MultiNetworkPipeline

The `MultiNetworkPipeline` directories have been removed. Use:

- `NpuBackend` interface for backend abstraction
- `NpuPipeline` for multi-model workflows
- `AsyncNpuBackend` for HailoRT-specific functionality

---

## References

- [CLAUDE.md](CLAUDE.md) - Developer guide for Claude Code
- [README.md](README.md) - User documentation
- HailoRT Documentation - Hailo Runtime API reference
