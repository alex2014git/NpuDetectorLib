# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build library
cmake -H. -Bbuild
cmake --build build

# Build with options
cmake -H. -Bbuild -DSHOW_LABEL=ON -DBUILD_TESTER=ON
cmake --build build

# Run tests
./build/tests/TestExecutable --help
./build/tests/TestExecutable -i input.mp4 -o output.mp4 -m models/yolov5s.json -a yolo_nms -f 200 -t 10
```

## Git Pre-commit Hook

To automatically validate builds and tests before each commit, install the pre-commit hook:

```bash
# Install the hook
./scripts/install-hooks.sh

# Or manually copy the hook
cp .git/hooks/pre-commit.sample .git/hooks/pre-commit  # if not already created
```

The pre-commit hook runs:
1. `cmake --build build` - Rebuilds the project
2. `./build/tests/TestExecutable -i 2.jpg` - Runs basic validation test

Commits are blocked if build or test fails.

### CMake Options
- `HAILORT_INCLUDE`: Path to HailoRT include directory (default: `/usr/include/hailo`)
- `HAILORT_LIB`: Path to HailoRT library (default: `/usr/lib/libhailort.so`)
- `LETTER_BOX`: Enable letterboxing (default: ON)
- `SHOW_LABEL`: Enable labels display (default: OFF)
- `TIME_TRACE_DEBUG`: Enable debug/timing output (default: OFF)
- `BUILD_TESTER`: Build test executable (default: OFF)
- `BUILD_UNIT_TESTS`: Build unit tests with mocks (default: OFF)
- `BUILD_INTEGRATION_TESTS`: Build integration tests (default: OFF)
- `BUILD_HARDWARE_TESTS`: Build hardware-dependent tests (default: OFF)

## Multi-Model Pipeline Architecture

The multi-model pipeline architecture enables chaining multiple NPU models with data transformations between them. This supports complex inference workflows like detection + LPR, detection + classification, and detection + tracking + secondary inference.

### Core Components

**PipelineContext** (`include/pipeline/npu_pipeline_context.hpp`)
- Shared state container for all pipeline execution
- Holds `FrameResults` for each frame (indexed by frame_id)
- Manages `TrackState` persistence across frames
- Batch accumulation for efficient NPU inference
- Thread-safe with mutex protection
- Key API:
  - `getFrame(frame_id)` - Get or create frame results
  - `writeNodeOutput()` - Nodes write results here
  - `accumulateForBatch()` - Batch ROIs for secondary inference
  - `getTrackState()` - Persistent object tracking

**PipelineGraph** (`include/pipeline/npu_pipeline_graph.hpp`)
- DAG (Directed Acyclic Graph) structure defining pipeline topology
- Nodes = processing stages (NPU inference, tracking, transforms)
- Edges = data flow with transforms (filter, crop, batch)
- Validates cycles and disconnected components
- Key API:
  - `addNode()` - Add processing node
  - `addEdge()` - Connect nodes with transform
  - `validate()` - Check for cycles
  - `topologicalSort()` - Execution order

**PipelineScheduler** (`include/pipeline/npu_pipeline_scheduler.hpp`)
- Three scheduling strategies:
  - `SEQUENTIAL` - One node at a time, deterministic
  - `PARALLEL` - Independent nodes in parallel (thread pool)
  - `BATCHED` - Accumulate ROIs, batch inference for efficiency
- Thread pool for parallel execution
- Completion tracking per frame
- Key API:
  - `submit(frame_id, ctx)` - Queue frame for processing
  - `waitForFrame(frame_id)` - Block until frame completes
  - `getStats()` - Performance metrics

**PipelineEdge** (`include/pipeline/npu_pipeline_edge.hpp`)
- Transforms data flowing between nodes
- Transform types:
  - `PASS_THROUGH` - No transformation
  - `FILTER_CLASS` - Filter by class ID(s)
  - `FILTER_CONFIDENCE` - Filter by confidence threshold
  - `CROP_ROI` - Crop region from source frame
  - `CROP_ROI_PADDED` - Crop with padding context
  - `FILTER_TRACK_NEW` - Only new tracks
  - `FILTER_TRACK_UNPROCESSED` - Tracks not yet processed by destination
  - `FILTER_TRACK_ACTIVE` - Active (not lost) tracks
  - `BATCH_ACCUMULATE` - Accumulate for batch inference
  - `CUSTOM` - User-defined transform function
- Factory functions in `Edge::` namespace:
  - `Edge::passThrough(from, to)`
  - `Edge::filterClass(from, to, class_id)`
  - `Edge::cropRoi(from, to, target_class)`
  - `Edge::cropRoiPadded(from, to, padding_ratio)`
  - `Edge::batch(from, to, batch_size, timeout)`
  - `Edge::filterUnprocessed(from, to)`
  - `Edge::lprTrigger(from, to)`

**PipelineNode** (`include/pipeline/npu_pipeline_node.hpp`)
- Base class for all pipeline nodes
- Types:
  - `NpuInferenceNode` - NPU model inference (supports batching)
  - `TransformNode` - Image preprocessing (resize, crop, normalize)
  - `AggregateNode` - Combine results from multiple branches
- Virtual methods:
  - `processObject()` - Single object processing
  - `processBatch()` - Batch processing (override for efficiency)
  - `supportsBatching()` - Return true for NPU nodes
  - `getPreferredBatchSize()` - Hint for scheduler

### Data Flow

```
Input Frame
    ↓
[Detector Node] (YOLOv8) → DetectionResult objects
    ↓ (Edge::cropRoi)
[Cropped ROIs]
    ↓ (Edge::batch)
[Secondary Node] (LPR/Classification) → LprResult/ClassificationResult
    ↓
[Aggregator] (optional)
    ↓
FrameResults with all stage_results
```

### Thread Safety Model

- `PipelineContext` - Mutex per frame, batch accumulators
- `PipelineScheduler` - Thread pool + atomic counters
- `FrameResults` - Atomic completion flags + condition variables
- `TrackState` - Mutex in PipelineContext for updates

## Architecture Overview

This is a C++17 library for Hailo NPU inference, supporting YOLO object detection, LPR, and classification models.

### Core Components

**Npu Interface** (`include/npu.hpp`)
- Abstract base class defining the NPU API
- Algorithm types: `ALG_BASE`, `ALG_YOLO_V5`, `ALG_YOLO_V8`, `ALG_POSE`, `ALG_YOLO_V8_SEG`, `ALG_LPR`, `ALG_CLASSIFICATION`

**Implementation Classes** (`src/include/core/*.hpp`, `src/core/*.cpp`)
- `NpuBaseImpl`: Base implementation with common functionality (preprocessing, NPU init, result drawing)
- `NpuYoloImpl`: YOLO-specific post-processing
- `NpuYolov8Impl`, `NpuYolov8PoseImpl`, `NpuYolov8SegImpl`: YOLOv8 variant implementations
- `NpuYoloNmsImpl`: Hardware NMS implementation

**Backend Abstraction** (`src/include/core/npu_backend.hpp`, `src/include/backend/async_npu_backend.hpp`)
- `NpuBackend`: Interface for NPU operations (dependency injection)
- `AsyncNpuBackend`: Adapter wrapping the legacy AsyncBackend singleton
- Enables unit testing with mock backends, future backend implementations

**Result Decoding** (`src/include/pipeline/result_decoder.hpp`)
- `ResultDecoder`: Interface for algorithm-specific result decoding
- `LprDecoder`: CTC decoding for license plate recognition
- `ClassificationDecoder`: Argmax/top-k for classification models

**Post-processing** (`src/yolov8/`)
- `nms.cpp`: Non-maximum suppression
- `tensors.cpp`: Tensor handling
- `yolov8_postprocess.cpp`, `yolov8seg_postprocess.cpp`, `yolov8pose_postprocess.cpp`: Model-specific output processing

### Dependencies
- HailoRT (Hailo Runtime Library)
- OpenCV (image preprocessing/display)
- xtensor/xtl (numerical operations)
- rapidjson (model config parsing)

### Architecture Documentation
- See [ARCHITECTURE.md](ARCHITECTURE.md) for comprehensive architecture documentation

## Pipeline API Reference

### NpuPipeline Main Class

`NpuPipeline` (`include/npu_pipeline.hpp`) is the main orchestrator class.

```cpp
npu_pipeline::NpuPipeline pipeline;

// Add nodes
pipeline.addNpuNode("detector", ALG_YOLO_V8, "yolov8s.json");
pipeline.addNpuNode("lpr", ALG_LPR, "lpr.json");

// Add edges with transforms
pipeline.addEdge(Edge::cropRoi("detector", "lpr"));
pipeline.addEdge(Edge::batch("detector", "lpr", 8));

// Build with configuration
pipeline.build({
    .scheduler = {
        .strategy = SchedulerConfig::BATCHED,
        .thread_pool_size = 4,
        .max_concurrent_frames = 16,
        .batch_timeout = std::chrono::milliseconds(5)
    }
});

// Submit frame (non-blocking)
uint64_t frame_id = pipeline.submit(image);

// Wait and get results
pipeline.waitForFrame(frame_id);
auto results = pipeline.getResults(frame_id);
```

### Synchronous vs Asynchronous Processing

The pipeline supports two processing patterns:

**Synchronous (blocking):**
```cpp
// Blocks until processing completes
FrameOutput output = pipeline.process(image);
```

**Asynchronous (non-blocking):**
```cpp
// Submit and return immediately
uint64_t frame_id = pipeline.submit(image);

// Do other work while processing...

// Wait for completion when needed
pipeline.waitForFrame(frame_id);
auto results = pipeline.getResults(frame_id);
```

- Use `process()` for simple, single-frame processing or when blocking is acceptable
- Use `submit()` + `waitForFrame()` + `getResults()` for:
  - Streaming/video processing with multiple frames in flight
  - When you need to do other work between submission and result retrieval
  - Batched scheduling (multiple frames accumulated before processing)

### Pipeline Construction Patterns

**Basic Detection Pipeline:**
```cpp
NpuPipeline pipeline;
pipeline.addNpuNode("detector", ALG_YOLO_V8, "yolov8s.json");
pipeline.build();
```

**Detection + LPR Pipeline:**
```cpp
NpuPipeline pipeline;
pipeline.addNpuNode("detector", ALG_YOLO_V8, "vehicle.json");
pipeline.addNpuNode("lpr", ALG_LPR, "lpr.json");
pipeline.addEdge(Edge::cropRoi("detector", "lpr", /*class=*/0));  // Crop vehicles
pipeline.addEdge(Edge::batch("detector", "lpr", 8));              // Batch size 8
pipeline.build({.scheduler = {.strategy = BATCHED}});
```

**Detection + Tracking + LPR:**
```cpp
NpuPipeline pipeline;
pipeline.addNpuNode("detector", ALG_YOLO_V8, "vehicle.json");
pipeline.addNode<TrackingNode>("tracker");
pipeline.addNpuNode("lpr", ALG_LPR, "lpr.json");

pipeline.addEdge(Edge::passThrough("detector", "tracker"));
pipeline.addEdge(Edge::filterUnprocessed("tracker", "lpr"));
pipeline.addEdge(Edge::batch("tracker", "lpr", 4));

pipeline.build();
```

**Detection + Classification:**
```cpp
NpuPipeline pipeline;
pipeline.addNpuNode("detector", ALG_YOLO_V8, "vehicle.json");
pipeline.addNpuNode("classifier", ALG_CLASSIFICATION, "vehicle_type.json");

pipeline.addEdge(Edge::cropRoi("detector", "classifier"));
pipeline.addEdge(Edge::batch("detector", "classifier", 16));

pipeline.build();
```

### Edge Transform Types and Use Cases

| Transform | Use Case | Example |
|-----------|----------|---------|
| `PASS_THROUGH` | Forward all detections | Detection → Tracking |
| `FILTER_CLASS` | Select specific object class | Vehicle detection → LPR (class 0) |
| `FILTER_CONFIDENCE` | Filter low-confidence detections | Post-detection filtering |
| `CROP_ROI` | Extract region for secondary model | Detection → Classification |
| `CROP_ROI_PADDED` | Extract with context padding | Face detection → recognition |
| `FILTER_TRACK_NEW` | Process only new tracks | Tracking → alert on new objects |
| `FILTER_TRACK_UNPROCESSED` | Avoid re-processing | Tracking → LPR (process each plate once) |
| `FILTER_TRACK_ACTIVE` | Skip lost tracks | Tracking → downstream only active |
| `BATCH_ACCUMULATE` | Efficient NPU batching | Any → NPU node |

### Scheduler Configuration Options

```cpp
struct SchedulerConfig {
    enum Strategy {
        SEQUENTIAL,     // Deterministic, slower
        PARALLEL,       // Independent nodes parallel
        BATCHED         // Accumulate ROIs for efficiency
    };

    Strategy strategy = PARALLEL;
    size_t thread_pool_size = 4;           // Threads for PARALLEL
    size_t max_concurrent_frames = 16;      // Pipeline depth
    std::chrono::milliseconds batch_timeout{5};  // Max wait for batch fill
    bool enable_profiling = false;
    bool dynamic_batching = true;           // Auto-adjust batch size
};
```

**Strategy Selection Guide:**
- `SEQUENTIAL` - Debugging, deterministic execution
- `PARALLEL` - Multi-branch pipelines (e.g., detection + pose + seg)
- `BATCHED` - Detection + secondary inference (LPR/classification)

### Batch Size Tuning Guidelines

| Model Type | Recommended Batch | Notes |
|------------|-------------------|-------|
| YOLO Detection | 1 | Full frame, no batching needed |
| LPR | 8-16 | Variable text lengths |
| Classification | 16-32 | Fixed input size |
| Face Recognition | 8-16 | Higher resolution crops |

**Tuning Rules:**
1. Start with batch size = 8 for secondary models
2. Increase if NPU utilization < 80%
3. Decrease if latency > target (batch_timeout tradeoff)
4. Monitor with `pipeline.getStats()`

## Two-Phase Inference API

The NPU interface provides a two-phase API for fine-grained control over inference:

```cpp
// Phase 1: Run inference (NPU execution)
npu->Infer(image_data, needPreProcess);

// Phase 2: Post-process (CPU decoding)
npu->PostProcess(image_data);

// Get unified results
auto results = npu->GetResults();
for (const auto& result : results) {
    std::visit([](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, npu::DetectionResult>) {
            // Handle detection: arg.class_id, arg.confidence, arg.bbox
        } else if constexpr (std::is_same_v<T, npu::LprResult>) {
            // Handle LPR: arg.text, arg.confidence
        } else if constexpr (std::is_same_v<T, npu::ClassificationResult>) {
            // Handle classification: arg.class_id, arg.label, arg.confidence
        }
    }, result);
}

// Clear for next inference
npu->ClearResults();
```

### Legacy API (Backward Compatible)

The `Detect()` method still works and is equivalent to `Infer()` + `PostProcess()`:

```cpp
// Equivalent to: Infer() + PostProcess()
npu->Detect(image_data, needPreProcess);
auto results = npu->GetResults();
```

### Algorithm-Specific Result Types

Each algorithm type returns results through the unified `npu::NpuResult` variant:

| Algorithm | Result Type | Access Pattern |
|-----------|-------------|----------------|
| `ALG_YOLO_V5/V8/NMS` | `npu::DetectionResult` | `result.bbox.x_min`, `result.class_id`, `result.confidence` |
| `ALG_POSE` | `npu::PoseResult` | `result.detection`, `result.keypoints[]` |
| `ALG_YOLO_V8_SEG` | `npu::SegmentationResult` | `result.detection`, `result.mask[]` |
| `ALG_LPR` | `npu::LprResult` | `result.text`, `result.confidence` |
| `ALG_CLASSIFICATION` | `npu::ClassificationResult` | `result.class_id`, `result.label`, `result.top_k[]` |
| `ALG_BASE` | None (raw outputs) | Use `GetRawOutputFloat()` or `GetRawOutputUint8()` |

### Raw Output Access (ALG_BASE)

For models without built-in post-processing, access raw NPU outputs:

```cpp
auto npu = NpuFactory::CreateNpu(ALG_BASE);
npu->Initialize("models/custom.json", 0);
npu->Infer(image, true);

// Access raw output tensors
const auto& float_outputs = npu->GetRawOutputFloat();   // Float format
const auto& uint8_outputs = npu->GetRawOutputUint8();   // Quantized format

// Process outputs manually
for (const auto& tensor : float_outputs) {
    // tensor is std::vector<float>
    process_tensor(tensor.data(), tensor.size());
}
```

## Model Support Matrix

| Model Type | Algorithm Enum | Implementation Class | Status | Notes |
|------------|----------------|---------------------|--------|-------|
| Generic Base | `ALG_BASE` | `NpuBaseAlgImpl` | Ready | Simple models without NMS (LPR, Classification) |
| YOLO w/ Hardware NMS | `ALG_YOLO_NMS` | `NpuYoloNmsImpl` | Ready | YOLO with HailoRT hardware NMS |
| YOLOv5 Detection | `ALG_YOLO_V5` | `NpuYoloImpl` | Ready | YOLOv5-v7 with software NMS |
| YOLOv8 Detection | `ALG_YOLO_V8` | `NpuYolov8Impl` | Ready | YOLOv8 with software NMS |
| YOLOv8 Pose | `ALG_POSE` | `NpuYolov8PoseImpl` | Ready | Keypoint detection |
| YOLOv8 Seg | `ALG_YOLO_V8_SEG` | `NpuYolov8SegImpl` | Ready | Instance segmentation |
| LPR | `ALG_LPR` | `NpuBaseAlgImpl` | Ready | Pipeline-ready with CTC decoding |
| Classification | `ALG_CLASSIFICATION` | `NpuBaseAlgImpl` | Ready | Pipeline-ready with argmax decoding |

### Factory Registration

Algorithm-to-implementation mappings in `src/factory.cpp`:

| Algorithm | Implementation | Use Case |
|-----------|---------------|----------|
| `ALG_BASE` | `NpuBaseAlgImpl` | Generic/simple models without NMS |
| `ALG_YOLO_NMS` | `NpuYoloNmsImpl` | YOLO with hardware NMS (yolo_nms_core: true) |
| `ALG_YOLO_V5` | `NpuYoloImpl` | YOLOv5-v7 with software NMS |
| `ALG_YOLO_V8` | `NpuYolov8Impl` | YOLOv8 with software NMS |
| `ALG_POSE` | `NpuYolov8PoseImpl` | YOLOv8 pose estimation |
| `ALG_YOLO_V8_SEG` | `NpuYolov8SegImpl` | YOLOv8 instance segmentation |
| `ALG_LPR` | `NpuBaseAlgImpl` | License plate recognition |
| `ALG_CLASSIFICATION` | `NpuBaseAlgImpl` | Image classification |

### Algorithm String Mapping

| String | Enum | Description |
|--------|------|-------------|
| `base` | `ALG_BASE` | Generic base implementation |
| `yolo_nms` | `ALG_YOLO_NMS` | YOLO with hardware NMS |
| `yolov5` | `ALG_YOLO_V5` | YOLOv5 detection |
| `yolov8` | `ALG_YOLO_V8` | YOLOv8 detection |
| `yolov8_pose` | `ALG_POSE` | Pose estimation |
| `yolov8_seg` | `ALG_YOLO_V8_SEG` | Instance segmentation |
| `lpr` | `ALG_LPR` | License plate recognition |
| `classification` | `ALG_CLASSIFICATION` | Image classification |

### Adding New Model Types

1. Add algorithm enum to `npu.hpp`
2. Implement post-processing in `src/yolov8/` or `src/algorithms/`
3. Create implementation class inheriting from `NpuBaseImpl` or `NpuBaseAlgImpl`
4. Register in `src/factory.cpp`
5. Add JSON config template to `models/`
6. Update model support matrix in this doc

## Code Validation Guidelines

Checklist for validating pipeline code:

- [ ] **Graph Validation** - Call `PipelineGraph::validate()` to check for cycles
- [ ] **Node Connections** - All nodes have valid input connections (except input nodes)
- [ ] **Memory Safety** - No raw `new/delete`, use `shared_ptr` for objects
- [ ] **Thread Safety** - Mutex usage in `PipelineContext` for shared state
- [ ] **Batch Size** - Appropriate for NPU (8-32 for secondary models)
- [ ] **Edge Transforms** - Match expected data flow (crop before batch)
- [ ] **Error Handling** - Node initialization failures caught and reported
- [ ] **Resource Cleanup** - `release()` called on all nodes in destructor
- [ ] **Frame Lifecycle** - Old frames cleaned up to prevent memory growth
- [ ] **Batched Scheduler Flush** - `processBatchAccumulator` must process remaining items even when not "ready"

### Critical Bug Fixes (Reference)

**BATCHED Scheduler Flush (2026-03-16):**
- Issue: `processBatchAccumulator` checked `isBatchReady()` before processing, causing items to never flush when batch size not reached
- Fix: Changed to check `peekBatch().isEmpty()` - process any accumulated items when flushing
- Location: `src/pipeline/npu_pipeline_scheduler.cpp:processBatchAccumulator()`

### Critical Files Reference

| Component | Header | Implementation |
|-----------|--------|----------------|
| Main API | `include/npu_pipeline.hpp` | `src/pipeline/npu_pipeline.cpp` |
| Types | `include/pipeline/npu_pipeline_types.hpp` | - |
| Context | `include/pipeline/npu_pipeline_context.hpp` | `src/pipeline/npu_pipeline_context.cpp` |
| Graph | `include/pipeline/npu_pipeline_graph.hpp` | `src/pipeline/npu_pipeline_graph.cpp` |
| Scheduler | `include/pipeline/npu_pipeline_scheduler.hpp` | `src/pipeline/npu_pipeline_scheduler.cpp` |
| Nodes | `include/pipeline/npu_pipeline_node.hpp` | `src/pipeline/npu_pipeline_node.cpp` |
| Edges | `include/pipeline/npu_pipeline_edge.hpp` | `src/pipeline/npu_pipeline_edge.cpp` |
| Tracking | - | `src/pipeline/tracking_node.cpp` |
| **Backend Interface** | `src/include/core/npu_backend.hpp` | - |
| **Async Adapter** | `src/include/backend/async_npu_backend.hpp` | `src/backend/async_npu_backend.cpp` |
| **Result Decoder** | `src/include/pipeline/result_decoder.hpp` | `src/pipeline/result_decoder.cpp` |
| **Transform Engine** | `src/include/pipeline/transform_engine.hpp` | `src/pipeline/npu_pipeline_scheduler.cpp` |
| **Batch Accumulator** | `src/include/pipeline/batch_accumulator.hpp` | `src/pipeline/npu_pipeline_scheduler.cpp` |
| **Debug Logger** | `src/include/common/debug_logger.hpp` | - |
| **Scope Guard** | `src/include/common/scope_guard.hpp` | - |

### New Architecture Key Files

| Component | File | Purpose |
|-----------|------|---------|
| Backend Interface | `src/include/core/npu_backend.hpp` | NpuBackend abstraction interface for dependency injection |
| Backend Types | `src/include/core/npu_types.hpp` | Shared types (MnpReturnCode, NetworkConfig, qp_zp_scale_t) |
| Async Adapter | `src/include/backend/async_npu_backend.hpp` | AsyncBackend adapter implementing NpuBackend interface |
| Result Decoder | `src/include/pipeline/result_decoder.hpp` | Algorithm-specific result decoding (LPR CTC, Classification argmax) |
| Transform Engine | `src/include/pipeline/transform_engine.hpp` | Edge transform application engine |
| Batch Accumulator | `src/include/pipeline/batch_accumulator.hpp` | Batch accumulation for batched scheduler |
| Debug Logger | `src/include/common/debug_logger.hpp` | Conditional NPU_DEBUG logging macros |
| Scope Guard | `src/include/common/scope_guard.hpp` | RAII cleanup utilities |

## Architecture Deep Dive

### Multi-threaded Inference Flow

**Key Components:**
1. **tests/hardware/main.cpp** - Test harness that spawn N threads
2. **NpuBaseImpl** - Base class with preprocessing and NPU initialization
3. **NpuBackend** - Backend abstraction interface (replaces AsyncBackend singleton access)
4. **AsyncNpuBackend** - Adapter wrapping HailoRT async API
5. **NPUHandler** - Per-network Hailo model wrapper (inside AsyncBackend)
6. **Post-processing** - Model-specific output parsing (nms, pose, seg)

**Thread Flow:**
```
main() → process_image() (N threads)
   ↓
NpuFactory::CreateNpu() → NpuYoloImpl / NpuYolo8Impl / etc.
   ↓
Initialize() → backend_->Initialize() → backend_->AddNetwork()
   ↓
Detect() → NpuProcessing<T>()
   ↓
PreProcessing() → backend_->Infer() → backend_->ReadOutput() → PostProcessing()
```

**Dependency Injection:**
```cpp
// Algorithm implementations receive backend via constructor
auto backend = std::make_shared<AsyncNpuBackend>();
auto npu = std::make_shared<NpuYoloImpl>(backend);

// For testing, use mock backend
auto mock_backend = std::make_shared<MockNpuBackend>();
auto npu = std::make_shared<NpuYoloImpl>(mock_backend);
```

**AsyncNpuBackend Double Buffering:**
- Each NetworkInstance has 2 sets of input/output buffers (ping-pong)
- Infer() submits async request and returns immediately
- Callback marks completion when NPU finishes
- ReadOutput() waits for completion then reads from completed buffer
- No global mutex on hot path (backend-level locking only)

**Key Files Reference:**
| Component | File | Purpose |
|-----------|------|---------|
| Test harness | `tests/hardware/main.cpp` | Multi-threaded test entry point |
| NPU Interface | `include/npu.hpp` | Abstract base class definition |
| Base Implementation | `src/core/npu_base_impl.cpp` | Preprocessing, NPU init, common utilities |
| YOLOv8 Detection | `src/implementations/npu_yolov8_impl.cpp` | YOLOv8 detection post-processing |
| YOLOv8 Pose | `src/implementations/npu_yolov8_pose_impl.cpp` | Keypoint detection post-processing |
| YOLOv8 Segmentation | `src/implementations/npu_yolov8_seg_impl.cpp` | Instance segmentation post-processing |
| Pipeline Scheduler | `src/pipeline/npu_pipeline_scheduler.cpp` | Execution scheduling (296 lines) |
| Pipeline Node | `src/pipeline/npu_pipeline_node.cpp` | NPU inference node (190 lines) |
| Backend Interface | `src/include/core/npu_backend.hpp` | Backend abstraction interface |
| Async Adapter | `src/include/backend/async_npu_backend.hpp` | AsyncBackend adapter |
| NPU Handler | `src/async_backend/npu_handler.cpp` | HailoRT model wrapper |
| Post-process (YOLOv8) | `src/yolov8/yolov8_postprocess.cpp` | Detection NMS |
| Post-process (Pose) | `src/yolov8/yolov8pose_postprocess.cpp` | Keypoint filtering |
| Post-process (Seg) | `src/yolov8/yolov8seg_postprocess.cpp` | Mask filtering |

### Performance Notes

### Performance Tuning

**Expected FPS (YOLOv5s/yolov8s on Hailo H8L):**
| Threads | Expected Total FPS | Notes |
|---------|-------------------|-------|
| 1 | ~75-80 FPS | Single stream, full pipeline |
| 4 | ~280-300 FPS | Good scaling |
| 8 | ~400-450 FPS | Approaching NPU saturation |
| 16 | ~500 FPS | Hardware limited (~30 FPS × 16 pipelined) |

**If performance is poor:**
1. Check that AsyncBackend::Infer() does NOT hold global mutex
2. Verify NPUHandler::run() uses async callback (not job.wait())
3. Confirm double-buffering is active (no buffer contention)
4. Use TIME_TRACE_DEBUG=ON to profile preprocess/infer/postprocess times
