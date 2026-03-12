# NpuDetectorLib

This is a Hailo NPU Detector library for Hailo inference.

**Note:**
- We use code from the Hailo official GitHub repository: [Hailo-Application-Code-Examples](https://github.com/hailo-ai/Hailo-Application-Code-Examples).
- For some third-party libraries, we've copied the header files directly into this repo for convenience.

**Third-Party Libraries**

The following table lists the third-party libraries used in this project:

| Library Name   | GitHub Repository URL                           | Version |
|----------------|-------------------------------------------------|---------|
| xtl            | [xtl](https://github.com/xtensor-stack/xtl)     | 0.7.5   |
| xtensor        | [xtensor](https://github.com/xtensor-stack/xtensor) | 0.25.0  |
| xtensor-blas   | [xtensor-blas](https://github.com/xtensor-stack/xtensor-blas) | 0.21.0  |
| xsimd          | [xsimd](https://github.com/xtensor-stack/xsimd) | 11.0.0  |
| rapidjson      | [rapidjson](https://github.com/Tencent/rapidjson) |   |

**How to Build:**

```sh
cmake -H. -Bbuild
cmake --build build
```

In the build step, you can also change some of the definitions in this library:
- `HAILORT_INCLUDE`: Path to HailoRT include directory.
- `HAILORT_LIB`: Path to HailoRT library.
- `LETTER_BOX`: Enable letterboxing functionality. If set to OFF, it will use resize as the preprocess function.
- `SHOW_LABEL`: Enable show labels functionality. You can use this to decide if you want to show the detected result label or not.
- `TIME_TRACE_DEBUG`: Enable debugging and time tracking functionality. Debugging settings will show the time tracking and the result details.
- `BUILD_TESTER`: Enable build the tester

You could try:  
```sh
cmake -H. -Bbuild -DSHOW_LABEL=ON -DBUILD_TESTER=ON
```

**How to Use the Demo:**

```sh
./build/tests/TestExecutable --help
Usage: ./build/tests/TestExecutable
 -i  input file name (default: input.mp4)
 -o  output file name (default: output.mp4)
 -m  model json file path (default: yolov5s.json)
 -a  running alg: base, yolov8_pose, yolov8_seg (default: base)
 -f  frame count (default: 1)
 -t  thread count (default: 1)
```

Example for checking the model performance:
```sh
./build/tests/TestExecutable -i 2.jpg  -m models/yolov5s.json -a base -f 200 -t 10
```

Example for loading the mp4 and saving it as mp4:
```sh
./build/tests/TestExecutable -i VID.mp4 -o VID_out.mp4  -m models/yolov8s_nms.json -a base
```

Example for loading different model on different thread:
```sh
./build/tests/TestExecutable -t 2 -m models/yolov8s_nms.json -m models/yolov8s_seg.json -a base -a yolov8_seg
```

Example for using the usb camera(only support one stream):
```sh
./build/tests/TestExecutable -i /dev/video0  -m models/yolov8s_pose.json -a yolov8_pose
```

**How to Run Validation Tests:**

Build with validation tests enabled:
```sh
cmake -H. -Bbuild -DBUILD_TESTER=ON -DBUILD_VALIDATION_TESTS=ON
cmake --build build
```

Available validation tests:
- `TestModelLoading` - Verifies model JSON configs can be parsed and HEF files are accessible
- `TestSingleInference` - Tests inference produces valid outputs on a test image
- `TestThreadSafety` - Verifies concurrent access from multiple threads
- `TestAsyncBackend` - Stress tests the async backend

Run individual tests:
```sh
./build/tests/TestModelLoading
./build/tests/TestSingleInference
./build/tests/TestThreadSafety
./build/tests/TestAsyncBackend
```

All tests should report "ALL TESTS PASSED" on successful completion.

## Multi-Model Pipeline Usage

The multi-model pipeline API enables chaining multiple NPU models for complex inference workflows.

### Detection + LPR Pipeline

```cpp
#include "npu_pipeline.hpp"
using namespace npu_pipeline;

NpuPipeline pipeline;

// Add detection node
pipeline.addNpuNode("detector", ALG_YOLO_V8, "yolov8s_lp.json");

// Add LPR node (when HEF model available)
pipeline.addNpuNode("lpr", ALG_LPR, "lpr.json");

// Connect with crop + batch transforms
pipeline.addEdge(Edge::cropRoi("detector", "lpr"));     // Crop plate ROIs
pipeline.addEdge(Edge::batch("detector", "lpr", 8));    // Batch size 8

// Build with batched scheduler
pipeline.build({
    .scheduler = {
        .strategy = SchedulerConfig::BATCHED,
        .thread_pool_size = 4,
        .max_concurrent_frames = 16
    }
});

// Process frames
auto frame_id = pipeline.submit(image);
pipeline.waitForFrame(frame_id);
auto results = pipeline.getResults(frame_id);

// Access results
for (const auto& obj : results->node_outputs["lpr"]) {
    if (auto lpr = obj.getResult<LprResult>("lpr")) {
        std::cout << "Plate: " << lpr->text
                  << " (" << lpr->confidence << ")\n";
    }
}
```

### Detection + Tracking + LPR

```cpp
NpuPipeline pipeline;

// Add nodes
pipeline.addNpuNode("detector", ALG_YOLO_V8, "vehicle.json");
pipeline.addNode<TrackingNode>("tracker");
pipeline.addNpuNode("lpr", ALG_LPR, "lpr.json");

// Edges: detector -> tracker -> lpr
pipeline.addEdge(Edge::passThrough("detector", "tracker"));
pipeline.addEdge(Edge::filterUnprocessed("tracker", "lpr"));
pipeline.addEdge(Edge::batch("tracker", "lpr", 4));

pipeline.build({.scheduler = {.strategy = SchedulerConfig::BATCHED}});

// Process - each track processed only once for LPR
auto frame_id = pipeline.submit(image);
pipeline.waitForFrame(frame_id);
```

### Detection + Classification

```cpp
NpuPipeline pipeline;

// Detection node
pipeline.addNpuNode("detector", ALG_YOLO_V8, "vehicle.json");

// Classification node for vehicle type
pipeline.addNpuNode("classifier", ALG_CLASSIFICATION, "vehicle_type.json");

// Crop vehicles and batch for classification
pipeline.addEdge(Edge::cropRoi("detector", "classifier", /*class=*/0));
pipeline.addEdge(Edge::batch("detector", "classifier", 16));

pipeline.build({.scheduler = {.strategy = SchedulerConfig::BATCHED}});

// Get classification results
auto results = pipeline.getResults(frame_id);
for (const auto& obj : results->node_outputs["classifier"]) {
    if (auto cls = obj.getResult<ClassificationResult>("classifier")) {
        std::cout << "Type: " << cls->label
                  << " conf: " << cls->confidence << "\n";
    }
}
```

### Multi-Branch Pipeline

```cpp
// Detection feeding multiple downstream models
pipeline.addNpuNode("detector", ALG_YOLO_V8, "multi_class.json");
pipeline.addNpuNode("lpr", ALG_LPR, "lpr.json");
pipeline.addNpuNode("color_cls", ALG_CLASSIFICATION, "color.json");

// Branch 1: vehicles -> LPR
pipeline.addEdge(Edge::filterClass("detector", "lpr", 0));  // class 0 = vehicle
pipeline.addEdge(Edge::cropRoi("detector", "lpr"));
pipeline.addEdge(Edge::batch("detector", "lpr", 8));

// Branch 2: vehicles -> color classification
pipeline.addEdge(Edge::filterClass("detector", "color_cls", 0));
pipeline.addEdge(Edge::cropRoi("detector", "color_cls"));
pipeline.addEdge(Edge::batch("detector", "color_cls", 16));

pipeline.build({.scheduler = {.strategy = SchedulerConfig::PARALLEL}});
```

## Pipeline Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scheduler.strategy` | `PARALLEL` | `SEQUENTIAL`, `PARALLEL`, or `BATCHED` |
| `scheduler.thread_pool_size` | 4 | Worker threads for PARALLEL strategy |
| `scheduler.max_concurrent_frames` | 16 | Pipeline depth (frames in flight) |
| `scheduler.batch_timeout` | 5ms | Max wait for batch fill before inference |
| `scheduler.dynamic_batching` | true | Auto-adjust batch size based on load |
| `enable_profiling` | false | Enable latency profiling |
| `auto_cleanup` | true | Automatic cleanup of old frames |
| `max_pending_frames` | 16 | Max frames before backpressure |

```cpp
PipelineConfig config;
config.scheduler.strategy = SchedulerConfig::BATCHED;
config.scheduler.thread_pool_size = 4;
config.scheduler.max_concurrent_frames = 16;
config.scheduler.batch_timeout = std::chrono::milliseconds(5);
config.enable_profiling = true;

pipeline.build(config);

// Get profiling report
std::cout << pipeline.getProfilingReport() << std::endl;
```

## Performance Tuning Guide

### Batch Size Guidelines

| Model Type | Recommended | Latency Impact | Throughput |
|------------|-------------|----------------|------------|
| YOLO Detection | 1 (full frame) | Baseline | ~75 FPS |
| LPR | 8-16 | +5-10ms | Optimal |
| Classification | 16-32 | +10-20ms | Optimal |
| Face Recognition | 8-16 | +5-15ms | Optimal |

**Tuning Process:**
1. Start with batch size = 8
2. Monitor NPU utilization (`cat /sys/class/hailo/hailo0/npu_utilization`)
3. If < 80%, increase batch size
4. If latency exceeds target, decrease batch size or batch_timeout

### Thread Pool Sizing

```cpp
// For Hailo H8L (8 cores)
config.scheduler.thread_pool_size = 4;  // Good default

// For multi-branch pipelines
config.scheduler.thread_pool_size = 6;  // More parallelism
```

### Pipeline Depth

```cpp
// Low latency mode (fewer frames in flight)
config.scheduler.max_concurrent_frames = 4;

// High throughput mode
config.scheduler.max_concurrent_frames = 32;
```

### Memory Monitoring

```cpp
auto stats = pipeline.getStats();
std::cout << "Frames: " << stats.frames_processed << "\n"
          << "Objects: " << stats.objects_detected << "\n"
          << "Avg latency: " << stats.avg_latency_ms << " ms\n"
          << "Throughput: " << stats.throughput_fps << " FPS\n";

// Per-node latency breakdown
for (const auto& [node, latency] : stats.node_latency_ms) {
    std::cout << node << ": " << latency << " ms\n";
}
```

## Troubleshooting

### Pipeline Deadlock

**Symptoms:** Frame never completes, `waitForFrame()` hangs indefinitely

**Diagnostic:**
```cpp
// Check if all nodes are completing
auto& ctx = pipeline.getContext();
auto& frame = ctx.getFrame(frame_id);
std::cout << "Pending inputs: " << frame.pending_inputs << "\n"
          << "Completed nodes: " << frame.completed_nodes << "\n";
```

**Resolution:**
- Ensure all graph edges form valid DAG (no cycles)
- Verify all nodes write output to `PipelineContext`
- Check that `markNodeCompleted()` is called

### Memory Leaks

**Symptoms:** RSS grows continuously during processing

**Diagnostic:**
```cpp
// Monitor frame count in context
size_t frame_count = pipeline.getContext().frameCount();
std::cout << "Active frames: " << frame_count << "\n";
```

**Resolution:**
- Enable `auto_cleanup` in config
- Call `cleanupFramesBefore(frame_id)` periodically
- Ensure `removeFrame()` called after results consumed

### Performance Bottlenecks

**Diagnostic Steps:**
```cpp
// 1. Enable profiling
pipeline.setProfiling(true);

// 2. Run workload
// ...

// 3. Get report
std::cout << pipeline.getProfilingReport() << std::endl;
```

**Common Issues:**

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Low throughput, high CPU | Batch size too small | Increase batch size |
| High latency spikes | Batch timeout too long | Decrease timeout |
| NPU underutilized | Sequential scheduling | Switch to PARALLEL/BATCHED |
| Frame drops | Pipeline depth too high | Reduce max_concurrent_frames |

### Batch Processing Issues

**Symptom:** Secondary model never receives inputs

**Check:**
```cpp
// Verify batch accumulator
auto& ctx = pipeline.getContext();
const auto& batch = ctx.peekBatch("lpr_node");
std::cout << "Batch size: " << batch.size() << "\n"
          << "Is ready: " << batch.isReady() << "\n";
```

**Common Fixes:**
- Ensure `Edge::batch()` has correct batch size
- Check that upstream node outputs valid `PipelineObject`s
- Verify edge transform passes objects (use `Edge::passThrough` to test)

## JSON Model Configuration Templates

### Detection Model (YOLOv8)

```json
{
    "name": "yolov8s",
    "model_path": "models/yolov8s.hef",
    "classes": 80,
    "feature_map_size": [80, 40, 20],
    "labels": ["__background__", "person", "bicycle", "car", "motorcycle",
               "airplane", "bus", "train", "truck", "boat", ...],
    "size": [640, 640, 3],
    "threshold": 0.5,
    "yolo_nms_core": false,
    "output_order_by_name": ["yolov8s/conv..."],
    "out_quantized": true,
    "out_format": "HAILO_FORMAT_TYPE_UINT8"
}
```

### LPR Model (Template - Pending HEF)

```json
{
    "name": "lpr",
    "model_path": "models/lpr.hef",
    "input_size": [224, 96, 3],
    "character_set": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "max_plate_length": 10,
    "preprocessing": {
        "color_format": "grayscale",
        "normalization": {"mean": 127.5, "std": 127.5}
    },
    "postprocessing": {
        "type": "ctc_greedy",
        "blank_index": 0
    },
    "confidence_threshold": 0.8
}
```

### Classification Model - Single Class Mode

```json
{
    "name": "vehicle_type",
    "model_path": "models/vehicle_type.hef",
    "input_size": [224, 224, 3],
    "mode": "single_class",
    "classes": ["sedan", "suv", "truck", "van", "bus"],
    "preprocessing": {
        "resize": [224, 224],
        "normalize": {"mean": [0.485, 0.456, 0.406],
                      "std": [0.229, 0.224, 0.225]}
    },
    "confidence_threshold": 0.7
}
```

### Classification Model - Top-K Mode

```json
{
    "name": "vehicle_attributes",
    "model_path": "models/vehicle_attr.hef",
    "input_size": [224, 224, 3],
    "mode": "top_k",
    "top_k": 3,
    "classes": ["red", "blue", "black", "white", "silver", ...],
    "preprocessing": {
        "resize": [224, 224],
        "normalize": {"mean": [0.485, 0.456, 0.406],
                      "std": [0.229, 0.224, 0.225]}
    },
    "output": {
        "type": "softmax",
        "return_top_k": true
    }
}
```
