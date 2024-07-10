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
 -a  running alg: base, yolov5, yolov8, yolov8_pose, yolov8_seg (default: yolov5)
 -f  frame count (default: 1)
 -t  thread count (default: 1)
```

Example for checking the model performance:
```sh
./build/tests/TestExecutable -i 2.jpg  -m models/yolov8s.json -a yolov8 -f 200 -t 10
```

Example for loading the mp4 and saving it as mp4:
```sh
./build/tests/TestExecutable -i VID.mp4 -o VID_out.mp4  -m models/yolov8s.json -a yolov8
```

Example for loading different model on different thread:
```sh
./build/tests/TestExecutable -t 2 -m models/yolov5s.json -m models/yolov8s_seg.json -a yolov5 -a yolov8_seg
```