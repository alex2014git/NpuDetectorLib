#include "npu_factory.hpp"
#include <opencv2/opencv.hpp>
#include <getopt.h>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>
#include <iostream>
#include <filesystem>

#define MAX_NAME_LEN 128

char input_file_name[MAX_NAME_LEN] = "input.mp4";
char output_file_name[MAX_NAME_LEN] = "output.mp4";
char model_json_path[MAX_NAME_LEN] = "models/yolov5s.json";
char running_alg[MAX_NAME_LEN] = "yolov5";
int frame_count = 1;
int thread_count = 1;

std::mutex mtx;

void parse_args(int argc, char **argv)
{
    int c;
    optind = 0;
    while (1) {
        int option_index = 0;
        static struct option long_options[] = {
            {"input_file_name", required_argument, 0, 'i'},
            {"output_file_name", required_argument, 0, 'o'},
            {"model_json_file", required_argument, 0, 'm'},
            {"running_alg", required_argument, 0, 'a'},
            {"frame_count", required_argument, 0, 'f'},
            {"thread_count", required_argument, 0, 't'},
            {"help", no_argument, 0, 'h'},
            {0, 0, 0, 0}
        };

        c = getopt_long(argc, argv, "i:o:m:a:f:t:h?", long_options, &option_index);
        if (c == -1)
            break;

        switch (c) {
        case 'i':
            memset(input_file_name, 0, MAX_NAME_LEN);
            snprintf(input_file_name, MAX_NAME_LEN, "%s", optarg);
            break;
        case 'o':
            memset(output_file_name, 0, MAX_NAME_LEN);
            snprintf(output_file_name, MAX_NAME_LEN, "%s", optarg);
            break;
        case 'm':
            memset(model_json_path, 0, MAX_NAME_LEN);
            snprintf(model_json_path, MAX_NAME_LEN, "%s", optarg);
            break;
        case 'a':
            memset(running_alg, 0, MAX_NAME_LEN);
            snprintf(running_alg, MAX_NAME_LEN, "%s", optarg);
            break;
        case 'f':
            frame_count = atoi(optarg);
            break;
        case 't':
            thread_count = atoi(optarg);
            break;
        case 'h':
        case '?':
            printf("Usage: %s\n"
                   " -i  input file name (default: input.mp4)\n"
                   " -o  output file name (default: output.mp4)\n"
                   " -m  model json file path (default: yolov5s.json)\n"
                   " -a  running alg: base, yolov5, yolov8, yolov8_pose, yolov8_seg (default: yolov5)\n"
                   " -f  frame count (default: 1)\n"
                   " -t  thread count (default: 1)\n",
                   argv[0]);
            exit(-1);
        default:
            printf("?? getopt returned character code 0%o ??\n", c);
        }
    }
}

void process_video(int thread_id, const std::string& input_file, const std::string& output_file, const std::string& model_file, const std::string& alg, int frame_cnt)
{
    auto npuBase = NpuFactory::CreateNpu(Npu::str2AlgEnum(alg.c_str()));
    npuBase->Initialize(model_file, thread_id);

    cv::VideoCapture cap(input_file);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file: " << input_file << std::endl;
        return;
    }
    int video_fps = cap.get(cv::CAP_PROP_FPS);
    cap.set(cv::CAP_PROP_FRAME_COUNT, 0);
    frame_cnt = cap.get(cv::CAP_PROP_FRAME_COUNT);

    cv::VideoWriter writer;
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    writer.open(output_file, codec, video_fps, cv::Size(frame_width, frame_height));

    if (!writer.isOpened()) {
        std::cerr << "Error opening video writer: " << output_file << std::endl;
        return;
    }

    cv::Mat frame;
    int processed_frames = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    while (cap.read(frame) && (frame_cnt == 0 || processed_frames < frame_cnt)) {
        image_share_t imgData;
        imgData.data = (void*)frame.datastart;
        imgData.width = frame.cols;
        imgData.height = frame.rows;
        imgData.ch = 3;

        npuBase->Detect(imgData, true);
        npuBase->DrawResult(imgData, false);

        writer.write(frame);
        processed_frames++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    double fps = processed_frames / duration.count();

    {
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << "Thread " << thread_id << " processed " << processed_frames << " frames in "
                  << duration.count() << " seconds. FPS: " << fps << std::endl;
    }

    cap.release();
    writer.release();
}

std::string ensure_jpg_extension(const std::string& filename) {
  std::string output_image_file = filename;
  // Check if there's a dot (.) in the filename
  if (output_image_file.find('.') == std::string::npos) {
    // No dot, assume no extension and add ".jpg"
    output_image_file += ".jpg";
  } else {
    // There's a dot, check if it's followed by "jpg" (case-insensitive)
    size_t dot_pos = output_image_file.find('.');
    // replace extension
    output_image_file.replace(dot_pos + 1, output_image_file.size() - dot_pos - 1, "jpg");
  }
  return output_image_file;
}

void process_image(int thread_id, const std::string& input_file, const std::string& output_file, const std::string& model_file, const std::string& alg, int frame_cnt)
{
    auto npuBase = NpuFactory::CreateNpu(Npu::str2AlgEnum(alg.c_str()));
    npuBase->Initialize(model_file, thread_id);

    cv::Mat frame = cv::imread(input_file);
    if (frame.empty()) {
        std::cerr << "Error opening image file: " << input_file << std::endl;
        return;
    }

    int processed_frames = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    while (processed_frames < frame_cnt) {
        image_share_t imgData;
        imgData.data = (void*)frame.datastart;
        imgData.width = frame.cols;
        imgData.height = frame.rows;

        npuBase->Detect(imgData, true);
        npuBase->DrawResult(imgData, false);

        processed_frames++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    double fps = processed_frames / duration.count();

    {
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << "Thread " << thread_id << " processed " << processed_frames << " frames in "
                  << duration.count() << " seconds. FPS: " << fps << std::endl;
    }

    std::string output_image_file = ensure_jpg_extension(output_file);
    cv::imwrite(output_image_file, frame);
}

int main(int argc, char **argv) {
    parse_args(argc, argv);

    std::vector<std::thread> threads;
    bool is_video = (std::filesystem::path(input_file_name).extension() == ".mp4");

    for (int i = 0; i < thread_count; ++i) {
        std::string thread_output_file = output_file_name;
        if (thread_count > 1) {
            thread_output_file.insert(thread_output_file.find_last_of('.'), "_" + std::to_string(i));
        }
        if (is_video) {
            threads.emplace_back(process_video, i, input_file_name, thread_output_file, model_json_path, running_alg, frame_count);
        } else {
            threads.emplace_back(process_image, i, input_file_name, thread_output_file, model_json_path, running_alg, frame_count);
        }
    }

    for (auto& t : threads) {
        t.join();
    }

    return 0;
}
