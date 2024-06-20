#include "npu_factory.hpp"
#include <opencv2/opencv.hpp>
#include <getopt.h>                /* getopt_long() */

#define MAX_NAME_LEN 128
char input_file_name[MAX_NAME_LEN];
char model_json_path[MAX_NAME_LEN];
char running_alg[MAX_NAME_LEN];

void parse_args(int argc, char **argv)
{
    int c;
    int digit_optind = 0;
    optind = 0;
    while (1) {
        int this_option_optind = optind ? optind : 1;
        int option_index = 0;
        static struct option long_options[] = {
            {"input_file_name",             required_argument,     0, 'i' },
            {"model_json_file",             required_argument,     0, 'm' },
            {"running_alg",                 required_argument,     0, 'a' },
            {"help",                        no_argument,           0, 'h' },
            {0,                             0,                     0,  0  }
        };

        c = getopt_long(argc, argv, "i:m:a:h?", long_options, &option_index);
        if (c == -1)
            break;

        switch (c) {
        case 'i':
            memset(input_file_name, 0, MAX_NAME_LEN);
            snprintf(input_file_name, MAX_NAME_LEN, "%s", optarg);
            break;
        case 'm':
            memset(model_json_path, 0, MAX_NAME_LEN);
            snprintf(model_json_path, MAX_NAME_LEN, "%s", optarg);
            break;
        case 'a':
            memset(running_alg, 0, MAX_NAME_LEN);
            snprintf(running_alg, MAX_NAME_LEN, "%s", optarg);
            break;
        case 'h':
        case '?':
            printf("Usage: %s to pull rtsp stream\n"
                       " -i  input file name(only support image for now)\n"
                       " -m  model json file path\n"
                       " -a  running alg: base, yolov5, yolov8, yolov8_pose, yolov8_seg\n",
                       argv[0]);
            exit(-1);
        default:
            printf("?? getopt returned character code 0%o ??\n", c);
        }
    }
}

int main(int argc, char **argv) {
    parse_args(argc, argv);
    // Create an instance of BASE algorithm
    auto npuBase = NpuFactory::CreateNpu(Npu::str2AlgEnum(running_alg));

    // Initialize the algorithm
    npuBase->Initialize(model_json_path, 0);

    // Use the algorithm
    image_share_t imgData;
    // Load image using cv::imread()
    auto infer_cv_img = cv::imread(input_file_name);
    imgData.data = (void*)infer_cv_img.datastart;
    imgData.width = infer_cv_img.cols;
    imgData.height = infer_cv_img.rows;
    // Fill imgData with image information
    npuBase->Detect(imgData, true);

    // Draw the result
    npuBase->DrawResult(imgData, false);

    // Save the processed image
    cv::imwrite("processed_image.jpg", infer_cv_img);
    return 0;
}
