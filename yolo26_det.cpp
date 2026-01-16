#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "cuda_utils.h"
#include "logging.h"
// #include "model.h"
// #include "postprocess.h"
// #include "preprocess.h"
#include "utils.h"
#include "types.h"

Logger gLogger;
using namespace nvinfer1;
const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

void serialize_engine(const std::string &wts_name, std::string &engine_name, float &gd, float &gw, int &max_channels, std::string &type)
{
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();
    IHostMemory *serialized_engine = buildEngineYolo26Det(builder, config, DataType::kFLOAT, wts_name, gd, gw, max_channels, type);

    assert(serialized_engine);
    std::ofstream p(engine_name, std::ios::binary);
    if (!p)
    {
        std::cout << "could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char *>(serialized_engine->data()), serialized_engine->size());
    delete serialized_engine;
    delete config;
    delete builder;
}

bool parse_args(int argc, char **argv, std::string &wts, std::string &engine, std::string &img_dir, std::string &type,
                std::string &cuda_post_process, float &gd, float &gw, int &max_channels)
{
    if (argc < 4)
        return false;
    if (std::string(argv[1]) == "-s" && (argc == 5))
    {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        auto sub_type = std::string(argv[4]);

        if (sub_type[0] == 'n')
        {
            gd = 0.50;
            gw = 0.25;
            max_channels = 1024;
            type = "n";
        }
        else if (sub_type[0] == 's')
        {
            gd = 0.50;
            gw = 0.50;
            max_channels = 1024;
            type = "s";
        }
        else if (sub_type[0] == 'm')
        {
            gd = 0.50;
            gw = 1.00;
            max_channels = 512;
            type = "m";
        }
        else if (sub_type[0] == 'l')
        {
            gd = 1.0;
            gw = 1.0;
            max_channels = 512;
            type = "l";
        }
        else if (sub_type[0] == 'x')
        {
            gd = 1.0;
            gw = 1.50;
            max_channels = 512;
            type = "x";
        }
        else
        {
            return false;
        }
    }
    else if (std::string(argv[1]) == "-d" && argc == 5)
    {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
        cuda_post_process = std::string(argv[4]);
    }
    else
    {
        return false;
    }
    return true;
}

int main(int argc, char **argv)
{
    // yolo26_det -s ../models/yolo26n.wts ../models/yolo26n.fp32.trt n
    // yolo26_det -d ../models/yolo26n.fp32.trt ../images c
    cudaSetDevice(kGpuId);
    std::string wts_name;
    std::string engine_name;
    std::string img_dir;
    std::string cuda_post_process;
    std::string type;
    int model_bboxes;
    float gd = 0, gw = 0;
    int max_channels = 0;

    if (!parse_args(argc, argv, wts_name, engine_name, img_dir, type, cuda_post_process, gd, gw, max_channels))
    {
        std::cerr << "Arguments not right!" << std::endl;
        std::cerr << "./yolo26_det -s [.wts] [.engine] [n/s/m/l/x]  // serialize model to "
                     "plan file"
                  << std::endl;
        std::cerr << "./yolo26_det -d [.engine] ../images  [c/g]// deserialize plan file and run inference"
                  << std::endl;
        return -1;
    }

    // Create a model using the API directly and serialize it to a file
    if (!wts_name.empty())
    {
        serialize_engine(wts_name, engine_name, gd, gw, max_channels, type);
        return 0;
    }
}