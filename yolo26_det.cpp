#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
// #include "postprocess.h"
#include "preprocess.h"
#include "types.h"
#include "utils.h"

Logger gLogger;
using namespace nvinfer1;
const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
int32_t debug_kOutputSize = 1 * 27001;  // TODO: remove after debugging

void serialize_engine(const std::string& wts_name, std::string& engine_name, float& gd, float& gw, int& max_channels,
                      std::string& type) {
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    IHostMemory* serialized_engine =
            buildEngineYolo26Det(builder, config, DataType::kFLOAT, wts_name, gd, gw, max_channels, type);

    assert(serialized_engine);
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cout << "could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());
    delete serialized_engine;
    delete config;
    delete builder;
}

void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine,
                        IExecutionContext** context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}

void prepare_buffer(ICudaEngine* engine, float** input_buffer_device, float** output_buffer_device,
                    float** output_buffer_host, float** decode_ptr_host, float** decode_ptr_device,
                    std::string cuda_post_process) {
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);

    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)input_buffer_device, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)output_buffer_device, kBatchSize * debug_kOutputSize * sizeof(float)));
    std::cout << "Input buffer size: " << kBatchSize * 3 * kInputH * kInputW * sizeof(float) << " bytes"
              << std::endl;  // TODO: remove after debugging
    std::cout << "Output buffer size: " << kBatchSize * debug_kOutputSize * sizeof(float) << " bytes"
              << std::endl;  // TODO: remove after debugging
    *output_buffer_host = new float[kBatchSize * debug_kOutputSize];
}

void infer(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output, int batchsize,
           float* decode_ptr_host, float* decode_ptr_device, int model_bboxes, std::string cuda_post_process) {
    // infer on the batch asynchronously, and DMA output back to host
    auto start = std::chrono::system_clock::now();
    context.enqueueV2(buffers, stream, nullptr);
    cudaMemcpyAsync(output, buffers[1], batchsize * debug_kOutputSize * sizeof(float), cudaMemcpyDeviceToHost,
                    stream);  // TODO: remove after debugging

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, std::string& img_dir, std::string& type,
                std::string& cuda_post_process, float& gd, float& gw, int& max_channels) {
    if (argc < 4)
        return false;
    if (std::string(argv[1]) == "-s" && (argc == 5)) {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        auto sub_type = std::string(argv[4]);

        if (sub_type[0] == 'n') {
            gd = 0.50;
            gw = 0.25;
            max_channels = 1024;
            type = "n";
        } else if (sub_type[0] == 's') {
            gd = 0.50;
            gw = 0.50;
            max_channels = 1024;
            type = "s";
        } else if (sub_type[0] == 'm') {
            gd = 0.50;
            gw = 1.00;
            max_channels = 512;
            type = "m";
        } else if (sub_type[0] == 'l') {
            gd = 1.0;
            gw = 1.0;
            max_channels = 512;
            type = "l";
        } else if (sub_type[0] == 'x') {
            gd = 1.0;
            gw = 1.50;
            max_channels = 512;
            type = "x";
        } else {
            return false;
        }
    } else if (std::string(argv[1]) == "-d" && argc == 5) {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
        cuda_post_process = std::string(argv[4]);
    } else {
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
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

    if (!parse_args(argc, argv, wts_name, engine_name, img_dir, type, cuda_post_process, gd, gw, max_channels)) {
        std::cerr << "Arguments not right!" << std::endl;
        std::cerr << "./yolo26_det -s [.wts] [.engine] [n/s/m/l/x]  // serialize model to "
                     "plan file"
                  << std::endl;
        std::cerr << "./yolo26_det -d [.engine] ../images  [c/g]// deserialize plan file and run inference"
                  << std::endl;
        return -1;
    }

    // Create a model using the API directly and serialize it to a file
    if (!wts_name.empty()) {
        serialize_engine(wts_name, engine_name, gd, gw, max_channels, type);
        return 0;
    }

    // Deserialize the engine from file
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    deserialize_engine(engine_name, &runtime, &engine, &context);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cuda_preprocess_init(kMaxInputImageSize);
    auto out_dims = engine->getBindingDimensions(1);

    std::cout << "Output tensor dimensions: ";  // TODO: remove after debugging
    for (int d = 0; d < out_dims.nbDims; ++d) {
        std::cout << out_dims.d[d] << " ";
    }
    std::cout << std::endl;

    model_bboxes = out_dims.d[0];
    // Prepare cpu and gpu buffers
    float* device_buffers[2];
    float* output_buffer_host = nullptr;
    float* decode_ptr_host = nullptr;
    float* decode_ptr_device = nullptr;

    // Read images from directory
    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &output_buffer_host, &decode_ptr_host,
                   &decode_ptr_device, cuda_post_process);

    cv::Mat disp_img;  // TODO: remove after debugging

    // batch predict
    for (size_t i = 0; i < file_names.size(); i += kBatchSize) {
        // Get a batch of images
        std::vector<cv::Mat> img_batch;
        std::vector<std::string> img_name_batch;
        for (size_t j = i; j < i + kBatchSize && j < file_names.size(); j++) {
            cv::Mat img = cv::imread(img_dir + "/" + file_names[j]);
            disp_img = img.clone();
            img_batch.push_back(img);
            img_name_batch.push_back(file_names[j]);
        }
        // Preprocess
        cuda_batch_preprocess(img_batch, device_buffers[0], kInputW, kInputH, stream);
        // Run inference
        infer(*context, stream, (void**)device_buffers, output_buffer_host, kBatchSize, decode_ptr_host,
              decode_ptr_device, model_bboxes, cuda_post_process);

        int det_size = sizeof(Detection) / sizeof(float);
        for (int i = 0; i < output_buffer_host[0]; i++) {
            std::cout << "Bbox: " << output_buffer_host[1 + det_size * i + 0] << " "
                      << output_buffer_host[1 + det_size * i + 1] << " " << output_buffer_host[1 + det_size * i + 2]
                      << " " << output_buffer_host[1 + det_size * i + 3]
                      << " Score: " << output_buffer_host[1 + det_size * i + 4] << std::endl;

            int xmin = static_cast<int>(output_buffer_host[1 + det_size * i + 0]);
            int ymin = static_cast<int>(output_buffer_host[1 + det_size * i + 1]);
            int xmax = static_cast<int>(output_buffer_host[1 + det_size * i + 2]);
            int ymax = static_cast<int>(output_buffer_host[1 + det_size * i + 3]);
            cv::rectangle(disp_img, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(0, 255, 0), 2);
        }
        std::cout << "disp_img.cols: " << disp_img.cols << " disp_img.rows: " << disp_img.rows << std::endl;
        cv::imwrite("result.jpg", disp_img);  // TODO: remove after debugging
        /*
        // Letterbox affine transformation parametreleri (preprocess.cu ile ayni)
        float scale = std::min((float)kInputW / disp_img.cols, (float)kInputH / disp_img.rows);
        float offset_x = -scale * disp_img.cols * 0.5f + kInputW * 0.5f;
        float offset_y = -scale * disp_img.rows * 0.5f + kInputH * 0.5f;

        // std::ofstream debug_out("full_result.txt");
        // for (int j = 0; j < kBatchSize * debug_kOutputSize; j++) {  // TODO: remove after debugging
        //     debug_out << output_buffer_host[j] << " ";
        //     if ((j + 1) % 84 == 0)
        //         debug_out << std::endl;
        // }
        // debug_out.close();

        for (int row = 0; row < 8400; row++) {  // TODO: remove after debugging
            float xmin = (output_buffer_host[row * 84 + 0] - offset_x) / scale;
            float ymin = (output_buffer_host[row * 84 + 1] - offset_y) / scale;
            float xmax = (output_buffer_host[row * 84 + 2] - offset_x) / scale;
            float ymax = (output_buffer_host[row * 84 + 3] - offset_y) / scale;
            float score = 0.0;

            for (int c = 4; c < 84; c++) {
                if (output_buffer_host[row * 84 + c] > 0.5) {
                    score = output_buffer_host[row * 84 + c];
                    printf("Class: %d, Score: %.2f\n", c - 4, score);
                    break;
                }
            }

            if (score > 0) {
                std::cout << "Box " << row << ": [" << xmin << ", " << ymin << ", " << xmax << ", " << ymax << "]"
                          << std::endl;
                cv::rectangle(disp_img, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(0, 255, 0), 2);
            }
        }

        

        std::ofstream out("output.txt");
        for (int j = 0; j < debug_kOutputSize; j++) {  // TODO: remove after debugging
            out << output_buffer_host[j] << std::endl;
        }
        out.close();
        */
    }
}