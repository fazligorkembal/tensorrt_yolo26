#include <math.h>
#include <iostream>

#include "block.h"
// #include "calibrator.h"
#include "config.h"
#include "model.h"

static int get_width(int x, float gw, int max_channels, int divisor = 8) {
    auto channel = std::min(x, max_channels);
    channel = int(ceil((channel * gw) / divisor)) * divisor;
    return channel;
}

static int get_depth(int x, float gd) {
    if (x == 1)
        return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0)
        --r;
    return std::max<int>(r, 1);
}

void calculateStrides(nvinfer1::IElementWiseLayer* conv_layers[], int size, int reference_size, int strides[]) {
    for (int i = 0; i < size; ++i) {
        nvinfer1::ILayer* layer = conv_layers[i];
        nvinfer1::Dims dims = layer->getOutput(0)->getDimensions();
        int feature_map_size = dims.d[2];
        strides[i] = reference_size / feature_map_size;
    }
}

nvinfer1::IHostMemory* buildEngineYolo26Det(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                            int& max_channels, std::string& type)

{
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);

    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    /*******************************************************************************************************
     ******************************************  YOLO26 INPUT  **********************************************
     *******************************************************************************************************/

    nvinfer1::ITensor* data = network->addInput(kInputTensorName, dt, nvinfer1::Dims4{kBatchSize, 3, kInputH, kInputW});
    assert(data);

    /*******************************************************************************************************
    *****************************************  YOLO26 BACKBONE  ********************************************
    *******************************************************************************************************/

    nvinfer1::IElementWiseLayer* block0 =
            convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), {3, 3}, 2, "model.0");

    nvinfer1::IElementWiseLayer* block1 = convBnSiLU(network, weightMap, *block0->getOutput(0),
                                                     get_width(128, gw, max_channels), {3, 3}, 2, "model.1");

    bool c3k = false;
    if (type == "m" || type == "l" || type == "x") {
        c3k = true;
    }

    nvinfer1::IElementWiseLayer* block2 =
            C3K2(network, weightMap, *block1->getOutput(0), get_width(128, gw, max_channels),
                 get_width(256, gw, max_channels), get_depth(2, gd), c3k, true, false, 0.25, "model.2");

    nvinfer1::IElementWiseLayer* block3 = convBnSiLU(network, weightMap, *block2->getOutput(0),
                                                     get_width(256, gw, max_channels), {3, 3}, 2, "model.3");

    nvinfer1::IElementWiseLayer* block4 =
            C3K2(network, weightMap, *block3->getOutput(0), get_width(256, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), c3k, true, false, 0.25, "model.4");

    nvinfer1::IElementWiseLayer* block5 = convBnSiLU(network, weightMap, *block4->getOutput(0),
                                                     get_width(512, gw, max_channels), {3, 3}, 2, "model.5");

    nvinfer1::IElementWiseLayer* block6 =
            C3K2(network, weightMap, *block5->getOutput(0), get_width(512, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), true, true, false, 0.5, "model.6");

    nvinfer1::IElementWiseLayer* block7 = convBnSiLU(network, weightMap, *block6->getOutput(0),
                                                     get_width(1024, gw, max_channels), {3, 3}, 2, "model.7");

    nvinfer1::IElementWiseLayer* block8 =
            C3K2(network, weightMap, *block7->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), get_depth(2, gd), true, true, false, 0.5, "model.8");

    nvinfer1::IElementWiseLayer* block9 = SPPF(network, weightMap, *block8->getOutput(0),
                                               get_width(1024, gw, max_channels), get_width(1024, gw, max_channels), 5,
                                               true, "model.9");  // TODO: VERIFY THIS BLOCK FOR OTHER YOLO26 MODELS

    nvinfer1::IElementWiseLayer* block10 =
            C2PSA(network, weightMap, *block9->getOutput(0), get_width(1024, gw, max_channels),
                  get_width(1024, gw, max_channels), get_depth(2, gd), 0.5, "model.10");

    /*******************************************************************************************************
    *********************************************  YOLO26 HEAD  ********************************************
    *******************************************************************************************************/

    block10->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*block10->getOutput(0));

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 16 * (1 << 20));

#if defined(USE_FP16)
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cerr << "INT8 not supported for YOLO26 model yet." << std::endl;
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    nvinfer1::IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    delete network;

    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }
    return serialized_model;
}
