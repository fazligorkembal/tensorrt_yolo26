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

void modelInfo(nvinfer1::INetworkDefinition* network)  // TODO: remove this after debugging
{
    std::cout << "=================== Model Information ===================" << std::endl;
    std::cout << "Number of layers: " << network->getNbLayers() << std::endl;
    for (int i = 0; i < network->getNbLayers(); ++i) {
        nvinfer1::ILayer* layer = network->getLayer(i);
        std::cout << "Layer " << i << ": " << layer->getName() << " [Type: " << static_cast<int>(layer->getType())
                  << "]" << std::endl;

        std::cout << "\tNumber of inputs: " << layer->getNbInputs() << std::endl;
        for (int j = 0; j < layer->getNbInputs(); ++j) {
            nvinfer1::ITensor* inputTensor = layer->getInput(j);
            if (inputTensor) {
                nvinfer1::Dims dims = inputTensor->getDimensions();
                std::cout << "\t\tInput " << j << " dimensions: ";
                for (int d = 0; d < dims.nbDims; ++d) {
                    std::cout << dims.d[d] << " ";
                }
                std::cout << std::endl;
            }
        }

        std::cout << "\tNumber of outputs: " << layer->getNbOutputs() << std::endl;
        for (int j = 0; j < layer->getNbOutputs(); ++j) {
            nvinfer1::ITensor* outputTensor = layer->getOutput(j);
            if (outputTensor) {
                nvinfer1::Dims dims = outputTensor->getDimensions();
                std::cout << "\t\tOutput " << j << " dimensions: ";
                for (int d = 0; d < dims.nbDims; ++d) {
                    std::cout << dims.d[d] << " ";
                }
                std::cout << std::endl;
            }
        }

        // Conv layer ise detaylari goster
        if (layer->getType() == nvinfer1::LayerType::kCONVOLUTION) {
            nvinfer1::IConvolutionLayer* convLayer = static_cast<nvinfer1::IConvolutionLayer*>(layer);

            nvinfer1::Dims kernelSize = convLayer->getKernelSizeNd();
            std::cout << "\t[CONV] Kernel shape: ";
            for (int d = 0; d < kernelSize.nbDims; ++d) {
                std::cout << kernelSize.d[d] << " ";
            }
            std::cout << std::endl;

            nvinfer1::Dims stride = convLayer->getStrideNd();
            std::cout << "\t[CONV] Strides: ";
            for (int d = 0; d < stride.nbDims; ++d) {
                std::cout << stride.d[d] << " ";
            }
            std::cout << std::endl;

            nvinfer1::Dims padding = convLayer->getPaddingNd();
            std::cout << "\t[CONV] Pads: ";
            for (int d = 0; d < padding.nbDims; ++d) {
                std::cout << padding.d[d] << " ";
            }
            std::cout << std::endl;

            nvinfer1::Dims dilation = convLayer->getDilationNd();
            std::cout << "\t[CONV] Dilations: ";
            for (int d = 0; d < dilation.nbDims; ++d) {
                std::cout << dilation.d[d] << " ";
            }
            std::cout << std::endl;

            std::cout << "\t[CONV] Groups: " << convLayer->getNbGroups() << std::endl;
            std::cout << "\t[CONV] Num output maps: " << convLayer->getNbOutputMaps() << std::endl;

            // Weight ve Bias bilgileri
            nvinfer1::Weights kernelWeights = convLayer->getKernelWeights();
            nvinfer1::Weights biasWeights = convLayer->getBiasWeights();

            int outCh = convLayer->getNbOutputMaps();
            int groups = convLayer->getNbGroups();
            int kH = kernelSize.d[0];
            int kW = kernelSize.d[1];

            // input channels hesapla: weight_count = outCh * (inCh/groups) * kH * kW
            int inChPerGroup = (kH * kW > 0 && outCh > 0) ? kernelWeights.count / (outCh * kH * kW) : 0;
            int inCh = inChPerGroup * groups;

            std::cout << "\t[CONV] Weight count: " << kernelWeights.count << " [" << outCh << " x " << inChPerGroup
                      << " x " << kH << " x " << kW << "]" << std::endl;
            std::cout << "\t[CONV] Bias count: " << biasWeights.count;
            if (biasWeights.count > 0) {
                std::cout << " [" << biasWeights.count << "]";
            } else {
                std::cout << " (no bias)";
            }
            std::cout << std::endl;
            std::cout << "\t[CONV] Input channels (calculated): " << inCh << std::endl;
        }
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

    nvinfer1::IElementWiseLayer* conv0 =
            convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), {3, 3}, 2, "model.0");

    nvinfer1::IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, *conv0->getOutput(0),
                                                    get_width(128, gw, max_channels), {3, 3}, 2, "model.1");

    bool c3k = false;
    if (type == "m" || type == "l" || type == "x") {
        c3k = true;
    }

    nvinfer1::IElementWiseLayer* conv2 =
            C3K2(network, weightMap, *conv1->getOutput(0), get_width(128, gw, max_channels),
                 get_width(256, gw, max_channels), get_depth(2, gd), c3k, true, false, 0.25, "model.2");

    nvinfer1::IElementWiseLayer* conv3 = convBnSiLU(network, weightMap, *conv2->getOutput(0),
                                                    get_width(256, gw, max_channels), {3, 3}, 2, "model.3");

    nvinfer1::IElementWiseLayer* conv4 =
            C3K2(network, weightMap, *conv3->getOutput(0), get_width(256, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), c3k, true, false, 0.25, "model.4");

    nvinfer1::IElementWiseLayer* conv5 = convBnSiLU(network, weightMap, *conv4->getOutput(0),
                                                    get_width(512, gw, max_channels), {3, 3}, 2, "model.5");

    nvinfer1::IElementWiseLayer* conv6 =
            C3K2(network, weightMap, *conv5->getOutput(0), get_width(512, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), true, true, false, 0.5, "model.6");

    nvinfer1::IElementWiseLayer* conv7 = convBnSiLU(network, weightMap, *conv6->getOutput(0),
                                                    get_width(1024, gw, max_channels), {3, 3}, 2, "model.7");

    nvinfer1::IElementWiseLayer* conv8 =
            C3K2(network, weightMap, *conv7->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), get_depth(2, gd), true, true, false, 0.5, "model.8");

    nvinfer1::IElementWiseLayer* conv9 =
            SPPF(network, weightMap, *conv8->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), 5, true, "model.9");

    nvinfer1::IElementWiseLayer* conv10 =
            C2PSA(network, weightMap, *conv9->getOutput(0), get_width(1024, gw, max_channels),
                  get_width(1024, gw, max_channels), get_depth(2, gd), 0.5, "model.10");
    /////////////////////////////////////////////////////

    conv10->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*conv10->getOutput(0));
    config->setMaxWorkspaceSize(1 << 30);
    // config->setFlag(nvinfer1::BuilderFlag::kFP16); // TODO: make this configurable with config file
    modelInfo(network);  // TODO: remove this after debugging
    //std::cout << "Output channels: " << c2 << ", " << c3 << std::endl;  // TODO: remove after debugging

    std::cout << "Building engine, please wait for a while..." << std::endl;
    nvinfer1::IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    delete network;

    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return serialized_model;
}
