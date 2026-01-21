#include <assert.h>
#include <math.h>
#include <iostream>
#include <vector>
#include "cuda_utils.h"
#include "types.h"
#include "yololayer.h"

namespace Tn {
template <typename T>
void write(char*& buffer, const T& val) {
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
void read(const char*& buffer, T& val) {
    val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
}
}  // namespace Tn

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

namespace nvinfer1 {
YoloLayerPlugin::YoloLayerPlugin(int classCount, int numberOfPoints, float confThresholdKeypoints, int inputWidth,
                                 int inputHeight, int maxDetections, bool isSegmentation, bool isPose, bool isObb,
                                 const int* strides, int stridesLength) {
    mClassCount = classCount;
    mNumberOfPoints = numberOfPoints;
    mConfThresholdKeypoints = confThresholdKeypoints;
    mInputWidth = inputWidth;
    mInputHeight = inputHeight;
    mMaxDetections = maxDetections;
    mIsSegmentation = isSegmentation;
    mIsPose = isPose;
    mIsObb = isObb;
    mStridesLength = stridesLength;
    mStrides = new int[stridesLength];
    memcpy(mStrides, strides, stridesLength * sizeof(int));

    std::cout << "YoloLayerPlugin created with the following parameters:" << std::endl;
    std::cout << "  Class Count: " << mClassCount << std::endl;
    std::cout << "  Number of Points: " << mNumberOfPoints << std::endl;
    std::cout << "  Confidence Threshold Keypoints: " << mConfThresholdKeypoints << std::endl;
    std::cout << "  Input Width: " << mInputWidth << std::endl;
    std::cout << "  Input Height: " << mInputHeight << std::endl;
    std::cout << "  Max Detections: " << mMaxDetections << std::endl;
    std::cout << "  Is Segmentation: " << mIsSegmentation << std::endl;
    std::cout << "  Is Pose: " << mIsPose << std::endl;
    std::cout << "  Is OBB: " << mIsObb << std::endl;
    std::cout << "  Strides Length: " << mStridesLength << std::endl;
    std::cout << "  Strides: ";
    for (int i = 0; i < mStridesLength; ++i) {
        std::cout << mStrides[i] << " ";
    }
    std::cout << std::endl;

}

YoloLayerPlugin::~YoloLayerPlugin() {
    if (mStrides != nullptr) {
        delete[] mStrides;
        mStrides = nullptr;
    }
}

YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length) {
    using namespace Tn;
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mClassCount);
    read(d, mNumberOfPoints);
    read(d, mConfThresholdKeypoints);
    read(d, mThreadCount);
    read(d, mInputWidth);
    read(d, mInputHeight);
    read(d, mMaxDetections);
    read(d, mStridesLength);
    mStrides = new int[mStridesLength];
    for (int i = 0; i < mStridesLength; ++i) {
        read(d, mStrides[i]);
    }
    read(d, mIsSegmentation);
    read(d, mIsPose);
    read(d, mIsObb);

    assert(d == a + length);
}

void YoloLayerPlugin::serialize(void* buffer) const TRT_NOEXCEPT {

    using namespace Tn;
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mClassCount);
    write(d, mNumberOfPoints);
    write(d, mConfThresholdKeypoints);
    write(d, mThreadCount);
    write(d, mInputWidth);
    write(d, mInputHeight);
    write(d, mMaxDetections);
    write(d, mStridesLength);
    for (int i = 0; i < mStridesLength; ++i) {
        write(d, mStrides[i]);
    }
    write(d, mIsSegmentation);
    write(d, mIsPose);
    write(d, mIsObb);

    assert(d == a + getSerializationSize());
}

size_t YoloLayerPlugin::getSerializationSize() const TRT_NOEXCEPT {
    return sizeof(mClassCount) + sizeof(mNumberOfPoints) + sizeof(mConfThresholdKeypoints) + sizeof(mThreadCount) +
           sizeof(mInputWidth) + sizeof(mInputHeight) + sizeof(mMaxDetections) + sizeof(mStridesLength) +
           mStridesLength * sizeof(int) + sizeof(mIsSegmentation) + sizeof(mIsPose) + sizeof(mIsObb);
}

int YoloLayerPlugin::initialize() TRT_NOEXCEPT {
    return 0;
}

nvinfer1::Dims YoloLayerPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                                    int nbInputDims) TRT_NOEXCEPT {
    int total_size = mMaxDetections * sizeof(Detection) / sizeof(float);
    return nvinfer1::Dims3(total_size + 1, 1, 1);
}

void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char* YoloLayerPlugin::getPluginNamespace() const TRT_NOEXCEPT {
    return mPluginNamespace;
}

nvinfer1::DataType YoloLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                      int nbInputs) const TRT_NOEXCEPT {
    return nvinfer1::DataType::kFLOAT;
}

bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted,
                                                   int nbInputs) const TRT_NOEXCEPT {
    return false;
}

bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT {
    return false;
}

void YoloLayerPlugin::configurePlugin(nvinfer1::PluginTensorDesc const* in, int32_t nbInput,
                                      nvinfer1::PluginTensorDesc const* out, int32_t nbOutput) TRT_NOEXCEPT {}

void YoloLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext,
                                      IGpuAllocator* gpuAllocator) TRT_NOEXCEPT {}

void YoloLayerPlugin::detachFromContext() TRT_NOEXCEPT {}

const char* YoloLayerPlugin::getPluginType() const TRT_NOEXCEPT {
    return "YoloLayer_TRT";
}

const char* YoloLayerPlugin::getPluginVersion() const TRT_NOEXCEPT {
    return "1";
}

void YoloLayerPlugin::destroy() TRT_NOEXCEPT {
    delete this;
}

nvinfer1::IPluginV2IOExt* YoloLayerPlugin::clone() const TRT_NOEXCEPT {

    YoloLayerPlugin* p =
            new YoloLayerPlugin(mClassCount, mNumberOfPoints, mConfThresholdKeypoints, mInputWidth, mInputHeight,
                                mMaxDetections, mIsSegmentation, mIsPose, mIsObb, mStrides, mStridesLength);
    p->setPluginNamespace(mPluginNamespace);
    return p;
}

int YoloLayerPlugin::enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace,
                             cudaStream_t stream) TRT_NOEXCEPT {
    gatherKernelLauncher(reinterpret_cast<const float* const*>(inputs), reinterpret_cast<float*>(outputs[0]), stream,
                         mInputWidth, mInputHeight, batchSize);
    return 0;
}

__device__ float Logist(float data) {
    return 1.f / (1.f + expf(-data));
}

__global__ void gatherKernel(const float* input, float* output, int numElements, int maxoutobject, const int grid_h,
                             int grid_w, const int stride, int classes, int nk, float confkeypoints, int outputElem,
                             bool is_segmentation, bool is_pose, bool is_obb) {}

void YoloLayerPlugin::gatherKernelLauncher(const float* const* inputs, float* outputs, cudaStream_t stream,
                                           int modelInputWidth, int modelInputHeight, int batchSize) {
    std::cout << "gatherKernelLauncher is not implemented yet." << std::endl;
}

PluginFieldCollection YoloLayerPluginCreator::mFC{};
std::vector<PluginField> YoloLayerPluginCreator::mPluginAttributes;

YoloLayerPluginCreator::YoloLayerPluginCreator() {
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* YoloLayerPluginCreator::getPluginName() const TRT_NOEXCEPT {
    return "YoloLayer_TRT";
}

const char* YoloLayerPluginCreator::getPluginVersion() const TRT_NOEXCEPT {
    return "1";
}

const PluginFieldCollection* YoloLayerPluginCreator::getFieldNames() TRT_NOEXCEPT {
    return &mFC;
}

IPluginV2IOExt* YoloLayerPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT {
    assert(fc->nbFields == 1);
    assert(strcmp(fc->fields[0].name, "combinedInfo") == 0);
    const int* combinedInfo = static_cast<const int*>(fc->fields[0].data);
    int net_info_count = 9;
    int class_count = combinedInfo[0];
    int number_of_points = combinedInfo[1];
    float conf_threshold_keypoints = combinedInfo[2];
    int input_width = combinedInfo[3];
    int input_height = combinedInfo[4];
    int max_detections = combinedInfo[5];
    bool is_segmentation = combinedInfo[6];
    bool is_pose = combinedInfo[7];
    bool is_obb = combinedInfo[8];
    const int* px_array = combinedInfo + net_info_count;
    int px_array_length = fc->fields[0].length - net_info_count;

    YoloLayerPlugin* plugin = new YoloLayerPlugin(class_count, number_of_points, conf_threshold_keypoints, input_width,
                                                  input_height, max_detections, is_segmentation, is_pose, is_obb,
                                                  px_array, px_array_length);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2IOExt* YoloLayerPluginCreator::deserializePlugin(const char* name, const void* serialData,
                                                           size_t serialLength) TRT_NOEXCEPT {
    YoloLayerPlugin* plugin = new YoloLayerPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

}  // namespace nvinfer1