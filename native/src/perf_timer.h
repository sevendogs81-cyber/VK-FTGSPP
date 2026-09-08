#pragma once

#include "gs_pipeline.h"

#include <chrono>
#include <mutex>
#include <map>


namespace PerfTimer {


#define PERF_TIMER_TRAIN_STAGES \
    _(Materialize) \
    _(ProjectionForward) \
    _(TileCount) \
    _(GenerateKeys) \
    _(ComputeTileRanges) \
    _(RasterizeForward) \
    _(RasterizeBackward) \
    _(LossForwardBackward) \
    _(DynamicBackward) \
    _(ExplicitAdam) \
    _(FusedDynamicBackwardAdam) \
    _(ColorCorrection) \
    _(SparseGradientPack) \
    _(Relocation) \
    _(_Cumsum) \
    _(CalculateIndexBufferOffset) \
    _(SortRTS) \
    _(CopyTrainImageToDevice) \
    _(ComputeSSIMGradient) \
    _(_RasterizeBackwardScheduling_PerPixel) \
    _(_RasterizeBackwardScheduling_PerSplat) \
    _(_RasterizeBackwardScheduling_Tensor_0_8_0) \
    _(_RasterizeBackwardScheduling_Tensor_0_8_8) \
    _(_RasterizeBackwardScheduling_Tensor_1_16_0) \
    _(_RasterizeBackwardScheduling_N) \

#define _(name) name ,
enum TrainStage {
    PERF_TIMER_TRAIN_STAGES
    END
};
#undef _


void reset();

void hostTic();
void hostToc();

template<TrainStage stage>
struct Timer {
    VulkanGSPipeline *module;
    std::chrono::time_point<std::chrono::high_resolution_clock> then;
    // static std::mutex mutex;

    Timer(VulkanGSPipeline *module);
    ~Timer();
};

void pushMarker(VulkanGSPipeline *module);
void popMarkers(VulkanGSPipeline *module);

std::vector<std::pair<int, int>> takeMarks();
std::vector<std::pair<size_t, double>> update(std::vector<double> times);
std::vector<std::pair<size_t, double>> update(
    const std::vector<double>& times,
    const std::vector<std::pair<int, int>>& markers);

std::map<std::string, std::tuple<size_t, double>> get_summary();


}  // namespace PerfTimer
