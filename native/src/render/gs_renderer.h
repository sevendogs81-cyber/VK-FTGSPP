#pragma once

#include "device/gs_pipeline.h"

#include "device/perf_timer.h"
#include "render/scheduler.h"

#include <array>
#include <map>


PACK_STRUCT(struct VulkanGSRendererUniforms {
    uint32_t image_height;
    uint32_t image_width;
    uint32_t grid_height;
    uint32_t grid_width;
    uint32_t num_splats;
    uint32_t active_sh;
    uint32_t step;
    uint32_t camera_model;
    float fx;
    float fy;
    float cx;
    float cy;
    float dist_coeffs[4];
    float world_view_transform[16];
});


class VulkanGSRenderer : public VulkanGSPipeline {
public:
    VulkanGSRenderer();
    ~VulkanGSRenderer();

    void initialize(const std::map<std::string, std::string> &spirv_paths, int device_id,
                    bool forward_only = false, const DeviceInitOptions& options = {});
    void cleanup();

    void executeProjectionForward(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers, size_t alloc_reserve=0);
    void executeDynamicProjectionForward(
        const VulkanGSRendererUniforms& uniforms,
        VulkanGSPipelineBuffers& buffers,
        float timestamp,
        float max_duration,
        float marginal_threshold,
        size_t alloc_reserve=0);
    void executeGenerateKeys(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers);
    void executeComputeTileRanges(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers);
    void executeRasterizeForward(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers);
    std::string rasterizerCheckpoint();
    void restoreRasterizerCheckpoint(const std::string& state);

    void executeRasterizeBackward(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers);

    void executeCalculateIndexBufferOffset(VulkanGSPipelineBuffers& buffers);
    void executeSort(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers, int num_bits);
    void executeTileWorkPrepare(
        const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers);

protected:
    enum class RasterBackwardImpl {
        PerPixel = 0,
        PerSplat = 1,
        Tensor_0_8_0 = 2,
        Tensor_0_8_8 = 3,
        Tensor_1_16_0 = 4,
        size = 5,
        Default = 0x3fffffff,
    };
    struct RasterBackwardKey {
        uint32_t image_width = 0;
        uint32_t image_height = 0;
        uint32_t tile_density_bucket = 0;
        uint32_t active_sh = 0;

        bool operator<(const RasterBackwardKey& other) const {
            if (image_width != other.image_width)
                return image_width < other.image_width;
            if (image_height != other.image_height)
                return image_height < other.image_height;
            if (tile_density_bucket != other.tile_density_bucket)
                return tile_density_bucket < other.tile_density_bucket;
            return active_sh < other.active_sh;
        }
    };
    struct RasterBackwardTuningState {
        ThompsonSamplingScheduler scheduler;
        RasterBackwardImpl frozen = RasterBackwardImpl::Default;
        size_t warmup_updates = 0;
    };
    std::vector<RasterBackwardImpl> rasterizeBackwardAlternatives;
    std::map<RasterBackwardKey, RasterBackwardTuningState>
        rasterizeBackwardTuningCache;
    std::array<
        std::vector<RasterBackwardKey>,
        static_cast<size_t>(RasterBackwardImpl::size)>
        rasterizeBackwardPendingKeys;
    bool rasterizeBackwardSchedulerInitialized = false;
    void initRasterizationBackwardScheduler();

protected:
    virtual DeviceRequirement getDeviceRequirement();

    void executeCumsum(
        VulkanGSPipelineBuffers &buffers,
        Buffer<int32_t> &input_buffer,
        Buffer<int32_t> &output_buffer
    );
    void executeFloatInclusiveScan(
        VulkanGSPipelineBuffers &buffers,
        Buffer<float> &values
    );

    _ComputePipeline pipeline_projection_forward = _ComputePipeline(12);
    _ComputePipeline pipeline_projection_forward_sh[4] = {
        _ComputePipeline(12), _ComputePipeline(12),
        _ComputePipeline(12), _ComputePipeline(12),
    };
    _ComputePipeline pipeline_dynamic_projection_forward = _ComputePipeline(14);
    _ComputePipeline pipeline_dynamic_projection_sh[4] = {
        _ComputePipeline(14), _ComputePipeline(14),
        _ComputePipeline(14), _ComputePipeline(14),
    };
    _ComputePipeline pipeline_generate_keys = _ComputePipeline(9);
    _ComputePipeline pipeline_compute_tile_ranges[2] = {
        _ComputePipeline(3),
        _ComputePipeline(3)
    };
    _ComputePipeline pipeline_ftgspp_tile_work_prepare = _ComputePipeline(4);
    _ComputePipelinePair pipeline_rasterize_forward = _ComputePipelinePair(8);
    _ComputePipelinePair pipeline_rasterize_backward[(size_t)RasterBackwardImpl::size] = {
        _ComputePipelinePair(12),  // per pixel
        _ComputePipelinePair(12),  // per splat
        _ComputePipelinePair(12),  // tensor
        _ComputePipelinePair(12),  // tensor
        _ComputePipelinePair(12),  // tensor
    };
    struct _CumsumComputePipeline {
        _ComputePipeline single_pass = _ComputePipeline(2);
        _ComputePipeline block_scan = _ComputePipeline(3);
        _ComputePipeline scan_block_sums = _ComputePipeline(3);
        _ComputePipeline add_block_offsets = _ComputePipeline(3);
    } pipeline_cumsum;
    _CumsumComputePipeline pipeline_cumsum_f32;
    struct _RadixSortComputePipeline {
        _ComputePipeline upsweep = _ComputePipeline(4);
        _ComputePipeline spine = _ComputePipeline(3);
        _ComputePipeline downsweep = _ComputePipeline(7);
    } pipeline_sorting_1, pipeline_sorting_2;

};
