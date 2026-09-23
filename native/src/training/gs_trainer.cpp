#include "training/gs_trainer.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace {

_VulkanBuffer& trainer_work_control(
    VulkanGSPipeline* pipeline, VulkanGSPipelineBuffers& buffers
) {
    pipeline->resizeDeviceBuffer(
        buffers.tile_work_control, ftgspp::kTileWorkControlU32Count);
    return buffers.tile_work_control.deviceBuffer;
}

template<typename Pipeline>
Pipeline& select_trainer_sh(
    Pipeline& fallback,
    Pipeline* sh,
    uint32_t degree
) {
    degree = std::min(degree, 3u);
    if (degree < 3u && sh[degree].pipeline != VK_NULL_HANDLE)
        return sh[degree];
    return fallback;
}

}  // namespace



PACK_STRUCT(struct FTGSPPMaterializeUniforms {
    uint32_t num_splats;
    float timestamp;
    float max_duration;
    float marginal_threshold;
});

PACK_STRUCT(struct FTGSPPBackwardUniforms {
    uint32_t step;
    uint32_t active_sh;
    uint32_t num_splats;
    uint32_t image_size;
    float fx;
    float fy;
    float cx;
    float cy;
    float dist_coeffs[4];
    float timestamp;
    float max_duration;
    float marginal_threshold;
    uint32_t reserved_layout;
    float lr_means;
    float lr_quats;
    float lr_scales;
    float lr_opacities;
    float lr_sh_dc;
    float lr_sh_rest;
    float lr_times;
    float lr_durations;
    float lr_velocities;
    float lr_gates;
    float reg_scale;
    float reg_opacity;
    float reg_gate;
    float reserved[3];
    float world_view_transform[16];
});

PACK_STRUCT(struct FTGSPPRelocationUniforms {
    uint32_t num_splats;
    uint32_t mode;
    float min_opacity;
    uint32_t reserved;
});

PACK_STRUCT(struct FTGSPPRelocationPlanUniforms {
    uint32_t num_splats;
    uint32_t include_gate;
    float opacity_threshold;
    uint32_t seed;
    uint32_t group_relocation;
});

PACK_STRUCT(struct FTGSPPPeerMeanUniforms {
    uint32_t count;
    uint32_t local_offset;
    uint32_t peer_offset;
    uint32_t reserved;
    float local_weight;
    float peer_weight;
});

PACK_STRUCT(struct FTGSPPSparseGradientUniforms {
    uint32_t num_splats;
    uint32_t peer_count;
    uint32_t reserved;
    uint32_t hard_separation;
    uint32_t world_size;
});

PACK_STRUCT(struct FTGSPPRegularizerMeanUniforms {
    uint32_t num_splats;
    float rank0_timestamp;
    float max_duration;
    float reg_scale;
    float reg_opacity;
    float reg_gate;
    float rank1_timestamp;
    uint32_t reserved;
});

static_assert(sizeof(FTGSPPMaterializeUniforms) == 16,
    "FTGSPP materialize push-constant ABI changed");
static_assert(sizeof(FTGSPPBackwardUniforms) == 192,
    "FTGSPP backward push-constant ABI changed");
static_assert(sizeof(FTGSPPRelocationUniforms) == 16,
    "FTGSPP relocation push-constant ABI changed");
static_assert(sizeof(FTGSPPRelocationPlanUniforms) == 20,
    "FTGSPP relocation-plan push-constant ABI changed");
static_assert(sizeof(FTGSPPPeerMeanUniforms) == 24,
    "FTGSPP peer-mean push-constant ABI changed");
static_assert(sizeof(FTGSPPSparseGradientUniforms) == 20,
    "FTGSPP sparse-gradient push-constant ABI changed");
static_assert(sizeof(FTGSPPRegularizerMeanUniforms) == 32,
    "FTGSPP regularizer-mean push-constant ABI changed");
static FTGSPPBackwardUniforms makeFTGSPPBackwardUniforms(
    const TrainerConfig& config,
    const VulkanGSRendererUniforms& uniforms,
    size_t num_splats,
    int step,
    float timestamp
) {
    FTGSPPBackwardUniforms u{};
    u.step = (uint32_t)step;
    u.active_sh = (uniforms.camera_model << 8) | uniforms.active_sh;
    u.num_splats = (uint32_t)num_splats;
    u.image_size = (uniforms.image_height << 16) | uniforms.image_width;
    u.fx = uniforms.fx;
    u.fy = uniforms.fy;
    u.cx = uniforms.cx;
    u.cy = uniforms.cy;
    for (int i = 0; i < 4; ++i) u.dist_coeffs[i] = uniforms.dist_coeffs[i];
    u.timestamp = timestamp;
    u.max_duration = config.max_duration;
    u.marginal_threshold = config.marginal_threshold;
    u.reserved_layout = std::max(config.optimizer_batch_size, 1u) & 0xffffu;
    const int schedule_step = std::max(step - 1, 0);
    u.lr_means = config.get_means_lr(schedule_step, config.global_scale);
    u.lr_quats = config.quats_lr;
    u.lr_scales = config.scales_lr;
    u.lr_opacities = config.opacities_lr;
    u.lr_sh_dc = config.features_dc_lr;
    u.lr_sh_rest = config.features_rest_lr;
    u.lr_times = config.times_lr;
    u.lr_durations = config.durations_lr;
    u.lr_velocities = config.get_velocities_lr(schedule_step, config.global_scale);
    u.lr_gates = config.gates_lr;
    u.reg_scale = config.scale_reg / std::max<size_t>(3 * num_splats, 1);
    u.reg_opacity = config.opacity_reg / std::max<size_t>(num_splats, 1);
    u.reg_gate = config.gate_reg / std::max<size_t>(num_splats, 1);
    u.reserved[0] = config.hard_separation ? 1.0f : 0.0f;
    u.reserved[1] = config.sparse_distributed ? 1.0f : 0.0f;
    for (int i = 0; i < 16; ++i)
        u.world_view_transform[i] = uniforms.world_view_transform[i];
    return u;
}

VulkanGSTrainer::VulkanGSTrainer()
    : VulkanGSRenderer() {
}

VulkanGSTrainer::~VulkanGSTrainer() {
    try {
        cleanup();
    } catch (const std::exception& error) {
        fprintf(stderr, "Vulkan trainer cleanup failed in destructor: %s\n", error.what());
    } catch (...) {
        fprintf(stderr, "Vulkan trainer cleanup failed in destructor\n");
    }
}

void VulkanGSTrainer::cleanup() {
    if (commandBatchInProgress)
        endCommandBatch(false);
    clear_external_dataset();
    VulkanGSRenderer::cleanup();
}

void VulkanGSTrainer::clear_external_dataset() {
    HOST_GUARD;
    auto clear = [this](std::vector<DatasetImage>& images) {
        for (auto& image : images) {
            destroyBuffer(image.buffer.deviceBuffer);
            image.buffer.deviceBuffer = _VulkanBuffer();
            image.buffer.clear();
            image.buffer.shrink_to_fit();
            destroyBuffer(image.validity.deviceBuffer);
            image.validity.deviceBuffer = _VulkanBuffer();
            image.validity.clear();
            image.validity.shrink_to_fit();
        }
        images.clear();
    };
    clear(dataset_train);
    clear(dataset_val);
}

void VulkanGSTrainer::add_external_view(
    DatasetImage&& image,
    bool is_train
) {
    HOST_GUARD;
    if (image.camera.h <= 0 || image.camera.w <= 0)
        throw std::runtime_error("external view dimensions must be positive");
    const size_t expected = (size_t)image.camera.h * (size_t)image.camera.w * 4;
    // Empty host buffers are allowed for lazy pixel binding: metadata/cameras
    // register first, RGBA is filled later via bind_dynamic_view_image.
    if (!image.buffer.empty() && image.buffer.size() != expected)
        throw std::runtime_error("external view must contain H*W*4 RGBA bytes");
    if (!image.validity.empty() && image.validity.size() != expected / 4)
        throw std::runtime_error("external view validity must contain H*W words");
    auto& target = is_train ? dataset_train : dataset_val;
    target.push_back(std::move(image));
}
void VulkanGSTrainer::initialize(const std::map<std::string, std::string> &spirv_paths, int device_id) {

    VulkanGSRenderer::initialize(spirv_paths, device_id);
    
    createComputePipeline(pipeline_ssim_forward, spirv_paths.at("ssim_forward"));
    createComputePipeline(pipeline_ssim_backward, spirv_paths.at("ssim_backward"));
    createComputePipeline(pipeline_ftgspp_materialize, spirv_paths.at("ftgspp_materialize"));
    createComputePipeline(pipeline_ftgspp_explicit_adam, spirv_paths.at("ftgspp_explicit_adam"));
    createComputePipeline(pipeline_ftgspp_peer_mean, spirv_paths.at("ftgspp_peer_mean"));
    createComputePipeline(pipeline_ftgspp_sparse_flag, spirv_paths.at("ftgspp_sparse_flag"));
    createComputePipeline(pipeline_ftgspp_sparse_pack, spirv_paths.at("ftgspp_sparse_pack"));
    createComputePipeline(pipeline_ftgspp_sparse_scatter, spirv_paths.at("ftgspp_sparse_scatter"));
    createComputePipeline(pipeline_ftgspp_regularizer_mean, spirv_paths.at("ftgspp_regularizer_mean"));
    createComputePipeline(pipeline_ftgspp_relocation_prepare, spirv_paths.at("ftgspp_relocation_prepare"));
    createComputePipeline(pipeline_ftgspp_relocation_copy, spirv_paths.at("ftgspp_relocation_copy"));
    createComputePipeline(pipeline_ftgspp_relocation_cdf, spirv_paths.at("ftgspp_relocation_cdf"));
    createComputePipeline(
        pipeline_ftgspp_relocation_cdf_normalize,
        spirv_paths.at("ftgspp_relocation_cdf_normalize"));
    createComputePipeline(pipeline_ftgspp_relocation_sample, spirv_paths.at("ftgspp_relocation_sample"));
    createComputePipeline(pipeline_ftgspp_flow_features, spirv_paths.at("ftgspp_flow_features"));
    createComputePipeline(
        pipeline_ftgspp_hard_separation,
        spirv_paths.at("ftgspp_hard_separation"));
    createComputePipeline(pipeline_ftgspp_color_correct_forward, spirv_paths.at("ftgspp_color_correct_forward"));
    // These shaders contain direct RWByteAddressBuffer/structured-buffer
    // InterlockedAddF32 operations.  USE_EMULATED_F32_ATOMIC only rewrites
    // the inherited VkSplat shaders, so creating these pipelines on a soft
    // device can fail before a pure VkSplat workload even starts.  Leave the
    // forward/non-atomic pipelines available and hard-gate FTGSPP training at
    // its public and defensive dispatch boundaries below.
    if (deviceInfo.hasFloat32AtomicAdd) {
        createComputePipeline(
            pipeline_ftgspp_loss_reduce,
            spirv_paths.at("ftgspp_loss_reduce"));
        createComputePipeline(
            pipeline_ftgspp_eval_metrics,
            spirv_paths.at("ftgspp_eval_metrics"));
        createComputePipeline(
            pipeline_ftgspp_fused_backward_optimizer,
            spirv_paths.at("ftgspp_fused_backward_optimizer"));
        createComputePipeline(
            pipeline_ftgspp_fused_dynamic_backward_adam,
            spirv_paths.at("ftgspp_fused_dynamic_backward_adam"));
        for (int degree = 0; degree < 3; ++degree) {
            auto load_optional = [&](_ComputePipeline& pipeline, const std::string& name) {
                auto it = spirv_paths.find(name);
                if (it == spirv_paths.end())
                    return;
                createComputePipeline(pipeline, it->second);
            };
            load_optional(
                pipeline_ftgspp_fused_backward_optimizer_sh[degree],
                "ftgspp_fused_backward_optimizer_sh" + std::to_string(degree));
            load_optional(
                pipeline_ftgspp_fused_dynamic_backward_adam_sh[degree],
                "ftgspp_fused_dynamic_backward_adam_sh" + std::to_string(degree));
        }
        createComputePipeline(
            pipeline_ftgspp_relocation_score,
            spirv_paths.at("ftgspp_relocation_score"));
        createComputePipeline(
            pipeline_ftgspp_color_correct_backward,
            spirv_paths.at("ftgspp_color_correct_backward"));
        createComputePipeline(
            pipeline_ftgspp_color_adam,
            spirv_paths.at("ftgspp_color_adam"));
    }
}

void VulkanGSTrainer::requireFTGSPPFloat32AtomicAdd() const {
    if (deviceInfo.hasFloat32AtomicAdd)
        return;
    throw std::runtime_error(
        "FTGSPP native training requires Vulkan "
        "shaderBufferFloat32AtomicAdd support. USE_EMULATED_F32_ATOMIC=1 "
        "and cannot be replaced by the alpha-blend CAS fallback. Forward-only "
        "rendering remains available on this device.");
}




void VulkanGSTrainer::camera_to_uniforms(const Camera &cam, VulkanGSRendererUniforms &uniforms) const {
    uniforms.image_height = cam.h;
    uniforms.image_width = cam.w;

    uniforms.grid_height = _CEIL_DIV(cam.h, TILE_HEIGHT);
    uniforms.grid_width = _CEIL_DIV(cam.w, TILE_WIDTH);

    // NativeDataset has already undistorted the SelfCap image and intrinsic.
    uniforms.camera_model = 0;

    uniforms.fx = cam.fx;
    uniforms.fy = cam.fy;
    uniforms.cx = cam.cx;
    uniforms.cy = cam.cy;

    for (int i = 0; i < 4; ++i)
        uniforms.dist_coeffs[i] = 0.0f;

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            uniforms.world_view_transform[4*i+j] = cam.world_view_transform[i][j];
}

void VulkanGSTrainer::get_train_camera(size_t idx, VulkanGSRendererUniforms &uniforms) const {
    camera_to_uniforms(dataset_train[idx].camera, uniforms);
}

void VulkanGSTrainer::get_val_camera(size_t idx, VulkanGSRendererUniforms &uniforms) const {
    camera_to_uniforms(dataset_val[idx].camera, uniforms);
}


void VulkanGSTrainer::executeComputeSSIMGradient(
    const TrainerConfig &config,
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    size_t train_idx,
    uint32_t prediction_clamp_mode,
    bool collect_loss_scalar,
    const std::array<float, 3>& background
) {
    PerfTimer::Timer<PerfTimer::LossForwardBackward> loss_timer(this);
    if (prediction_clamp_mode > 2u)
        throw std::runtime_error("prediction clamp mode must be 0, 1 or 2");
    const size_t w = uniforms.image_width, h = uniforms.image_height;
    const size_t num_pixels = w*h;
    struct LossUniforms {
        uint32_t width;
        uint32_t height;
        float grad_weight_l1;
        float grad_weight_ssim;
        uint32_t prediction_clamp_mode;
        uint32_t objective_mode;
        float background_r;
        float background_g;
        float background_b;
        uint32_t valid_pixel_count;
        uint32_t valid_window_count;
        float ssim_c1;
        float ssim_c2;
    } shader_uniforms{};
    static_assert(sizeof(LossUniforms) == 13 * sizeof(uint32_t),
        "FTGSPP SSIM push-constant ABI changed");
    shader_uniforms.width = (uint32_t)w;
    shader_uniforms.height = (uint32_t)h;
    auto& image = dataset_train.at(train_idx);
    auto& validity = image.validity;
    uint32_t valid_pixels = image.has_custom_validity
        ? image.valid_pixel_count : (uint32_t)num_pixels;
    uint32_t valid_windows = image.has_custom_validity
        ? image.valid_window_count
        : (uint32_t)((w > 10 && h > 10) ? (w-10)*(h-10) : 0);
    if (valid_pixels == 0)
        throw std::runtime_error("train view has no valid loss pixels");
    if (config.image_objective > 1u)
        throw std::runtime_error("image objective must be 0 or 1");
    const bool l1_only = valid_windows == 0;
    shader_uniforms.grad_weight_l1 = (
        (l1_only ? 1.0f : 1.0f-config.ssim_lambda)
        / (3.0f*float(valid_pixels)));
    shader_uniforms.grad_weight_ssim = l1_only
        ? 0.0f : -config.ssim_lambda/(3.0f*float(valid_windows));
    shader_uniforms.prediction_clamp_mode = prediction_clamp_mode;
    shader_uniforms.objective_mode = config.image_objective;
    shader_uniforms.background_r = background[0];
    shader_uniforms.background_g = background[1];
    shader_uniforms.background_b = background[2];
    shader_uniforms.valid_pixel_count = valid_pixels;
    shader_uniforms.valid_window_count = valid_windows;
    shader_uniforms.ssim_c1 = 0.01f * 0.01f;
    shader_uniforms.ssim_c2 = 0.03f * 0.03f;

    auto& reference = image.buffer;
    _VulkanBuffer reference_device;
    bool buffer_swapped = false;
    if (pending_reference_ticket_.valid()) {
        reference_device = referenceBufferForTicket(pending_reference_ticket_);
        if (reference_device.buffer == VK_NULL_HANDLE)
            throw std::runtime_error(
                "pending reference upload ticket is stale");
    } else if (reference.deviceBuffer.buffer != VK_NULL_HANDLE) {
        reference_device = reference.deviceBuffer;
    } else {
        if (reference.empty())
            throw std::runtime_error(
                "train view pixels are not bound; call bind_dynamic_view_image before training");
        PerfTimer::Timer<PerfTimer::CopyTrainImageToDevice> copy_timer(this);
        std::swap(reference.deviceBuffer, buffers.ref_image.deviceBuffer);
        copyToDevice(reference);
        buffer_swapped = true;
        reference_device = reference.deviceBuffer;
    }
    _VulkanBuffer validity_device;
    if (image.has_custom_validity) {
        if (validity.deviceBuffer.buffer == VK_NULL_HANDLE)
            copyToDevice(validity);
        validity_device = validity.deviceBuffer;
    } else {
        if (buffers.ref_validity.size() != num_pixels
            || buffers.ref_validity_width != w
            || buffers.ref_validity_height != h) {
            buffers.ref_validity.assign(num_pixels, 1u);
            if (w > 10 && h > 10) {
                for (size_t y = 5; y + 5 < h; ++y)
                    for (size_t x = 5; x + 5 < w; ++x)
                        buffers.ref_validity[y*w+x] |= 2u;
            }
            copyToDevice(buffers.ref_validity);
            buffers.ref_validity_width = w;
            buffers.ref_validity_height = h;
        }
        validity_device = buffers.ref_validity.deviceBuffer;
    }
    bufferMemoryBarrier({
        {buffers.pixel_state.deviceBuffer, COMPUTE_SHADER_WRITE},
        {reference_device, TRANSFER_WRITE},
        {validity_device, TRANSFER_WRITE},
    }, COMPUTE_SHADER_READ);

    PerfTimer::Timer<PerfTimer::ComputeSSIMGradient> ssim_timer(this);
    DEVICE_GUARD;
    auto& ssim_map = buffers._temp_gauss_attr;
    auto& wc = trainer_work_control(this, buffers);
    executeCompute({{w,16},{h,16}}, &shader_uniforms, sizeof(shader_uniforms),
        pipeline_ssim_forward,
        {buffers.pixel_state.deviceBuffer, reference_device, validity_device,
         resizeDeviceBuffer(ssim_map, 12*num_pixels), wc});
    memoryBarrier(COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ);
    if (collect_loss_scalar && deviceInfo.hasFloat32AtomicAdd) {
        clearDeviceBuffer(buffers.loss_accum, 6);
        memoryBarrier(TRANSFER_WRITE, COMPUTE_SHADER_READ_WRITE);
        executeCompute({{num_pixels,256}}, &shader_uniforms, 11 * sizeof(uint32_t),
            pipeline_ftgspp_loss_reduce,
            {buffers.pixel_state.deviceBuffer, reference_device, validity_device,
             ssim_map.deviceBuffer, buffers.loss_accum.deviceBuffer, wc});
        bufferMemoryBarrier({
            {ssim_map.deviceBuffer, COMPUTE_SHADER_WRITE},
            {buffers.loss_accum.deviceBuffer, COMPUTE_SHADER_WRITE},
        }, COMPUTE_SHADER_READ_WRITE);
    } else {
        resizeDeviceBuffer(buffers.loss_accum, 6);
    }
    executeCompute({{w,16},{h,16}}, &shader_uniforms, sizeof(shader_uniforms),
        pipeline_ssim_backward,
        {buffers.pixel_state.deviceBuffer, reference_device, validity_device,
         ssim_map.deviceBuffer, resizeDeviceBuffer(buffers.v_pixel_state, 4*num_pixels), wc});

    if (buffer_swapped)
        std::swap(reference.deviceBuffer, buffers.ref_image.deviceBuffer);
}

void VulkanGSTrainer::executeEvalMetrics(
    const TrainerConfig &config,
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    size_t val_idx,
    const std::array<float, 3>& background,
    const std::array<uint32_t, 4>& roi_yxyx,
    bool upload_reference
) {
    requireFTGSPPFloat32AtomicAdd();
    const size_t w = uniforms.image_width, h = uniforms.image_height;
    const size_t num_pixels = w * h;
    if (num_pixels == 0)
        throw std::runtime_error("eval metrics require a rendered view");
    if (config.image_objective > 1u)
        throw std::runtime_error("image objective must be 0 or 1");
    auto& image = dataset_val.at(val_idx);
    auto& reference = image.buffer;
    if (reference.empty())
        throw std::runtime_error(
            "eval view pixels are not bound; call bind_dynamic_view_image");
    struct EvalUniforms {
        uint32_t width;
        uint32_t height;
        float grad_weight_l1;
        float grad_weight_ssim;
        uint32_t prediction_clamp_mode;
        uint32_t objective_mode;
        float background_r;
        float background_g;
        float background_b;
        uint32_t valid_pixel_count;
        uint32_t valid_window_count;
        float ssim_c1;
        float ssim_c2;
    } shader_uniforms{};
    static_assert(sizeof(EvalUniforms) == 13 * sizeof(uint32_t),
        "FTGSPP eval SSIM push-constant ABI changed");
    shader_uniforms.width = (uint32_t)w;
    shader_uniforms.height = (uint32_t)h;
    shader_uniforms.grad_weight_l1 = 0.0f;
    shader_uniforms.grad_weight_ssim = 0.0f;
    shader_uniforms.prediction_clamp_mode = 2u;
    shader_uniforms.objective_mode = config.image_objective;
    shader_uniforms.background_r = background[0];
    shader_uniforms.background_g = background[1];
    shader_uniforms.background_b = background[2];
    shader_uniforms.ssim_c1 = 0.02f * 0.02f;
    shader_uniforms.ssim_c2 = 0.06f * 0.06f;
    struct EvalReduceUniforms {
        uint32_t width;
        uint32_t height;
        uint32_t roi_y0;
        uint32_t roi_y1;
        uint32_t prediction_clamp_mode;
        uint32_t objective_mode;
        float background_r;
        float background_g;
        float background_b;
        uint32_t roi_x0;
        uint32_t roi_x1;
    } reduce_uniforms{
        shader_uniforms.width,
        shader_uniforms.height,
        roi_yxyx[0],
        roi_yxyx[1],
        shader_uniforms.prediction_clamp_mode,
        shader_uniforms.objective_mode,
        shader_uniforms.background_r,
        shader_uniforms.background_g,
        shader_uniforms.background_b,
        roi_yxyx[2],
        roi_yxyx[3],
    };
    static_assert(sizeof(EvalReduceUniforms) == 11 * sizeof(uint32_t),
        "FTGSPP eval reduction push-constant ABI changed");

    std::swap(reference.deviceBuffer, buffers.ref_image.deviceBuffer);
    if (upload_reference)
        copyToDevice(reference);
    _VulkanBuffer reference_device = reference.deviceBuffer;
    _VulkanBuffer validity_device;
    if (image.has_custom_validity) {
        if (image.validity.empty())
            throw std::runtime_error("eval view validity is empty");
        if (upload_reference)
            copyToDevice(image.validity);
        validity_device = image.validity.deviceBuffer;
        shader_uniforms.valid_pixel_count = image.valid_pixel_count;
        shader_uniforms.valid_window_count = image.valid_window_count;
    } else {
        if (buffers.eval_validity.size() != num_pixels
            || buffers.eval_validity_width != w
            || buffers.eval_validity_height != h
            || buffers.eval_validity.deviceBuffer.buffer == VK_NULL_HANDLE) {
            buffers.eval_validity.assign(num_pixels, 1u);
            if (w > 10 && h > 10) {
                for (size_t y = 5; y + 5 < h; ++y)
                    for (size_t x = 5; x + 5 < w; ++x)
                        buffers.eval_validity[y * w + x] |= 2u | 4u;
            }
            copyToDevice(buffers.eval_validity);
            buffers.eval_validity_width = w;
            buffers.eval_validity_height = h;
        }
        validity_device = buffers.eval_validity.deviceBuffer;
        shader_uniforms.valid_pixel_count = (uint32_t)num_pixels;
        shader_uniforms.valid_window_count = (uint32_t)(
            (w > 10 && h > 10) ? (w - 10) * (h - 10) : 0
        );
    }
    bufferMemoryBarrier({
        {buffers.pixel_state.deviceBuffer, COMPUTE_SHADER_WRITE},
        {reference_device, TRANSFER_WRITE},
        {validity_device, TRANSFER_WRITE},
    }, COMPUTE_SHADER_READ);

    DEVICE_GUARD;
    auto& ssim_map = buffers._temp_gauss_attr;
    auto& wc = trainer_work_control(this, buffers);
    executeCompute({{w,16},{h,16}}, &shader_uniforms, sizeof(shader_uniforms),
        pipeline_ssim_forward,
        {buffers.pixel_state.deviceBuffer, reference_device, validity_device,
         resizeDeviceBuffer(ssim_map, 12*num_pixels), wc});
    memoryBarrier(COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ);
    clearDeviceBuffer(buffers.eval_metrics, 10);
    memoryBarrier(TRANSFER_WRITE, COMPUTE_SHADER_READ_WRITE);
    executeCompute({{num_pixels,256}}, &reduce_uniforms, sizeof(reduce_uniforms),
        pipeline_ftgspp_eval_metrics,
        {buffers.pixel_state.deviceBuffer, reference_device, validity_device,
         ssim_map.deviceBuffer, buffers.eval_metrics.deviceBuffer, wc});
    bufferMemoryBarrier({
        {ssim_map.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.eval_metrics.deviceBuffer, COMPUTE_SHADER_WRITE},
    }, TRANSFER_COMPUTE_SHADER_READ);
    std::swap(reference.deviceBuffer, buffers.ref_image.deviceBuffer);
}

void VulkanGSTrainer::executeFTGSPPMaterialize(
    const TrainerConfig &config,
    VulkanGSPipelineBuffers& buffers,
    float timestamp
) {
    PerfTimer::Timer<PerfTimer::Materialize> timer(this);
    const size_t n = buffers.num_splats;
    FTGSPPMaterializeUniforms u{};
    u.num_splats = (uint32_t)n;
    u.timestamp = timestamp;
    u.max_duration = config.max_duration;
    u.marginal_threshold = config.marginal_threshold;

    bufferMemoryBarrier({
        { buffers.base_xyz_ws.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.base_log_scales_opacity.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.times_durations.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.velocities_gates.deviceBuffer, COMPUTE_SHADER_WRITE },
    }, COMPUTE_SHADER_READ_WRITE);
    executeCompute(
        {{n, SUBGROUP_SIZE}}, &u, sizeof(u), pipeline_ftgspp_materialize,
        {
            buffers.base_xyz_ws.deviceBuffer,
            buffers.base_log_scales_opacity.deviceBuffer,
            buffers.times_durations.deviceBuffer,
            buffers.velocities_gates.deviceBuffer,
            resizeDeviceBuffer(buffers.xyz_ws, 3*n),
            resizeDeviceBuffer(buffers.scales_opacs, 4*n),
        }
    );
    bufferMemoryBarrier({
        { buffers.xyz_ws.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.scales_opacs.deviceBuffer, COMPUTE_SHADER_WRITE },
    }, TRANSFER_COMPUTE_SHADER_READ);
}

void VulkanGSTrainer::executeFTGSPPBackward(
    const TrainerConfig &config,
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    int step,
    float timestamp,
    bool collect_loss_scalar
) {
    PerfTimer::Timer<PerfTimer::DynamicBackward> timer(this);
    requireFTGSPPFloat32AtomicAdd();
    const size_t n = buffers.num_splats;
    FTGSPPBackwardUniforms u = makeFTGSPPBackwardUniforms(
        config, uniforms, n, step, timestamp);
    if (collect_loss_scalar)
        u.reserved_layout |= 0x80000000u;

    bufferMemoryBarrier({
        { buffers.tiles_touched.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.v_xy_vs.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.v_inv_cov_vs_opacity.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
        { buffers.v_rgb.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
        { buffers.loss_accum.deviceBuffer, COMPUTE_SHADER_WRITE },
    }, COMPUTE_SHADER_READ_WRITE);

    executeCompute(
        {{n, SUBGROUP_SIZE}}, &u, sizeof(u),
        select_trainer_sh(
            pipeline_ftgspp_fused_backward_optimizer,
            pipeline_ftgspp_fused_backward_optimizer_sh,
            uniforms.active_sh),
        {
            buffers.base_xyz_ws.deviceBuffer,
            buffers.base_log_scales_opacity.deviceBuffer,
            buffers.times_durations.deviceBuffer,
            buffers.velocities_gates.deviceBuffer,
            buffers.sh_coeffs.deviceBuffer,
            buffers.rotations.deviceBuffer,
            buffers.tiles_touched.deviceBuffer,
            buffers.v_xy_vs.deviceBuffer,
            buffers.v_inv_cov_vs_opacity.deviceBuffer,
            buffers.v_rgb.deviceBuffer,
            resizeAndCopyDeviceBuffer(buffers.relocation_stats, 2*n, true),
            resizeDeviceBuffer(buffers.grad_base_xyz_ws, 3*n),
            resizeDeviceBuffer(buffers.grad_base_log_scales_opacity, 4*n),
            resizeDeviceBuffer(buffers.grad_times_durations, 2*n),
            resizeDeviceBuffer(buffers.grad_velocities_gates, 4*n),
            resizeDeviceBuffer(
                buffers.grad_sh_coeffs,
                16*3*_CEIL_ROUND(n,SH_REORDER_SIZE)),
            resizeDeviceBuffer(buffers.grad_rotations, 4*n),
            buffers.loss_accum.deviceBuffer,
            trainer_work_control(this, buffers),
        }
    );
    bufferMemoryBarrier({
        { buffers.grad_base_xyz_ws.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.grad_base_log_scales_opacity.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.grad_times_durations.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.grad_velocities_gates.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.grad_sh_coeffs.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.grad_rotations.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.relocation_stats.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.loss_accum.deviceBuffer, COMPUTE_SHADER_WRITE },
    }, TRANSFER_COMPUTE_SHADER_READ_WRITE);
}

void VulkanGSTrainer::executeFTGSPPFusedDynamicBackwardAdam(
    const TrainerConfig &config,
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    int step,
    float timestamp,
    bool collect_loss_scalar,
    bool collect_geometry_grad_stats
) {
    PerfTimer::Timer<PerfTimer::FusedDynamicBackwardAdam> timer(this);
    requireFTGSPPFloat32AtomicAdd();
    const size_t n = buffers.num_splats;
    FTGSPPBackwardUniforms u = makeFTGSPPBackwardUniforms(
        config, uniforms, n, step, timestamp);
    if (collect_loss_scalar)
        u.reserved_layout |= 0x80000000u;
    u.reserved[2] = collect_geometry_grad_stats ? 1.0f : 0.0f;
    if (collect_geometry_grad_stats)
        clearDeviceBuffer(buffers.grad_stats, 6);
    else
        resizeDeviceBuffer(buffers.grad_stats, 6);
    resizeDeviceBuffer(buffers.loss_accum, 6);

    bufferMemoryBarrier({
        {buffers.base_xyz_ws.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.base_log_scales_opacity.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.times_durations.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.velocities_gates.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.sh_coeffs.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.rotations.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.tiles_touched.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.v_xy_vs.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.v_inv_cov_vs_opacity.deviceBuffer, COMPUTE_SHADER_READ_WRITE},
        {buffers.v_rgb.deviceBuffer, COMPUTE_SHADER_READ_WRITE},
        {buffers.loss_accum.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
        {buffers.grad_stats.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
    }, COMPUTE_SHADER_READ_WRITE);

    executeCompute(
        {{n, SUBGROUP_SIZE}}, &u, sizeof(u),
        select_trainer_sh(
            pipeline_ftgspp_fused_dynamic_backward_adam,
            pipeline_ftgspp_fused_dynamic_backward_adam_sh,
            uniforms.active_sh),
        {
            buffers.base_xyz_ws.deviceBuffer,
            buffers.base_log_scales_opacity.deviceBuffer,
            buffers.times_durations.deviceBuffer,
            buffers.velocities_gates.deviceBuffer,
            buffers.sh_coeffs.deviceBuffer,
            buffers.rotations.deviceBuffer,
            buffers.tiles_touched.deviceBuffer,
            buffers.v_xy_vs.deviceBuffer,
            buffers.v_inv_cov_vs_opacity.deviceBuffer,
            buffers.v_rgb.deviceBuffer,
            resizeAndCopyDeviceBuffer(buffers.relocation_stats, 2*n, true),
            resizeAndCopyDeviceBuffer(buffers.g_xyz_ws, 2*3*n, true),
            resizeAndCopyDeviceBuffer(buffers.g_scales_opacs, 2*4*n, true),
            resizeAndCopyDeviceBuffer(buffers.g_times_durations, 2*2*n, true),
            resizeAndCopyDeviceBuffer(buffers.g_velocities_gates, 2*4*n, true),
            resizeAndCopyDeviceBuffer(
                buffers.g_sh_coeffs_1,
                16*3*_CEIL_ROUND(n,SH_REORDER_SIZE), true),
            resizeAndCopyDeviceBuffer(
                buffers.g_sh_coeffs_2,
                16*3*_CEIL_ROUND(n,SH_REORDER_SIZE), true),
            resizeAndCopyDeviceBuffer(buffers.g_rotations, 2*4*n, true),
            buffers.loss_accum.deviceBuffer,
            buffers.grad_stats.deviceBuffer,
            trainer_work_control(this, buffers),
        });
    bufferMemoryBarrier({
        {buffers.base_xyz_ws.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.base_log_scales_opacity.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.times_durations.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.velocities_gates.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.sh_coeffs.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.rotations.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.g_xyz_ws.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.g_scales_opacs.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.g_times_durations.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.g_velocities_gates.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.g_sh_coeffs_1.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.g_sh_coeffs_2.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.g_rotations.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.relocation_stats.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.loss_accum.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.grad_stats.deviceBuffer, COMPUTE_SHADER_WRITE},
    }, TRANSFER_COMPUTE_SHADER_READ_WRITE);
}

void VulkanGSTrainer::executeFTGSPPExplicitAdam(
    const TrainerConfig &config,
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    int step,
    bool collect_geometry_grad_stats
) {
    PerfTimer::Timer<PerfTimer::ExplicitAdam> timer(this);
    const size_t n = buffers.num_splats;
    FTGSPPBackwardUniforms u = makeFTGSPPBackwardUniforms(
        config, uniforms, n, step, 0.0f);
    // reserved[2] gates the optional WaveActiveSum path inside Adam. Ordinary
    // steps keep it zero so the shader skips the atomics entirely.
    u.reserved[2] = collect_geometry_grad_stats ? 1.0f : 0.0f;
    if (collect_geometry_grad_stats)
        clearDeviceBuffer(buffers.grad_stats, 6);
    else
        resizeDeviceBuffer(buffers.grad_stats, 6);

    // A local backward writes these buffers from compute; distributed training
    // subsequently replaces them through staging transfers.  Accept either
    // producer before the optimizer reads the globally reduced gradients.
    bufferMemoryBarrier({
        { buffers.grad_base_xyz_ws.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE },
        { buffers.grad_base_log_scales_opacity.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE },
        { buffers.grad_times_durations.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE },
        { buffers.grad_velocities_gates.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE },
        { buffers.grad_sh_coeffs.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE },
        { buffers.grad_rotations.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE },
        { buffers.grad_stats.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE },
    }, COMPUTE_SHADER_READ_WRITE);

    executeCompute(
        {{n, SUBGROUP_SIZE}}, &u, sizeof(u), pipeline_ftgspp_explicit_adam,
        {
            buffers.base_xyz_ws.deviceBuffer,
            buffers.base_log_scales_opacity.deviceBuffer,
            buffers.times_durations.deviceBuffer,
            buffers.velocities_gates.deviceBuffer,
            buffers.sh_coeffs.deviceBuffer,
            buffers.rotations.deviceBuffer,
            buffers.grad_base_xyz_ws.deviceBuffer,
            buffers.grad_base_log_scales_opacity.deviceBuffer,
            buffers.grad_times_durations.deviceBuffer,
            buffers.grad_velocities_gates.deviceBuffer,
            buffers.grad_sh_coeffs.deviceBuffer,
            buffers.grad_rotations.deviceBuffer,
            resizeAndCopyDeviceBuffer(buffers.g_xyz_ws, 2*3*n, true),
            resizeAndCopyDeviceBuffer(buffers.g_scales_opacs, 2*4*n, true),
            resizeAndCopyDeviceBuffer(buffers.g_times_durations, 2*2*n, true),
            resizeAndCopyDeviceBuffer(buffers.g_velocities_gates, 2*4*n, true),
            resizeAndCopyDeviceBuffer(
                buffers.g_sh_coeffs_1,
                16*3*_CEIL_ROUND(n,SH_REORDER_SIZE), true),
            resizeAndCopyDeviceBuffer(
                buffers.g_sh_coeffs_2,
                16*3*_CEIL_ROUND(n,SH_REORDER_SIZE), true),
            resizeAndCopyDeviceBuffer(buffers.g_rotations, 2*4*n, true),
            buffers.grad_stats.deviceBuffer,
            trainer_work_control(this, buffers),
        }
    );
    bufferMemoryBarrier({
        { buffers.base_xyz_ws.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.base_log_scales_opacity.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.times_durations.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.velocities_gates.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.sh_coeffs.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.rotations.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.g_xyz_ws.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.g_scales_opacs.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.g_times_durations.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.g_velocities_gates.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.g_sh_coeffs_1.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.g_sh_coeffs_2.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.g_rotations.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.grad_base_xyz_ws.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.grad_base_log_scales_opacity.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.grad_times_durations.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.grad_velocities_gates.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.grad_sh_coeffs.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.grad_rotations.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.grad_stats.deviceBuffer, COMPUTE_SHADER_WRITE },
    }, COMPUTE_SHADER_READ_WRITE);
}


void VulkanGSTrainer::executeFTGSPPPeerMean(
    const _VulkanBuffer& local,
    const _VulkanBuffer& peer,
    size_t float_count,
    float local_weight,
    float peer_weight,
    size_t local_offset,
    size_t peer_offset,
    bool reset
) {
    if (float_count == 0 || float_count > uint32_t(-1) ||
        local_offset > uint32_t(-1) || peer_offset > uint32_t(-1))
        throw std::runtime_error("peer-gradient mean count is empty or exceeds uint32");
    if (local_offset > std::numeric_limits<size_t>::max() - float_count ||
        peer_offset > std::numeric_limits<size_t>::max() - float_count)
        throw std::overflow_error("peer-gradient mean range overflow");
    const size_t local_bytes = (local_offset + float_count) * sizeof(float);
    const size_t peer_bytes = (peer_offset + float_count) * sizeof(float);
    if (local.buffer == VK_NULL_HANDLE || peer.buffer == VK_NULL_HANDLE ||
        local.size < local_bytes || peer.size < peer_bytes)
        throw std::runtime_error("peer-gradient mean buffer is missing or undersized");
    FTGSPPPeerMeanUniforms u{
        static_cast<uint32_t>(float_count),
        static_cast<uint32_t>(local_offset),
        static_cast<uint32_t>(peer_offset), reset ? 1u : 0u,
        local_weight, peer_weight,
    };
    bufferMemoryBarrier({
        {local, TRANSFER_COMPUTE_SHADER_WRITE},
        {peer, TRANSFER_COMPUTE_SHADER_WRITE},
    }, COMPUTE_SHADER_READ_WRITE);
    executeCompute(
        {{float_count, 256}}, &u, sizeof(u), pipeline_ftgspp_peer_mean,
        {local, peer});
    bufferMemoryBarrier({
        {local, COMPUTE_SHADER_WRITE},
    }, COMPUTE_SHADER_READ_WRITE);
}


void VulkanGSTrainer::writeFTGSPPSparseActiveFlags(
    VulkanGSPipelineBuffers& buffers
) {
    const size_t n = buffers.num_splats;
    if (n == 0 || n > uint32_t(-1))
        throw std::runtime_error("sparse active flags have invalid Gaussian count");
    resizeDeviceBuffer(buffers.sparse_active_flags, n);
    FTGSPPSparseGradientUniforms u{
        static_cast<uint32_t>(n), 0u, 0u, 0u, 1u,
    };
    bufferMemoryBarrier({
        {buffers.tiles_touched.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.sparse_active_flags.deviceBuffer,
         TRANSFER_COMPUTE_SHADER_WRITE},
    }, COMPUTE_SHADER_READ_WRITE);
    executeCompute(
        {{n, 256}}, &u, sizeof(u), pipeline_ftgspp_sparse_flag,
        {
            buffers.tiles_touched.deviceBuffer,
            buffers.sparse_active_flags.deviceBuffer,
        });
    bufferMemoryBarrier({
        {buffers.sparse_active_flags.deviceBuffer, COMPUTE_SHADER_WRITE},
    }, COMPUTE_SHADER_READ_WRITE);
}


uint32_t VulkanGSTrainer::executeFTGSPPSparseGradientPack(
    const TrainerConfig& config,
    VulkanGSPipelineBuffers& buffers,
    uint32_t active_sh
) {
    PerfTimer::Timer<PerfTimer::SparseGradientPack> timer(this);
    static constexpr size_t kRecordWords = 66;
    const size_t n = buffers.num_splats;
    if (!config.sparse_distributed)
        throw std::runtime_error("sparse gradient pack requested while disabled");
    if (n == 0 || n > uint32_t(-1))
        throw std::runtime_error("sparse gradient pack has invalid Gaussian count");
    if (buffers.sparse_active_flags.deviceSize() != n)
        throw std::runtime_error(
            "sparse gradient pack lacks the active Gaussian mask");
    clearDeviceBuffer(buffers.sparse_gradient_count, 1);
    resizeDeviceBuffer(buffers.sparse_gradient_records, kRecordWords*n);
    executeCumsum(
        buffers, buffers.sparse_active_flags, buffers.sparse_active_offsets);
    FTGSPPSparseGradientUniforms u{
        static_cast<uint32_t>(n), 0u, std::min(active_sh, 3u),
        config.hard_separation ? 1u : 0u,
        1u,
    };
    std::vector<std::pair<_VulkanBuffer, BarrierMask>> pack_barriers{
        {buffers.sparse_active_flags.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.sparse_active_offsets.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.grad_base_xyz_ws.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.grad_base_log_scales_opacity.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.grad_times_durations.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.grad_velocities_gates.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.grad_sh_coeffs.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.grad_rotations.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.sparse_gradient_count.deviceBuffer, TRANSFER_WRITE},
    };
    bufferMemoryBarrier(pack_barriers, COMPUTE_SHADER_READ_WRITE);
    executeCompute(
        {{n, 256}}, &u, sizeof(u), pipeline_ftgspp_sparse_pack,
        {
            buffers.sparse_active_flags.deviceBuffer,
            buffers.sparse_active_offsets.deviceBuffer,
            buffers.grad_base_xyz_ws.deviceBuffer,
            buffers.grad_base_log_scales_opacity.deviceBuffer,
            buffers.grad_times_durations.deviceBuffer,
            buffers.grad_velocities_gates.deviceBuffer,
            buffers.grad_sh_coeffs.deviceBuffer,
            buffers.grad_rotations.deviceBuffer,
            buffers.sparse_gradient_records.deviceBuffer,
            buffers.sparse_gradient_count.deviceBuffer,
        });
    bufferMemoryBarrier({
        {buffers.sparse_gradient_records.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.sparse_gradient_count.deviceBuffer, COMPUTE_SHADER_WRITE},
    }, TRANSFER_READ);
    return readElement<uint32_t>(buffers.sparse_gradient_count.deviceBuffer, 0);
}


void VulkanGSTrainer::executeFTGSPPRegularizerMean(
    const TrainerConfig& config,
    VulkanGSPipelineBuffers& buffers,
    float rank0_timestamp,
    float rank1_timestamp
) {
    const size_t n = buffers.num_splats;
    FTGSPPRegularizerMeanUniforms regularizer{
        static_cast<uint32_t>(n),
        rank0_timestamp,
        config.max_duration,
        config.scale_reg / std::max<size_t>(3 * n, 1),
        config.opacity_reg / std::max<size_t>(n, 1),
        config.gate_reg / std::max<size_t>(n, 1),
        rank1_timestamp,
        0u,
    };
    bufferMemoryBarrier({
        {buffers.grad_base_log_scales_opacity.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.grad_velocities_gates.deviceBuffer, COMPUTE_SHADER_WRITE},
    }, COMPUTE_SHADER_READ_WRITE);
    executeCompute(
        {{n, 256}}, &regularizer, sizeof(regularizer),
        pipeline_ftgspp_regularizer_mean,
        {
            buffers.base_log_scales_opacity.deviceBuffer,
            buffers.times_durations.deviceBuffer,
            buffers.velocities_gates.deviceBuffer,
            buffers.grad_base_log_scales_opacity.deviceBuffer,
            buffers.grad_velocities_gates.deviceBuffer,
        });
    memoryBarrier(COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ_WRITE);
}

void VulkanGSTrainer::executeFTGSPPSparseGradientMerge(
    const TrainerConfig& config,
    VulkanGSPipelineBuffers& buffers,
    uint32_t peer_count,
    float rank0_timestamp,
    float rank1_timestamp,
    uint32_t active_sh,
    uint32_t local_rank,
    uint32_t local_count
) {
    const size_t n = buffers.num_splats;
    if (!config.sparse_distributed)
        throw std::runtime_error("sparse gradient merge requested while disabled");
    FTGSPPSparseGradientUniforms sparse{
        static_cast<uint32_t>(n), peer_count, std::min(active_sh, 3u),
        config.hard_separation ? 1u : 0u,
        2u,
    };
    auto clear_image_gradients = [&]() {
        clearDeviceBuffer(buffers.grad_base_xyz_ws, 3 * n);
        clearDeviceBuffer(buffers.grad_base_log_scales_opacity, 4 * n);
        clearDeviceBuffer(buffers.grad_times_durations, 2 * n);
        clearDeviceBuffer(buffers.grad_velocities_gates, 4 * n);
        clearDeviceBuffer(
            buffers.grad_sh_coeffs,
            16 * 3 * _CEIL_ROUND(n, SH_REORDER_SIZE));
        clearDeviceBuffer(buffers.grad_rotations, 4 * n);
    };
    auto scatter_sparse_records = [&](
        const _VulkanBuffer& records, uint32_t count
    ) {
        if (count == 0)
            return;
        if (records.buffer == VK_NULL_HANDLE)
            throw std::runtime_error("sparse scatter records buffer is missing");
        sparse.peer_count = count;
        bufferMemoryBarrier({
            {records, TRANSFER_COMPUTE_SHADER_WRITE},
            {buffers.grad_base_xyz_ws.deviceBuffer,
             TRANSFER_COMPUTE_SHADER_WRITE},
            {buffers.grad_base_log_scales_opacity.deviceBuffer,
             TRANSFER_COMPUTE_SHADER_WRITE},
            {buffers.grad_times_durations.deviceBuffer,
             TRANSFER_COMPUTE_SHADER_WRITE},
            {buffers.grad_velocities_gates.deviceBuffer,
             TRANSFER_COMPUTE_SHADER_WRITE},
            {buffers.grad_sh_coeffs.deviceBuffer,
             TRANSFER_COMPUTE_SHADER_WRITE},
            {buffers.grad_rotations.deviceBuffer,
             TRANSFER_COMPUTE_SHADER_WRITE},
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{count, 256}}, &sparse, sizeof(sparse),
            pipeline_ftgspp_sparse_scatter,
            {
                records,
                buffers.grad_base_xyz_ws.deviceBuffer,
                buffers.grad_base_log_scales_opacity.deviceBuffer,
                buffers.grad_times_durations.deviceBuffer,
                buffers.grad_velocities_gates.deviceBuffer,
                buffers.grad_sh_coeffs.deviceBuffer,
                buffers.grad_rotations.deviceBuffer,
            });
        bufferMemoryBarrier({
            {buffers.grad_base_xyz_ws.deviceBuffer, COMPUTE_SHADER_WRITE},
            {buffers.grad_base_log_scales_opacity.deviceBuffer,
             COMPUTE_SHADER_WRITE},
            {buffers.grad_times_durations.deviceBuffer, COMPUTE_SHADER_WRITE},
            {buffers.grad_velocities_gates.deviceBuffer, COMPUTE_SHADER_WRITE},
            {buffers.grad_sh_coeffs.deviceBuffer, COMPUTE_SHADER_WRITE},
            {buffers.grad_rotations.deviceBuffer, COMPUTE_SHADER_WRITE},
        }, COMPUTE_SHADER_READ_WRITE);
    };
    memoryBarrier(COMPUTE_SHADER_READ, COMPUTE_SHADER_READ_WRITE);
    clear_image_gradients();
    memoryBarrier(COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ_WRITE);
        if (local_rank > 1)
            throw std::runtime_error(
                "pair-sparse merge local_rank must be 0 or 1");
        // scatter pair-sparse sources in rank order so 0.5*g0 + 0.5*g1 uses
        // the same FMA operand order on both replicas under -fp-mode fast.
        const _VulkanBuffer& local_records =
            buffers.sparse_gradient_records.deviceBuffer;
        const _VulkanBuffer& peer_records =
            buffers.peer_sparse_gradient_records.deviceBuffer;
        if (local_rank == 0) {
            scatter_sparse_records(local_records, local_count);
            scatter_sparse_records(peer_records, peer_count);
        } else {
            scatter_sparse_records(peer_records, peer_count);
            scatter_sparse_records(local_records, local_count);
        }
    executeFTGSPPRegularizerMean(config, buffers, rank0_timestamp, rank1_timestamp);
}

void VulkanGSTrainer::executeFTGSPPRelocation(
    const TrainerConfig &config,
    VulkanGSPipelineBuffers& buffers,
    int mode
) {
    PerfTimer::Timer<PerfTimer::Relocation> timer(this);
    const size_t n = buffers.num_splats;
    FTGSPPRelocationUniforms u{
        (uint32_t)n, (uint32_t)mode, config.min_opacity, 0u
    };
    executeCompute(
        {{n, SUBGROUP_SIZE}}, &u, sizeof(u), pipeline_ftgspp_relocation_prepare,
        {
            buffers.relocation_counts.deviceBuffer,
            buffers.base_log_scales_opacity.deviceBuffer,
            buffers.g_xyz_ws.deviceBuffer,
            buffers.g_sh_coeffs_1.deviceBuffer,
            buffers.g_sh_coeffs_2.deviceBuffer,
            buffers.g_rotations.deviceBuffer,
            buffers.g_scales_opacs.deviceBuffer,
            buffers.g_times_durations.deviceBuffer,
            buffers.g_velocities_gates.deviceBuffer,
        }
    );
    memoryBarrier(COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ);
    executeCompute(
        {{n, SUBGROUP_SIZE}}, &u, sizeof(u), pipeline_ftgspp_relocation_copy,
        {
            buffers.relocation_map.deviceBuffer,
            buffers.base_xyz_ws.deviceBuffer,
            buffers.base_log_scales_opacity.deviceBuffer,
            buffers.times_durations.deviceBuffer,
            buffers.velocities_gates.deviceBuffer,
            buffers.sh_coeffs.deviceBuffer,
            buffers.rotations.deviceBuffer,
            buffers.g_xyz_ws.deviceBuffer,
            buffers.g_sh_coeffs_1.deviceBuffer,
            buffers.g_sh_coeffs_2.deviceBuffer,
            buffers.g_rotations.deviceBuffer,
            buffers.g_scales_opacs.deviceBuffer,
            buffers.g_times_durations.deviceBuffer,
            buffers.g_velocities_gates.deviceBuffer,
        }
    );
    clearDeviceBuffer(buffers.relocation_stats, 2*n);
}

void VulkanGSTrainer::executeFTGSPPRelocationPlanNative(
    const TrainerConfig &config,
    VulkanGSPipelineBuffers& buffers,
    int step,
    bool include_gate,
    uint32_t seed
) {
    requireFTGSPPFloat32AtomicAdd();
    const size_t n = buffers.num_splats;
    if (n == 0) return;
    FTGSPPRelocationPlanUniforms u{
        (uint32_t)n, include_gate ? 1u : 0u,
        config.min_opacity, seed ^ (uint32_t)step,
        config.hard_separation ? 1u : 0u
    };

    clearDeviceBuffer(buffers.relocation_totals, 4);
    memoryBarrier(TRANSFER_WRITE, COMPUTE_SHADER_READ_WRITE);
    executeCompute({{n,SUBGROUP_SIZE}}, &u, sizeof(u),
        pipeline_ftgspp_relocation_score,
        {buffers.base_log_scales_opacity.deviceBuffer,
         buffers.velocities_gates.deviceBuffer,
         buffers.relocation_stats.deviceBuffer,
         resizeDeviceBuffer(buffers.relocation_raw, 4*n),
         buffers.relocation_totals.deviceBuffer});
    memoryBarrier(COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ);

    resizeDeviceBuffer(
        buffers.relocation_cdf, config.hard_separation ? 2 * n : n);
    clearDeviceBuffer(buffers.relocation_status, 4);
    memoryBarrier(TRANSFER_WRITE, COMPUTE_SHADER_READ_WRITE);
    executeCompute({{n,SUBGROUP_SIZE}}, &u, sizeof(u),
        pipeline_ftgspp_relocation_cdf,
        {buffers.relocation_raw.deviceBuffer,
         buffers.relocation_totals.deviceBuffer,
         buffers.relocation_cdf.deviceBuffer});
    memoryBarrier(COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ_WRITE);
    executeFloatInclusiveScan(buffers, buffers.relocation_cdf);
    memoryBarrier(COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ_WRITE);
    executeCompute({{1,SUBGROUP_SIZE}}, &u, sizeof(u),
        pipeline_ftgspp_relocation_cdf_normalize,
        {buffers.relocation_totals.deviceBuffer,
         buffers.relocation_status.deviceBuffer});
    memoryBarrier(COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ);

    clearDeviceBuffer(buffers.relocation_counts, n);
    resizeDeviceBuffer(buffers.relocation_map, n);
    memoryBarrier(TRANSFER_WRITE, COMPUTE_SHADER_READ_WRITE);
    executeCompute({{n,SUBGROUP_SIZE}}, &u, sizeof(u),
        pipeline_ftgspp_relocation_sample,
        {buffers.relocation_raw.deviceBuffer,
         buffers.relocation_cdf.deviceBuffer,
         buffers.relocation_map.deviceBuffer,
         buffers.relocation_counts.deviceBuffer,
         buffers.relocation_status.deviceBuffer});
    memoryBarrier(COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ);
}

void VulkanGSTrainer::executeFTGSPPRelocationNative(
    const TrainerConfig &config,
    VulkanGSPipelineBuffers& buffers,
    int mode,
    int step,
    bool include_gate,
    uint32_t seed
) {
    executeFTGSPPRelocationPlanNative(
        config, buffers, step, include_gate, seed);
    executeFTGSPPRelocation(config, buffers, mode);
}

void VulkanGSTrainer::executeFTGSPPFlowFeatures(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers
) {
    const size_t n = buffers.num_splats;
    bufferMemoryBarrier({
        { buffers.flow_end_xyz.deviceBuffer, TRANSFER_WRITE },
        { buffers.xy_vs.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.tiles_touched.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.rgb.deviceBuffer, COMPUTE_SHADER_WRITE },
    }, COMPUTE_SHADER_READ_WRITE);
    executeCompute(
        {{n, SUBGROUP_SIZE}}, &uniforms, sizeof(uniforms),
        pipeline_ftgspp_flow_features,
        {
            buffers.flow_end_xyz.deviceBuffer,
            buffers.xy_vs.deviceBuffer,
            resizeDeviceBuffer(buffers.rgb, 3*n),
            buffers.tiles_touched.deviceBuffer,
        }
    );
    bufferMemoryBarrier({
        { buffers.rgb.deviceBuffer, COMPUTE_SHADER_WRITE },
    }, COMPUTE_SHADER_READ);
}

void VulkanGSTrainer::executeFTGSPPHardSeparation(
    VulkanGSPipelineBuffers& buffers
) {
    const size_t n = buffers.num_splats;
    // The separation stage constructs a fresh Adam optimizer. Reset every
    // native moment buffer before starting its configured recovery pass.
    clearDeviceBuffer(buffers.g_xyz_ws, 2*3*n);
    clearDeviceBuffer(buffers.g_sh_coeffs_1, 16*3*_CEIL_ROUND(n,SH_REORDER_SIZE));
    clearDeviceBuffer(buffers.g_sh_coeffs_2, 16*3*_CEIL_ROUND(n,SH_REORDER_SIZE));
    clearDeviceBuffer(buffers.g_rotations, 2*4*n);
    clearDeviceBuffer(buffers.g_scales_opacs, 2*4*n);
    clearDeviceBuffer(buffers.g_times_durations, 2*2*n);
    clearDeviceBuffer(buffers.g_velocities_gates, 2*4*n);
    clearDeviceBuffer(buffers.relocation_stats, 2*n);
    clearDeviceBuffer(buffers.relocation_status, 4);
    memoryBarrier(TRANSFER_WRITE, COMPUTE_SHADER_READ_WRITE);
    uint32_t uniforms[4] = {(uint32_t)n, 0u, 0u, 0u};
    executeCompute({{n,SUBGROUP_SIZE}}, uniforms, sizeof(uniforms),
        pipeline_ftgspp_hard_separation,
        {buffers.velocities_gates.deviceBuffer,
         buffers.g_velocities_gates.deviceBuffer,
         buffers.relocation_status.deviceBuffer});
    memoryBarrier(COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ);
}

void VulkanGSTrainer::ensureFTGSPPColorState(
    const TrainerConfig& config,
    VulkanGSPipelineBuffers& buffers
) {
    if (!config.color_correction || config.color_num_cameras == 0)
        return;
    const size_t param_count = size_t(config.color_num_cameras) * 12u;
    if (buffers.color_params.deviceSize() == param_count &&
        buffers.color_m.deviceSize() == param_count &&
        buffers.color_v.deviceSize() == param_count)
        return;

    // A first allocation or camera-count change creates the same complete
    // zero state on every replica.  Matching device sizes mean a restored or
    // already-trained state and are deliberately never overwritten.
    buffers.color_params.assign(param_count, 0.0f);
    buffers.color_m.assign(param_count, 0.0f);
    buffers.color_v.assign(param_count, 0.0f);
    copyToDevice(buffers.color_params);
    copyToDevice(buffers.color_m);
    copyToDevice(buffers.color_v);
}

void VulkanGSTrainer::executeFTGSPPColorCorrectForwardWH(
    const TrainerConfig& config,
    VulkanGSPipelineBuffers& buffers,
    uint32_t width,
    uint32_t height,
    uint32_t camera_index
) {
    if (!config.color_correction || config.color_num_cameras == 0)
        return;
    PerfTimer::Timer<PerfTimer::ColorCorrection> timer(this);
    if (camera_index >= config.color_num_cameras)
        throw std::runtime_error("color correction camera_index out of range");
    ensureFTGSPPColorState(config, buffers);
    const size_t num_pixels = size_t(width) * size_t(height);
    resizeDeviceBuffer(buffers.color_raw_rgb, 4 * num_pixels);

    struct ColorCorrectUniforms {
        uint32_t width, height, camera_index, enabled, objective_mode;
    } u{width, height, camera_index, 1u, config.image_objective};
    static_assert(sizeof(ColorCorrectUniforms) == 20,
        "FTGSPP color-forward push-constant ABI changed");
    bufferMemoryBarrier({
        {buffers.pixel_state.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.color_params.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
        {buffers.color_raw_rgb.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
    }, COMPUTE_SHADER_READ_WRITE);
    executeCompute(
        {{num_pixels, SUBGROUP_SIZE}}, &u, sizeof(u),
        pipeline_ftgspp_color_correct_forward,
        {buffers.pixel_state.deviceBuffer, buffers.color_params.deviceBuffer,
         buffers.color_raw_rgb.deviceBuffer,
         trainer_work_control(this, buffers)});
    bufferMemoryBarrier({
        {buffers.pixel_state.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.color_raw_rgb.deviceBuffer, COMPUTE_SHADER_WRITE},
    }, COMPUTE_SHADER_READ_WRITE);
}

void VulkanGSTrainer::executeFTGSPPColorCorrectBackwardRawWH(
    const TrainerConfig& config,
    VulkanGSPipelineBuffers& buffers,
    uint32_t width,
    uint32_t height,
    uint32_t camera_index
) {
    if (!config.color_correction || config.color_num_cameras == 0)
        return;
    PerfTimer::Timer<PerfTimer::ColorCorrection> timer(this);
    requireFTGSPPFloat32AtomicAdd();
    if (camera_index >= config.color_num_cameras)
        throw std::runtime_error("color correction camera_index out of range");
    const size_t num_pixels = size_t(width) * size_t(height);
    const size_t param_count = size_t(config.color_num_cameras) * 12u;
    const bool accumulate_parameter_grads = !config.hard_separation;
    ensureFTGSPPColorState(config, buffers);
    if (accumulate_parameter_grads) {
        resizeDeviceBuffer(buffers.color_grad, param_count);
        clearDeviceBuffer(buffers.color_grad, param_count);
        bufferMemoryBarrier({
            // SSIM backward is the compute producer of v_pixel_state.  The
            // color backward pass reads it and overwrites RGB in place.
            {buffers.v_pixel_state.deviceBuffer, COMPUTE_SHADER_WRITE},
            {buffers.color_grad.deviceBuffer, TRANSFER_WRITE},
        }, COMPUTE_SHADER_READ_WRITE);
    } else {
        // The descriptor remains valid, but enabled=2 never indexes it.
        resizeDeviceBuffer(buffers.color_grad, 1u);
        bufferMemoryBarrier(
            {{buffers.v_pixel_state.deviceBuffer, COMPUTE_SHADER_WRITE}},
            COMPUTE_SHADER_READ_WRITE);
    }

    struct ColorCorrectUniforms {
        uint32_t width, height, camera_index, enabled, objective_mode;
    } u{
        width, height, camera_index,
        accumulate_parameter_grads ? 1u : 2u,
        config.image_objective,
    };
    static_assert(sizeof(ColorCorrectUniforms) == 20,
        "FTGSPP color-backward push-constant ABI changed");
    executeCompute(
        {{num_pixels, SUBGROUP_SIZE}}, &u, sizeof(u),
        pipeline_ftgspp_color_correct_backward,
        {buffers.pixel_state.deviceBuffer, buffers.v_pixel_state.deviceBuffer,
         buffers.color_params.deviceBuffer, buffers.color_grad.deviceBuffer,
         buffers.color_raw_rgb.deviceBuffer,
         trainer_work_control(this, buffers)});
    if (accumulate_parameter_grads) {
        bufferMemoryBarrier({
            {buffers.v_pixel_state.deviceBuffer, COMPUTE_SHADER_WRITE},
            {buffers.color_grad.deviceBuffer, COMPUTE_SHADER_WRITE},
        }, TRANSFER_COMPUTE_SHADER_READ_WRITE);
    } else {
        bufferMemoryBarrier(
            {{buffers.v_pixel_state.deviceBuffer, COMPUTE_SHADER_WRITE}},
            TRANSFER_COMPUTE_SHADER_READ_WRITE);
    }
}

void VulkanGSTrainer::executeFTGSPPColorAdam(
    const TrainerConfig& config,
    VulkanGSPipelineBuffers& buffers,
    int step,
    bool collect_loss_scalar
) {
    if (!config.color_correction || config.color_num_cameras == 0)
        return;
    // The hard optimizer does not own color parameters or an L_cc term.  The
    // fixed affine transform still participates in forward/backward through
    // executeFTGSPPColorCorrect*; there is no optimizer/scalar work here.
    if (config.hard_separation)
        return;
    PerfTimer::Timer<PerfTimer::ColorCorrection> timer(this);
    requireFTGSPPFloat32AtomicAdd();
    const size_t param_count = size_t(config.color_num_cameras) * 12u;
    if (buffers.color_grad.deviceSize() != param_count)
        throw std::runtime_error(
            "color Adam requires a preceding color-correction backward pass");
    struct ColorAdamUniforms {
        uint32_t count;
        uint32_t step;
        float lr;
        float reg;
        uint32_t update_parameters;
    } adam{
        (uint32_t)param_count,
        (uint32_t)std::max(step, 1),
        config.color_lr,
        config.color_reg,
        1u | (collect_loss_scalar ? 2u : 0u),
    };
    static_assert(sizeof(ColorAdamUniforms) == 20,
        "FTGSPP color-Adam push-constant ABI changed");
    bufferMemoryBarrier({
        {buffers.color_params.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
        {buffers.color_grad.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
        {buffers.color_m.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.color_v.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.loss_accum.deviceBuffer, COMPUTE_SHADER_WRITE},
    }, COMPUTE_SHADER_READ_WRITE);
    executeCompute(
        {{param_count, 256}}, &adam, sizeof(adam),
        pipeline_ftgspp_color_adam,
        {buffers.color_params.deviceBuffer, buffers.color_grad.deviceBuffer,
         buffers.color_m.deviceBuffer, buffers.color_v.deviceBuffer,
         buffers.loss_accum.deviceBuffer,
         trainer_work_control(this, buffers)});
    bufferMemoryBarrier({
        {buffers.color_params.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.color_grad.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.color_m.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.color_v.deviceBuffer, COMPUTE_SHADER_WRITE},
        {buffers.loss_accum.deviceBuffer, COMPUTE_SHADER_WRITE},
    }, TRANSFER_COMPUTE_SHADER_READ_WRITE);
}

// End of VulkanGSTrainer implementation.
