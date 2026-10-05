#pragma once

#include <glm/glm.hpp>
#include <array>

#include "render/gs_renderer.h"

// Configuration owned by the Python SelfCap runtime.
struct TrainerConfig {
    float global_scale;

    // optimizer
    int max_steps;
    int max_sh_degree;
    float ssim_lambda;
    // 0=scene_rgb_v1, 1=composite_v1.
    uint32_t image_objective;
    float means_lr;
    float means_lr_final;
    float features_dc_lr;
    float features_rest_lr;
    float opacities_lr;
    float scales_lr;
    float quats_lr;
    float scale_reg;
    float opacity_reg;

    // FTGSPP dynamic optimizer and temporal model.
    float times_lr;
    float durations_lr;
    float velocities_lr;
    float velocities_lr_final;
    float gates_lr;
    float gate_reg;
    float max_duration;
    float marginal_threshold;
    bool hard_separation;
    bool color_correction;
    float color_lr;
    float color_reg;
    uint32_t color_num_cameras;
    int color_correction_start;
    uint32_t seed;
    bool sparse_distributed;
    // One preserves the historical Adam.  B>1 enables FTGSPP's effective
    // batch rule: beta1=0.9^B, beta2=0.999^B, eps=1e-15/sqrt(B).
    uint32_t optimizer_batch_size;
    // Distinct from the optional pair Adam beta/epsilon compensation above.
    uint32_t local_batch_size = 1u;

    inline float get_means_lr(int step, float scene_scale) const {
        float progress = fmin(float(step)/float(this->max_steps), 1.0f);
        return this->means_lr * pow(this->means_lr_final/this->means_lr, progress) * scene_scale;
    }

    inline float get_velocities_lr(int step, float scene_scale) const {
        if (this->velocities_lr <= 0.0f || this->velocities_lr_final <= 0.0f)
            return 0.0f;
        float progress = fmin(float(step)/float(this->max_steps), 1.0f);
        return this->velocities_lr *
            pow(this->velocities_lr_final/this->velocities_lr, progress) * scene_scale;
    }

    float min_opacity;
};



class VulkanGSTrainer : public VulkanGSRenderer {
public:
    VulkanGSTrainer();
    ~VulkanGSTrainer();

    struct Camera {
        int h, w;
        float fx, fy, cx, cy;
        glm::mat4 world_view_transform;
    };

    struct DatasetImage {
        Camera camera;
        Buffer<uint8_t> buffer;  // (h, w, 4)
        Buffer<uint32_t> validity;  // (h, w), pixel/window validity bits
        uint32_t valid_pixel_count = 0;
        uint32_t valid_window_count = 0;
        uint32_t foreground_window_count = 0;
        bool has_custom_validity = false;
        float timestamp = 0.0f;
        int32_t camera_id = -1;
        int32_t frame_id = -1;
    };

    void initialize(const std::map<std::string, std::string> &spirv_paths, int device_id);
    void cleanup();
    // FTGSPP raw-loss/gradient shaders use native float atomics directly.
    void requireFTGSPPFloat32AtomicAdd() const;

    void clear_external_dataset();
    void add_external_view(DatasetImage&& image, bool is_train);

    size_t num_train() const { return dataset_train.size(); }
    size_t num_val() const { return dataset_val.size(); }

    DatasetImage& get_train_image(size_t idx) { return dataset_train[idx]; }
    DatasetImage& get_val_image(size_t idx) { return dataset_val[idx]; }
    const DatasetImage& get_train_image(size_t idx) const { return dataset_train.at(idx); }
    const DatasetImage& get_val_image(size_t idx) const { return dataset_val.at(idx); }
    float get_train_timestamp(size_t idx) const { return dataset_train.at(idx).timestamp; }
    float get_val_timestamp(size_t idx) const { return dataset_val.at(idx).timestamp; }
    void camera_to_uniforms(const Camera &cam, VulkanGSRendererUniforms &uniforms) const;
    void get_train_camera(size_t idx, VulkanGSRendererUniforms &uniforms) const;
    void get_val_camera(size_t idx, VulkanGSRendererUniforms &uniforms) const;

    void executeComputeSSIMGradient(
        const TrainerConfig &config,
        const VulkanGSRendererUniforms& uniforms,
        VulkanGSPipelineBuffers& buffers,
        size_t train_idx,
        uint32_t prediction_clamp_mode = 1u,
        bool collect_loss_scalar = true,
        const std::array<float, 3>& background = {0.0f, 0.0f, 0.0f});
    void executeEvalMetrics(
        const TrainerConfig &config,
        const VulkanGSRendererUniforms& uniforms,
        VulkanGSPipelineBuffers& buffers,
        size_t val_idx,
        const std::array<float, 3>& background,
        const std::array<uint32_t, 4>& roi_yxyx = {0u, 0u, 0u, 0u},
        bool upload_reference = true);
    void executeFTGSPPMaterialize(const TrainerConfig &config, VulkanGSPipelineBuffers& buffers, float timestamp);
    void executeFTGSPPBackward(
        const TrainerConfig &config, const VulkanGSRendererUniforms& uniforms,
        VulkanGSPipelineBuffers& buffers, int step, float timestamp,
        bool collect_loss_scalar = true);
    void executeFTGSPPFusedDynamicBackwardAdam(
        const TrainerConfig &config, const VulkanGSRendererUniforms& uniforms,
        VulkanGSPipelineBuffers& buffers, int step, float timestamp,
        bool collect_loss_scalar = true,
        bool collect_geometry_grad_stats = false);
    void executeFTGSPPExplicitAdam(
        const TrainerConfig &config, const VulkanGSRendererUniforms& uniforms,
        VulkanGSPipelineBuffers& buffers, int step,
        bool collect_geometry_grad_stats = false);
    void executeFTGSPPPeerMean(
        const _VulkanBuffer& local, const _VulkanBuffer& peer,
        size_t float_count, float local_weight=0.5f,
        float peer_weight=0.5f, size_t local_offset=0,
        size_t peer_offset=0, bool reset=false);
    void writeFTGSPPSparseActiveFlags(
        VulkanGSPipelineBuffers& buffers);
    uint32_t executeFTGSPPSparseGradientPack(
        const TrainerConfig& config, VulkanGSPipelineBuffers& buffers,
        uint32_t active_sh);
    void executeFTGSPPSparseGradientMerge(
        const TrainerConfig& config, VulkanGSPipelineBuffers& buffers,
        uint32_t peer_count,
        float rank0_timestamp, float rank1_timestamp,
        uint32_t active_sh=3u,
        uint32_t local_rank=0, uint32_t local_count=0);
    void executeFTGSPPRegularizerMean(
        const TrainerConfig& config, VulkanGSPipelineBuffers& buffers,
        float rank0_timestamp, float rank1_timestamp);
    void executeFTGSPPRelocation(const TrainerConfig &config, VulkanGSPipelineBuffers& buffers, int mode);
    void executeFTGSPPRelocationPlanNative(
        const TrainerConfig &config, VulkanGSPipelineBuffers& buffers,
        int step, bool include_gate, uint32_t seed);
    void executeFTGSPPRelocationNative(
        const TrainerConfig &config, VulkanGSPipelineBuffers& buffers,
        int mode, int step, bool include_gate, uint32_t seed);
    void executeFTGSPPFlowFeatures(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers);
    void executeFTGSPPHardSeparation(VulkanGSPipelineBuffers& buffers);
    void executeFTGSPPColorCorrectForwardWH(
        const TrainerConfig& config, VulkanGSPipelineBuffers& buffers,
        uint32_t width, uint32_t height, uint32_t camera_index);
    void ensureFTGSPPColorState(
        const TrainerConfig& config, VulkanGSPipelineBuffers& buffers);
    void executeFTGSPPColorCorrectBackwardRawWH(
        const TrainerConfig& config, VulkanGSPipelineBuffers& buffers,
        uint32_t width, uint32_t height, uint32_t camera_index);
    void executeFTGSPPColorAdam(
        const TrainerConfig& config, VulkanGSPipelineBuffers& buffers, int step,
        bool collect_loss_scalar = true);
private:
    std::vector<DatasetImage> dataset_train;
    std::vector<DatasetImage> dataset_val;

    _ComputePipeline pipeline_ssim_forward = _ComputePipeline(5);
    _ComputePipeline pipeline_ssim_backward = _ComputePipeline(6);
    _ComputePipeline pipeline_ftgspp_loss_reduce = _ComputePipeline(6);
    _ComputePipeline pipeline_ftgspp_eval_metrics = _ComputePipeline(6);
    _ComputePipeline pipeline_ftgspp_materialize = _ComputePipeline(6);
    _ComputePipeline pipeline_ftgspp_fused_backward_optimizer = _ComputePipeline(19);
    _ComputePipeline pipeline_ftgspp_fused_dynamic_backward_adam =
        _ComputePipeline(21);
    _ComputePipeline pipeline_ftgspp_fused_dynamic_backward_adam_sh[4] = {
        _ComputePipeline(21), _ComputePipeline(21),
        _ComputePipeline(21), _ComputePipeline(21),
    };
    _ComputePipeline pipeline_ftgspp_fused_backward_optimizer_sh[4] = {
        _ComputePipeline(19), _ComputePipeline(19),
        _ComputePipeline(19), _ComputePipeline(19),
    };
    _ComputePipeline pipeline_ftgspp_explicit_adam = _ComputePipeline(21);
    _ComputePipeline pipeline_ftgspp_peer_mean = _ComputePipeline(2);
    _ComputePipeline pipeline_ftgspp_sparse_flag = _ComputePipeline(2);
    _ComputePipeline pipeline_ftgspp_sparse_pack = _ComputePipeline(10);
    _ComputePipeline pipeline_ftgspp_sparse_scatter = _ComputePipeline(7);
    _ComputePipeline pipeline_ftgspp_regularizer_mean = _ComputePipeline(5);
    _ComputePipeline pipeline_ftgspp_relocation_prepare = _ComputePipeline(9);
    _ComputePipeline pipeline_ftgspp_relocation_copy = _ComputePipeline(14);
    _ComputePipeline pipeline_ftgspp_relocation_score = _ComputePipeline(5);
    _ComputePipeline pipeline_ftgspp_relocation_cdf = _ComputePipeline(3);
    _ComputePipeline pipeline_ftgspp_relocation_cdf_normalize = _ComputePipeline(2);
    _ComputePipeline pipeline_ftgspp_relocation_sample = _ComputePipeline(5);
    _ComputePipeline pipeline_ftgspp_flow_features = _ComputePipeline(4);
    _ComputePipeline pipeline_ftgspp_hard_separation = _ComputePipeline(3);
    _ComputePipeline pipeline_ftgspp_color_correct_forward = _ComputePipeline(4);
    _ComputePipeline pipeline_ftgspp_color_correct_backward = _ComputePipeline(6);
    _ComputePipeline pipeline_ftgspp_color_adam = _ComputePipeline(6);
};
