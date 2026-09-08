#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "gs_renderer.h"
#include "buffer.h"
#include "gs_trainer.h"
#include "host_mailbox_owner.h"
#include "hierarchical_host_collective.h"
#include "sync_generation_state.h"
#include "tile_work_control.h"

#include <glm/glm.hpp>
#include <glm/matrix.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace py = pybind11;

namespace {
std::shared_ptr<HostMailboxOwner> g_host_mailbox_owner;
std::shared_ptr<HostMailboxOwner> g_sparse_mailbox_owner;
std::shared_ptr<HostMailboxOwner> g_relocation_mailbox_owner;
std::shared_ptr<SyncGenerationState> g_sync_generation;
std::mutex g_sync_generation_mutex;

constexpr const char* kTileOverflowAbortReason = "tile_overflow_generation";

bool is_tile_overflow_abort(const std::exception& error) {
    const std::string what = error.what();
    return what.find("tile_overflow") != std::string::npos;
}

double median_of(std::vector<double> values) {
    if (values.empty())
        return 0.0;
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    return values[mid];
}

void ensure_sync_generation(uint32_t world_size) {
    std::lock_guard<std::mutex> lock(g_sync_generation_mutex);
    if (!g_sync_generation) {
        g_sync_generation = std::make_shared<SyncGenerationState>(world_size);
        return;
    }
    if (g_sync_generation->world_size() != world_size)
        throw std::runtime_error(
            "sync generation world size does not match the active collective");
}

void release_sync_generation() {
    std::lock_guard<std::mutex> lock(g_sync_generation_mutex);
    g_sync_generation.reset();
}

struct HierarchicalHostBindingState {
    uint32_t world_size = 0;
    size_t requested_bucket_bytes = 0;
    uint32_t local_microsteps = 1;
    uint32_t reducer_shards = 0;
    HierarchicalHostAlgorithm requested_algorithm =
        HierarchicalHostAlgorithm::Auto;
    HierarchicalHostAlgorithm selected_algorithm =
        HierarchicalHostAlgorithm::Central;
    std::mutex mutex;
    std::shared_ptr<HierarchicalHostCollective> gradient;
    std::shared_ptr<HierarchicalHostCollective> gradient_sparse;
    std::shared_ptr<HierarchicalHostCollective> relocation;

    std::mutex codec_mutex;
    std::condition_variable codec_cv;
    uint64_t codec_generation = 0;
    std::vector<HierarchicalExactCodecVote> codec_votes;
    std::vector<char> codec_submitted;
    std::string codec_decision;
    uint64_t codec_union_count = 0;
    bool codec_ready = false;
    bool codec_aborted = false;
    std::string codec_abort_reason;

    void reset_codec_locked() {
        codec_generation = 0;
        codec_votes.assign(world_size, {});
        codec_submitted.assign(world_size, 0);
        codec_decision.clear();
        codec_union_count = 0;
        codec_ready = false;
        codec_aborted = false;
        codec_abort_reason.clear();
    }

    void abort_codec(const std::string& reason) {
        std::lock_guard<std::mutex> lock(codec_mutex);
        codec_aborted = true;
        codec_abort_reason = reason;
        codec_cv.notify_all();
    }

    std::string agree_exact_codec(
        uint32_t rank,
        uint64_t generation,
        HierarchicalExactCodecVote vote,
        const std::string& policy) {
        if (policy == "dense")
            return "dense";
        if (policy == "sorted_sparse")
            return "exact_sorted_sparse";
        std::unique_lock<std::mutex> lock(codec_mutex);
        if (codec_aborted)
            throw std::runtime_error(
                "hierarchical codec agreement aborted: " + codec_abort_reason);
        if (codec_generation != generation) {
            codec_generation = generation;
            codec_votes.assign(world_size, {});
            codec_submitted.assign(world_size, 0);
            codec_decision.clear();
            codec_union_count = 0;
            codec_ready = false;
        }
        if (rank >= world_size)
            throw std::out_of_range(
                "hierarchical codec rank is out of range");
        if (codec_submitted[rank])
            throw std::runtime_error(
                "hierarchical codec rank voted twice in generation " +
                std::to_string(generation));
        codec_votes[rank] = vote;
        codec_submitted[rank] = 1;
        if (std::all_of(
                codec_submitted.begin(), codec_submitted.end(),
                [](char value) { return value != 0; })) {
            const auto snapshot = codec_votes;
            lock.unlock();
            uint64_t union_count = 0;
            std::string decision;
            try {
                decision = decide_hierarchical_exact_codec(
                    snapshot.data(), world_size, 0.05, &union_count);
            } catch (const std::exception& error) {
                lock.lock();
                codec_aborted = true;
                codec_abort_reason = error.what();
                codec_cv.notify_all();
                throw;
            } catch (...) {
                lock.lock();
                codec_aborted = true;
                codec_abort_reason =
                    "unknown hierarchical codec agreement failure";
                codec_cv.notify_all();
                throw;
            }
            lock.lock();
            if (codec_aborted)
                throw std::runtime_error(
                    "hierarchical codec agreement aborted: " +
                    codec_abort_reason);
            codec_decision = std::move(decision);
            codec_union_count = union_count;
            codec_ready = true;
            codec_cv.notify_all();
        } else {
            codec_cv.wait(lock, [&]() {
                return codec_ready || codec_aborted;
            });
        }
        if (codec_aborted)
            throw std::runtime_error(
                "hierarchical codec agreement aborted: " + codec_abort_reason);
        return codec_decision;
    }
};

std::mutex g_hierarchical_host_mutex;
std::shared_ptr<HierarchicalHostBindingState> g_hierarchical_host_state;

void abort_sync_collectives(const std::string& reason) {
    if (g_host_mailbox_owner)
        g_host_mailbox_owner->abort(reason);
    if (g_sparse_mailbox_owner)
        g_sparse_mailbox_owner->abort(reason);
    if (g_relocation_mailbox_owner)
        g_relocation_mailbox_owner->abort(reason);
    std::shared_ptr<HierarchicalHostBindingState> hierarchical;
    {
        std::lock_guard<std::mutex> lock(g_hierarchical_host_mutex);
        hierarchical = g_hierarchical_host_state;
    }
    if (!hierarchical)
        return;
    hierarchical->abort_codec(reason);
    std::lock_guard<std::mutex> lock(hierarchical->mutex);
    if (hierarchical->gradient)
        hierarchical->gradient->abort(reason);
    if (hierarchical->gradient_sparse)
        hierarchical->gradient_sparse->abort(reason);
    if (hierarchical->relocation)
        hierarchical->relocation->abort(reason);
}

void clear_sync_collectives_for_replay() {
    if (g_host_mailbox_owner)
        g_host_mailbox_owner->clear_abort();
    if (g_sparse_mailbox_owner)
        g_sparse_mailbox_owner->clear_abort();
    if (g_relocation_mailbox_owner)
        g_relocation_mailbox_owner->clear_abort();
    std::shared_ptr<HierarchicalHostBindingState> hierarchical;
    {
        std::lock_guard<std::mutex> lock(g_hierarchical_host_mutex);
        hierarchical = g_hierarchical_host_state;
    }
    if (hierarchical) {
        {
            std::lock_guard<std::mutex> lock(hierarchical->codec_mutex);
            hierarchical->reset_codec_locked();
        }
        std::lock_guard<std::mutex> lock(hierarchical->mutex);
        if (hierarchical->gradient)
            hierarchical->gradient->clear_abort();
        if (hierarchical->gradient_sparse)
            hierarchical->gradient_sparse->clear_abort();
        if (hierarchical->relocation)
            hierarchical->relocation->clear_abort();
    }
    std::lock_guard<std::mutex> lock(g_sync_generation_mutex);
    if (g_sync_generation)
        g_sync_generation->clear_for_replay();
}
}

class VkSplat {
public:
    // VulkanGSRenderer trainer;
    VulkanGSTrainer trainer;
    VulkanGSRendererUniforms uniforms;
    VulkanGSPipelineBuffers buffers;
    TrainerConfig config{};
    bool initialized = false;
    bool closed = false;
    bool gradient_ready = false;
    bool gradient_finalized = false;
    int gradient_step = -1;
    uint32_t local_gradient_microsteps = 0;
    int host_mailbox_rank = -1;
    int sparse_mailbox_rank = -1;
    int relocation_mailbox_rank = -1;
    int hierarchical_gradient_rank = -1;
    int hierarchical_relocation_rank = -1;
    bool hierarchical_gradient_attached_ = false;
    bool hierarchical_sparse_attached_ = false;
    bool hierarchical_relocation_attached_ = false;
    bool hierarchical_sparse_mean_ready_ = false;
    bool hierarchical_needs_regularizer_ = false;
    uint32_t pending_peer_sparse_count = 0;
    uint32_t pending_local_sparse_count = 0;
    bool pair_sparse_mean_ready_ = false;
    bool pair_needs_regularizer_ = false;
    std::string pair_codec_policy_ = "dense";
    std::string pair_gaussian_codec_ = "dense";
    std::string hierarchical_gaussian_codec_ = "dense";
    std::string hierarchical_codec_policy_ = "auto_exact";
    // Lightweight geometry / SH gradient logging: enabled only on log steps.
    // Adam accumulates sum-of-squares into grad_stats when the flag is set.
    bool collect_geometry_grad_stats_ = false;
    bool geometry_grad_stats_ready_ = false;
    float geometry_grad_sum_sq_[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    size_t geometry_grad_num_splats_ = 0;
    bool pending_collect_loss_scalar_ = true;
    bool resolving_tile_overflow_ = false;
    bool force_sync_overflow_ = false;
    uint32_t force_sync_overflow_exact_ = 0;
    bool relocation_stats_checkpoint_valid_ = false;
    std::optional<VulkanGSPipeline::PackedTransferTicket> eval_rgb_ticket_;
    size_t last_eval_view_idx_ = 0;
    struct PendingTileStep {
        bool valid = false;
        bool is_train_view = true;
        size_t view_idx = 0;
        int step = 0;
        bool collect_loss = true;
        float timestamp = 0.0f;
    };
    static constexpr size_t kPendingTileDepth = 4;
    std::array<PendingTileStep, kPendingTileDepth> pending_tile_ring_{};
    size_t pending_tile_count_ = 0;

    void record_pending_tile_step(PendingTileStep pending) {
        pending.valid = true;
        if (pending_tile_count_ >= pending_tile_ring_.size()) {
            for (size_t i = 1; i < pending_tile_count_; ++i)
                pending_tile_ring_[i - 1] = pending_tile_ring_[i];
            pending_tile_count_ = pending_tile_ring_.size() - 1;
        }
        pending_tile_ring_[pending_tile_count_++] = pending;
    }

    template<typename Ti, typename To, bool on_device=true>
    py::array_t<To> buffer_to_array(
        Buffer<Ti> &buffer,
        size_t n1=(size_t)(-1),
        size_t n2=(size_t)(-1),
        std::string reorder = "",
        size_t n1_actual=(size_t)-1,
        size_t n1_offset=(size_t)0
    ) {
        if (on_device) {
            // Device readback can block on the Vulkan queue.  Let sibling
            // rank workers enter their own native backend while this rank is
            // waiting; Python array construction below still runs with the
            // GIL held.
            py::gil_scoped_release release;
            trainer.copyFromDevice(buffer);
        }
        if (reorder == "sh") {
            py::gil_scoped_release release;
            buffers.undoReorderSH(buffer, buffers.num_splats);
        }

        size_t n0 = buffer.size();
        if (n1 != (size_t)(-1)) n0 /= n1;
        if (n2 != (size_t)(-1)) n0 /= n2;

        if (n1 == (size_t)(-1) && n2 == (size_t)(-1)) {
            py::array_t<To> result({(py::ssize_t)n0});
            To* out = result.mutable_data();
            {
                py::gil_scoped_release release;
                for (size_t i = 0; i < n0; ++i)
                    out[i] = sizeof(Ti) == sizeof(To)
                        ? (To)buffer[i]
                        : (To)halfToFloat((uint16_t)buffer[i]);
            }
            return result;
        }
        else if (n2 == (size_t)(-1)) {
            if (n1_actual == (size_t)-1)
                n1_actual = n1;
            py::array_t<To> result({(py::ssize_t)n0, (py::ssize_t)n1_actual});
            To* out = result.mutable_data();
            {
                py::gil_scoped_release release;
                for (size_t row = 0; row < n0; ++row) {
                    for (size_t col = 0; col < n1_actual; ++col) {
                        Ti value = buffer[row*n1+n1_offset+col];
                        out[row*n1_actual+col] = sizeof(Ti) == sizeof(To)
                            ? (To)value : (To)halfToFloat((uint16_t)value);
                    }
                }
            }
            return result;
        }
        else {
            py::array_t<To> result({
                (py::ssize_t)n0, (py::ssize_t)n1, (py::ssize_t)n2});
            To* out = result.mutable_data();
            const size_t total = n0*n1*n2;
            {
                py::gil_scoped_release release;
                for (size_t i = 0; i < total; ++i)
                    out[i] = sizeof(Ti) == sizeof(To)
                        ? (To)buffer[i]
                        : (To)halfToFloat((uint16_t)buffer[i]);
            }
            return result;
        }
    }

    template<typename T>
    void array_to_buffer(
        py::array_t<T> array,
        Buffer<T> &buffer
    ) {
        auto array_buf = array.request();
        buffer.assign((T*)(array_buf.ptr), (T*)(array_buf.ptr) + array_buf.size);
        trainer.copyToDevice(buffer);
    }

public:

    VkSplat() {
        uniforms = {};
        // Safe defaults for the Python-owned SelfCap path.  Unlike the
        // original VkSplat COLMAP CLI, this path is configured without loading
        // a second native dataset.
        config.max_steps = 30000;
        config.max_sh_degree = 3;
        config.ssim_lambda = 0.2f;
        config.global_scale = 1.0f;
        config.means_lr = 1.6e-4f;
        config.means_lr_final = 1.6e-6f;
        config.features_dc_lr = 2.5e-3f;
        config.features_rest_lr = 1.25e-4f;
        config.opacities_lr = 5e-2f;
        config.scales_lr = 5e-3f;
        config.quats_lr = 1e-3f;
        config.times_lr = 1.6e-4f;
        config.durations_lr = 5e-3f;
        config.velocities_lr = 1.6e-4f;
        config.velocities_lr_final = 1.6e-6f;
        config.gates_lr = 1e-3f;
        config.scale_reg = 1e-2f;
        config.opacity_reg = 1e-2f;
        config.gate_reg = 0.0f;
        config.max_duration = -1.0f;
        config.marginal_threshold = 0.05f;
        config.hard_separation = false;
        config.color_correction = false;
        config.color_lr = 1e-3f;
        config.color_reg = 1e-3f;
        config.color_num_cameras = 0u;
        config.color_correction_start = 0;
        config.seed = 0u;
        config.sparse_distributed = false;
        config.optimizer_batch_size = 1u;
        config.min_opacity = 0.005f;
    }

    ~VkSplat() {
        try { cleanup(); } catch (...) { /* destructors must not throw */ }
    }

    void initialize(std::string spirv_dir, int device_id) {
        if (initialized && !closed)
            throw std::runtime_error("FTGSPPVulkan is already initialized");
        buffers.num_splats = 0;

        std::vector<std::string> spirv_paths = {
            "projection_forward",
            "projection_forward_sh0",
            "projection_forward_sh1",
            "projection_forward_sh2",
            "ftgspp_dynamic_projection",
            "ftgspp_dynamic_projection_sh0",
            "ftgspp_dynamic_projection_sh1",
            "ftgspp_dynamic_projection_sh2",
            "generate_keys",
            "ftgspp_tile_work_prepare",
            "compute_tile_ranges",
            "rasterize_forward",
            "rasterize_backward_0",
            "rasterize_backward_1",
            "rasterize_backward_2",
            "rasterize_backward_3",
            "rasterize_backward_4",
            "cumsum_single_pass",
            "cumsum_block_scan",
            "cumsum_scan_block_sums",
            "cumsum_add_block_offsets",
            "cumsum_f32_single_pass",
            "cumsum_f32_block_scan",
            "cumsum_f32_scan_block_sums",
            "cumsum_f32_add_block_offsets",
            "radix_sort/upsweep",
            "radix_sort/spine",
            "radix_sort/downsweep",
            "ssim_forward",
            "ssim_backward",
            "ftgspp_loss_reduce",
            "ftgspp_materialize",
            "ftgspp_fused_backward_optimizer",
            "ftgspp_fused_backward_optimizer_sh0",
            "ftgspp_fused_backward_optimizer_sh1",
            "ftgspp_fused_backward_optimizer_sh2",
            "ftgspp_fused_dynamic_backward_adam",
            "ftgspp_fused_dynamic_backward_adam_sh0",
            "ftgspp_fused_dynamic_backward_adam_sh1",
            "ftgspp_fused_dynamic_backward_adam_sh2",
            "ftgspp_explicit_adam",
            "ftgspp_peer_mean",
            "ftgspp_sparse_flag",
            "ftgspp_sparse_pack",
            "ftgspp_sparse_prepare",
            "ftgspp_sparse_scatter",
            "ftgspp_regularizer_mean",
            "ftgspp_relocation_prepare",
            "ftgspp_relocation_copy",
            "ftgspp_relocation_score",
            "ftgspp_relocation_cdf",
            "ftgspp_relocation_cdf_normalize",
            "ftgspp_relocation_sample",
            "ftgspp_flow_features", "ftgspp_hard_separation",
            "ftgspp_color_correct_forward",
            "ftgspp_color_correct_backward",
            "ftgspp_color_adam",
        };
        std::map<std::string, std::string> spirv_paths_dict;
        for (std::string name : spirv_paths) {
            spirv_paths_dict[name] = 
                (name.find('/') == std::string::npos) ?
                spirv_dir + "generated/" + name + ".spv" :
                spirv_dir + name + ".spv";
        }

        trainer.initialize(spirv_paths_dict, device_id);
        initialized = true;
        closed = false;

        if (config.color_correction && config.color_num_cameras > 0) {
            py::gil_scoped_release release;
            trainer.ensureFTGSPPColorState(config, buffers);
        }

        PerfTimer::reset();
    }

    void configure_ftgspp(py::dict cfg) {
        auto set_float = [&](const char* key, float& value) {
            if (cfg.contains(key)) value = cfg[key].cast<float>();
        };
        auto set_int = [&](const char* key, int& value) {
            if (cfg.contains(key)) value = cfg[key].cast<int>();
        };
        set_int("max_steps", config.max_steps);
        set_int("max_sh_degree", config.max_sh_degree);
        if (cfg.contains("optimizer_batch_size")) {
            const uint32_t value = cfg["optimizer_batch_size"].cast<uint32_t>();
            if (value == 0u || value > 32u)
                throw std::invalid_argument(
                    "optimizer_batch_size must be in [1,32]");
            config.optimizer_batch_size = value;
        }
        if (cfg.contains("seed")) config.seed = cfg["seed"].cast<uint32_t>();
        set_float("ssim_lambda", config.ssim_lambda);
        set_float("scene_scale", config.global_scale);
        set_float("means_lr", config.means_lr);
        set_float("means_lr_final", config.means_lr_final);
        set_float("features_dc_lr", config.features_dc_lr);
        set_float("features_rest_lr", config.features_rest_lr);
        set_float("opacities_lr", config.opacities_lr);
        set_float("scales_lr", config.scales_lr);
        set_float("quats_lr", config.quats_lr);
        set_float("times_lr", config.times_lr);
        set_float("durations_lr", config.durations_lr);
        set_float("velocities_lr", config.velocities_lr);
        set_float("velocities_lr_final", config.velocities_lr_final);
        set_float("gates_lr", config.gates_lr);
        set_float("scale_reg", config.scale_reg);
        set_float("opacity_reg", config.opacity_reg);
        set_float("gate_reg", config.gate_reg);
        set_float("max_duration", config.max_duration);
        set_float("marginal_threshold", config.marginal_threshold);
        set_float("min_opacity", config.min_opacity);
        if (cfg.contains("color_correction"))
            config.color_correction = cfg["color_correction"].cast<bool>();
        set_float("color_lr", config.color_lr);
        set_float("color_reg", config.color_reg);
        if (cfg.contains("color_num_cameras"))
            config.color_num_cameras = cfg["color_num_cameras"].cast<uint32_t>();
        if (cfg.contains("color_correction_start"))
            config.color_correction_start = cfg["color_correction_start"].cast<int>();
        if (initialized && config.color_correction &&
            config.color_num_cameras > 0) {
            py::gil_scoped_release release;
            trainer.ensureFTGSPPColorState(config, buffers);
        }
    }

    void set_dynamic_gaussians(
        py::array_t<float, py::array::c_style | py::array::forcecast> means,
        py::array_t<float, py::array::c_style | py::array::forcecast> log_scales,
        py::array_t<float, py::array::c_style | py::array::forcecast> rotations,
        py::array_t<float, py::array::c_style | py::array::forcecast> opacity_logits,
        py::array_t<float, py::array::c_style | py::array::forcecast> sh_coeffs,
        py::array_t<float, py::array::c_style | py::array::forcecast> times,
        py::array_t<float, py::array::c_style | py::array::forcecast> durations,
        py::array_t<float, py::array::c_style | py::array::forcecast> velocities,
        py::array_t<float, py::array::c_style | py::array::forcecast> gates
    ) {
        auto mean = means.request();
        auto scale = log_scales.request();
        auto rot = rotations.request();
        auto opacity = opacity_logits.request();
        auto sh = sh_coeffs.request();
        auto time = times.request();
        auto duration = durations.request();
        auto velocity = velocities.request();
        auto gate = gates.request();
        if (mean.ndim != 2 || mean.shape[1] != 3)
            throw std::runtime_error("means must have shape (N, 3)");
        const size_t n = (size_t)mean.shape[0];
        auto require_2d = [n](const py::buffer_info& a, size_t width, const char* name) {
            if (a.ndim != 2 || (size_t)a.shape[0] != n || (size_t)a.shape[1] != width)
                throw std::runtime_error(std::string(name) + " has an invalid shape");
        };
        require_2d(scale, 3, "log_scales");
        require_2d(rot, 4, "rotations");
        require_2d(opacity, 1, "opacity_logits");
        require_2d(time, 1, "times");
        require_2d(duration, 1, "durations");
        require_2d(velocity, 3, "velocities");
        require_2d(gate, 1, "gates");
        if (sh.ndim != 3 || (size_t)sh.shape[0] != n || sh.shape[1] != 16 || sh.shape[2] != 3)
            throw std::runtime_error("sh_coeffs must have shape (N, 16, 3)");

        gradient_ready = false;
        gradient_finalized = false;
        gradient_step = -1;
        local_gradient_microsteps = 0;
        buffers.num_splats = n;
        const float* mean_ptr = static_cast<float*>(mean.ptr);
        const float* scale_ptr = static_cast<float*>(scale.ptr);
        const float* rot_ptr = static_cast<float*>(rot.ptr);
        const float* opacity_ptr = static_cast<float*>(opacity.ptr);
        const float* sh_ptr = static_cast<float*>(sh.ptr);
        const float* time_ptr = static_cast<float*>(time.ptr);
        const float* duration_ptr = static_cast<float*>(duration.ptr);
        const float* velocity_ptr = static_cast<float*>(velocity.ptr);
        const float* gate_ptr = static_cast<float*>(gate.ptr);

        buffers.base_xyz_ws.assign(mean_ptr, mean_ptr + 3*n);
        buffers.rotations.assign(rot_ptr, rot_ptr + 4*n);
        buffers.sh_coeffs.assign(sh_ptr, sh_ptr + 16*3*n);
        buffers.base_log_scales_opacity.resize(4*n);
        buffers.times_durations.resize(2*n);
        buffers.velocities_gates.resize(4*n);
        for (size_t i = 0; i < n; ++i) {
            for (int j = 0; j < 3; ++j) {
                buffers.base_log_scales_opacity[4*i+j] = scale_ptr[3*i+j];
                buffers.velocities_gates[4*i+j] = velocity_ptr[3*i+j];
            }
            buffers.base_log_scales_opacity[4*i+3] = opacity_ptr[i];
            buffers.times_durations[2*i] = time_ptr[i];
            buffers.times_durations[2*i+1] = duration_ptr[i];
            buffers.velocities_gates[4*i+3] = gate_ptr[i];
        }
        buffers.reorderSH(buffers.sh_coeffs);
        trainer.copyToDevice(buffers.base_xyz_ws);
        trainer.copyToDevice(buffers.base_log_scales_opacity);
        trainer.copyToDevice(buffers.times_durations);
        trainer.copyToDevice(buffers.velocities_gates);
        trainer.copyToDevice(buffers.rotations);
        trainer.copyToDevice(buffers.sh_coeffs);
    }

    void set_reference_image(
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> image
    ) {
        auto src = image.request();
        if (src.ndim != 3 || src.shape[0] != uniforms.image_height ||
            src.shape[1] != uniforms.image_width || (src.shape[2] != 3 && src.shape[2] != 4))
            throw std::runtime_error("reference image must have shape (H, W, 3|4) matching the camera");
        const size_t pixels = (size_t)src.shape[0] * (size_t)src.shape[1];
        const size_t channels = (size_t)src.shape[2];
        const uint8_t* ptr = static_cast<uint8_t*>(src.ptr);
        buffers.ref_image.resize(4*pixels);
        if (channels == 4) {
            memcpy(buffers.ref_image.data(), ptr, 4 * pixels);
        } else {
            for (size_t i = 0; i < pixels; ++i) {
                buffers.ref_image[4*i] = ptr[3*i];
                buffers.ref_image[4*i+1] = ptr[3*i+1];
                buffers.ref_image[4*i+2] = ptr[3*i+2];
                buffers.ref_image[4*i+3] = 255;
            }
        }
        trainer.copyToDevice(buffers.ref_image);
    }

    void release_model_upload_host_copies() {
        release_model_host_copies();
    }

    void clear_dynamic_views() {
        trainer.clear_external_dataset();
    }

    void add_dynamic_view(
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> image,
        py::array_t<float, py::array::c_style | py::array::forcecast> world_view_transform,
        py::array_t<float, py::array::c_style | py::array::forcecast> intrinsic,
        float timestamp,
        int32_t camera_id,
        int32_t frame_id,
        bool is_train,
        bool cache_on_device
    ) {
        auto im = image.request();
        auto transform = world_view_transform.request();
        auto k = intrinsic.request();
        if (im.ndim != 3 || (im.shape[2] != 3 && im.shape[2] != 4))
            throw std::runtime_error("image must have shape (H, W, 3|4)");
        if (transform.ndim != 2 || transform.shape[0] != 4 || transform.shape[1] != 4)
            throw std::runtime_error("world_view_transform must have shape (4, 4)");
        if (k.ndim != 2 || k.shape[0] != 3 || k.shape[1] != 3)
            throw std::runtime_error("intrinsic must have shape (3, 3)");

        VulkanGSTrainer::DatasetImage view;
        view.camera = {};
        view.camera.h = (int)im.shape[0];
        view.camera.w = (int)im.shape[1];
        const float* k_ptr = static_cast<float*>(k.ptr);
        view.camera.fx = k_ptr[0];
        view.camera.fy = k_ptr[4];
        view.camera.cx = k_ptr[2];
        view.camera.cy = k_ptr[5];
        const float* transform_ptr = static_cast<float*>(transform.ptr);
        for (int col = 0; col < 4; ++col)
            for (int row = 0; row < 4; ++row)
                view.camera.world_view_transform[col][row] = transform_ptr[4*row+col];
        view.timestamp = timestamp;
        view.camera_id = camera_id;
        view.frame_id = frame_id;

        const size_t pixels = (size_t)im.shape[0] * (size_t)im.shape[1];
        const size_t channels = (size_t)im.shape[2];
        const uint8_t* src = static_cast<uint8_t*>(im.ptr);
        view.buffer.resize(4*pixels);
        if (channels == 4) {
            memcpy(view.buffer.data(), src, 4 * pixels);
        } else {
            for (size_t i = 0; i < pixels; ++i) {
                view.buffer[4*i] = src[3*i];
                view.buffer[4*i+1] = src[3*i+1];
                view.buffer[4*i+2] = src[3*i+2];
                view.buffer[4*i+3] = 255;
            }
        }
        trainer.add_external_view(std::move(view), is_train, cache_on_device);
    }

    void add_dynamic_view_meta(
        int height,
        int width,
        py::array_t<float, py::array::c_style | py::array::forcecast> world_view_transform,
        py::array_t<float, py::array::c_style | py::array::forcecast> intrinsic,
        float timestamp,
        int32_t camera_id,
        int32_t frame_id,
        bool is_train
    ) {
        auto transform = world_view_transform.request();
        auto k = intrinsic.request();
        if (height <= 0 || width <= 0)
            throw std::runtime_error("view height/width must be positive");
        if (transform.ndim != 2 || transform.shape[0] != 4 || transform.shape[1] != 4)
            throw std::runtime_error("world_view_transform must have shape (4, 4)");
        if (k.ndim != 2 || k.shape[0] != 3 || k.shape[1] != 3)
            throw std::runtime_error("intrinsic must have shape (3, 3)");

        VulkanGSTrainer::DatasetImage view;
        view.camera = {};
        view.camera.h = height;
        view.camera.w = width;
        const float* k_ptr = static_cast<float*>(k.ptr);
        view.camera.fx = k_ptr[0];
        view.camera.fy = k_ptr[4];
        view.camera.cx = k_ptr[2];
        view.camera.cy = k_ptr[5];
        const float* transform_ptr = static_cast<float*>(transform.ptr);
        for (int col = 0; col < 4; ++col)
            for (int row = 0; row < 4; ++row)
                view.camera.world_view_transform[col][row] = transform_ptr[4*row+col];
        view.timestamp = timestamp;
        view.camera_id = camera_id;
        view.frame_id = frame_id;
        trainer.add_external_view(std::move(view), is_train, /*cache_on_device=*/false);
    }

    void bind_dynamic_view_image(
        size_t view_idx,
        bool is_train,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> image
    ) {
        auto& target = is_train ? trainer.get_train_image(view_idx) : trainer.get_val_image(view_idx);
        auto im = image.request();
        if (im.ndim != 3 || (im.shape[2] != 3 && im.shape[2] != 4))
            throw std::runtime_error("image must have shape (H, W, 3|4)");
        if ((int)im.shape[0] != target.camera.h || (int)im.shape[1] != target.camera.w)
            throw std::runtime_error("bound image shape must match registered view camera size");
        const size_t pixels = (size_t)im.shape[0] * (size_t)im.shape[1];
        const size_t channels = (size_t)im.shape[2];
        const uint8_t* src = static_cast<uint8_t*>(im.ptr);
        {
            py::gil_scoped_release release;
            target.buffer.resize(4 * pixels);
            if (channels == 4) {
                memcpy(target.buffer.data(), src, 4 * pixels);
            } else {
                for (size_t i = 0; i < pixels; ++i) {
                    target.buffer[4*i] = src[3*i];
                    target.buffer[4*i+1] = src[3*i+1];
                    target.buffer[4*i+2] = src[3*i+2];
                    target.buffer[4*i+3] = 255;
                }
            }
            if (!target.cache_on_device) {
                auto ticket = trainer.enqueueReferenceRgba(
                    target.buffer.data(), 4 * pixels,
                    (uint32_t)im.shape[1], (uint32_t)im.shape[0]);
                trainer.bindPendingReferenceTicket(ticket);
            } else {
                trainer.destroyBuffer(target.buffer.deviceBuffer);
                target.buffer.deviceBuffer = _VulkanBuffer();
            }
        }
    }

    void release_gradient_host_copies() {
        // Gradient NumPy arrays and device buffers own their data after an
        // export/upload.  Keeping another std::vector copy per rank would add
        // hundreds of MiB at large N without serving any later dispatch.
        Buffer<float>* values[] = {
            &buffers.grad_base_xyz_ws,
            &buffers.grad_base_log_scales_opacity,
            &buffers.grad_times_durations,
            &buffers.grad_velocities_gates,
            &buffers.grad_sh_coeffs,
            &buffers.grad_rotations,
            &buffers.color_grad,
        };
        for (Buffer<float>* value : values) {
            value->clear();
            value->shrink_to_fit();
        }
    }

    void release_model_host_copies() {
        Buffer<float>* values[] = {
            &buffers.base_xyz_ws,
            &buffers.base_log_scales_opacity,
            &buffers.times_durations,
            &buffers.velocities_gates,
            &buffers.sh_coeffs,
            &buffers.rotations,
        };
        for (Buffer<float>* value : values) {
            value->clear();
            value->shrink_to_fit();
        }
    }

    void release_optimizer_host_copies() {
        Buffer<float>* values[] = {
            &buffers.g_xyz_ws,
            &buffers.g_sh_coeffs_1,
            &buffers.g_sh_coeffs_2,
            &buffers.g_rotations,
            &buffers.g_scales_opacs,
            &buffers.g_times_durations,
            &buffers.g_velocities_gates,
            &buffers.relocation_stats,
            &buffers.color_params,
            &buffers.color_m,
            &buffers.color_v,
        };
        for (Buffer<float>* value : values) {
            value->clear();
            value->shrink_to_fit();
        }
    }

    struct PackedGradientEntry {
        const char* name;
        Buffer<float>* buffer;
        size_t offset;
        size_t count;
    };

    static size_t active_sh_float_count(uint32_t active_sh) {
        const size_t degree = std::min<uint32_t>(active_sh, 3u);
        return 3u * (degree + 1u) * (degree + 1u);
    }

    size_t sparse_record_words() const {
        return 18u + active_sh_float_count(uniforms.active_sh)
            - (config.hard_separation ? 4u : 0u);
    }

    std::vector<PackedGradientEntry> packed_gradient_entries(
        bool require_ready=true
    ) {
        if (require_ready && !gradient_ready)
            throw std::runtime_error(
                "packed gradient state requires a preceding backward_dynamic_view");
        const size_t n = buffers.num_splats;
        std::vector<PackedGradientEntry> entries;
        size_t offset = 0;
        auto add = [&](const char* name, Buffer<float>& value, size_t expected) {
            if (value.deviceSize() != expected) {
                if (require_ready)
                    throw std::runtime_error(
                        std::string("packed raw gradient buffer ") + name +
                        " has " + std::to_string(value.deviceSize()) +
                        " floats, expected " + std::to_string(expected));
                trainer.resizeDeviceBuffer(value, expected);
            }
            entries.push_back({name, &value, offset, expected});
            offset += expected;
        };
        // ABI v2 deliberately keeps grad_sh_coeffs in the exact reordered and
        // padded float4 layout consumed by the Vulkan optimizer. Reduction is
        // elementwise and does not require the diagnostic (N,16,3) layout.
        add("grad_base_xyz_ws", buffers.grad_base_xyz_ws, 3*n);
        add(
            "grad_base_log_scales_opacity",
            buffers.grad_base_log_scales_opacity, 4*n);
        add("grad_times_durations", buffers.grad_times_durations, 2*n);
        // Hard separation freezes velocity/gate parameters and both moments;
        // their rank-local raw buffer is cleared by the optimizer but is never
        // consumed. Omitting it from the exchange is therefore exact.
        if (!config.hard_separation)
            add("grad_velocities_gates", buffers.grad_velocities_gates, 4*n);
        add(
            "grad_sh_coeffs_reordered", buffers.grad_sh_coeffs,
            16*3*_CEIL_ROUND(n, SH_REORDER_SIZE));
        add("grad_rotations", buffers.grad_rotations, 4*n);
        // The affine transform remains in the backward chain, but color Adam
        // is frozen in hard separation and does not consume color_grad.
        if (!config.hard_separation && color_correction_active(gradient_step))
            add(
                "color_grad", buffers.color_grad,
                size_t(config.color_num_cameras) * 12u);
        return entries;
    }

    Buffer<float>* peer_buffer_for(const char* name) {
            if (std::strcmp(name, "grad_base_xyz_ws") == 0)
                return &buffers.peer_grad_base_xyz_ws;
            if (std::strcmp(name, "grad_base_log_scales_opacity") == 0)
                return &buffers.peer_grad_base_log_scales_opacity;
            if (std::strcmp(name, "grad_times_durations") == 0)
                return &buffers.peer_grad_times_durations;
            if (std::strcmp(name, "grad_velocities_gates") == 0)
                return &buffers.peer_grad_velocities_gates;
            if (std::strcmp(name, "grad_sh_coeffs_reordered") == 0)
                return &buffers.peer_grad_sh_coeffs;
            if (std::strcmp(name, "grad_rotations") == 0)
                return &buffers.peer_grad_rotations;
            if (std::strcmp(name, "color_grad") == 0)
                return &buffers.peer_color_grad;
            throw std::runtime_error(
                std::string("host mailbox has no peer buffer for packed entry ") + name);
    }

    std::vector<PackedGradientEntry> peer_entries_for(
        const std::vector<PackedGradientEntry>& local) {
        std::vector<PackedGradientEntry> entries;
        entries.reserve(local.size());
        for (const auto& source : local) {
            Buffer<float>* peer = peer_buffer_for(source.name);
            trainer.resizeDeviceBuffer(*peer, source.count);
            entries.push_back({source.name, peer, source.offset, source.count});
        }
        return entries;
    }

    std::vector<PackedGradientEntry> peer_gradient_entries() {
        return peer_entries_for(packed_gradient_entries());
    }

    void require_finalized_local_gradient(const char* operation) const {
        if (!gradient_ready || !gradient_finalized)
            throw std::runtime_error(
                std::string(operation) +
                " requires finalize_local_gradient after raw backward");
    }

    void accumulate_current_raw_gradient() {
        const auto current = packed_gradient_entries();
        const auto accumulator = peer_entries_for(current);
        if (current.size() != accumulator.size())
            throw std::runtime_error(
                "local gradient accumulator ABI entry count differs");
        auto device_guard = DeviceGuard(&trainer);
        if (local_gradient_microsteps == 0) {
            for (size_t i = 0; i < current.size(); ++i)
                trainer.copyFromDeviceToDevice(
                    *current[i].buffer, *accumulator[i].buffer);
        } else {
            for (size_t i = 0; i < current.size(); ++i) {
                if (current[i].count != accumulator[i].count)
                    throw std::runtime_error(
                        "local gradient accumulator ABI entry differs");
                trainer.executeFTGSPPPeerMean(
                    accumulator[i].buffer->deviceBuffer,
                    current[i].buffer->deviceBuffer,
                    current[i].count,
                    1.0f, 1.0f);
            }
        }
        ++local_gradient_microsteps;
    }

    void finalize_local_gradient(uint32_t microsteps) {
        if (microsteps != 1 && microsteps != 2 && microsteps != 4)
            throw std::invalid_argument(
                "finalize_local_gradient requires K in {1,2,4}");
        if (!gradient_ready || gradient_finalized)
            throw std::runtime_error(
                "finalize_local_gradient requires an unfinished raw gradient");
        if (local_gradient_microsteps != microsteps)
            throw std::runtime_error(
                "finalize_local_gradient microstep count mismatch: accumulated " +
                std::to_string(local_gradient_microsteps) + ", expected " +
                std::to_string(microsteps));
        // K=1 is deliberately a no-op so the original raw buffers remain
        // bit-identical. Larger K values copy sum/K from the device-local
        // accumulator into the public collective buffers.
        if (microsteps > 1) {
            const auto output = packed_gradient_entries();
            const auto accumulator = peer_entries_for(output);
            const float inverse = 1.0f / static_cast<float>(microsteps);
            auto device_guard = DeviceGuard(&trainer);
            for (size_t i = 0; i < output.size(); ++i)
                trainer.executeFTGSPPPeerMean(
                    output[i].buffer->deviceBuffer,
                    accumulator[i].buffer->deviceBuffer,
                    output[i].count,
                    0.0f, inverse, 0u, 0u, true);
        }
        gradient_finalized = true;
    }

    std::vector<PackedGradientEntry> mailbox_dense_entries(bool auxiliary_only=false) {
        if (!gradient_ready)
            throw std::runtime_error(
                "host mailbox requires a preceding backward_dynamic_view");
        const size_t n = buffers.num_splats;
        std::vector<PackedGradientEntry> entries;
        size_t offset = 0;
        auto add = [&](const char* name, Buffer<float>& value, size_t expected) {
            trainer.resizeDeviceBuffer(value, expected);
            entries.push_back({name, &value, offset, expected});
            offset += expected;
        };
        if (!auxiliary_only) {
            add("grad_base_xyz_ws", buffers.grad_base_xyz_ws, 3*n);
            add(
                "grad_base_log_scales_opacity",
                buffers.grad_base_log_scales_opacity, 4*n);
            add("grad_times_durations", buffers.grad_times_durations, 2*n);
            // Dense fallback keeps the explicit ABI stable inside one phase;
            // the coordinator recreates its mailbox at hard separation.
            if (!config.hard_separation)
                add("grad_velocities_gates", buffers.grad_velocities_gates, 4*n);
            add(
                "grad_sh_coeffs_reordered", buffers.grad_sh_coeffs,
                16*3*_CEIL_ROUND(n, SH_REORDER_SIZE));
            add("grad_rotations", buffers.grad_rotations, 4*n);
        }
        if (config.color_correction && config.color_num_cameras > 0 &&
            !config.hard_separation) {
            const size_t color_count = size_t(config.color_num_cameras) * 12u;
            if (!color_correction_active(gradient_step))
                trainer.clearDeviceBuffer(buffers.color_grad, color_count);
            add(
                "color_grad", buffers.color_grad,
                color_count);
        }
        return entries;
    }

    static std::vector<VulkanGSPipeline::PackedBufferRegion> packed_regions(
        const std::vector<PackedGradientEntry>& entries) {
        std::vector<VulkanGSPipeline::PackedBufferRegion> regions;
        regions.reserve(entries.size());
        for (const auto& entry : entries)
            regions.push_back({
                entry.buffer->deviceBuffer, entry.offset * sizeof(float)});
        return regions;
    }

    static size_t packed_bytes(
        const std::vector<PackedGradientEntry>& entries) {
        if (entries.empty())
            throw std::runtime_error("host mailbox packed ABI is empty");
        const auto& last = entries.back();
        return (last.offset + last.count) * sizeof(float);
    }

    static std::string packed_signature(
        const std::vector<PackedGradientEntry>& entries) {
        std::ostringstream out;
        out << "abi=2;";
        for (const auto& entry : entries)
            out << entry.name << ':' << entry.offset << ':' << entry.count << ';';
        return out.str();
    }

    struct MailboxTransferTiming {
        size_t payload_bytes = 0;
        size_t bucket_bytes = 0;
        size_t bucket_count = 0;
        double download_seconds = 0.0;
        double upload_seconds = 0.0;
        double host_memcpy_seconds = 0.0;
        // Time blocked in the first publish_borrowed_and_wait (peer not ready yet).
        // Excluded from a useful transfer-overlap ratio; still included in wall_seconds.
        double peer_wait_seconds = 0.0;
        // wall_seconds - peer_wait_seconds (copy pipeline + local GPU work in-loop).
        double transfer_wall_seconds = 0.0;
        double wall_seconds = 0.0;
        bool async_pipeline = false;
    };

    py::dict aborted_exchange_report(
        const SyncGenerationState::Decision& decision) {
        trainer.telemetry().sync_abort += 1;
        py::dict result;
        result["aborted"] = true;
        result["fatal"] = decision.fatal;
        result["overflow"] = true;
        result["reason"] = decision.reason;
        result["max_exact_count"] = decision.max_exact_count;
        result["local_exact_count"] = decision.local_exact_count;
        result["local_overflow"] = decision.local_overflow;
        result["abort_code"] = decision.abort_code;
        result["transport"] = "host_mailbox";
        result["download_seconds"] = 0.0;
        result["upload_seconds"] = 0.0;
        result["host_memcpy_seconds"] = 0.0;
        result["peer_wait_seconds"] = 0.0;
        result["transfer_wall_seconds"] = 0.0;
        result["wall_seconds"] = 0.0;
        result["cpu_reduction_seconds"] = 0.0;
        result["payload_bytes"] = 0;
        result["pcie_d2h_bytes"] = 0;
        result["pcie_h2d_bytes"] = 0;
        result["async_pipeline"] = false;
        return result;
    }

    SyncGenerationState::Decision harvest_sync_generation(
        uint32_t rank, uint64_t generation, bool wait_all_ranks) {
        const bool force_overflow = force_sync_overflow_;
        const uint32_t force_exact = force_sync_overflow_exact_;
        force_sync_overflow_ = false;
        SyncGenerationState::Decision decision;
        {
            // Rank threads share one GIL. report_and_wait must not hold it,
            // otherwise P>=3 hierarchical exchange deadlocks.
            py::gil_scoped_release release;
            trainer.waitAllComputeSlots("collective");
            auto retired = trainer.collectRetiredTileOverflow();
            if (force_overflow) {
                retired.overflow = true;
                retired.is_train = true;
                retired.exact_count = std::max(
                    retired.exact_count, force_exact);
            }
            if (retired.fatal) {
                throw std::runtime_error(
                    "tile invariant/prefix abort; refusing automatic replay");
            }
            SyncGenerationState::RankReport report;
            report.overflow = retired.overflow;
            report.exact_count = retired.exact_count;
            report.abort_code = retired.abort_code;
            std::shared_ptr<SyncGenerationState> sync;
            {
                std::lock_guard<std::mutex> lock(g_sync_generation_mutex);
                sync = g_sync_generation;
            }
            if (!sync)
                throw std::runtime_error(
                    "sync generation state is not configured");
            decision = wait_all_ranks
                ? sync->report_and_wait(rank, generation, report)
                : sync->report_local(rank, generation, report);
        }
        if (decision.fatal)
            throw std::runtime_error(
                "tile invariant/prefix abort; refusing automatic replay");
        if (decision.aborted)
            abort_sync_collectives(kTileOverflowAbortReason);
        return decision;
    }

    MailboxTransferTiming exchange_fixed_mailbox(
        HostMailboxOwner& owner,
        int& attached_rank,
        uint32_t rank,
        uint64_t generation,
        const std::vector<PackedGradientEntry>& local,
        const std::vector<PackedGradientEntry>& peer,
        const std::string& signature,
        bool mean_reduction) {
        const size_t total_bytes = packed_bytes(local);
        if (packed_bytes(peer) != total_bytes)
            throw std::runtime_error("host mailbox local/peer packed sizes differ");
        if (attached_rank < 0) {
            owner.attach(rank, total_bytes, signature);
            attached_rank = static_cast<int>(rank);
        } else if (attached_rank != static_cast<int>(rank)) {
            throw std::runtime_error("native backend changed host mailbox rank");
        }
        const auto local_regions = packed_regions(local);
        const auto peer_regions = packed_regions(peer);
        MailboxTransferTiming timing;
        timing.payload_bytes = total_bytes;
        timing.bucket_bytes = owner.bucket_bytes();
        timing.bucket_count = owner.bucket_count();
        struct PendingDownload {
            VulkanGSPipeline::PackedTransferTicket ticket;
            size_t bucket = 0;
            size_t offset = 0;
            size_t bytes = 0;
        };
        std::deque<PendingDownload> downloads;
        auto add_download_timing = [&](const auto& value) {
            timing.download_seconds += value.gpu_copy_seconds
                + value.host_memcpy_seconds + value.invalidate_seconds;
            timing.host_memcpy_seconds += value.host_memcpy_seconds;
        };
        auto add_upload_timing = [&](const auto& value) {
            timing.upload_seconds += value.gpu_copy_seconds
                + value.host_memcpy_seconds + value.invalidate_seconds;
            timing.host_memcpy_seconds += value.host_memcpy_seconds;
        };
        auto reduce_bucket = [&](size_t offset, size_t bytes, uint32_t source_rank) {
            if (offset % sizeof(float) != 0 || bytes % sizeof(float) != 0)
                throw std::runtime_error(
                    "host mailbox reduction range is not float aligned");
            const size_t first = offset / sizeof(float);
            const size_t end = first + bytes / sizeof(float);
            const float peer_weight = mean_reduction
                ? 1.0f / static_cast<float>(owner.world_size()) : 1.0f;
            auto device_guard = DeviceGuard(&trainer);
            for (size_t i = 0; i < local.size(); ++i) {
                const size_t entry_first = local[i].offset;
                const size_t entry_end = entry_first + local[i].count;
                const size_t overlap_first = std::max(first, entry_first);
                const size_t overlap_end = std::min(end, entry_end);
                if (overlap_first >= overlap_end)
                    continue;
                trainer.executeFTGSPPPeerMean(
                    local[i].buffer->deviceBuffer,
                    peer[i].buffer->deviceBuffer,
                    overlap_end - overlap_first,
                    1.0f, peer_weight,
                    overlap_first - entry_first,
                    overlap_first - peer[i].offset,
                    source_rank == 0);
            }
        };
        auto submit_download = [&](size_t bucket) {
            const size_t offset = bucket * timing.bucket_bytes;
            const size_t bytes = std::min(
                timing.bucket_bytes, total_bytes - offset);
            // Protect the borrowed Vulkan staging slot before queueing a write
            // into it. The peer marks the prior bucket consumed only after it
            // has copied that pointer into its private upload stager.
            owner.wait_writable(
                rank, generation, static_cast<uint32_t>(bucket));
            downloads.push_back({
                trainer.submitBuffersFromDevicePackedRange(
                    local_regions, offset, bytes),
                bucket, offset, bytes,
            });
        };
        const auto begin = std::chrono::steady_clock::now();
        const size_t initial = std::min<size_t>(
            timing.bucket_count, VulkanGSPipeline::packed_transfer_ring_depth);
        for (size_t bucket = 0; bucket < initial; ++bucket)
            submit_download(bucket);
        size_t next_download = initial;
        bool measured_first_peer_wait = false;
        for (size_t bucket = 0; bucket < timing.bucket_count; ++bucket) {
            if (downloads.empty() || downloads.front().bucket != bucket)
                throw std::runtime_error(
                    "host mailbox readback pipeline lost bucket ordering");
            const PendingDownload current = downloads.front();
            downloads.pop_front();
            const auto mapped =
                trainer.completeBuffersFromDevicePackedRangeMapped(
                    current.ticket);
            add_download_timing(mapped.timing);
            const auto peer_wait_begin = std::chrono::steady_clock::now();
            owner.publish_borrowed_and_wait(
                rank, generation, static_cast<uint32_t>(bucket),
                mapped.data, current.bytes);
            if (!measured_first_peer_wait) {
                // Fast rank blocks here until the slow rank finishes backward
                // and reaches the same bucket; later buckets are usually lockstep.
                timing.peer_wait_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - peer_wait_begin).count();
                measured_first_peer_wait = true;
            }

            // Every rank reduces sources in the same rank order. This preserves
            // replica determinism while using only one reusable peer buffer.
            // The local source is re-uploaded from its borrowed snapshot, so
            // source rank zero can reset the accumulator before ordered adds.
            for (uint32_t source_rank = 0;
                 source_rank < owner.world_size(); ++source_rank) {
                const void* source_slot = source_rank == rank
                    ? mapped.data
                    : owner.peer_borrowed_pointer(
                        rank, source_rank, generation,
                        static_cast<uint32_t>(bucket), current.bytes);
                const auto upload = trainer.submitBuffersToDevicePackedRange(
                    source_slot, current.offset, current.bytes, peer_regions);
                add_upload_timing(
                    trainer.completeBuffersToDevicePackedRange(upload));
                reduce_bucket(
                    current.offset, current.bytes, source_rank);
            }
            // Every remote producer can recycle this ring slot only after this
            // rank has copied and reduced all published sources.
            owner.mark_consumed(
                rank, generation, static_cast<uint32_t>(bucket));

            if (next_download < timing.bucket_count)
                submit_download(next_download++);
        }
        timing.async_pipeline = true;
        timing.wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();
        timing.transfer_wall_seconds = std::max(
            0.0, timing.wall_seconds - timing.peer_wait_seconds);
        return timing;
    }

    static py::dict create_pair_exchange_gpu_merge_owner(
        uint32_t world_size, size_t bucket_bytes, bool sparse) {
        if (g_host_mailbox_owner || g_sparse_mailbox_owner ||
            g_relocation_mailbox_owner)
            throw std::runtime_error("host mailbox owner already exists");
        auto dense_owner = std::make_shared<HostMailboxOwner>(
            world_size, bucket_bytes);
        std::shared_ptr<HostMailboxOwner> sparse_owner;
        // The N-rank dense path is mathematically identical to sparse exchange
        // and avoids retaining P-1 variable-length record streams. Keep the
        // optimized sparse pair path for the exact two-rank case.
        // Sparse payloads are ~90 MiB at SH degree 3; 8/16 MiB buckets add
        // mailbox round-trips without helping PCIe. Keep the calibrated size
        // for the small dense auxiliary mailbox.
        constexpr size_t kMinSparsePairBucketBytes = 32ull * 1024ull * 1024ull;
        const size_t sparse_bucket_bytes =
            std::max(bucket_bytes, kMinSparsePairBucketBytes);
        if (sparse && world_size == 2)
            sparse_owner = std::make_shared<HostMailboxOwner>(
                world_size, sparse_bucket_bytes);
        auto relocation_owner = std::make_shared<HostMailboxOwner>(
            world_size, bucket_bytes);
        g_host_mailbox_owner = std::move(dense_owner);
        g_sparse_mailbox_owner = std::move(sparse_owner);
        g_relocation_mailbox_owner = std::move(relocation_owner);
        ensure_sync_generation(world_size);
        py::dict result;
        for (const auto& [key, value] : g_host_mailbox_owner->report()) {
            std::visit([&](const auto& typed) {
                result[py::str(key)] = typed;
            }, value);
        }
        result["sparse"] = bool(g_sparse_mailbox_owner);
        result["sparse_requested"] = sparse;
        result["sparse_bucket_bytes"] = g_sparse_mailbox_owner
            ? g_sparse_mailbox_owner->bucket_bytes() : size_t(0);
        result["gpu_peer_mean"] = true;
        result["relocation_gpu_sum"] = true;
        result["async_pipeline"] = true;
        result["borrowed_readback"] = true;
        result["remaining_host_copy"] =
            "borrowed-readback-to-private-upload-stager";
        return result;
    }

    static void release_pair_exchange_gpu_merge_owner() {
        g_relocation_mailbox_owner.reset();
        g_sparse_mailbox_owner.reset();
        g_host_mailbox_owner.reset();
        release_sync_generation();
    }

    void reset_pair_exchange_gpu_merge_attachments() {
        if (host_mailbox_rank >= 0 && g_host_mailbox_owner)
            g_host_mailbox_owner->detach(
                static_cast<uint32_t>(host_mailbox_rank));
        if (sparse_mailbox_rank >= 0 && g_sparse_mailbox_owner)
            g_sparse_mailbox_owner->detach(
                static_cast<uint32_t>(sparse_mailbox_rank));
        if (relocation_mailbox_rank >= 0 && g_relocation_mailbox_owner)
            g_relocation_mailbox_owner->detach(
                static_cast<uint32_t>(relocation_mailbox_rank));
        host_mailbox_rank = -1;
        sparse_mailbox_rank = -1;
        relocation_mailbox_rank = -1;
        pending_peer_sparse_count = 0;
        pending_local_sparse_count = 0;
        pair_sparse_mean_ready_ = false;
        pair_needs_regularizer_ = false;
    }

    py::dict exchange_pair_gradient(
        uint32_t rank, uint64_t generation) {
        if (!g_host_mailbox_owner)
            throw std::runtime_error("host mailbox owner is not configured");
        require_finalized_local_gradient("host mailbox exchange");
        const auto harvest = harvest_sync_generation(
            rank, generation, /*wait_all_ranks=*/false);
        if (harvest.aborted)
            return aborted_exchange_report(harvest);
        const auto begin = std::chrono::steady_clock::now();
        MailboxTransferTiming dense_timing;
        double sparse_download_seconds = 0.0;
        double sparse_upload_seconds = 0.0;
        double sparse_host_memcpy_seconds = 0.0;
        double sparse_pack_seconds = 0.0;
        double sparse_peer_wait_seconds = 0.0;
        double sparse_transfer_wall_seconds = 0.0;
        size_t local_sparse_bytes = 0;
        size_t peer_sparse_bytes = 0;
        uint32_t local_sparse_count = 0;
        bool used_sparse = false;
        pair_sparse_mean_ready_ = false;
        pair_needs_regularizer_ = config.sparse_distributed;
        pair_gaussian_codec_ = "dense";
        pending_peer_sparse_count = 0;
        pending_local_sparse_count = 0;
        try {
            py::gil_scoped_release release;
            if (!config.sparse_distributed) {
                const auto local = mailbox_dense_entries(false);
                const auto peer = peer_entries_for(local);
                dense_timing = exchange_fixed_mailbox(
                    *g_host_mailbox_owner, host_mailbox_rank,
                    rank, generation, local, peer,
                    "dense;" + packed_signature(local), true);
            } else {
                if (!g_sparse_mailbox_owner)
                    throw std::runtime_error("sparse host mailbox was not created");
                static constexpr size_t kMaxRecordWords = 66;
                const size_t record_words = sparse_record_words();
                const size_t record_bytes = record_words * sizeof(uint32_t);
                const size_t capacity_bytes =
                    kMaxRecordWords * sizeof(uint32_t) * buffers.num_splats;
                std::ostringstream signature;
                signature << "sparse-gradient-v2;n=" << buffers.num_splats
                          << ";max_record_words=" << kMaxRecordWords;
                if (sparse_mailbox_rank < 0) {
                    g_sparse_mailbox_owner->attach(
                        rank, capacity_bytes, signature.str());
                    sparse_mailbox_rank = static_cast<int>(rank);
                } else if (sparse_mailbox_rank != static_cast<int>(rank)) {
                    throw std::runtime_error(
                        "native backend changed sparse mailbox rank");
                }

                // Sparse pack runs Cumsum/PerfTimer timestamps and must own a
                // primary command batch. End the guard before async packed
                // D2H/H2D tickets, which require an idle primary batch.
                {
                    const auto pack_begin = std::chrono::steady_clock::now();
                    auto device_guard = DeviceGuard(&trainer);
                    local_sparse_count =
                        trainer.executeFTGSPPSparseGradientPack(
                            config, buffers, uniforms.active_sh);
                    sparse_pack_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - pack_begin).count();
                }
                local_sparse_bytes = size_t(local_sparse_count) * record_bytes;
                const std::vector<VulkanGSPipeline::PackedBufferRegion>
                    local_regions{{buffers.sparse_gradient_records.deviceBuffer, 0u}};
                const size_t bucket_bytes = g_sparse_mailbox_owner->bucket_bytes();
                const size_t local_buckets = local_sparse_bytes == 0 ? 0 :
                    (local_sparse_bytes + bucket_bytes - 1) / bucket_bytes;
                std::vector<std::optional<
                    VulkanGSPipeline::PackedTransferTicket>> downloads(
                        local_buckets);
                auto submit_sparse_download = [&](size_t bucket) {
                    const size_t offset = bucket * bucket_bytes;
                    const size_t bytes = std::min(
                        bucket_bytes, local_sparse_bytes - offset);
                    downloads[bucket] =
                        trainer.submitBuffersFromDevicePackedRange(
                            local_regions, offset, bytes);
                };
                // Start local D2H immediately after pack. Mailbox wait_writable
                // is only required before publishing the borrowed mapping; the
                // per-rank packed ring does not need the peer GPU.
                const size_t speculative = std::min<size_t>(
                    local_buckets, VulkanGSPipeline::packed_transfer_ring_depth);
                for (size_t bucket = 0; bucket < speculative; ++bucket)
                    submit_sparse_download(bucket);
                {
                    const auto wait_begin = std::chrono::steady_clock::now();
                    g_sparse_mailbox_owner->publish_message_and_wait(
                        rank, generation, local_sparse_bytes);
                    sparse_peer_wait_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - wait_begin).count();
                }
                const auto transfer_begin = std::chrono::steady_clock::now();
                peer_sparse_bytes = g_sparse_mailbox_owner->peer_message_bytes(
                    rank, 1u-rank, generation);
                if (peer_sparse_bytes % record_bytes != 0 ||
                    peer_sparse_bytes > capacity_bytes)
                    throw std::runtime_error(
                        "peer sparse-gradient message has an invalid size");
                pending_peer_sparse_count = static_cast<uint32_t>(
                    peer_sparse_bytes / record_bytes);
                const auto dense_all = mailbox_dense_entries(false);
                const auto local_aux = mailbox_dense_entries(true);
                const size_t dense_bytes = packed_bytes(dense_all);
                const size_t aux_bytes = local_aux.empty()
                    ? 0u : packed_bytes(local_aux);
                const double dense_cost = 2.0 * double(dense_bytes);
                const double sparse_cost = double(
                    local_sparse_bytes + peer_sparse_bytes + 2u * aux_bytes);
                used_sparse = pair_codec_policy_ == "sorted_sparse" || (
                    pair_codec_policy_ == "auto_exact" &&
                    std::strcmp(choose_exact_bucket_codec(
                        dense_cost, sparse_cost, true, 0.05),
                        "exact_sorted_sparse") == 0);
                pair_gaussian_codec_ = used_sparse
                    ? "exact_sorted_sparse" : "dense";
                auto discard_speculative_downloads = [&]() {
                    for (auto& ticket : downloads) {
                        if (!ticket.has_value())
                            continue;
                        trainer.completeBuffersFromDevicePackedRangeMapped(
                            *ticket);
                        ticket.reset();
                    }
                };
                if (!used_sparse) {
                    discard_speculative_downloads();
                    const auto peer = peer_entries_for(dense_all);
                    dense_timing = exchange_fixed_mailbox(
                        *g_host_mailbox_owner, host_mailbox_rank,
                        rank, generation, dense_all, peer,
                        "dense-auto;" + packed_signature(dense_all), true);
                    pending_peer_sparse_count = 0;
                    pending_local_sparse_count = 0;
                } else {
                std::vector<VulkanGSPipeline::PackedBufferRegion> peer_regions;
                if (peer_sparse_bytes > 0) {
                    trainer.resizeDeviceBuffer(
                        buffers.peer_sparse_gradient_records,
                        size_t(pending_peer_sparse_count) * record_words);
                    peer_regions.push_back({
                        buffers.peer_sparse_gradient_records.deviceBuffer, 0u});
                }
                const size_t transfer_bytes = std::max(
                    local_sparse_bytes, peer_sparse_bytes);
                const size_t transfer_buckets = transfer_bytes == 0 ? 0 :
                    (transfer_bytes + bucket_bytes - 1) / bucket_bytes;
                if (downloads.size() < transfer_buckets)
                    downloads.resize(transfer_buckets);
                std::deque<VulkanGSPipeline::PackedTransferTicket> uploads;
                auto submit_sparse_download = [&](size_t bucket) {
                    if (bucket >= local_buckets || downloads[bucket].has_value())
                        return;
                    const size_t offset = bucket * bucket_bytes;
                    const size_t bytes = std::min(
                        bucket_bytes, local_sparse_bytes - offset);
                    downloads[bucket] =
                        trainer.submitBuffersFromDevicePackedRange(
                            local_regions, offset, bytes);
                };
                size_t next_local = speculative;
                for (size_t bucket = 0; bucket < transfer_buckets; ++bucket) {
                    const size_t offset = bucket * bucket_bytes;
                    if (offset < local_sparse_bytes) {
                        const size_t bytes = std::min(
                            bucket_bytes, local_sparse_bytes - offset);
                        g_sparse_mailbox_owner->wait_writable(
                            rank, generation, static_cast<uint32_t>(bucket));
                        if (!downloads[bucket].has_value())
                            submit_sparse_download(bucket);
                        if (!downloads[bucket].has_value())
                            throw std::runtime_error(
                                "sparse mailbox lost an async readback ticket");
                        const auto mapped =
                            trainer.completeBuffersFromDevicePackedRangeMapped(
                                *downloads[bucket]);
                        downloads[bucket].reset();
                        sparse_download_seconds += mapped.timing.gpu_copy_seconds
                            + mapped.timing.host_memcpy_seconds
                            + mapped.timing.invalidate_seconds;
                        sparse_host_memcpy_seconds +=
                            mapped.timing.host_memcpy_seconds;
                        g_sparse_mailbox_owner->publish_borrowed_and_wait(
                            rank, generation, static_cast<uint32_t>(bucket),
                            mapped.data, bytes);
                    } else {
                        g_sparse_mailbox_owner->wait_writable(
                            rank, generation, static_cast<uint32_t>(bucket));
                        g_sparse_mailbox_owner->publish_and_wait(
                            rank, generation, static_cast<uint32_t>(bucket));
                    }
                    if (offset < peer_sparse_bytes) {
                        const size_t bytes = std::min(
                            bucket_bytes, peer_sparse_bytes - offset);
                        const void* slot =
                            g_sparse_mailbox_owner->peer_borrowed_pointer(
                                rank, 1u-rank, generation,
                                static_cast<uint32_t>(bucket), bytes);
                        if (uploads.size() >=
                            VulkanGSPipeline::packed_transfer_ring_depth) {
                            const auto timing =
                                trainer.completeBuffersToDevicePackedRange(
                                    uploads.front());
                            uploads.pop_front();
                            sparse_upload_seconds += timing.gpu_copy_seconds
                                + timing.host_memcpy_seconds
                                + timing.invalidate_seconds;
                            sparse_host_memcpy_seconds += timing.host_memcpy_seconds;
                        }
                        uploads.push_back(
                            trainer.submitBuffersToDevicePackedRange(
                                slot, offset, bytes, peer_regions));
                    }
                    g_sparse_mailbox_owner->mark_consumed(
                        rank, generation, static_cast<uint32_t>(bucket));
                    if (next_local < local_buckets)
                        submit_sparse_download(next_local++);
                }
                while (!uploads.empty()) {
                    const auto timing =
                        trainer.completeBuffersToDevicePackedRange(uploads.front());
                    uploads.pop_front();
                    sparse_upload_seconds += timing.gpu_copy_seconds
                        + timing.host_memcpy_seconds + timing.invalidate_seconds;
                    sparse_host_memcpy_seconds += timing.host_memcpy_seconds;
                }

                // Color gradients remain a dense auxiliary collective;
                // Gaussian image gradients use exact visible records.
                if (!local_aux.empty()) {
                    const auto peer_aux = peer_entries_for(local_aux);
                    dense_timing = exchange_fixed_mailbox(
                        *g_host_mailbox_owner, host_mailbox_rank,
                        rank, generation, local_aux, peer_aux,
                        "auxiliary;" + packed_signature(local_aux), true);
                }
                sparse_transfer_wall_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - transfer_begin).count();
                pending_local_sparse_count = local_sparse_count;
                pair_sparse_mean_ready_ = true;
                }
            }
        } catch (const std::exception& error) {
            if (is_tile_overflow_abort(error) && g_sync_generation) {
                const auto decision = g_sync_generation->wait_decision(
                    rank, generation);
                if (decision.aborted && !decision.fatal)
                    return aborted_exchange_report(decision);
            }
            g_host_mailbox_owner->abort(error.what());
            if (g_sparse_mailbox_owner)
                g_sparse_mailbox_owner->abort(error.what());
            throw;
        } catch (...) {
            g_host_mailbox_owner->abort("unknown native mailbox rank failure");
            if (g_sparse_mailbox_owner)
                g_sparse_mailbox_owner->abort(
                    "unknown native sparse mailbox rank failure");
            throw;
        }
        const double wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();
        // Dense-only path: peer wait is the first bucket rendezvous inside
        // exchange_fixed_mailbox. Sparse path: length rendezvous dominates;
        // aux dense peer_wait is usually near-zero (both ranks already synced).
        const double peer_wait_seconds = used_sparse
            ? sparse_peer_wait_seconds + dense_timing.peer_wait_seconds
            : dense_timing.peer_wait_seconds;
        const double transfer_wall_seconds = used_sparse
            ? sparse_transfer_wall_seconds
            : dense_timing.transfer_wall_seconds;
        py::dict result;
        result["transport"] = "host_mailbox";
        result["sparse"] = used_sparse;
        result["gaussian_codec"] = pair_gaussian_codec_;
        result["codec_policy"] = pair_codec_policy_;
        result["active_sh_degree"] = uint32_t(uniforms.active_sh);
        result["active_sh_floats"] = uint64_t(
            active_sh_float_count(uniforms.active_sh));
        result["payload_bytes"] = dense_timing.payload_bytes
            + (used_sparse ? local_sparse_bytes : 0u);
        result["peer_payload_bytes"] =
            dense_timing.payload_bytes + (used_sparse ? peer_sparse_bytes : 0u);
        const size_t sparse_bucket_bytes = used_sparse && g_sparse_mailbox_owner
            ? g_sparse_mailbox_owner->bucket_bytes()
            : g_host_mailbox_owner->bucket_bytes();
        result["bucket_bytes"] = sparse_bucket_bytes;
        result["bucket_count"] = dense_timing.bucket_count +
            (!used_sparse || std::max(local_sparse_bytes, peer_sparse_bytes) == 0 ? 0 :
             (std::max(local_sparse_bytes, peer_sparse_bytes) +
              sparse_bucket_bytes - 1) /
             sparse_bucket_bytes);
        result["local_visible_count"] = local_sparse_count;
        result["peer_visible_count"] = pending_peer_sparse_count;
        result["download_seconds"] =
            dense_timing.download_seconds + sparse_download_seconds;
        result["upload_seconds"] =
            dense_timing.upload_seconds + sparse_upload_seconds;
        result["host_memcpy_seconds"] =
            dense_timing.host_memcpy_seconds + sparse_host_memcpy_seconds;
        result["sparse_pack_seconds"] = sparse_pack_seconds;
        result["peer_wait_seconds"] = peer_wait_seconds;
        result["transfer_wall_seconds"] = transfer_wall_seconds;
        result["async_pipeline"] = true;
        result["transfer_ring_depth"] =
            VulkanGSPipeline::packed_transfer_ring_depth;
        result["borrowed_readback"] = true;
        result["vulkan_persistent_mapping"] = true;
        result["host_storage_bytes"] = size_t(0);
        result["wall_seconds"] = wall_seconds;
        result["cpu_reduction_seconds"] = 0.0;
        result["aborted"] = false;
        result["pcie_d2h_bytes"] = trainer.telemetry().pcie_d2h_bytes;
        result["pcie_h2d_bytes"] = trainer.telemetry().pcie_h2d_bytes;
        return result;
    }

    void mean_local_with_peer_gradients(bool auxiliary_only) {
        if (!gradient_ready)
            throw std::runtime_error("peer mean requires pending raw gradients");
        const auto local = mailbox_dense_entries(auxiliary_only);
        if (local.empty())
            return;
        const auto peer = peer_entries_for(local);
        if (local.size() != peer.size())
            throw std::runtime_error("host mailbox peer ABI entry count differs");
        auto device_guard = DeviceGuard(&trainer);
        for (size_t i = 0; i < local.size(); ++i) {
            if (local[i].offset != peer[i].offset || local[i].count != peer[i].count)
                throw std::runtime_error("host mailbox peer ABI entry differs");
            trainer.executeFTGSPPPeerMean(
                local[i].buffer->deviceBuffer,
                peer[i].buffer->deviceBuffer,
                local[i].count);
        }
    }

    void configure_pair_exchange_gpu_merge(const std::string& codec_policy) {
        if (gradient_ready)
            throw std::runtime_error(
                "host-mailbox mode cannot change while gradients are pending");
        if (codec_policy != "auto_exact" && codec_policy != "dense" &&
            codec_policy != "sorted_sparse")
            throw std::invalid_argument(
                "pair codec policy must be auto_exact, dense or sorted_sparse");
        pair_codec_policy_ = codec_policy;
        config.sparse_distributed = codec_policy != "dense"
            && bool(g_sparse_mailbox_owner);
        if (codec_policy != "dense" && !config.sparse_distributed)
            throw std::runtime_error(
                "pair sparse/auto codec requested without a sparse mailbox");
    }

    py::dict optimizer_step_pair_exchange_gpu_merge(
        int step, const std::vector<float>& timestamps) {
        if (!g_host_mailbox_owner)
            throw std::invalid_argument(
                "host mailbox optimizer requires an active pair-exchange owner");
        // Prefer the full PK view set; 1..32 covers P<=8 and K<=4.
        if (timestamps.empty() || timestamps.size() > 32u)
            throw std::invalid_argument(
                "pair-exchange optimizer requires 1..32 PK view timestamps");
        double reduction_seconds = 0.0;
        double adam_seconds = 0.0;
        {
            py::gil_scoped_release release;
            const auto reduction_begin = std::chrono::steady_clock::now();
            if (pair_sparse_mean_ready_) {
                {
                    if (sparse_mailbox_rank < 0)
                        throw std::runtime_error(
                            "pair-sparse merge requires an attached sparse mailbox rank");
                    auto device_guard = DeviceGuard(&trainer);
                    trainer.executeFTGSPPSparseGradientMerge(
                        config, buffers, pending_peer_sparse_count, 2u,
                        timestamps.data(),
                        static_cast<uint32_t>(timestamps.size()),
                        false, uniforms.active_sh,
                        static_cast<uint32_t>(sparse_mailbox_rank),
                        pending_local_sparse_count);
                }
            } else if (pair_needs_regularizer_) {
                auto device_guard = DeviceGuard(&trainer);
                trainer.executeFTGSPPRegularizerMean(
                    config, buffers, timestamps.data(),
                    static_cast<uint32_t>(timestamps.size()));
            }
            reduction_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - reduction_begin).count();
            const auto adam_begin = std::chrono::steady_clock::now();
            optimizer_step(step);
            adam_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - adam_begin).count();
        }
        py::dict result;
        result["gpu_reduction_seconds"] = reduction_seconds;
        result["optimizer_seconds"] = adam_seconds;
        pair_sparse_mean_ready_ = false;
        pair_needs_regularizer_ = false;
        pending_local_sparse_count = 0;
        pending_peer_sparse_count = 0;
        return result;
    }

    py::dict exchange_pair_relocation(
        uint32_t rank, uint64_t generation) {
        if (!g_relocation_mailbox_owner)
            throw std::runtime_error(
                "relocation host mailbox owner is not configured");
        trainer.waitAllComputeSlots("relocation");
        trainer.drainPackedTransfers();
        const size_t count = 2 * buffers.num_splats;
        if (count == 0 || buffers.relocation_stats.deviceSize() != count)
            throw std::runtime_error(
                "relocation mailbox requires initialized (N,2) statistics");
        trainer.resizeDeviceBuffer(buffers.peer_relocation_stats, count);
        const std::vector<PackedGradientEntry> local{{
            "relocation_stats", &buffers.relocation_stats, 0u, count}};
        const std::vector<PackedGradientEntry> peer{{
            "peer_relocation_stats", &buffers.peer_relocation_stats, 0u, count}};
        MailboxTransferTiming timing;
        try {
            py::gil_scoped_release release;
            timing = exchange_fixed_mailbox(
                *g_relocation_mailbox_owner, relocation_mailbox_rank,
                rank, generation, local, peer,
                "relocation-sum-v1;n=" + std::to_string(buffers.num_splats),
                false);
        } catch (const std::exception& error) {
            g_relocation_mailbox_owner->abort(error.what());
            throw;
        } catch (...) {
            g_relocation_mailbox_owner->abort(
                "unknown native relocation mailbox rank failure");
            throw;
        }
        py::dict result;
        result["transport"] = "host_mailbox";
        result["operation"] = "relocation_sum";
        result["payload_bytes"] = timing.payload_bytes;
        result["bucket_count"] = timing.bucket_count;
        result["download_seconds"] = timing.download_seconds;
        result["upload_seconds"] = timing.upload_seconds;
        result["host_memcpy_seconds"] = timing.host_memcpy_seconds;
        result["peer_wait_seconds"] = timing.peer_wait_seconds;
        result["transfer_wall_seconds"] = timing.transfer_wall_seconds;
        result["async_pipeline"] = timing.async_pipeline;
        result["transfer_ring_depth"] =
            VulkanGSPipeline::packed_transfer_ring_depth;
        result["wall_seconds"] = timing.wall_seconds;
        result["cpu_reduction_seconds"] = 0.0;
        return result;
    }

    static HierarchicalHostAlgorithm parse_hierarchical_algorithm(
        const std::string& algorithm) {
        if (algorithm == "auto")
            return HierarchicalHostAlgorithm::Auto;
        if (algorithm == "central_hierarchical_reduce_broadcast")
            return HierarchicalHostAlgorithm::Central;
        throw std::invalid_argument(
            "unsupported hierarchical host algorithm: " + algorithm);
    }

    static std::shared_ptr<HierarchicalHostBindingState>
    hierarchical_binding_state() {
        std::lock_guard<std::mutex> lock(g_hierarchical_host_mutex);
        if (!g_hierarchical_host_state)
            throw std::runtime_error(
                "hierarchical host collective owner is not configured");
        return g_hierarchical_host_state;
    }

    static py::dict create_hierarchical_host_collective(
        uint32_t world_size,
        size_t bucket_bytes,
        const std::string& algorithm,
        uint32_t local_microsteps,
        uint32_t reducer_shards,
        bool sparse_requested) {
        if (world_size < 3 || world_size > 8)
            throw std::invalid_argument(
                "native hierarchical central collective requires P in [3, 8]");
        if (bucket_bytes == 0 || bucket_bytes % sizeof(float) != 0)
            throw std::invalid_argument(
                "hierarchical bucket bytes must be positive and float aligned");
        if (local_microsteps != 1 &&
            local_microsteps != 2 &&
            local_microsteps != 4)
            throw std::invalid_argument(
                "native hierarchical gradient accumulation requires K in {1,2,4}");
        const HierarchicalHostAlgorithm requested =
            parse_hierarchical_algorithm(algorithm);
        const HierarchicalHostAlgorithm selected =
            requested == HierarchicalHostAlgorithm::Auto
                ? HierarchicalHostAlgorithm::Central
                : requested;
        auto state = std::make_shared<HierarchicalHostBindingState>();
        state->world_size = world_size;
        state->requested_bucket_bytes = bucket_bytes;
        state->local_microsteps = local_microsteps;
        state->reducer_shards = reducer_shards;
        state->requested_algorithm = requested;
        state->selected_algorithm = selected;
        state->codec_votes.assign(world_size, {});
        state->codec_submitted.assign(world_size, 0);
        {
            std::lock_guard<std::mutex> lock(g_hierarchical_host_mutex);
            if (g_hierarchical_host_state)
                throw std::runtime_error(
                    "hierarchical host collective owner already exists");
            g_hierarchical_host_state = state;
        }
        ensure_sync_generation(world_size);

        py::dict result;
        result["abi"] = "ftgspp-hierarchical-host-v1";
        result["requested_algorithm"] =
            HierarchicalHostCollective::algorithm_name(requested);
        result["selected_algorithm"] =
            HierarchicalHostCollective::algorithm_name(selected);
        result["execution_backend"] = "native_cpu_fixed_rank_reduce";
        result["selection_reason"] =
            requested == HierarchicalHostAlgorithm::Auto
                ? "P>=3 uses fixed-rank central reduction"
                : "explicit native candidate selection";
        result["collective_ready"] = true;
        result["gpu_transport_ready"] = false;
        result["central_native_transport_ready"] = true;
        result["world_size"] = world_size;
        result["bucket_bytes"] = bucket_bytes;
        result["local_microsteps"] = local_microsteps;
        result["reducer_shards"] = reducer_shards;
        result["sparse_requested"] = sparse_requested;
        // Gaussian exact sorted sparse uses CPU k-way merge + Vulkan scatter;
        // the small color-correction gradient remains dense.
        result["sparse"] = sparse_requested;
        result["exact_sorted_sparse_ready"] = true;
        result["gaussian_codec"] = sparse_requested
            ? "exact_sorted_sparse"
            : "dense";
        result["color_codec"] = "dense";
        result["packed_transfer_ticket"] = true;
        result["single_d2h_per_rank"] = false;
        result["single_h2d_per_rank"] = false;
        result["per_bucket_reduce"] = true;
        result["python_gradient_payload"] = false;
        result["deterministic_fp32_rank_order"] = true;
        return result;
    }

    static void release_hierarchical_host_collective() {
        std::shared_ptr<HierarchicalHostBindingState> state;
        {
            std::lock_guard<std::mutex> lock(g_hierarchical_host_mutex);
            state = std::move(g_hierarchical_host_state);
        }
        if (!state)
            return;
        std::shared_ptr<HierarchicalHostCollective> gradient;
        std::shared_ptr<HierarchicalHostCollective> gradient_sparse;
        std::shared_ptr<HierarchicalHostCollective> relocation;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            gradient = state->gradient;
            gradient_sparse = state->gradient_sparse;
            relocation = state->relocation;
        }
        if (gradient)
            gradient->abort("hierarchical host collective released");
        if (gradient_sparse)
            gradient_sparse->abort(
                "hierarchical sparse collective released");
        if (relocation)
            relocation->abort("hierarchical relocation collective released");
        release_sync_generation();
    }

    void configure_hierarchical_host_collective(
        uint32_t rank, bool sparse, const std::string& codec_policy) {
        auto state = hierarchical_binding_state();
        if (rank >= state->world_size)
            throw std::out_of_range(
                "hierarchical host configured rank is out of range");
        if (codec_policy != "auto_exact" &&
            codec_policy != "dense" &&
            codec_policy != "sorted_sparse")
            throw std::invalid_argument(
                "hierarchical codec_policy must be auto_exact, dense, or "
                "sorted_sparse");
        config.sparse_distributed = sparse;
        hierarchical_codec_policy_ = sparse ? codec_policy : "dense";
        hierarchical_sparse_mean_ready_ = false;
        hierarchical_needs_regularizer_ = false;
        hierarchical_gaussian_codec_ = sparse ? "exact_sorted_sparse" : "dense";
        hierarchical_gradient_rank = static_cast<int>(rank);
        hierarchical_relocation_rank = static_cast<int>(rank);
    }

    std::shared_ptr<HierarchicalHostCollective>
    ensure_hierarchical_collective(
        const std::shared_ptr<HierarchicalHostBindingState>& state,
        bool relocation,
        uint32_t rank,
        size_t payload_bytes,
        const std::string& signature,
        bool sparse_gaussian=false) {
        if (payload_bytes == 0 || payload_bytes % sizeof(float) != 0)
            throw std::runtime_error(
                "hierarchical host packed payload is empty or misaligned");
        std::shared_ptr<HierarchicalHostCollective> collective;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            auto& target = relocation
                ? state->relocation
                : (sparse_gaussian ? state->gradient_sparse : state->gradient);
            if (!target) {
                HierarchicalHostCollectiveConfig collective_config;
                collective_config.world_size = state->world_size;
                collective_config.local_microsteps = state->local_microsteps;
                collective_config.reducer_shards = state->reducer_shards;
                collective_config.algorithm = state->selected_algorithm;
                collective_config.mean_reduction = !relocation;
                if (sparse_gaussian) {
                    collective_config.bucket_count = 1;
                    collective_config.bucket_float_capacity =
                        payload_bytes / sizeof(float);
                    collective_config.ring_depth = 1;
                } else {
                    const size_t bucket_floats = std::max<size_t>(
                        1u, state->requested_bucket_bytes / sizeof(float));
                    const size_t payload_floats =
                        payload_bytes / sizeof(float);
                    const size_t bucket_count = std::max<size_t>(
                        1u,
                        (payload_floats + bucket_floats - 1) / bucket_floats);
                    collective_config.bucket_count = bucket_count;
                    collective_config.bucket_float_capacity = bucket_floats;
                    collective_config.ring_depth = std::min<size_t>(
                        3u, bucket_count);
                }
                target = std::make_shared<HierarchicalHostCollective>(
                    collective_config);
            }
            collective = target;
        }
        bool& attached = relocation
            ? hierarchical_relocation_attached_
            : (sparse_gaussian
                ? hierarchical_sparse_attached_
                : hierarchical_gradient_attached_);
        if (!attached) {
            try {
                collective->attach(rank, payload_bytes, signature);
                attached = true;
            } catch (const std::exception& error) {
                collective->abort(error.what());
                throw;
            } catch (...) {
                collective->abort(
                    "unknown hierarchical attachment failure");
                throw;
            }
        }
        return collective;
    }

    py::dict exchange_hierarchical_entries(
        uint32_t rank,
        uint64_t generation,
        const std::vector<PackedGradientEntry>& entries,
        bool relocation) {
        auto state = hierarchical_binding_state();
        const int configured_rank = relocation
            ? hierarchical_relocation_rank
            : hierarchical_gradient_rank;
        if (configured_rank != static_cast<int>(rank))
            throw std::runtime_error(
                "hierarchical exchange rank differs from configured rank");
        const size_t total_bytes = packed_bytes(entries);
        const std::string signature =
            std::string(relocation ? "relocation-sum;" : "gradient-mean;") +
            packed_signature(entries);
        auto collective = ensure_hierarchical_collective(
            state, relocation, rank, total_bytes, signature);
        const auto regions = packed_regions(entries);
        const size_t bucket_bytes = std::max<size_t>(
            sizeof(float), state->requested_bucket_bytes);
        const size_t bucket_count = std::max<size_t>(
            1u, (total_bytes + bucket_bytes - 1) / bucket_bytes);

        VulkanGSPipeline::PackedTransferTiming download_timing;
        VulkanGSPipeline::PackedTransferTiming upload_timing;
        double reduction_seconds = 0.0;
        double peer_wait_seconds = 0.0;
        const auto wall_begin = std::chrono::steady_clock::now();
        try {
            py::gil_scoped_release release;
            struct PendingDownload {
                VulkanGSPipeline::PackedTransferTicket ticket;
                size_t bucket = 0;
                size_t offset = 0;
                size_t bytes = 0;
            };
            std::deque<PendingDownload> downloads;
            std::deque<VulkanGSPipeline::PackedTransferTicket> uploads;
            auto drain_transfers = [&]() {
                while (!downloads.empty()) {
                    try {
                        trainer.completeBuffersFromDevicePackedRangeMapped(
                            downloads.front().ticket);
                    } catch (...) {}
                    downloads.pop_front();
                }
                while (!uploads.empty()) {
                    try {
                        trainer.completeBuffersToDevicePackedRange(
                            uploads.front());
                    } catch (...) {}
                    uploads.pop_front();
                }
            };
            auto complete_oldest_upload = [&]() {
                if (uploads.empty())
                    return;
                const auto uploaded =
                    trainer.completeBuffersToDevicePackedRange(uploads.front());
                uploads.pop_front();
                upload_timing.gpu_copy_seconds += uploaded.gpu_copy_seconds;
                upload_timing.host_memcpy_seconds +=
                    uploaded.host_memcpy_seconds;
                upload_timing.invalidate_seconds += uploaded.invalidate_seconds;
                upload_timing.used_persistent_map =
                    uploaded.used_persistent_map;
            };
            auto submit_download = [&](size_t bucket) {
                const size_t offset = bucket * bucket_bytes;
                const size_t bytes = std::min(
                    bucket_bytes, total_bytes - offset);
                downloads.push_back({
                    trainer.submitBuffersFromDevicePackedRange(
                        regions, offset, bytes),
                    bucket, offset, bytes,
                });
            };
            const size_t initial = std::min<size_t>(
                bucket_count, VulkanGSPipeline::packed_transfer_ring_depth);
            for (size_t bucket = 0; bucket < initial; ++bucket)
                submit_download(bucket);
            size_t next_download = initial;
            bool measured_first_wait = false;
            try {
            for (size_t bucket = 0; bucket < bucket_count; ++bucket) {
                if (downloads.empty() || downloads.front().bucket != bucket)
                    throw std::runtime_error(
                        "hierarchical readback pipeline lost bucket ordering");
                const PendingDownload current = downloads.front();
                downloads.pop_front();
                const auto mapped =
                    trainer.completeBuffersFromDevicePackedRangeMapped(
                        current.ticket);
                download_timing.gpu_copy_seconds +=
                    mapped.timing.gpu_copy_seconds;
                download_timing.host_memcpy_seconds +=
                    mapped.timing.host_memcpy_seconds;
                download_timing.invalidate_seconds +=
                    mapped.timing.invalidate_seconds;
                download_timing.used_persistent_map =
                    mapped.timing.used_persistent_map;
                const auto reduction_begin = std::chrono::steady_clock::now();
                const float* reduced = collective->acquire_reduced_bucket(
                    rank, generation, static_cast<uint32_t>(bucket),
                    static_cast<const float*>(mapped.data),
                    current.bytes / sizeof(float));
                const double reduced_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - reduction_begin).count();
                reduction_seconds += reduced_seconds;
                if (!measured_first_wait) {
                    peer_wait_seconds = reduced_seconds;
                    measured_first_wait = true;
                }
                // submitBuffersToDevicePackedRange memcpy's into a private
                // stager before returning, so the shared reduced slab can
                // be released immediately and the GPU H2D can overlap the
                // next D2H/reduce.
                uploads.push_back(trainer.submitBuffersToDevicePackedRange(
                    reduced, current.offset, current.bytes, regions));
                collective->release_bucket(
                    rank, generation, static_cast<uint32_t>(bucket));
                while (uploads.size() >=
                       VulkanGSPipeline::packed_transfer_ring_depth)
                    complete_oldest_upload();
                if (next_download < bucket_count)
                    submit_download(next_download++);
            }
            while (!uploads.empty())
                complete_oldest_upload();
            } catch (...) {
                drain_transfers();
                throw;
            }
        } catch (const std::exception& error) {
            if (is_tile_overflow_abort(error) && g_sync_generation) {
                const auto decision = g_sync_generation->wait_decision(
                    rank, generation);
                if (decision.aborted && !decision.fatal)
                    return aborted_exchange_report(decision);
            }
            collective->abort(error.what());
            throw;
        } catch (...) {
            collective->abort(
                "unknown native hierarchical rank failure");
            throw;
        }

        const double wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall_begin).count();
        py::dict result;
        result["abi"] = "ftgspp-hierarchical-host-v1";
        result["transport"] = "hierarchical_host";
        result["operation"] = relocation ? "relocation_sum" : "gradient_mean";
        result["algorithm"] =
            HierarchicalHostCollective::algorithm_name(
                state->selected_algorithm);
        result["execution_backend"] = "native_cpu_fixed_rank_reduce";
        result["payload_bytes"] = total_bytes;
        result["bucket_count"] = bucket_count;
        result["bucket_bytes"] = bucket_bytes;
        result["download_seconds"] =
            download_timing.gpu_copy_seconds +
            download_timing.host_memcpy_seconds +
            download_timing.invalidate_seconds;
        result["upload_seconds"] =
            upload_timing.gpu_copy_seconds +
            upload_timing.host_memcpy_seconds +
            upload_timing.invalidate_seconds;
        result["host_memcpy_seconds"] =
            download_timing.host_memcpy_seconds +
            upload_timing.host_memcpy_seconds;
        result["cpu_reduction_seconds"] = reduction_seconds;
        result["peer_wait_seconds"] = peer_wait_seconds;
        result["transfer_wall_seconds"] = std::max(
            0.0, wall_seconds - peer_wait_seconds);
        result["wall_seconds"] = wall_seconds;
        result["single_d2h"] = bucket_count == 1;
        result["single_h2d"] = bucket_count == 1;
        result["per_bucket_reduce"] = true;
        result["async_pipeline"] = true;
        result["vulkan_persistent_mapping"] =
            download_timing.used_persistent_map &&
            upload_timing.used_persistent_map;
        result["python_gradient_payload"] = false;
        result["finite_checked"] = true;
        result["aborted"] = false;
        return result;
    }

    static float bit_cast_float(uint32_t value) {
        float out = 0.0f;
        std::memcpy(&out, &value, sizeof(out));
        return out;
    }

    static uint32_t bit_cast_u32(float value) {
        uint32_t out = 0;
        std::memcpy(&out, &value, sizeof(out));
        return out;
    }

    py::dict exchange_hierarchical_sparse_gradient(
        uint32_t rank, uint64_t generation) {
        auto state = hierarchical_binding_state();
        if (hierarchical_gradient_rank != static_cast<int>(rank))
            throw std::runtime_error(
                "hierarchical sparse exchange rank differs from configured rank");
        const size_t record_words = sparse_record_words();
        const size_t record_width = record_words - 1u;
        const size_t record_bytes = record_words * sizeof(uint32_t);
        const size_t n = buffers.num_splats;
        size_t dense_total_floats = 0;
        for (const auto& entry : mailbox_dense_entries(false))
            dense_total_floats = entry.offset + entry.count;
        size_t dense_aux_floats = 0;
        for (const auto& entry : mailbox_dense_entries(true))
            dense_aux_floats = entry.offset + entry.count;
        const size_t dense_gaussian_floats =
            dense_total_floats - dense_aux_floats;
        const size_t dense_bytes = dense_gaussian_floats * sizeof(float);
        const size_t aux_bytes = dense_aux_floats * sizeof(float);
        const auto wall_begin = std::chrono::steady_clock::now();
        uint32_t local_count = 0;
        size_t local_bytes = 0;
        double pack_seconds = 0.0;
        double sparse_download_seconds = 0.0;
        double sparse_host_memcpy_seconds = 0.0;
        double sparse_reduce_seconds = 0.0;
        double sparse_upload_seconds = 0.0;
        std::string codec;
        py::dict aux_report;
        hierarchical_sparse_mean_ready_ = false;
        hierarchical_needs_regularizer_ = true;
        pending_peer_sparse_count = 0;

        size_t transfer_bucket_bytes = state->requested_bucket_bytes;
        if (transfer_bucket_bytes < 4096u)
            transfer_bucket_bytes = 32ull * 1024ull * 1024ull;

        try {
            std::vector<uint32_t> local_ids;
            std::vector<float> local_values;
            if (hierarchical_codec_policy_ != "dense") {
                py::gil_scoped_release release;
                {
                    auto device_guard = DeviceGuard(&trainer);
                    const auto pack_begin = std::chrono::steady_clock::now();
                    local_count =
                        trainer.executeFTGSPPSparseGradientPack(
                            config, buffers, uniforms.active_sh);
                    pack_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - pack_begin).count();
                }
                local_bytes = size_t(local_count) * record_bytes;
                const std::vector<VulkanGSPipeline::PackedBufferRegion>
                    local_regions{
                        {buffers.sparse_gradient_records.deviceBuffer, 0u}};
                std::vector<uint32_t> local_packed(
                    size_t(local_count) * record_words);
                if (local_bytes != 0) {
                    struct PendingDownload {
                        VulkanGSPipeline::PackedTransferTicket ticket;
                        size_t bucket = 0;
                        size_t offset = 0;
                        size_t bytes = 0;
                    };
                    std::deque<PendingDownload> downloads;
                    const size_t chunk = transfer_bucket_bytes;
                    const size_t bucket_count =
                        (local_bytes + chunk - 1) / chunk;
                    auto drain_downloads = [&]() {
                        while (!downloads.empty()) {
                            try {
                                trainer.completeBuffersFromDevicePackedRangeMapped(
                                    downloads.front().ticket);
                            } catch (...) {}
                            downloads.pop_front();
                        }
                    };
                    auto submit_download = [&](size_t bucket) {
                        const size_t offset = bucket * chunk;
                        const size_t bytes = std::min(
                            chunk, local_bytes - offset);
                        downloads.push_back({
                            trainer.submitBuffersFromDevicePackedRange(
                                local_regions, offset, bytes),
                            bucket, offset, bytes,
                        });
                    };
                    try {
                        const size_t initial = std::min<size_t>(
                            bucket_count,
                            VulkanGSPipeline::packed_transfer_ring_depth);
                        for (size_t bucket = 0; bucket < initial; ++bucket)
                            submit_download(bucket);
                        size_t next_download = initial;
                        for (size_t bucket = 0; bucket < bucket_count;
                             ++bucket) {
                            if (downloads.empty() ||
                                downloads.front().bucket != bucket)
                                throw std::runtime_error(
                                    "hierarchical sparse readback lost "
                                    "bucket ordering");
                            const PendingDownload current = downloads.front();
                            downloads.pop_front();
                            const auto mapped =
                                trainer.completeBuffersFromDevicePackedRangeMapped(
                                    current.ticket);
                            sparse_download_seconds +=
                                mapped.timing.gpu_copy_seconds +
                                mapped.timing.host_memcpy_seconds +
                                mapped.timing.invalidate_seconds;
                            sparse_host_memcpy_seconds +=
                                mapped.timing.host_memcpy_seconds;
                            std::memcpy(
                                reinterpret_cast<char*>(local_packed.data()) +
                                    current.offset,
                                mapped.data,
                                current.bytes);
                            if (next_download < bucket_count)
                                submit_download(next_download++);
                        }
                    } catch (...) {
                        drain_downloads();
                        throw;
                    }
                }
                local_ids.resize(local_count);
                local_values.resize(size_t(local_count) * record_width);
                for (uint32_t i = 0; i < local_count; ++i) {
                    const size_t base = size_t(i) * record_words;
                    local_ids[i] = local_packed[base];
                    for (size_t j = 0; j < record_width; ++j)
                        local_values[size_t(i) * record_width + j] =
                            bit_cast_float(local_packed[base + 1u + j]);
                }
            }

            {
                py::gil_scoped_release release;
                codec = state->agree_exact_codec(
                    rank, generation,
                    HierarchicalExactCodecVote{
                        local_bytes,
                        dense_bytes,
                        aux_bytes,
                        local_count == 0 ? nullptr : local_ids.data(),
                        local_count,
                        record_bytes,
                    },
                    hierarchical_codec_policy_);
            }
            hierarchical_gaussian_codec_ = codec;
            if (codec == "dense") {
                aux_report = exchange_hierarchical_entries(
                    rank, generation, mailbox_dense_entries(false), false);
                hierarchical_needs_regularizer_ = true;
            } else {
                static constexpr size_t kMaxRecordWidth = 65u;
                const size_t attach_bytes = std::max<size_t>(
                    sizeof(float), n * kMaxRecordWidth * sizeof(float));
                auto collective = ensure_hierarchical_collective(
                    state, false, rank, attach_bytes,
                    "exact-sorted-sparse-v2;max_record_width=" +
                        std::to_string(kMaxRecordWidth),
                    true);
                try {
                    py::gil_scoped_release release;
                    const auto reduce_begin = std::chrono::steady_clock::now();
                    const SortedSparseMeanResult& reduced =
                        collective->acquire_reduced_sorted_sparse(
                            rank, generation,
                            local_count == 0 ? nullptr : local_ids.data(),
                            local_count == 0 ? nullptr : local_values.data(),
                            local_ids.size(), record_width,
                            static_cast<uint32_t>(n));
                    sparse_reduce_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - reduce_begin).count();
                    pending_peer_sparse_count =
                        static_cast<uint32_t>(reduced.ids.size());
                    std::vector<uint32_t> packed(
                        size_t(pending_peer_sparse_count) * record_words);
                    for (uint32_t i = 0; i < pending_peer_sparse_count; ++i) {
                        const size_t base = size_t(i) * record_words;
                        packed[base] = reduced.ids[i];
                        for (size_t j = 0; j < record_width; ++j)
                            packed[base + 1u + j] = bit_cast_u32(
                                reduced.values[size_t(i) * record_width + j]);
                    }
                    trainer.resizeDeviceBuffer(
                        buffers.peer_sparse_gradient_records,
                        packed.size());
                    const size_t upload_bytes =
                        packed.size() * sizeof(uint32_t);
                    if (upload_bytes != 0) {
                        const std::vector<VulkanGSPipeline::PackedBufferRegion>
                            peer_regions{{
                                buffers.peer_sparse_gradient_records.deviceBuffer,
                                0u}};
                        std::deque<VulkanGSPipeline::PackedTransferTicket>
                            uploads;
                        const size_t chunk = transfer_bucket_bytes;
                        auto complete_oldest_upload = [&]() {
                            if (uploads.empty())
                                return;
                            const auto uploaded =
                                trainer.completeBuffersToDevicePackedRange(
                                    uploads.front());
                            uploads.pop_front();
                            sparse_upload_seconds +=
                                uploaded.gpu_copy_seconds +
                                uploaded.host_memcpy_seconds +
                                uploaded.invalidate_seconds;
                            sparse_host_memcpy_seconds +=
                                uploaded.host_memcpy_seconds;
                        };
                        auto drain_uploads = [&]() {
                            while (!uploads.empty()) {
                                try {
                                    trainer.completeBuffersToDevicePackedRange(
                                        uploads.front());
                                } catch (...) {}
                                uploads.pop_front();
                            }
                        };
                        try {
                            for (size_t offset = 0; offset < upload_bytes;
                                 offset += chunk) {
                                const size_t bytes = std::min(
                                    chunk, upload_bytes - offset);
                                while (uploads.size() >=
                                       VulkanGSPipeline::packed_transfer_ring_depth)
                                    complete_oldest_upload();
                                uploads.push_back(
                                    trainer.submitBuffersToDevicePackedRange(
                                        reinterpret_cast<const char*>(
                                            packed.data()) + offset,
                                        offset,
                                        bytes,
                                        peer_regions));
                            }
                            while (!uploads.empty())
                                complete_oldest_upload();
                        } catch (...) {
                            drain_uploads();
                            throw;
                        }
                    }
                    collective->release_sorted_sparse(rank, generation);
                } catch (const std::exception& error) {
                    collective->abort(error.what());
                    throw;
                } catch (...) {
                    collective->abort(
                        "unknown hierarchical sparse rank failure");
                    throw;
                }
                hierarchical_sparse_mean_ready_ = true;
                const auto aux = mailbox_dense_entries(true);
                if (!aux.empty()) {
                    aux_report = exchange_hierarchical_entries(
                        rank, generation, aux, false);
                }
            }
        } catch (const std::exception& error) {
            state->abort_codec(error.what());
            if (is_tile_overflow_abort(error) && g_sync_generation) {
                const auto decision = g_sync_generation->wait_decision(
                    rank, generation);
                if (decision.aborted && !decision.fatal)
                    return aborted_exchange_report(decision);
            }
            throw;
        } catch (...) {
            state->abort_codec("unknown hierarchical sparse rank failure");
            throw;
        }

        py::dict result;
        result["abi"] = "ftgspp-hierarchical-host-v1";
        result["transport"] = "hierarchical_host";
        result["operation"] = "gradient_mean";
        result["algorithm"] =
            HierarchicalHostCollective::algorithm_name(
                state->selected_algorithm);
        result["execution_backend"] = "native_cpu_fixed_rank_reduce";
        result["sparse"] = true;
        result["gaussian_codec"] = hierarchical_gaussian_codec_;
        result["color_codec"] = "dense";
        result["local_visible_count"] = local_count;
        result["union_visible_count"] = pending_peer_sparse_count;
        result["pack_seconds"] = pack_seconds;
        result["cpu_reduction_seconds"] = sparse_reduce_seconds + float(
            aux_report.contains("cpu_reduction_seconds")
                ? aux_report["cpu_reduction_seconds"].cast<double>() : 0.0);
        result["download_seconds"] = sparse_download_seconds + float(
            aux_report.contains("download_seconds")
                ? aux_report["download_seconds"].cast<double>() : 0.0);
        result["upload_seconds"] = sparse_upload_seconds + float(
            aux_report.contains("upload_seconds")
                ? aux_report["upload_seconds"].cast<double>() : 0.0);
        result["host_memcpy_seconds"] = sparse_host_memcpy_seconds + float(
            aux_report.contains("host_memcpy_seconds")
                ? aux_report["host_memcpy_seconds"].cast<double>() : 0.0);
        result["wall_seconds"] = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall_begin).count();
        result["payload_bytes"] = local_bytes;
        result["dense_payload_bytes"] = dense_bytes;
        result["pcie_d2h_bytes"] = local_bytes;
        result["pcie_h2d_bytes"] =
            size_t(pending_peer_sparse_count) * record_bytes;
        result["python_gradient_payload"] = false;
        result["finite_checked"] = true;
        result["codec_policy"] = hierarchical_codec_policy_;
        result["async_pipeline"] = true;
        return result;
    }

    py::dict exchange_hierarchical_gradient(
        uint32_t rank, uint64_t generation) {
        require_finalized_local_gradient("hierarchical exchange");
        const auto harvest = harvest_sync_generation(
            rank, generation, /*wait_all_ranks=*/true);
        if (harvest.aborted)
            return aborted_exchange_report(harvest);
        hierarchical_sparse_mean_ready_ = false;
        hierarchical_needs_regularizer_ = false;
        if (config.sparse_distributed)
            return exchange_hierarchical_sparse_gradient(rank, generation);
        return exchange_hierarchical_entries(
            rank, generation, mailbox_dense_entries(false), false);
    }

    py::dict exchange_hierarchical_relocation(
        uint32_t rank, uint64_t generation) {
        trainer.waitAllComputeSlots("relocation");
        trainer.drainPackedTransfers();
        const size_t count = 2 * buffers.num_splats;
        if (count == 0 || buffers.relocation_stats.deviceSize() != count)
            throw std::runtime_error(
                "hierarchical relocation requires initialized (N,2) statistics");
        const std::vector<PackedGradientEntry> entries{{
            "relocation_stats", &buffers.relocation_stats, 0u, count}};
        return exchange_hierarchical_entries(
            rank, generation, entries, true);
    }

    py::dict optimizer_step_hierarchical(
        int step, const std::vector<float>& timestamps) {
        if (config.sparse_distributed || hierarchical_needs_regularizer_) {
            if (timestamps.empty() || timestamps.size() > 32u)
                throw std::invalid_argument(
                    "hierarchical sparse optimizer requires 1..32 PK timestamps");
        }
        const auto begin = std::chrono::steady_clock::now();
        double reduction_seconds = 0.0;
        double adam_seconds = 0.0;
        {
            py::gil_scoped_release release;
            const auto reduction_begin = std::chrono::steady_clock::now();
            if (config.sparse_distributed) {
                auto device_guard = DeviceGuard(&trainer);
                if (hierarchical_sparse_mean_ready_) {
                    trainer.executeFTGSPPSparseGradientMerge(
                        config, buffers, pending_peer_sparse_count,
                        /*world_size=*/1u,
                        timestamps.data(),
                        static_cast<uint32_t>(timestamps.size()),
                        /*peer_already_meaned=*/true,
                        uniforms.active_sh);
                } else {
                    // Dense gaussian fallback already meaned image grads;
                    // only the deferred analytic regularizer remains.
                    trainer.executeFTGSPPRegularizerMean(
                        config, buffers, timestamps.data(),
                        static_cast<uint32_t>(timestamps.size()));
                }
            }
            reduction_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - reduction_begin).count();
            const auto adam_begin = std::chrono::steady_clock::now();
            optimizer_step(step);
            adam_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - adam_begin).count();
        }
        hierarchical_sparse_mean_ready_ = false;
        hierarchical_needs_regularizer_ = false;
        pending_peer_sparse_count = 0;
        py::dict result;
        result["gpu_reduction_seconds"] = reduction_seconds;
        result["optimizer_seconds"] = adam_seconds;
        result["execution_backend"] = "native_cpu_fixed_rank_reduce";
        result["gaussian_codec"] = hierarchical_gaussian_codec_;
        result["wall_seconds"] = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();
        return result;
    }

    void abort_hierarchical_host_collective(const std::string& reason) {
        std::shared_ptr<HierarchicalHostBindingState> state;
        {
            std::lock_guard<std::mutex> lock(g_hierarchical_host_mutex);
            state = g_hierarchical_host_state;
        }
        if (!state)
            return;
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->gradient)
            state->gradient->abort(reason);
        if (state->gradient_sparse)
            state->gradient_sparse->abort(reason);
        if (state->relocation)
            state->relocation->abort(reason);
    }

    void reset_hierarchical_host_attachments() {
        std::shared_ptr<HierarchicalHostBindingState> state;
        {
            std::lock_guard<std::mutex> lock(g_hierarchical_host_mutex);
            state = g_hierarchical_host_state;
        }
        if (state) {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (hierarchical_gradient_attached_ && state->gradient)
                state->gradient->detach(
                    static_cast<uint32_t>(hierarchical_gradient_rank));
            if (hierarchical_sparse_attached_ && state->gradient_sparse)
                state->gradient_sparse->detach(
                    static_cast<uint32_t>(hierarchical_gradient_rank));
            if (hierarchical_relocation_attached_ && state->relocation)
                state->relocation->detach(
                    static_cast<uint32_t>(hierarchical_relocation_rank));
        }
        hierarchical_gradient_attached_ = false;
        hierarchical_sparse_attached_ = false;
        hierarchical_relocation_attached_ = false;
        hierarchical_sparse_mean_ready_ = false;
        hierarchical_needs_regularizer_ = false;
        hierarchical_gradient_rank = -1;
        hierarchical_relocation_rank = -1;
    }

    py::dict packed_gradient_layout() {
        const auto entries = packed_gradient_entries(false);
        py::dict result;
        result["abi_version"] = 2;
        result["sh_layout"] = "native_reordered_padded_float4";
        result["sh_padding"] = "canonical_zero";
        result["sh_reorder_size"] = SH_REORDER_SIZE;
        size_t total = 0;
        for (const auto& entry : entries) {
            result[py::str(entry.name)] = py::make_tuple(entry.offset, entry.count);
            total = entry.offset + entry.count;
        }
        result["total_floats"] = total;
        return result;
    }

    size_t packed_gradient_size() {
        const auto entries = packed_gradient_entries(false);
        const auto& last = entries.back();
        return last.offset + last.count;
    }

    void gradient_state_packed_into(
        py::array_t<float, py::array::c_style> output
    ) {
        auto data = output.request();
        const auto entries = packed_gradient_entries();
        const size_t expected = entries.back().offset + entries.back().count;
        if (data.ndim != 1 || (size_t)data.size != expected)
            throw std::runtime_error(
                "packed gradient output must be one-dimensional with " +
                std::to_string(expected) + " float32 values");
        if (!output.writeable())
            throw std::runtime_error("packed gradient output must be writable");
        std::vector<VulkanGSPipeline::PackedBufferRegion> regions;
        regions.reserve(entries.size());
        for (const auto& entry : entries)
            regions.push_back({entry.buffer->deviceBuffer, entry.offset*sizeof(float)});
        {
            py::gil_scoped_release release;
            trainer.copyBuffersFromDevicePacked(
                regions, data.ptr, expected*sizeof(float));
            // Backward dispatches only idx<n. The reordered SH buffer rounds N
            // to SH_REORDER_SIZE, so at most 15 lanes in each of the final 12
            // float4 coefficient rows are otherwise undefined. Canonicalize
            // only those <=180 float4 lanes on the host; clearing all 48*N
            // values on-device every step would add a large redundant pass.
            const size_t padded_n = _CEIL_ROUND(
                buffers.num_splats, SH_REORDER_SIZE);
            if (padded_n != buffers.num_splats) {
                const size_t block_base = padded_n - SH_REORDER_SIZE;
                const size_t valid_lanes = buffers.num_splats - block_base;
                auto sh_entry = std::find_if(
                    entries.begin(), entries.end(),
                    [](const PackedGradientEntry& entry) {
                        return std::strcmp(
                            entry.name, "grad_sh_coeffs_reordered") == 0;
                    });
                if (sh_entry == entries.end())
                    throw std::runtime_error(
                        "packed gradient ABI has no reordered SH entry");
                float* sh = static_cast<float*>(data.ptr) + sh_entry->offset;
                for (size_t coefficient = 0; coefficient < 12; ++coefficient) {
                    for (size_t lane = valid_lanes;
                         lane < SH_REORDER_SIZE; ++lane) {
                        const size_t float4_index =
                            12*block_base + coefficient*SH_REORDER_SIZE + lane;
                        std::fill_n(sh + 4*float4_index, 4, 0.0f);
                    }
                }
            }
        }
    }

    void set_gradient_state_packed(
        py::array_t<float, py::array::c_style> state,
        bool validate_finite
    ) {
        auto data = state.request();
        const auto entries = packed_gradient_entries(false);
        const size_t expected = entries.back().offset + entries.back().count;
        if (data.ndim != 1 || (size_t)data.size != expected)
            throw std::runtime_error(
                "packed gradient input must be one-dimensional with " +
                std::to_string(expected) + " float32 values");
        const float* values = static_cast<const float*>(data.ptr);
        if (validate_finite) {
            for (size_t index = 0; index < expected; ++index) {
                if (!std::isfinite(values[index]))
                    throw std::runtime_error(
                        "packed gradient input contains a non-finite value at index " +
                        std::to_string(index));
            }
        }
        std::vector<VulkanGSPipeline::PackedBufferRegion> regions;
        regions.reserve(entries.size());
        for (const auto& entry : entries)
            regions.push_back({entry.buffer->deviceBuffer, entry.offset*sizeof(float)});
        {
            py::gil_scoped_release release;
            trainer.copyBuffersToDevicePacked(
                values, expected*sizeof(float), regions);
        }
        gradient_ready = true;
        gradient_finalized = true;
        local_gradient_microsteps = 1;
    }

    void mean_packed_gradients_into(
        py::sequence per_rank,
        py::array_t<float, py::array::c_style> output
    ) {
        const size_t ranks = (size_t)py::len(per_rank);
        if (ranks == 0)
            throw std::runtime_error("packed FP32 mean requires at least one rank");
        std::vector<py::array_t<float, py::array::c_style>> owners;
        std::vector<const float*> inputs;
        owners.reserve(ranks);
        inputs.reserve(ranks);
        size_t count = 0;
        for (py::handle item : per_rank) {
            auto array = py::cast<py::array_t<float, py::array::c_style>>(item);
            auto data = array.request();
            if (data.ndim != 1)
                throw std::runtime_error("packed rank gradients must be one-dimensional");
            if (owners.empty())
                count = (size_t)data.size;
            else if ((size_t)data.size != count)
                throw std::runtime_error("packed rank gradient sizes differ");
            inputs.push_back(static_cast<const float*>(data.ptr));
            owners.push_back(std::move(array));
        }
        if (count == 0)
            throw std::runtime_error("packed FP32 mean does not accept empty arenas");
        auto out = output.request();
        if (out.ndim != 1 || (size_t)out.size != count)
            throw std::runtime_error(
                "packed mean output must match the rank gradient size");
        if (!output.writeable())
            throw std::runtime_error("packed mean output must be writable");
        float* destination = static_cast<float*>(out.ptr);
        const float inverse = 1.0f / static_cast<float>(ranks);
        std::atomic<bool> invalid{false};
        // Exact aliasing with an input (the coordinator uses input 0) is part
        // of the ABI: every source at one index is read before that destination
        // index is written, and worker ranges never overlap.
        auto reduce_range = [&](size_t begin, size_t end) {
            for (size_t index = begin; index < end; ++index) {
                float sum = 0.0f;
                for (size_t rank = 0; rank < ranks; ++rank) {
                    const float value = inputs[rank][index];
                    if (!std::isfinite(value))
                        invalid.store(true, std::memory_order_relaxed);
                    // The loop order is the public deterministic FP32 contract.
                    sum = sum + value;
                }
                const float mean = sum * inverse;
                if (!std::isfinite(mean))
                    invalid.store(true, std::memory_order_relaxed);
                destination[index] = mean;
            }
        };
        {
            py::gil_scoped_release release;
            const size_t minimum_chunk = 1u << 18;
            const size_t available = std::max<size_t>(
                1, (size_t)std::thread::hardware_concurrency());
            const size_t workers = std::min<size_t>(
                16, std::min(available, std::max<size_t>(1, count/minimum_chunk)));
            std::vector<std::thread> threads;
            threads.reserve(workers > 0 ? workers - 1 : 0);
            const size_t chunk = (count + workers - 1) / workers;
            try {
                for (size_t worker = 1; worker < workers; ++worker) {
                    const size_t begin = worker*chunk;
                    const size_t end = std::min(count, begin+chunk);
                    if (begin < end)
                        threads.emplace_back(reduce_range, begin, end);
                }
            }
            catch (...) {
                for (auto& thread : threads)
                    if (thread.joinable())
                        thread.join();
                throw;
            }
            reduce_range(0, std::min(count, chunk));
            for (auto& thread : threads)
                thread.join();
        }
        if (invalid.load(std::memory_order_relaxed))
            throw std::runtime_error(
                "packed FP32 reduction encountered non-finite input or overflow");
    }

    py::dict last_packed_download_timing() const {
        const auto timing = trainer.lastPackedDownloadTiming();
        py::dict result;
        result["gpu_copy_seconds"] = timing.gpu_copy_seconds;
        result["host_memcpy_seconds"] = timing.host_memcpy_seconds;
        result["map_seconds"] = timing.map_seconds;
        result["invalidate_seconds"] = timing.invalidate_seconds;
        result["used_cached_memory"] = timing.used_cached_memory;
        result["used_persistent_map"] = timing.used_persistent_map;
        return result;
    }

    py::dict last_packed_upload_timing() const {
        const auto timing = trainer.lastPackedUploadTiming();
        py::dict result;
        result["gpu_copy_seconds"] = timing.gpu_copy_seconds;
        result["host_memcpy_seconds"] = timing.host_memcpy_seconds;
        result["map_seconds"] = timing.map_seconds;
        result["invalidate_seconds"] = timing.invalidate_seconds;
        result["used_cached_memory"] = timing.used_cached_memory;
        result["used_persistent_map"] = timing.used_persistent_map;
        return result;
    }

    auto device_info() const {
        return trainer.get_device_info();
    }

    std::string native_abi() const {
        return "ftgspp-vulkan-community-explicit-v5-abi-v1";
    }

    void release_dynamic_view_image(size_t view_idx, bool is_train) {
        auto& target = is_train ? trainer.get_train_image(view_idx) : trainer.get_val_image(view_idx);
        py::gil_scoped_release release;
        trainer.destroyBuffer(target.buffer.deviceBuffer);
        target.buffer.deviceBuffer = _VulkanBuffer();
        target.buffer.clear();
        target.buffer.shrink_to_fit();
    }

    void set_dynamic_view_cache_on_device(
        size_t view_idx, bool is_train, bool enabled
    ) {
        auto& target = is_train
            ? trainer.get_train_image(view_idx)
            : trainer.get_val_image(view_idx);
        target.cache_on_device = enabled;
    }

    void resolve_tile_overflow(bool force_drain = false) {
        if (resolving_tile_overflow_)
            return;
        resolving_tile_overflow_ = true;
        for (int attempt = 0; attempt < 4; ++attempt) {
            if (force_drain || attempt > 0)
                trainer.drain(attempt == 0 ? "tile_overflow_force" : "tile_overflow_replay");
            else
                trainer.harvestNextComputeSlot();
            const auto retired = trainer.collectRetiredTileOverflow();
            if (retired.overflow) {
                trainer.drain("tile_overflow_grow");
                const size_t next = ftgspp::growTilePairCapacity(
                    buffers.tile_pair_capacity, retired.exact_count);
                buffers.tile_pair_capacity = next;
                trainer.telemetry().tile_overflow = 1;
                trainer.clearDeviceBuffer(buffers.pipeline_abort_latch, 4);
                trainer.clearRetiredTileOverflow();

                std::vector<PendingTileStep> replay;
                if (retired.is_train && retired.step >= 0) {
                    for (size_t i = 0; i < pending_tile_count_; ++i) {
                        const PendingTileStep& pending = pending_tile_ring_[i];
                        if (pending.valid && pending.is_train_view &&
                            pending.step >= (int)retired.step)
                            replay.push_back(pending);
                    }
                }
                pending_tile_count_ = 0;
                for (const PendingTileStep& pending : replay) {
                    trainer.telemetry().tile_replay += 1;
                    trainer.telemetry().step = (uint64_t)std::max(pending.step, 0);
                    trainer.setSubmissionMeta(pending.step, true);
                    auto deviceGuard = DeviceGuard(&trainer);
                    train_dynamic_view_to_loss(
                        pending.view_idx, pending.step, pending.collect_loss);
                    train_dynamic_view_from_loss(pending.view_idx, pending.step);
                    record_pending_tile_step(pending);
                }
                continue;
            }
            const uint32_t exact = trainer.telemetry().exact_tile_count;
            const size_t cap = buffers.tile_pair_capacity;
            if (cap > 0 && exact > (uint32_t)(cap * 4 / 5)) {
                trainer.drain("tile_capacity_headroom");
                buffers.tile_pair_capacity = ftgspp::growTilePairCapacity(
                    cap, exact);
                trainer.clearDeviceBuffer(buffers.pipeline_abort_latch, 4);
                trainer.clearRetiredTileOverflow();
            }
            break;
        }
        resolving_tile_overflow_ = false;
    }

    std::map<std::string, double> get_pipeline_telemetry() {
        return trainer.snapshotTelemetry().as_map();
    }

    void drain_eval() {
        if (eval_rgb_ticket_.has_value()) {
            trainer.completeBuffersFromDevicePackedRangeMapped(*eval_rgb_ticket_);
            eval_rgb_ticket_.reset();
        }
        // Eval needs the last Adam generation, not a full overflow replay of
        // pending train steps. Train overflow is a collective concern.
        trainer.waitAllComputeSlots("eval");
    }

    void drain_for_structure_change() {
        trainer.drain("relocation");
    }

    void apply_sync_tile_capacity(uint32_t exact_count) {
        trainer.drain("sync_tile_overflow");
        if (exact_count > 0) {
            buffers.tile_pair_capacity = ftgspp::growTilePairCapacity(
                buffers.tile_pair_capacity, exact_count);
        }
        trainer.clearDeviceBuffer(buffers.pipeline_abort_latch, 4);
        trainer.clearRetiredTileOverflow();
        const size_t splat_count = buffers.num_splats;
        if (relocation_stats_checkpoint_valid_ &&
            buffers.relocation_stats_generation.deviceSize() == 2 * splat_count) {
            trainer.copyFromDeviceToDevice(
                buffers.relocation_stats_generation, buffers.relocation_stats);
        } else if (
            !relocation_stats_checkpoint_valid_ &&
            buffers.relocation_stats_generation.deviceSize() == 0 &&
            buffers.relocation_stats.deviceSize() == 2 * splat_count) {
            trainer.clearDeviceBuffer(buffers.relocation_stats, 2 * splat_count);
        }
        relocation_stats_checkpoint_valid_ = false;
        gradient_ready = false;
        gradient_finalized = false;
        gradient_step = -1;
        local_gradient_microsteps = 0;
        pending_tile_count_ = 0;
        trainer.telemetry().tile_overflow = 1;
        trainer.telemetry().tile_replay += 1;
    }

    void inject_sync_tile_overflow(uint32_t exact_count) {
        force_sync_overflow_ = true;
        force_sync_overflow_exact_ = std::max<uint32_t>(exact_count, 1u);
    }

    static void clear_sync_collectives_for_replay_py() {
        clear_sync_collectives_for_replay();
    }

    void begin_eval_val_view(size_t view_idx) {
        if (view_idx >= trainer.num_val())
            throw std::out_of_range("dynamic val view index is out of range");
        if (eval_rgb_ticket_.has_value())
            throw std::runtime_error(
                "complete_eval_rgb must finish the previous eval readback "
                "before beginning another eval view");
        last_eval_view_idx_ = view_idx;
        trainer.get_val_camera(view_idx, uniforms);
        uniforms.step = 0;
        uniforms.active_sh = config.max_sh_degree;
        trainer.setSubmissionMeta(-1, false);
        dynamic_forward(trainer.get_val_timestamp(view_idx));
    }

    void submit_eval_rgb_readback() {
        if (eval_rgb_ticket_.has_value())
            throw std::runtime_error("eval RGB readback is already in flight");
        const auto& pixels = buffers.pixel_state.deviceBuffer;
        if (pixels.buffer == VK_NULL_HANDLE || pixels.size == 0)
            throw std::runtime_error("eval RGB readback requires a rendered pixel_state");
        eval_rgb_ticket_ = trainer.submitComputeReadback(pixels, pixels.size);
    }

    void grow_eval_tile_capacity(uint32_t exact_count) {
        trainer.drain("eval_tile_overflow_grow");
        buffers.tile_pair_capacity = ftgspp::growTilePairCapacity(
            buffers.tile_pair_capacity, exact_count);
        trainer.clearDeviceBuffer(buffers.pipeline_abort_latch, 4);
        trainer.clearRetiredTileOverflow();
    }

    void rerender_last_eval_view() {
        if (last_eval_view_idx_ >= trainer.num_val())
            throw std::out_of_range("dynamic val view index is out of range");
        trainer.get_val_camera(last_eval_view_idx_, uniforms);
        uniforms.step = 0;
        uniforms.active_sh = config.max_sh_degree;
        trainer.setSubmissionMeta(-1, false);
        dynamic_forward(trainer.get_val_timestamp(last_eval_view_idx_));
        const auto& pixels = buffers.pixel_state.deviceBuffer;
        if (pixels.buffer == VK_NULL_HANDLE || pixels.size == 0)
            throw std::runtime_error("eval RGB readback requires a rendered pixel_state");
        eval_rgb_ticket_ = trainer.submitComputeReadback(pixels, pixels.size);
    }

    py::array_t<float> complete_eval_rgb() {
        if (!eval_rgb_ticket_.has_value())
            throw std::runtime_error("complete_eval_rgb has no in-flight readback");
        const size_t height = uniforms.image_height;
        const size_t width = uniforms.image_width;
        const size_t pixels = height * width;
        if (height == 0 || width == 0)
            throw std::runtime_error("complete_eval_rgb requires a rendered view");
        const size_t expected_bytes = 4 * pixels * sizeof(float);
        py::array_t<float> result({
            (py::ssize_t)height, (py::ssize_t)width, (py::ssize_t)3});
        float* out = result.mutable_data();
        {
            py::gil_scoped_release release;
            const float* src = nullptr;
            for (int attempt = 0; attempt < 4; ++attempt) {
                if (!eval_rgb_ticket_.has_value())
                    throw std::runtime_error("complete_eval_rgb has no in-flight readback");
                const auto ticket = *eval_rgb_ticket_;
                eval_rgb_ticket_.reset();
                const auto mapped =
                    trainer.completeBuffersFromDevicePackedRangeMapped(ticket);
                trainer.harvestNextComputeSlot();
                const auto retired = trainer.collectLastHarvestedOverflow();
                if (retired.overflow && attempt + 1 < 4) {
                    grow_eval_tile_capacity(retired.exact_count);
                    rerender_last_eval_view();
                    continue;
                }
                if (retired.overflow)
                    throw std::runtime_error(
                        "eval tile overflow persisted after capacity retries");
                if (ticket.range_size < expected_bytes)
                    throw std::runtime_error("eval RGB readback is smaller than HxWx4");
                src = static_cast<const float*>(mapped.data);
                break;
            }
            if (src == nullptr)
                throw std::runtime_error("complete_eval_rgb failed to map RGB");
            for (size_t i = 0; i < pixels; ++i) {
                const float* rgba = src + 4 * i;
                float* rgb = out + 3 * i;
                rgb[0] = rgba[0];
                rgb[1] = rgba[1];
                rgb[2] = rgba[2];
            }
        }
        return result;
    }

    void render_dynamic(float timestamp) {
        resolve_tile_overflow(true);
        uniforms.active_sh = config.max_sh_degree;
        trainer.setSubmissionMeta(-1, false);
        auto deviceGuard = DeviceGuard(&trainer);
        dynamic_forward(timestamp);
    }

    void render_flow_dynamic(
        float timestamp,
        py::array_t<float, py::array::c_style | py::array::forcecast> end_means
    ) {
        auto end = end_means.request();
        if (end.ndim != 2 || (size_t)end.shape[0] != buffers.num_splats || end.shape[1] != 3)
            throw std::runtime_error("end_means must have shape (N, 3)");
        const float* ptr = static_cast<float*>(end.ptr);
        buffers.flow_end_xyz.assign(ptr, ptr + end.size);
        trainer.copyToDevice(buffers.flow_end_xyz);

        auto deviceGuard = DeviceGuard(&trainer);
        trainer.executeFTGSPPMaterialize(config, buffers, timestamp);
        projection_forward();
        trainer.executeFTGSPPFlowFeatures(uniforms, buffers);
        process_tiles();
        rasterize_forward();
    }

    void materialize_dynamic(float timestamp) {
        auto deviceGuard = DeviceGuard(&trainer);
        trainer.executeFTGSPPMaterialize(config, buffers, timestamp);
    }

    void render_flow_dynamic_times(float timestamp, float end_timestamp) {
        // Diagnostic-only model-flow rendering; RGB training/eval never calls it.
        uniforms.active_sh = config.max_sh_degree;
        auto deviceGuard = DeviceGuard(&trainer);
        trainer.executeFTGSPPMaterialize(config, buffers, end_timestamp);
        trainer.copyFromDeviceToDevice(buffers.xyz_ws, buffers.flow_end_xyz);
        trainer.executeFTGSPPMaterialize(config, buffers, timestamp);
        projection_forward();
        trainer.executeFTGSPPFlowFeatures(uniforms, buffers);
        process_tiles();
        rasterize_forward();
    }

    void render_flow_dynamic_train_view(size_t view_idx, float end_timestamp) {
        if (view_idx >= trainer.num_train())
            throw std::out_of_range("dynamic train view index is out of range");
        trainer.get_train_camera(view_idx, uniforms);
        render_flow_dynamic_times(trainer.get_train_timestamp(view_idx), end_timestamp);
    }

    void render_flow_dynamic_val_view(size_t view_idx, float end_timestamp) {
        if (view_idx >= trainer.num_val())
            throw std::out_of_range("dynamic val view index is out of range");
        trainer.get_val_camera(view_idx, uniforms);
        render_flow_dynamic_times(trainer.get_val_timestamp(view_idx), end_timestamp);
    }

    bool color_correction_active(int step) const {
        return config.color_correction && config.color_num_cameras > 0
            && (config.hard_separation || step > config.color_correction_start);
    }

    void train_dynamic_view_to_loss(
        size_t view_idx, int step, bool collect_loss_scalar=true
    ) {
        if (view_idx >= trainer.num_train())
            throw std::out_of_range("dynamic train view index is out of range");
        if (gradient_ready && (
            gradient_finalized || gradient_step != step))
            throw std::runtime_error(
                "train_dynamic_view_to_loss cannot overwrite pending gradients "
                "from a different/finalized generation");
        // Reject before recording a partial forward/loss command sequence.
        // The shared SSIM helper intentionally remains usable by pure
        // VkSplat, but every FTGSPP training step needs native float atomics.
        trainer.requireFTGSPPFloat32AtomicAdd();
        pending_collect_loss_scalar_ = collect_loss_scalar;
        const float timestamp = trainer.get_train_timestamp(view_idx);
        trainer.get_train_camera(view_idx, uniforms);
        uniforms.step = step;
        uniforms.active_sh = config.hard_separation
            ? config.max_sh_degree : std::min(step / 1000, config.max_sh_degree);
        auto deviceGuard = DeviceGuard(&trainer);
        dynamic_forward(timestamp);
        const bool corrected_for_loss = color_correction_active(step);
        if (corrected_for_loss) {
            const uint32_t cam = (uint32_t)std::max(
                trainer.get_train_image(view_idx).camera_id, 0);
            trainer.executeFTGSPPColorCorrectForwardWH(
                config, buffers, uniforms.image_width, uniforms.image_height, cam);
        }
        // Without color correction the loss owns the renderer [0,1] clamp.
        // With correction, its forward shader already clamps the renderer
        // input and the affine output remains unbounded, as in FTGSPP.
        trainer.executeComputeSSIMGradient(
            config, uniforms, buffers, view_idx,
            corrected_for_loss ? 0u : 2u,
            collect_loss_scalar);
    }

    void accumulate_pixel_state_grad(py::array_t<float> extra_grad) {
        auto buf = extra_grad.request();
        if (buf.ndim != 3 ||
            buf.shape[0] != uniforms.image_height ||
            buf.shape[1] != uniforms.image_width ||
            buf.shape[2] != 4)
            throw std::runtime_error("extra pixel grad must be (H, W, 4) array");
        const size_t n = (size_t)buf.size;
        const float* src = static_cast<const float*>(buf.ptr);
        {
            py::gil_scoped_release release;
            trainer.copyFromDevice(buffers.v_pixel_state);
            if (buffers.v_pixel_state.size() != n)
                throw std::runtime_error(
                    "v_pixel_state size mismatch for accumulate");
            for (size_t i = 0; i < n; ++i)
                buffers.v_pixel_state[i] += src[i];
            trainer.copyToDevice(buffers.v_pixel_state);
        }
    }

    void backward_dynamic_view_from_loss_raw(size_t view_idx, int step) {
        if (view_idx >= trainer.num_train())
            throw std::out_of_range("dynamic train view index is out of range");
        if (gradient_ready && (
            gradient_finalized || gradient_step != step))
            throw std::runtime_error(
                "backward_dynamic_view_from_loss_raw cannot overwrite pending "
                "gradients from a different/finalized generation");
        trainer.requireFTGSPPFloat32AtomicAdd();
        const float timestamp = trainer.get_train_timestamp(view_idx);
        if (!gradient_ready) {
            const size_t n = buffers.num_splats;
            if (buffers.relocation_stats.deviceSize() == 2 * n)
                trainer.resizeDeviceBuffer(
                    buffers.relocation_stats_generation, 2 * n);
        }
        {
            auto deviceGuard = DeviceGuard(&trainer);
            if (!gradient_ready) {
                const size_t n = buffers.num_splats;
                if (buffers.relocation_stats.deviceSize() == 2 * n) {
                    trainer.copyFromDeviceToDevice(
                        buffers.relocation_stats,
                        buffers.relocation_stats_generation);
                    relocation_stats_checkpoint_valid_ = true;
                } else {
                    relocation_stats_checkpoint_valid_ = false;
                }
            }
            if (config.sparse_distributed && !gradient_ready)
                trainer.resetFTGSPPSparseActiveFlags(buffers);
            if (color_correction_active(step)) {
                const uint32_t cam = (uint32_t)std::max(
                    trainer.get_train_image(view_idx).camera_id, 0);
                trainer.executeFTGSPPColorCorrectBackwardRawWH(
                    config, buffers, uniforms.image_width, uniforms.image_height, cam);
            }
            rasterize_backward();
            trainer.executeFTGSPPBackward(
                config, uniforms, buffers, step + 1, timestamp,
                pending_collect_loss_scalar_);
            if (config.sparse_distributed)
                trainer.accumulateFTGSPPSparseActiveFlags(buffers);
        }
        if (!gradient_ready) {
            gradient_ready = true;
            gradient_finalized = false;
            gradient_step = step;
            local_gradient_microsteps = 0;
        }
        accumulate_current_raw_gradient();
    }

    void set_collect_geometry_grad_stats(bool enabled) {
        collect_geometry_grad_stats_ = enabled;
        if (!enabled) {
            geometry_grad_stats_ready_ = false;
            geometry_grad_num_splats_ = 0;
            for (int i = 0; i < 6; ++i)
                geometry_grad_sum_sq_[i] = 0.f;
        }
    }

    void cache_geometry_grad_stats_from_device() {
        if (!collect_geometry_grad_stats_) {
            geometry_grad_stats_ready_ = false;
            return;
        }
        if (buffers.grad_stats.deviceBuffer.size != 6 * sizeof(float))
            throw std::runtime_error(
                "geometry gradient stats buffer was not produced by Adam");
        trainer.copyFromDevice(buffers.grad_stats);
        for (int i = 0; i < 6; ++i)
            geometry_grad_sum_sq_[i] = buffers.grad_stats[i];
        geometry_grad_num_splats_ = buffers.num_splats;
        geometry_grad_stats_ready_ = true;
    }

    py::dict get_geometry_gradient_stats() {
        if (!geometry_grad_stats_ready_)
            throw std::runtime_error(
                "geometry gradient stats are unavailable; enable collection "
                "before optimizer_step on a log step");
        const double n = std::max<size_t>(geometry_grad_num_splats_, 1);
        auto rms = [&](double sum_sq, double elems_per_splat) {
            return std::sqrt(sum_sq / (n * elems_per_splat));
        };
        py::dict result;
        result["grad_xyz_rms"] = rms(geometry_grad_sum_sq_[0], 3.0);
        result["grad_scaling_rms"] = rms(geometry_grad_sum_sq_[1], 3.0);
        result["grad_oc_rms"] = rms(geometry_grad_sum_sq_[2], 1.0);
        result["grad_rotation_rms"] = rms(geometry_grad_sum_sq_[3], 4.0);
        // SH packs 16 bands x 3 channels = 48 floats / splat; DC is RGB.
        result["grad_sh_dc_rms"] = rms(geometry_grad_sum_sq_[4], 3.0);
        result["grad_sh_rest_rms"] = rms(geometry_grad_sum_sq_[5], 45.0);
        result["grad_xyz_l2"] = std::sqrt(
            std::max(geometry_grad_sum_sq_[0], 0.f));
        result["grad_scaling_l2"] = std::sqrt(
            std::max(geometry_grad_sum_sq_[1], 0.f));
        result["grad_oc_l2"] = std::sqrt(
            std::max(geometry_grad_sum_sq_[2], 0.f));
        result["grad_rotation_l2"] = std::sqrt(
            std::max(geometry_grad_sum_sq_[3], 0.f));
        result["grad_sh_dc_l2"] = std::sqrt(
            std::max(geometry_grad_sum_sq_[4], 0.f));
        result["grad_sh_rest_l2"] = std::sqrt(
            std::max(geometry_grad_sum_sq_[5], 0.f));
        result["grad_num_splats"] = (double)geometry_grad_num_splats_;
        return result;
    }

    void optimizer_step(int step) {
        require_finalized_local_gradient("optimizer_step");
        trainer.requireFTGSPPFloat32AtomicAdd();
        {
            auto deviceGuard = DeviceGuard(&trainer);
            trainer.executeFTGSPPExplicitAdam(
                config, uniforms, buffers, step + 1,
                collect_geometry_grad_stats_);
            if (color_correction_active(step))
                trainer.executeFTGSPPColorAdam(
                    config, buffers, step + 1,
                    pending_collect_loss_scalar_);
        }
        // Readback must happen after DeviceGuard submits the Adam batch.
        if (collect_geometry_grad_stats_)
            cache_geometry_grad_stats_from_device();
        gradient_ready = false;
        gradient_finalized = false;
        gradient_step = -1;
        local_gradient_microsteps = 0;
    }

    void backward_dynamic_view(
        size_t view_idx, int step, bool collect_loss_scalar=true
    ) {
        // A complete distributed backward boundary: this method may allocate
        // raw-gradient storage, but it never mutates Gaussian or color
        // parameters/Adam moments. Overflow harvest lives at the collective
        // D2H boundary so this entry does not wait the previous generation.
        if (gradient_ready && (
            gradient_finalized || gradient_step != step))
            throw std::runtime_error(
                "backward_dynamic_view cannot overwrite pending gradients "
                "from a different/finalized generation");
        if (!gradient_ready) {
            trainer.telemetry().reset_step();
            trainer.telemetry().step = (uint64_t)std::max(step, 0);
        }
        trainer.setSubmissionMeta(step, true);
        train_dynamic_view_to_loss(view_idx, step, collect_loss_scalar);
        backward_dynamic_view_from_loss_raw(view_idx, step);
        PendingTileStep pending;
        pending.valid = true;
        pending.is_train_view = true;
        pending.view_idx = view_idx;
        pending.step = step;
        pending.collect_loss = collect_loss_scalar;
        record_pending_tile_step(pending);
    }

    void train_dynamic_view_from_loss(size_t view_idx, int step) {
        if (view_idx >= trainer.num_train())
            throw std::out_of_range("dynamic train view index is out of range");
        trainer.requireFTGSPPFloat32AtomicAdd();
        const float timestamp = trainer.get_train_timestamp(view_idx);
        {
            auto deviceGuard = DeviceGuard(&trainer);
            if (color_correction_active(step)) {
                const uint32_t cam = (uint32_t)std::max(
                    trainer.get_train_image(view_idx).camera_id, 0);
                trainer.executeFTGSPPColorCorrectBackwardRawWH(
                    config, buffers, uniforms.image_width,
                    uniforms.image_height, cam);
            }
            rasterize_backward();
            trainer.executeFTGSPPFusedDynamicBackwardAdam(
                config, uniforms, buffers, step + 1, timestamp,
                pending_collect_loss_scalar_, collect_geometry_grad_stats_);
            if (color_correction_active(step))
                trainer.executeFTGSPPColorAdam(
                    config, buffers, step + 1,
                    pending_collect_loss_scalar_);
        }
        if (collect_geometry_grad_stats_)
            cache_geometry_grad_stats_from_device();
    }

    void train_dynamic_view(
        size_t view_idx, int step, bool collect_loss_scalar=true
    ) {
        trainer.telemetry().reset_step();
        trainer.telemetry().step = (uint64_t)std::max(step, 0);
        resolve_tile_overflow(false);
        trainer.telemetry().step = (uint64_t)std::max(step, 0);
        trainer.setSubmissionMeta(step, true);
        auto deviceGuard = DeviceGuard(&trainer);
        train_dynamic_view_to_loss(view_idx, step, collect_loss_scalar);
        train_dynamic_view_from_loss(view_idx, step);
        PendingTileStep pending;
        pending.valid = true;
        pending.is_train_view = true;
        pending.view_idx = view_idx;
        pending.step = step;
        pending.collect_loss = collect_loss_scalar;
        record_pending_tile_step(pending);
    }

    py::array_t<float> copy_pixel_rgb() {
        const size_t height = uniforms.image_height;
        const size_t width = uniforms.image_width;
        const size_t pixels = height * width;
        if (height == 0 || width == 0)
            throw std::runtime_error("copy_pixel_rgb requires a rendered view");
        if (buffers.pixel_state.deviceSize() < 4 * pixels)
            throw std::runtime_error("pixel_state is smaller than HxWx4");
        {
            py::gil_scoped_release release;
            trainer.copyFromDevice(buffers.pixel_state);
        }
        py::array_t<float> result({
            (py::ssize_t)height, (py::ssize_t)width, (py::ssize_t)3});
        float* out = result.mutable_data();
        const float* src = buffers.pixel_state.data();
        {
            py::gil_scoped_release release;
            for (size_t i = 0; i < pixels; ++i) {
                const float* rgba = src + 4 * i;
                float* rgb = out + 3 * i;
                rgb[0] = rgba[0];
                rgb[1] = rgba[1];
                rgb[2] = rgba[2];
            }
        }
        return result;
    }

    void render_dynamic_train_view(
        size_t view_idx, bool apply_color_correction=true
    ) {
        if (view_idx >= trainer.num_train())
            throw std::out_of_range("dynamic train view index is out of range");
        trainer.get_train_camera(view_idx, uniforms);
        render_dynamic(trainer.get_train_timestamp(view_idx));
        if (apply_color_correction &&
            config.color_correction && config.color_num_cameras > 0) {
            const uint32_t cam = (uint32_t)std::max(
                trainer.get_train_image(view_idx).camera_id, 0);
            auto deviceGuard = DeviceGuard(&trainer);
            trainer.executeFTGSPPColorCorrectForwardWH(
                config, buffers, uniforms.image_width, uniforms.image_height, cam);
        }
    }

    void set_color_correction_params(
        py::array_t<float, py::array::c_style | py::array::forcecast> params
    ) {
        if (!config.color_correction || config.color_num_cameras == 0)
            throw std::runtime_error("color correction is not configured");
        const size_t param_count = size_t(config.color_num_cameras) * 12u;
        auto data = params.request();
        if ((size_t)data.size != param_count)
            throw std::runtime_error(
                "color_params must contain " + std::to_string(param_count) +
                " floats");
        const float* ptr = static_cast<const float*>(data.ptr);
        {
            py::gil_scoped_release release;
            buffers.color_params.assign(ptr, ptr + param_count);
            trainer.copyToDevice(buffers.color_params);
        }
    }

    void set_sh_coefficients(
        py::array_t<float, py::array::c_style | py::array::forcecast> sh_coeffs
    ) {
        auto sh = sh_coeffs.request();
        const size_t n = buffers.num_splats;
        if (sh.ndim != 3 || (size_t)sh.shape[0] != n ||
            sh.shape[1] != 16 || sh.shape[2] != 3)
            throw std::runtime_error("sh_coeffs must have shape (N, 16, 3)");
        const float* ptr = static_cast<const float*>(sh.ptr);
        {
            py::gil_scoped_release release;
            buffers.sh_coeffs.assign(ptr, ptr + 16 * 3 * n);
            buffers.reorderSH(buffers.sh_coeffs);
            trainer.copyToDevice(buffers.sh_coeffs);
        }
    }

    void render_dynamic_val_view(size_t view_idx) {
        // Eval/final renders must match main-repo eval: raw renderer RGB only.
        // ColorCorrector is training-view / train-loss exclusive.
        if (view_idx >= trainer.num_val())
            throw std::out_of_range("dynamic val view index is out of range");
        trainer.get_val_camera(view_idx, uniforms);
        render_dynamic(trainer.get_val_timestamp(view_idx));
    }

    py::array_t<float> relocation_stats_state() {
        const size_t n = buffers.num_splats;
        if (buffers.relocation_stats.deviceSize() != 2 * n)
            throw std::runtime_error(
                "relocation statistics are unavailable before the first backward step");
        return buffer_to_array<float, float>(buffers.relocation_stats, 2);
    }

    void set_relocation_stats_state(
        py::array_t<float, py::array::c_style | py::array::forcecast> values
    ) {
        auto data = values.request();
        const size_t n = buffers.num_splats;
        if (data.ndim != 2 || (size_t)data.shape[0] != n || data.shape[1] != 2)
            throw std::runtime_error(
                "relocation statistics must have shape (N, 2)");
        const float* ptr = static_cast<const float*>(data.ptr);
        {
            py::gil_scoped_release release;
            buffers.relocation_stats.assign(ptr, ptr + data.size);
            trainer.copyToDevice(buffers.relocation_stats);
        }
    }

    uint32_t get_tile_invariant_status() {
        if (buffers.tile_invariant_status.deviceSize() == 0)
            return 0u;
        py::gil_scoped_release release;
        trainer.copyFromDevice(buffers.tile_invariant_status);
        return buffers.tile_invariant_status[0];
    }

    py::list benchmark_pair_transfer_buckets(
        const std::vector<size_t>& candidate_bytes,
        uint32_t rounds = 3u
    ) {
        if (candidate_bytes.empty() || rounds == 0u)
            throw std::invalid_argument(
                "transfer benchmark requires candidates and positive rounds");
        size_t payload_bytes = 0u;
        for (const size_t bytes : candidate_bytes) {
            if (bytes == 0u || bytes % sizeof(float) != 0u ||
                bytes > 256u * 1024u * 1024u)
                throw std::invalid_argument(
                    "transfer benchmark candidates must be 4-byte aligned "
                    "and no larger than 256 MiB");
            payload_bytes = std::max(payload_bytes, bytes);
        }
        struct Result {
            size_t bucket_bytes = 0u;
            size_t payload_bytes = 0u;
            double wall_seconds = 0.0;
            double download_seconds = 0.0;
            double upload_seconds = 0.0;
            bool persistent_mapping = true;
        };
        std::vector<Result> results;
        results.reserve(candidate_bytes.size());
        {
            py::gil_scoped_release release;
            trainer.clearDeviceBuffer(
                buffers._temp_gauss_attr, payload_bytes / sizeof(float));
            const std::vector<VulkanGSPipeline::PackedBufferRegion> regions{{
                buffers._temp_gauss_attr.deviceBuffer, 0u}};
            for (const size_t bucket_bytes : candidate_bytes) {
                Result result{};
                result.bucket_bytes = bucket_bytes;
                result.payload_bytes = payload_bytes;
                std::vector<double> round_walls;
                std::vector<double> round_downloads;
                std::vector<double> round_uploads;
                round_walls.reserve(rounds);
                round_downloads.reserve(rounds);
                round_uploads.reserve(rounds);
                for (uint32_t round = 0u; round < rounds; ++round) {
                    const auto round_begin = std::chrono::steady_clock::now();
                    double download_seconds = 0.0;
                    double upload_seconds = 0.0;
                    for (size_t offset = 0u; offset < payload_bytes;
                         offset += bucket_bytes) {
                        const size_t bytes = std::min(
                            bucket_bytes, payload_bytes - offset);
                        const auto download =
                            trainer.submitBuffersFromDevicePackedRange(
                                regions, offset, bytes);
                        const auto mapped =
                            trainer.completeBuffersFromDevicePackedRangeMapped(
                                download);
                        const auto upload =
                            trainer.submitBuffersToDevicePackedRange(
                                mapped.data, offset, bytes, regions);
                        const auto upload_timing =
                            trainer.completeBuffersToDevicePackedRange(upload);
                        download_seconds += mapped.timing.gpu_copy_seconds;
                        upload_seconds +=
                            upload_timing.gpu_copy_seconds +
                            upload_timing.host_memcpy_seconds;
                        result.persistent_mapping =
                            result.persistent_mapping &&
                            mapped.timing.used_persistent_map &&
                            upload_timing.used_persistent_map;
                    }
                    round_walls.push_back(std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - round_begin).count());
                    round_downloads.push_back(download_seconds);
                    round_uploads.push_back(upload_seconds);
                }
                result.wall_seconds = median_of(std::move(round_walls));
                result.download_seconds = median_of(std::move(round_downloads));
                result.upload_seconds = median_of(std::move(round_uploads));
                results.push_back(result);
            }
        }
        py::list output;
        for (const Result& result : results) {
            py::dict item;
            item["bucket_bytes"] = result.bucket_bytes;
            item["payload_bytes"] = result.payload_bytes;
            item["rounds"] = rounds;
            item["wall_aggregator"] = "median";
            item["wall_seconds"] = result.wall_seconds;
            item["download_seconds"] = result.download_seconds;
            item["upload_seconds"] = result.upload_seconds;
            item["persistent_mapping"] = result.persistent_mapping;
            output.append(std::move(item));
        }
        return output;
    }

    py::dict get_optimizer_state() {
        py::dict state;
        state["g_xyz_ws"] = buffer_to_array<float, float>(buffers.g_xyz_ws);
        state["g_sh_coeffs_1"] = buffer_to_array<float, float>(buffers.g_sh_coeffs_1);
        state["g_sh_coeffs_2"] = buffer_to_array<float, float>(buffers.g_sh_coeffs_2);
        state["g_rotations"] = buffer_to_array<float, float>(buffers.g_rotations);
        state["g_scales_opacs"] = buffer_to_array<float, float>(buffers.g_scales_opacs);
        state["g_times_durations"] = buffer_to_array<float, float>(buffers.g_times_durations);
        state["g_velocities_gates"] = buffer_to_array<float, float>(buffers.g_velocities_gates);
        state["relocation_stats"] = buffer_to_array<float, float>(
            buffers.relocation_stats, 2);
        if (config.color_correction && config.color_num_cameras > 0) {
            const size_t param_count = size_t(config.color_num_cameras) * 12u;
            if (buffers.color_params.deviceSize() == param_count) {
                state["color_params"] = buffer_to_array<float, float>(buffers.color_params);
                state["color_m"] = buffer_to_array<float, float>(buffers.color_m);
                state["color_v"] = buffer_to_array<float, float>(buffers.color_v);
            }
        }
        {
            py::gil_scoped_release release;
            release_optimizer_host_copies();
        }
        return state;
    }

    py::dict get_last_loss() {
        if (buffers.loss_accum.deviceBuffer.size != 6*sizeof(float))
            throw std::runtime_error("no native loss is available before the first training step");
        {
            py::gil_scoped_release release;
            trainer.copyFromDevice(buffers.loss_accum);
        }
        py::dict result;
        const float l1 = buffers.loss_accum[0];
        const float ssim = buffers.loss_accum[1];
        const float reg_opacity = buffers.loss_accum[2];
        const float reg_scale = buffers.loss_accum[3];
        const float gate = buffers.loss_accum[4];
        const float color_cc = buffers.loss_accum[5];
        const float image_loss = l1 + ssim;
        const float total_loss = image_loss + reg_opacity + reg_scale + gate
            + color_cc;
        result["l1"] = l1;
        result["ssim"] = ssim;
        result["reg_opacity"] = reg_opacity;
        result["reg_scale"] = reg_scale;
        result["gate"] = gate;
        result["color_cc"] = color_cc;
        result["image_loss"] = image_loss;
        result["total_loss"] = total_loss;
        return result;
    }

    void set_optimizer_state(py::dict state) {
        auto restore = [&](const char* key, Buffer<float>& target, size_t expected) {
            if (!state.contains(key))
                throw std::runtime_error(std::string("optimizer state is missing ") + key);
            py::array_t<float, py::array::c_style | py::array::forcecast> value =
                state[key].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
            auto data = value.request();
            if ((size_t)data.size != expected)
                throw std::runtime_error(
                    std::string("optimizer state ") + key + " has " +
                    std::to_string(data.size) + " floats, expected " +
                    std::to_string(expected));
            target.assign(static_cast<float*>(data.ptr), static_cast<float*>(data.ptr) + data.size);
            trainer.copyToDevice(target);
        };
        const size_t n = buffers.num_splats;
        const size_t sh_moments = 16*3*_CEIL_ROUND(n, SH_REORDER_SIZE);
        restore("g_xyz_ws", buffers.g_xyz_ws, 2*3*n);
        restore("g_sh_coeffs_1", buffers.g_sh_coeffs_1, sh_moments);
        restore("g_sh_coeffs_2", buffers.g_sh_coeffs_2, sh_moments);
        restore("g_rotations", buffers.g_rotations, 2*4*n);
        restore("g_scales_opacs", buffers.g_scales_opacs, 2*4*n);
        restore("g_times_durations", buffers.g_times_durations, 2*2*n);
        restore("g_velocities_gates", buffers.g_velocities_gates, 2*4*n);
        restore("relocation_stats", buffers.relocation_stats, 2*n);
        if (config.color_correction && config.color_num_cameras > 0
            && state.contains("color_params")) {
            const size_t param_count = size_t(config.color_num_cameras) * 12u;
            restore("color_params", buffers.color_params, param_count);
            restore("color_m", buffers.color_m, param_count);
            restore("color_v", buffers.color_v, param_count);
        }
        release_optimizer_host_copies();
    }

    void relocate_dynamic(
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> relocation_map,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> relocation_counts,
        const std::string& mode,
        bool require_dead_destinations = true,
        bool require_alive_donors = true
    ) {
        auto map = relocation_map.request();
        auto counts = relocation_counts.request();
        if (map.ndim != 1 || counts.ndim != 1 ||
            (size_t)map.shape[0] != buffers.num_splats ||
            (size_t)counts.shape[0] != buffers.num_splats)
            throw std::runtime_error("relocation map/counts must have shape (N,)");
        int mode_id = mode == "partial_copy" ? 0 : mode == "exact_copy" ? 1 :
            mode == "3d_mcmc" ? 2 : -1;
        if (mode_id < 0)
            throw std::runtime_error("unknown FTGSPP relocation mode: " + mode);
        const int32_t* map_ptr = static_cast<const int32_t*>(map.ptr);
        const int32_t* counts_ptr = static_cast<const int32_t*>(counts.ptr);
        // Explicit mapping is a replay/broadcast API. Validate it against this
        // replica's current state before the shader can turn an invalid donor
        // into non-finite scales. Broadcast replay already validated mapping
        // and counts on the host, including the 50-copy cap.
        const bool check_opacity_roles =
            require_dead_destinations || require_alive_donors;
        trainer.waitAllComputeSlots("relocate");
        if (check_opacity_roles) {
            trainer.copyFromDevice(buffers.base_log_scales_opacity);
            if (buffers.base_log_scales_opacity.size() != 4 * buffers.num_splats)
                throw std::runtime_error(
                    "relocation precondition check could not read current state");
            std::vector<int32_t> observed_counts(buffers.num_splats, 0);
            auto opacity_at = [&](size_t index) {
                const float logit = buffers.base_log_scales_opacity[4*index+3];
                return 1.0f / (1.0f + std::exp(-logit));
            };
            constexpr int32_t relocation_max_ratio = 51;
            for (size_t destination = 0; destination < buffers.num_splats;
                 ++destination) {
                const int32_t source = map_ptr[destination];
                if (source < -1 || source >= (int32_t)buffers.num_splats)
                    throw std::runtime_error(
                        "relocation mapping contains an out-of-range donor at "
                        "destination " + std::to_string(destination));
                if (source < 0)
                    continue;
                if ((size_t)source == destination)
                    throw std::runtime_error(
                        "relocation mapping contains an illegal self-reference at " +
                        std::to_string(destination));
                const float destination_opacity = opacity_at(destination);
                const float donor_opacity = opacity_at((size_t)source);
                const float opacity_tolerance =
                    8.0f * std::numeric_limits<float>::epsilon() *
                    std::max(1.0f, std::abs(config.min_opacity));
                if (!std::isfinite(destination_opacity))
                    throw std::runtime_error(
                        "relocation destination opacity is non-finite: " +
                        std::to_string(destination));
                // Rank 0 owns the mapping. Pair-mean FP32 and independent local
                // views can flip Gaussians across min_opacity on followers.
                // Broadcast replay still applies dest/donor roles so replicas
                // reconverge. Direct replay keeps the strict liveness checks.
                if (require_dead_destinations &&
                    destination_opacity > config.min_opacity + opacity_tolerance)
                    throw std::runtime_error(
                        "relocation destination is not dead in the current state: " +
                        std::to_string(destination));
                if (!std::isfinite(donor_opacity))
                    throw std::runtime_error(
                        "relocation donor opacity is non-finite: " +
                        std::to_string(source));
                if (require_alive_donors &&
                    donor_opacity <= config.min_opacity - opacity_tolerance)
                    throw std::runtime_error(
                        "relocation donor is dead/non-finite in the current state: " +
                        std::to_string(source));
                // Donor scales must still be finite on every replica; a
                // non-finite donor would copy NaNs into destinations.
                for (size_t component = 0; component < 3; ++component) {
                    if (!std::isfinite(
                            buffers.base_log_scales_opacity[4*(size_t)source+component]))
                        throw std::runtime_error(
                            "relocation donor scale is non-finite: " +
                            std::to_string(source));
                }
                ++observed_counts[(size_t)source];
            }
            for (size_t source = 0; source < buffers.num_splats; ++source) {
                if (counts_ptr[source] < 0 ||
                    counts_ptr[source] >= relocation_max_ratio)
                    throw std::runtime_error(
                        "relocation donor copy count is outside [0, 50] at " +
                        std::to_string(source));
                if (counts_ptr[source] != observed_counts[source])
                    throw std::runtime_error(
                        "relocation donor counts do not match the mapping at " +
                        std::to_string(source));
            }
        }
        {
            // Mapping/count arrays stay alive for the duration of this call;
            // all rank-local uploads and relocation dispatches can overlap.
            py::gil_scoped_release release;
            buffers.relocation_map.assign(map_ptr, map_ptr + map.size);
            buffers.relocation_counts.assign(
                counts_ptr, counts_ptr + counts.size);
            trainer.copyToDevice(buffers.relocation_map);
            trainer.copyToDevice(buffers.relocation_counts);
            auto deviceGuard = DeviceGuard(&trainer);
            trainer.executeFTGSPPRelocation(config, buffers, mode_id);
        }
    }

    void apply_planned_relocation(const std::string& mode) {
        int mode_id = mode == "partial_copy" ? 0 : mode == "exact_copy" ? 1 :
            mode == "3d_mcmc" ? 2 : -1;
        if (mode_id < 0)
            throw std::runtime_error("unknown FTGSPP relocation mode: " + mode);
        {
            py::gil_scoped_release release;
            trainer.waitAllComputeSlots("relocate");
            auto deviceGuard = DeviceGuard(&trainer);
            trainer.executeFTGSPPRelocation(config, buffers, mode_id);
        }
    }

    void clear_relocation_stats_state() {
        const size_t n = buffers.num_splats;
        if (buffers.relocation_stats.deviceSize() != 2 * n)
            throw std::runtime_error(
                "relocation statistics are unavailable before the first backward step");
        py::gil_scoped_release release;
        trainer.clearDeviceBuffer(buffers.relocation_stats, 2 * n);
    }

    py::dict relocate_dynamic_native(
        int step, const std::string& mode, bool include_gate, uint32_t seed
    ) {
        const int mode_id = mode == "partial_copy" ? 0 :
            mode == "exact_copy" ? 1 : mode == "3d_mcmc" ? 2 : -1;
        if (mode_id < 0)
            throw std::runtime_error("unknown FTGSPP relocation mode: " + mode);
        {
            py::gil_scoped_release release;
            {
                auto deviceGuard = DeviceGuard(&trainer);
                trainer.executeFTGSPPRelocationNative(
                    config, buffers, mode_id, step, include_gate, seed);
            }
            trainer.copyFromDevice(buffers.relocation_status);
        }
        py::dict result;
        result["relocated"] = buffers.relocation_status.empty()
            ? 0u : buffers.relocation_status[0];
        result["alive"] = buffers.relocation_status.size() < 2
            ? 0u : buffers.relocation_status[1];
        result["all_dead"] = buffers.relocation_status.size() < 3
            ? true : buffers.relocation_status[2] != 0u;
        return result;
    }

    py::dict plan_relocation_dynamic_native(
        int step, bool include_gate, uint32_t seed
    ) {
        {
            py::gil_scoped_release release;
            {
                auto deviceGuard = DeviceGuard(&trainer);
                trainer.executeFTGSPPRelocationPlanNative(
                    config, buffers, step, include_gate, seed);
            }
            trainer.copyFromDevice(buffers.relocation_map);
            trainer.copyFromDevice(buffers.relocation_counts);
            trainer.copyFromDevice(buffers.relocation_status);
        }
        py::dict result;
        result["mapping"] = buffer_to_array<int32_t, int32_t, false>(
            buffers.relocation_map);
        result["counts"] = buffer_to_array<int32_t, int32_t, false>(
            buffers.relocation_counts);
        result["relocated"] = buffers.relocation_status.empty()
            ? 0u : buffers.relocation_status[0];
        result["alive"] = buffers.relocation_status.size() < 2
            ? 0u : buffers.relocation_status[1];
        result["all_dead"] = buffers.relocation_status.size() < 3
            ? true : buffers.relocation_status[2] != 0u;
        return result;
    }

    py::dict get_dynamic_gaussians() {
        py::dict result;
        result["means"] = buffer_to_array<float, float>(buffers.base_xyz_ws, 3);
        result["log_scales_opacity"] = buffer_to_array<float, float>(buffers.base_log_scales_opacity, 4);
        result["times_durations"] = buffer_to_array<float, float>(buffers.times_durations, 2);
        result["velocities_gates"] = buffer_to_array<float, float>(buffers.velocities_gates, 4);
        result["rotations"] = buffer_to_array<float, float>(buffers.rotations, 4);
        result["sh_coeffs"] = buffer_to_array<float, float>(buffers.sh_coeffs, 16, 3, "sh");
        {
            py::gil_scoped_release release;
            release_model_host_copies();
        }
        return result;
    }

    py::array_t<float> hard_separation_gate_state() {
        // Return an owning diagnostic snapshot. buffer_to_array performs the
        // required D2H copy; releasing native model host vectors afterwards
        // prevents this phase-boundary check from pinning another 4*N floats
        // (or any earlier model readback) for the rest of training.
        py::array_t<float> result =
            buffer_to_array<float, float>(buffers.velocities_gates, 4);
        {
            py::gil_scoped_release release;
            release_model_host_copies();
        }
        return result;
    }

    void configure_hard_separation_phase() {
        config.hard_separation = true;
        config.max_steps = 5000;
        config.means_lr = 0.001f*1.6e-4f;
        config.means_lr_final = config.means_lr;
        // The reference hard phase creates a fresh Gaussian optimizer with
        // these fixed learning rates.  Re-state every trainable group here so
        // a main-phase Hydra override cannot leak into hard separation.
        config.features_dc_lr = 2.5e-3f;
        config.features_rest_lr = 1.25e-4f;
        config.opacities_lr = 5e-2f;
        config.scales_lr = 5e-3f;
        config.quats_lr = 1e-3f;
        config.times_lr = 1.6e-4f;
        config.durations_lr = 5e-3f;
        config.scale_reg = 0.001f;
        config.opacity_reg = 0.001f;
        config.gate_reg = 0.0f;
        // Hard separation keeps the already-learned affine transform in the
        // image-loss chain, but it is not part of the fresh hard-phase Adam
        // optimizer and FTGSPP assigns no L_cc regularizer in this phase.
        config.color_reg = 0.0f;
        config.velocities_lr = 0.0f;
        config.velocities_lr_final = 0.0f;
        config.gates_lr = 0.0f;
        gradient_ready = false;
        gradient_step = -1;
    }

    py::dict begin_hard_separation() {
        trainer.requireFTGSPPFloat32AtomicAdd();
        configure_hard_separation_phase();
        {
            // Every replica executes the same split concurrently.  Keep only
            // Python result construction under the GIL.
            py::gil_scoped_release release;
            {
                auto deviceGuard = DeviceGuard(&trainer);
                trainer.executeFTGSPPHardSeparation(buffers);
            }
            trainer.copyFromDevice(buffers.relocation_status);
        }
        py::dict result;
        const uint32_t marginalized = buffers.relocation_status.empty()
            ? 0u : buffers.relocation_status[0];
        result["marginalized"] = marginalized;
        result["kept"] = (uint32_t)buffers.num_splats-marginalized;
        return result;
    }

    py::dict resume_hard_separation() {
        // A resumed checkpoint already contains binary gates, zeroed dynamic
        // velocities, and the exact Adam moments saved during separation.
        // Only restore phase-local runtime hyperparameters; never dispatch the
        // destructive split shader or clear any moment buffer here.
        trainer.requireFTGSPPFloat32AtomicAdd();
        configure_hard_separation_phase();
        uint32_t marginalized = 0u;
        {
            py::gil_scoped_release release;
            trainer.copyFromDevice(buffers.velocities_gates);
            for (size_t i = 0; i < buffers.num_splats; ++i)
                marginalized +=
                    buffers.velocities_gates[4*i+3] > 0.0f ? 1u : 0u;
        }
        py::dict result;
        result["marginalized"] = marginalized;
        result["kept"] = (uint32_t)buffers.num_splats - marginalized;
        return result;
    }


    size_t num_train() { return trainer.num_train(); }
    size_t num_val() { return trainer.num_val(); }


    void projection_forward() {
        uniforms.num_splats = (uint32_t)buffers.num_splats;
        try {
            trainer.executeProjectionForward(uniforms, buffers, 0);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeProjectionForward failed");
        }
    }

    void dynamic_projection_forward(float timestamp) {
        uniforms.num_splats = (uint32_t)buffers.num_splats;
        try {
            trainer.executeDynamicProjectionForward(
                uniforms, buffers, timestamp, config.max_duration,
                config.marginal_threshold, 0);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(
                std::string(err.what()) +
                ". trainer.executeDynamicProjectionForward failed");
        }
    }
    
    void process_tiles() {
        auto deviceGuard = DeviceGuard(&trainer);

        try {
            trainer.executeCalculateIndexBufferOffset(buffers);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeCalculateIndexBufferOffset failed");
        }
        try {
            trainer.executeTileWorkPrepare(uniforms, buffers);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeTileWorkPrepare failed");
        }

        try {
            trainer.executeGenerateKeys(uniforms, buffers);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeGenerateKeys failed");
        }

        try {
            trainer.executeSort(uniforms, buffers, -1);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeSort failed");
        }

        try {
            trainer.executeComputeTileRanges(uniforms, buffers);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeComputeTileRanges failed");
        }

    }
    
    void rasterize_forward() {
        try {
            trainer.executeRasterizeForward(uniforms, buffers);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeRasterizeForward failed");
        }
    }
    
    void rasterize_backward() {
        try {
            trainer.executeRasterizeBackward(uniforms, buffers);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". trainer.executeRasterizeBackward failed");
        }
    }

    void dynamic_forward(float timestamp) {
        auto deviceGuard = DeviceGuard(&trainer);
        dynamic_projection_forward(timestamp);
        process_tiles();
        rasterize_forward();
    }


    size_t get_vram_usage() {
        return buffers.getTotalAllocSize();
    }

    size_t get_peak_vram_usage() {
        return trainer.getPeakAllocSize();
    }

    py::dict get_workload_counters() const {
        py::dict result;
        const auto tel = trainer.snapshotTelemetry();
        result["num_splats"] = buffers.num_splats;
        result["tile_pair_count"] = tel.exact_tile_count > 0
            ? tel.exact_tile_count : buffers.num_indices;
        result["sort_storage_bytes"] = (
            buffers.tile_pair_capacity * 2u
            * (sizeof(sortingKey_t) + sizeof(int32_t)));
        result["active_sh_degree"] = uniforms.active_sh;
        result["image_width"] = uniforms.image_width;
        result["image_height"] = uniforms.image_height;
        return result;
    }

    std::map<std::string, std::tuple<size_t, double>> get_timing_breakdown() {
        return PerfTimer::get_summary();
    }

    std::map<std::string, size_t> get_vram_breakdown() {
        return buffers.getVramBreakdown();
    }

    void cleanup() {
        if (!initialized || closed)
            return;
        if (host_mailbox_rank >= 0 && g_host_mailbox_owner)
            g_host_mailbox_owner->detach(
                static_cast<uint32_t>(host_mailbox_rank));
        if (sparse_mailbox_rank >= 0 && g_sparse_mailbox_owner)
            g_sparse_mailbox_owner->detach(
                static_cast<uint32_t>(sparse_mailbox_rank));
        if (relocation_mailbox_rank >= 0 && g_relocation_mailbox_owner)
            g_relocation_mailbox_owner->detach(
                static_cast<uint32_t>(relocation_mailbox_rank));
        host_mailbox_rank = -1;
        sparse_mailbox_rank = -1;
        relocation_mailbox_rank = -1;
        reset_hierarchical_host_attachments();
        trainer.cleanupBuffers(buffers);
        trainer.cleanup();
        closed = true;
    }
};

#define DEF_BUFFER_0(key, buffer_name, dtype, ...) \
    def_property_readonly(#key, [] (VkSplat &self) { \
        return self.buffer_to_array<dtype, dtype>(self.buffers.buffer_name, __VA_ARGS__); \
    })

#define DEF_BUFFER(key, dtype, ...) \
    DEF_BUFFER_0(key, key, dtype, __VA_ARGS__)

#define DEF_ARRAY(key, dtype) \
    def("_set_"#key, [] (VkSplat &self, py::array_t<dtype> array) { \
        self.array_to_buffer(array, self.buffers.key); \
    }, "Directly set `"#key"` buffer (use at own risk)", py::arg(#key))

#define DEF_BUFFER_ARRAY(key, dtype, ...) \
    DEF_BUFFER(key, dtype, __VA_ARGS__) \
    .DEF_ARRAY(key, dtype)

PYBIND11_MODULE(_ftgspp_vk, m) {
    m.doc() = "Standalone FTGSPP dynamic Vulkan training kernel";
    
    py::class_<VkSplat>(m, "FTGSPPVulkan")
        .def(py::init<>())
        .def("initialize", &VkSplat::initialize, "Initialize the trainer with SPIRV code")
        .def("configure_ftgspp", &VkSplat::configure_ftgspp)
        .def("set_dynamic_gaussians", &VkSplat::set_dynamic_gaussians,
             py::arg("means"), py::arg("log_scales"), py::arg("rotations"),
             py::arg("opacity_logits"), py::arg("sh_coeffs"), py::arg("times"),
             py::arg("durations"), py::arg("velocities"), py::arg("gates"))
        .def("get_dynamic_gaussians", &VkSplat::get_dynamic_gaussians)
        .def("hard_separation_gate_state",
             &VkSplat::hard_separation_gate_state)
        .def("begin_hard_separation", &VkSplat::begin_hard_separation)
        .def("resume_hard_separation", &VkSplat::resume_hard_separation)
        .def("set_reference_image", &VkSplat::set_reference_image)
        .def("clear_dynamic_views", &VkSplat::clear_dynamic_views)
        .def("add_dynamic_view", &VkSplat::add_dynamic_view,
             py::arg("image"), py::arg("world_view_transform"), py::arg("intrinsic"),
             py::arg("timestamp"), py::arg("camera_id"), py::arg("frame_id"),
             py::arg("is_train"), py::arg("cache_on_device") = false)
        .def("add_dynamic_view_meta", &VkSplat::add_dynamic_view_meta,
             py::arg("height"), py::arg("width"),
             py::arg("world_view_transform"), py::arg("intrinsic"),
             py::arg("timestamp"), py::arg("camera_id"), py::arg("frame_id"),
             py::arg("is_train"))
        .def("bind_dynamic_view_image", &VkSplat::bind_dynamic_view_image,
             py::arg("view_idx"), py::arg("is_train"), py::arg("image"))
        .def("get_pipeline_telemetry", &VkSplat::get_pipeline_telemetry)
        .def("release_dynamic_view_image", &VkSplat::release_dynamic_view_image,
             py::arg("view_idx"), py::arg("is_train"))
        .def("set_dynamic_view_cache_on_device",
             &VkSplat::set_dynamic_view_cache_on_device,
             py::arg("view_idx"), py::arg("is_train"), py::arg("enabled"))
        .def("release_model_upload_host_copies",
             &VkSplat::release_model_upload_host_copies)
        .def("render_dynamic", &VkSplat::render_dynamic, py::arg("timestamp"))
        .def("materialize_dynamic", &VkSplat::materialize_dynamic, py::arg("timestamp"))
        .def("render_flow_dynamic", &VkSplat::render_flow_dynamic,
             py::arg("timestamp"), py::arg("end_means"))
        .def("render_flow_dynamic_times", &VkSplat::render_flow_dynamic_times,
             py::arg("timestamp"), py::arg("end_timestamp"))
        .def("render_flow_dynamic_train_view", &VkSplat::render_flow_dynamic_train_view,
             py::arg("view_idx"), py::arg("end_timestamp"),
             py::call_guard<py::gil_scoped_release>())
        .def("render_flow_dynamic_val_view", &VkSplat::render_flow_dynamic_val_view,
             py::arg("view_idx"), py::arg("end_timestamp"),
             py::call_guard<py::gil_scoped_release>())
        .def("train_dynamic_view", &VkSplat::train_dynamic_view,
             py::arg("view_idx"), py::arg("step"),
             py::arg("collect_loss_scalar") = true,
             py::call_guard<py::gil_scoped_release>())
        .def("backward_dynamic_view", &VkSplat::backward_dynamic_view,
             py::arg("view_idx"), py::arg("step"),
             py::arg("collect_loss_scalar") = true,
             py::call_guard<py::gil_scoped_release>())
        .def(
            "backward_dynamic_view_from_loss_raw",
            &VkSplat::backward_dynamic_view_from_loss_raw,
            py::arg("view_idx"),
            py::arg("step"),
            py::call_guard<py::gil_scoped_release>(),
            "Raw backward from an already computed train_dynamic_view_to_loss state"
        )
        .def("finalize_local_gradient", &VkSplat::finalize_local_gradient,
             py::arg("microsteps"),
             py::call_guard<py::gil_scoped_release>())
        .def("configure_pair_exchange_gpu_merge", &VkSplat::configure_pair_exchange_gpu_merge,
             py::arg("codec_policy"))
        .def("packed_gradient_layout", &VkSplat::packed_gradient_layout)
        .def("packed_gradient_size", &VkSplat::packed_gradient_size)
        .def("gradient_state_packed_into", &VkSplat::gradient_state_packed_into,
             py::arg("output"))
        .def("set_gradient_state_packed", &VkSplat::set_gradient_state_packed,
             py::arg("state"), py::arg("validate_finite") = true)
        .def("mean_packed_gradients_into", &VkSplat::mean_packed_gradients_into,
             py::arg("per_rank"), py::arg("output"))
        .def_static(
            "create_pair_exchange_gpu_merge_owner",
            &VkSplat::create_pair_exchange_gpu_merge_owner,
            py::arg("world_size"), py::arg("bucket_bytes"),
            py::arg("sparse") = false)
        .def_static(
            "release_pair_exchange_gpu_merge_owner",
            &VkSplat::release_pair_exchange_gpu_merge_owner)
        .def(
            "reset_pair_exchange_gpu_merge_attachments",
            &VkSplat::reset_pair_exchange_gpu_merge_attachments)
        .def(
            "exchange_pair_gradient",
            &VkSplat::exchange_pair_gradient,
            py::arg("rank"), py::arg("generation"))
        .def(
            "optimizer_step_pair_exchange_gpu_merge",
            &VkSplat::optimizer_step_pair_exchange_gpu_merge,
            py::arg("step"), py::arg("timestamps"))
        .def(
            "exchange_pair_relocation",
            &VkSplat::exchange_pair_relocation,
            py::arg("rank"), py::arg("generation"))
        .def_static(
            "create_hierarchical_host_collective",
            &VkSplat::create_hierarchical_host_collective,
            py::arg("world_size"), py::arg("bucket_bytes"),
            py::arg("algorithm") = "auto",
            py::arg("local_microsteps") = 1,
            py::arg("reducer_shards") = 0,
            py::arg("sparse_requested") = false)
        .def_static(
            "release_hierarchical_host_collective",
            &VkSplat::release_hierarchical_host_collective)
        .def(
            "configure_hierarchical_host_collective",
            &VkSplat::configure_hierarchical_host_collective,
            py::arg("rank"), py::arg("sparse") = false,
            py::arg("codec_policy") = "auto_exact")
        .def(
            "exchange_hierarchical_gradient",
            &VkSplat::exchange_hierarchical_gradient,
            py::arg("rank"), py::arg("generation"))
        .def(
            "exchange_hierarchical_relocation",
            &VkSplat::exchange_hierarchical_relocation,
            py::arg("rank"), py::arg("generation"))
        .def(
            "optimizer_step_hierarchical",
            &VkSplat::optimizer_step_hierarchical,
            py::arg("step"),
            py::arg("timestamps") = std::vector<float>{})
        .def(
            "abort_hierarchical_host_collective",
            &VkSplat::abort_hierarchical_host_collective,
            py::arg("reason"))
        .def(
            "reset_hierarchical_host_attachments",
            &VkSplat::reset_hierarchical_host_attachments)
        .def("last_packed_download_timing", &VkSplat::last_packed_download_timing)
        .def("last_packed_upload_timing", &VkSplat::last_packed_upload_timing)
        .def("device_info", &VkSplat::device_info)
        .def("native_abi", &VkSplat::native_abi)
        .def("relocation_stats_state", &VkSplat::relocation_stats_state)
        .def("set_relocation_stats_state", &VkSplat::set_relocation_stats_state,
             py::arg("statistics"))
        .def("clear_relocation_stats_state", &VkSplat::clear_relocation_stats_state)
        .def("apply_planned_relocation", &VkSplat::apply_planned_relocation,
             py::arg("mode"))
        .def("optimizer_step", &VkSplat::optimizer_step,
             py::arg("step"), py::call_guard<py::gil_scoped_release>())
        .def("train_dynamic_view_to_loss", &VkSplat::train_dynamic_view_to_loss,
             py::arg("view_idx"), py::arg("step"),
             py::arg("collect_loss_scalar") = true,
             py::call_guard<py::gil_scoped_release>())
        .def("train_dynamic_view_from_loss", &VkSplat::train_dynamic_view_from_loss,
             py::arg("view_idx"), py::arg("step"),
             py::call_guard<py::gil_scoped_release>())
        .def("accumulate_pixel_state_grad", &VkSplat::accumulate_pixel_state_grad,
             py::arg("extra_grad"))
        .def("render_dynamic_train_view", &VkSplat::render_dynamic_train_view,
             py::arg("view_idx"), py::arg("apply_color_correction") = true,
             py::call_guard<py::gil_scoped_release>())
        .def("set_color_correction_params", &VkSplat::set_color_correction_params,
             py::arg("params"))
        .def("set_sh_coefficients", &VkSplat::set_sh_coefficients,
             py::arg("sh_coeffs"))
        .def("render_dynamic_val_view", &VkSplat::render_dynamic_val_view,
             py::arg("view_idx"), py::call_guard<py::gil_scoped_release>())
        .def("drain_eval", &VkSplat::drain_eval,
             py::call_guard<py::gil_scoped_release>())
        .def("drain_for_structure_change", &VkSplat::drain_for_structure_change,
             py::call_guard<py::gil_scoped_release>())
        .def("apply_sync_tile_capacity", &VkSplat::apply_sync_tile_capacity,
             py::arg("exact_count"),
             py::call_guard<py::gil_scoped_release>())
        .def("inject_sync_tile_overflow", &VkSplat::inject_sync_tile_overflow,
             py::arg("exact_count"))
        .def_static(
            "clear_sync_collectives_for_replay",
            &VkSplat::clear_sync_collectives_for_replay_py)
        .def("begin_eval_val_view", &VkSplat::begin_eval_val_view,
             py::arg("view_idx"), py::call_guard<py::gil_scoped_release>())
        .def("submit_eval_rgb_readback", &VkSplat::submit_eval_rgb_readback,
             py::call_guard<py::gil_scoped_release>())
        .def("complete_eval_rgb", &VkSplat::complete_eval_rgb)
        .def("copy_pixel_rgb", &VkSplat::copy_pixel_rgb)
        .def("get_optimizer_state", &VkSplat::get_optimizer_state)
        .def("get_last_loss", &VkSplat::get_last_loss)
        .def("set_collect_geometry_grad_stats",
             &VkSplat::set_collect_geometry_grad_stats,
             py::arg("enabled"))
        .def("get_geometry_gradient_stats",
             &VkSplat::get_geometry_gradient_stats)
        .def("set_optimizer_state", &VkSplat::set_optimizer_state)
        .def("relocate_dynamic", &VkSplat::relocate_dynamic,
             py::arg("relocation_map"), py::arg("relocation_counts"),
             py::arg("mode"),
             py::arg("require_dead_destinations") = true,
             py::arg("require_alive_donors") = true)
        .def("relocate_dynamic_native", &VkSplat::relocate_dynamic_native,
             py::arg("step"), py::arg("mode"), py::arg("include_gate"),
             py::arg("seed"))
        .def("plan_relocation_dynamic_native", &VkSplat::plan_relocation_dynamic_native,
             py::arg("step"), py::arg("include_gate"), py::arg("seed"))
        .def_property_readonly("num_train", &VkSplat::num_train)
        .def_property_readonly("num_val", &VkSplat::num_val)
        .def("get_vram_usage", &VkSplat::get_vram_usage, "Get VRAM Usage")
        .def("get_peak_vram_usage", &VkSplat::get_peak_vram_usage, "Get peak VRAM Usage")
        .def("get_workload_counters", &VkSplat::get_workload_counters,
             "Get current Gaussian/tile/sort workload counters")
        .def("get_tile_invariant_status", &VkSplat::get_tile_invariant_status,
             "Read latest exact tile count/emit diagnostic bits")
        .def(
            "benchmark_pair_transfer_buckets",
            &VkSplat::benchmark_pair_transfer_buckets,
            py::arg("candidate_bytes"), py::arg("rounds") = 3u,
            "Benchmark fixed-payload Vulkan D2H/H2D bucket candidates"
        )
        .def("get_timing_breakdown", &VkSplat::get_timing_breakdown, "Get timing for each step since start")
        .def("get_vram_breakdown", &VkSplat::get_vram_breakdown, "Get VRAM for each buffer at the current moment")
        .def("cleanup", &VkSplat::cleanup, "Cleanup Vulkan resources",
             py::call_guard<py::gil_scoped_release>())
        .DEF_BUFFER_ARRAY(xyz_ws, float, 3, (size_t)-1, "mean")
        .DEF_BUFFER_ARRAY(base_xyz_ws, float, 3)
        .DEF_BUFFER_ARRAY(base_log_scales_opacity, float, 4)
        .DEF_BUFFER_ARRAY(times_durations, float, 2)
        .DEF_BUFFER_ARRAY(velocities_gates, float, 4)
        .DEF_BUFFER_ARRAY(sh_coeffs, float, 16, 3, "sh")
        .DEF_BUFFER_ARRAY(rotations, float, 4)
        .DEF_BUFFER_0(scales, scales_opacs, float, 4, (size_t)-1, "", 3, 0)
        .DEF_BUFFER_0(opacities, scales_opacs, float, 4, (size_t)-1, "", 1, 3)
        .DEF_BUFFER_ARRAY(tiles_touched, int32_t, (size_t)(-1))
        .DEF_BUFFER_ARRAY(rect_tile_space, int64_t, (size_t)(-1))
        .DEF_BUFFER_ARRAY(radii, int32_t, (size_t)(-1))
        .DEF_BUFFER_ARRAY(xy_vs, float, 2)
        .DEF_BUFFER_ARRAY(depths, float, 1)
        .DEF_BUFFER_ARRAY(inv_cov_vs_opacity, float, 4)
        .DEF_BUFFER_ARRAY(rgb, float, 3)
        .DEF_BUFFER_ARRAY(projection_cull_reason, uint32_t, (size_t)(-1))
        .DEF_BUFFER_ARRAY(index_buffer_offset, int32_t, (size_t)(-1))
        // .DEF_BUFFER_ARRAY(sorting_keys_1, sortingKey_t, (size_t)(-1))
        // .DEF_BUFFER_ARRAY(sorting_gauss_idx_1, int32_t, (size_t)(-1))
        // .DEF_BUFFER_ARRAY(sorting_keys_2, sortingKey_t, (size_t)(-1))
        // .DEF_BUFFER_ARRAY(sorting_gauss_idx_2, int32_t, (size_t)(-1))
        .DEF_BUFFER_ARRAY(tile_ranges, int32_t, 2)
        .DEF_BUFFER_ARRAY(pixel_state, float, self.uniforms.image_width, 4)
        .DEF_BUFFER_ARRAY(n_contributors, int32_t, self.uniforms.image_width, 1)
        .DEF_BUFFER_ARRAY(ssim_map, float, self.uniforms.image_width, 12)
        .DEF_BUFFER_ARRAY(v_pixel_state, float, self.uniforms.image_width, 4)
        .DEF_BUFFER_ARRAY(v_xy_vs, float, 2)
        .DEF_BUFFER_ARRAY(v_depths, float, 1)
        .DEF_BUFFER_ARRAY(v_inv_cov_vs_opacity, float, 4)
        .DEF_BUFFER_ARRAY(v_rgb, float, 3)
        .DEF_BUFFER_ARRAY(relocation_stats, float, 2)
        .def("_set_num_splats", [] (VkSplat &self, size_t n) {
            self.buffers.num_splats = n;
        }, "Directly set buffers.num_splats (use at own risk)", py::arg("num_splats"))
        ;

    py::register_exception<std::runtime_error>(m, "RuntimeError");
}
