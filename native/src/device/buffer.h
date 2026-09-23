#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <cstring>  // memcpy
#include <mutex>
#include <cmath>
#include <cstdint>

#include "common/config.h"

// https://stackoverflow.com/a/3312896
#ifdef __GNUC__
#define PACK_STRUCT( __Declaration__ ) __Declaration__ __attribute__((__packed__))
#endif
#ifdef _MSC_VER
#define PACK_STRUCT( __Declaration__ ) __pragma( pack(push, 1) ) __Declaration__ __pragma( pack(pop))
#endif


// Buffers
struct _VulkanBuffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t allocSize;  // allocated size in bytes
    size_t size;  // actual size in bytes
    uint64_t generation; // changes on every allocation, even if handles are reused

    _VulkanBuffer()
        : buffer(VK_NULL_HANDLE), memory(VK_NULL_HANDLE),
          allocSize(0), size(0), generation(0) {}
    
    _VulkanBuffer(const _VulkanBuffer &other)
        : buffer(other.buffer), memory(other.memory),
          allocSize(other.allocSize), size(other.size),
          generation(other.generation) {}

    _VulkanBuffer& operator=(const _VulkanBuffer &other) {
        memcpy((void*)this, &other, sizeof(_VulkanBuffer));
        return *this;
    }

    // used to test if descriptor needs to be updated
    bool operator==(const _VulkanBuffer &other) const {
        // printf("%d %d %d %d  ", int(buffer == other.buffer), int(memory == other.memory),
        //     int(allocSize == other.allocSize), int(size == other.size));
        // return buffer == other.buffer && memory == other.memory &&
        //        allocSize == other.allocSize && size == other.size;
        return buffer == other.buffer && memory == other.memory &&
            allocSize == other.allocSize && generation == other.generation;
    }
};



template<typename T>
class Buffer : public std::vector<T> {
public:
    _VulkanBuffer deviceBuffer;

    Buffer() : std::vector<T>(), deviceBuffer() {}
    Buffer(const Buffer &other) = delete;
    Buffer& operator=(const Buffer &other) = delete;
    Buffer(Buffer&& other) noexcept
        : std::vector<T>(std::move(other)), deviceBuffer(other.deviceBuffer) {
        other.deviceBuffer = _VulkanBuffer();
    }
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            // A Buffer cannot release a Vulkan allocation without its owning
            // pipeline. Move-assignment is therefore only valid into an empty
            // device slot; all ownership transfers in the runtime satisfy it.
            assert(deviceBuffer.buffer == VK_NULL_HANDLE);
            std::vector<T>::operator=(std::move(other));
            deviceBuffer = other.deviceBuffer;
            other.deviceBuffer = _VulkanBuffer();
        }
        return *this;
    }

    size_t byteLength() const { return this->size() * sizeof(T); }
    size_t deviceSize() const { return deviceBuffer.size / sizeof(T); }

};

#define _DECL_DEVICE_BUFFER(dtype, name) Buffer<dtype> name = Buffer<dtype>(false) 
// #define _DECL_DEVICE_BUFFER(dtype, name) Buffer<dtype> name



struct VulkanGSPipelineBuffers {
    size_t num_splats = 0;
    size_t num_indices = 0;

    // projection inputs
    Buffer<float> xyz_ws;  // (N, 3)
    Buffer<float> sh_coeffs;  // (N, 16, 3)
    Buffer<float> rotations;  // (N, 4)
    Buffer<float> scales_opacs;  // (N, 4)

    // FTGSPP 4D parameters.  These are the persistent trainable values;
    // xyz_ws/scales_opacs above are materialized for the current timestamp and
    // remain the exact inputs expected by VkSplat's projection/tile/blend path.
    Buffer<float> base_xyz_ws;             // (N, 3)
    Buffer<float> base_log_scales_opacity; // (N, 4): log-scales + opacity logit
    Buffer<float> times_durations;         // (N, 2): center + raw duration
    Buffer<float> velocities_gates;        // (N, 4): explicit velocity + raw gate
    Buffer<float> flow_end_xyz;            // (N, 3), optical-flow endpoint

    // Complete explicit-model FP32 gradients.  They are deliberately separate
    // from Adam moments so a distributed caller can reduce them before the
    // optimizer mutates any persistent Gaussian parameter.
    Buffer<float> grad_base_xyz_ws;             // (N, 3)
    Buffer<float> grad_base_log_scales_opacity; // (N, 4)
    Buffer<float> grad_times_durations;         // (N, 2)
    Buffer<float> grad_velocities_gates;        // (N, 4)
    Buffer<float> grad_sh_coeffs;                // reordered/padded (N, 16, 3)
    Buffer<float> grad_rotations;                // (N, 4)

    // Host-mailbox peer payload.  Local raw gradients stay on the device;
    // only the other rank's bucket is uploaded here.  A Vulkan compute pass
    // forms the mean in the local buffers before the existing Adam shaders.
    Buffer<float> peer_grad_base_xyz_ws;
    Buffer<float> peer_grad_base_log_scales_opacity;
    Buffer<float> peer_grad_times_durations;
    Buffer<float> peer_grad_velocities_gates;
    Buffer<float> peer_grad_sh_coeffs;
    Buffer<float> peer_grad_rotations;

    // Per-camera affine color correction (N_cam * 12 floats).
    Buffer<float> color_params;
    Buffer<float> color_grad;
    Buffer<float> color_m;
    Buffer<float> color_v;
    Buffer<float> peer_color_grad;
    Buffer<float> color_raw_rgb;  // pre-correct RGBA snapshot for backward

    // Visibility-sparse exchange. V2 records contain an id plus active
    // gradients and only the current active-SH payload (up to 66/62 words).
    Buffer<uint32_t> sparse_gradient_records;
    Buffer<uint32_t> peer_sparse_gradient_records;
    Buffer<uint32_t> sparse_gradient_count;
    // Persistent per-generation visible union and its GPU inclusive scan.
    // No visibility mask is read or sorted by the host.
    Buffer<int32_t> sparse_active_flags;
    Buffer<int32_t> sparse_active_offsets;
    // One timestamp per rank for the exact analytic regularizer mean.
    Buffer<float> peer_relocation_stats;

    // projection outputs
    Buffer<int32_t> tiles_touched;  // (N,)
    Buffer<int64_t> rect_tile_space;  // (N,)
    Buffer<int32_t> radii;  // (N,)
    Buffer<float> xy_vs;  // (N, 2)
    Buffer<float> depths;  // (N, 1)
    Buffer<float> inv_cov_vs_opacity;  // (N, 4)
    Buffer<float> rgb;  // (N, 3)
    Buffer<uint32_t> projection_cull_reason;  // (N,), diagnostic bitmask

    // tiles
    Buffer<int32_t> index_buffer_offset;  // N
    Buffer<sortingKey_t> sorting_keys_1;  // NInt [no_shrink]
    Buffer<sortingKey_t> sorting_keys_2;  // NInt [no_shrink]
    Buffer<int32_t> sorting_gauss_idx_1;  // NInt [no_shrink]
    Buffer<int32_t> sorting_gauss_idx_2;  // NInt [no_shrink]
    Buffer<int32_t> tile_ranges;  // (Gh*Gw, 2)
    // GPU-side exact-culling diagnostic bits for the latest key generation:
    // bit 0 = count/emit mismatch, bit 1 = emitted tile outside the grid.
    Buffer<uint32_t> tile_invariant_status;  // (1,)
    Buffer<uint32_t> tile_work_control;      // TileWorkControl words
    Buffer<uint32_t> pipeline_abort_latch;   // PipelineAbortLatch words
    size_t tile_pair_capacity = 0;
    bool is_unsorted_1 = true;
    Buffer<sortingKey_t>& unsorted_keys()
      { return is_unsorted_1 ? sorting_keys_1 : sorting_keys_2; }
    Buffer<sortingKey_t>& sorted_keys()
      { return is_unsorted_1 ? sorting_keys_2 : sorting_keys_1; }
    Buffer<sortingKey_t>& unsorted_gauss_idx()
      { return is_unsorted_1 ? sorting_gauss_idx_1 : sorting_gauss_idx_2; }
    Buffer<sortingKey_t>& sorted_gauss_idx()
      { return is_unsorted_1 ? sorting_gauss_idx_2 : sorting_gauss_idx_1; }

    // pixels
    Buffer<float> pixel_state;  // (H, W, 4)
    Buffer<int32_t> n_contributors;  // (H, W, 1)

    // rasterization v_outputs
    Buffer<float> ssim_map;  // (H, W, 3, 4)
    Buffer<float> v_pixel_state;  // (H, W, 4)
    Buffer<uint8_t> ref_image;  // (H, W, 4)
    // Bit 0: valid L1 pixel. Bit 1: center has a fully valid 11x11 SSIM
    // window. Bit 2: that window lies entirely inside the reference mask.
    Buffer<uint32_t> ref_validity;  // (H, W)
    size_t ref_validity_width = 0;
    size_t ref_validity_height = 0;
    // Native objective terms for the most recent training step:
    // L1, SSIM, opacity-reg, scale-reg, gate, color-cc.
    // Kept on-device until Python requests a log row.
    Buffer<float> loss_accum;  // (6,)
    // Hold-out ranking: se_sum, ssim_sum, fg_se_sum, fg_pixels,
    // valid_pixels, valid_windows, roi_ssim_sum, roi_windows,
    // foreground_ssim_sum, foreground_windows.
    Buffer<float> eval_metrics;  // (10,)
    Buffer<uint32_t> eval_validity;  // (H, W) ranking validity, not training's
    size_t eval_validity_width = 0;
    size_t eval_validity_height = 0;
    // Optional geometry / SH gradient sum-of-squares collected inside Adam when
    // logging is enabled: xyz, log-scales, opacity-logit, rotations, sh_dc,
    // sh_rest.
    Buffer<float> grad_stats;  // (6,)

    // rasterization v_inputs, projection v_outputs
    Buffer<float> v_xy_vs;  // (N, 2)
    Buffer<float> v_depths;  // (N, 1)
    Buffer<float> v_inv_cov_vs_opacity;  // (N, 4)
    Buffer<float> v_rgb;  // (N, 3)

    // optimizer parameters (Adam)
    Buffer<float> g_xyz_ws;  // (N, 2, 3)
    Buffer<float> g_sh_coeffs_1;  // (N, 16, 3)
    Buffer<float> g_sh_coeffs_2;  // (N, 16, 3)
    Buffer<float> g_rotations;  // (N, 2, 4)
    Buffer<float> g_scales_opacs;  // (N, 2, 4)

    // Adam moments for persistent FTGSPP 4D parameters.
    Buffer<float> g_times_durations;  // (N, 2, 2)
    Buffer<float> g_velocities_gates; // (N, 2, 4)

    // Fixed-capacity FTGSPP relocation: destination i copies donor map[i].
    Buffer<int32_t> relocation_map;  // (N,), -1 keeps the current splat
    Buffer<int32_t> relocation_counts; // (N,), number of copies sourced here
    Buffer<float> relocation_raw;      // (N,4): opacity, grad2d, static score, alive
    Buffer<float> relocation_cdf;      // (N,), inclusive donor scan (unnormalized)
    Buffer<float> relocation_totals;   // (4,), reduction scratch
    Buffer<uint32_t> relocation_status; // dead, alive, all-dead, alive-uniform fallback

    Buffer<float> relocation_stats;  // (N, 2), grad magnitude sum and count
    // Device snapshot taken before the view backward of a sync generation.
    // Overflow abort restores this so replay does not double-count.
    Buffer<float> relocation_stats_generation;

    // intermediate buffers
    Buffer<float> _temp_gauss_attr;
    Buffer<int32_t> _cumsum_blockSums;
    Buffer<int32_t> _cumsum_blockSums2;
    Buffer<float> _cumsum_f32_blockSums;
    Buffer<float> _cumsum_f32_blockSums2;
    Buffer<int32_t> _sorting_histogram;
    Buffer<int32_t> _sorting_histogram_cumsum;
    
    size_t getTotalAllocSize();
    void updateTotalAllocSize();
    size_t maxTotalAllocSize = 0;
    std::map<std::string, size_t> getVramBreakdown();

    template<typename T> static void reorderSH(Buffer<T> &coeffs);
    template<typename T> static void undoReorderSH(Buffer<T> &coeffs, size_t num_splats);

    static void assignScalesOpacs(Buffer<float> &scales_opacs, size_t n, const float* scales, const float* opacs);
};

float halfToFloat(uint16_t h);
uint16_t floatToHalf(float f);
