#define NO_MAIN
#include "vertex_shader.slang"
#undef NO_MAIN

#include "ftgspp_dynamic.slang"
#include "ftgspp_work_control.slang"

import strategy_utils;

// Shared parameter backward. The ordinary variant writes one raw gradient;
// the local-batch specialization writes the four-view FP32 mean directly into
// the same buffers consumed by ftgspp_explicit_adam.slang. No intermediate
// gradient, separate reduction dispatch, or optimizer variant is needed.
#ifndef LOCAL_BATCH_MEAN
#define LOCAL_BATCH_MEAN 0
#endif

#if LOCAL_BATCH_MEAN
// Preserve the previous reduction's multiply-then-add FP32 order. In particular,
// do not contract the accumulated sum into an FMA or read stale first-view data.
__generic<let dim: int>
vector<float,dim> add_batch_gradient(
    vector<float,dim> previous, vector<float,dim> gradient)
{
    precise vector<float,dim> scaled = gradient * 0.25f;
    precise vector<float,dim> result = previous + scaled;
    return result;
}
#define STORE_GRADIENT(target, gradient) \
    if (reset_mean) { target = (gradient) * 0.25f; } \
    else { target = add_batch_gradient(target, gradient); }
#else
#define STORE_GRADIENT(target, gradient) target = gradient;
#endif

// Persistent FTGSPP parameters (read-only during backward).
layout(binding=0) StructuredBuffer<float> base_xyz_ws;
layout(binding=1) StructuredBuffer<float4> base_log_scales_opacity;
layout(binding=2) StructuredBuffer<float2> times_durations;
layout(binding=3) StructuredBuffer<float4> velocities_gates;
layout(binding=4) StructuredBuffer<float4> sh_coeffs;
layout(binding=5) StructuredBuffer<float4> rotations;

// Gradients produced by VkSplat's unchanged tile alpha-blending backward pass.
layout(binding=6) StructuredBuffer<int> tiles_touched;
layout(binding=7) StructuredBuffer<float2> v_output_xy_vs;
layout(binding=8) StructuredBuffer<float4> v_output_inv_cov_vs;
layout(binding=9) StructuredBuffer<float> v_output_rgb;

// Relocation statistics are sums/counts, not optimizer gradients.  They remain
// local to this view and are reduced separately by the distributed runtime.
layout(binding=10) RWStructuredBuffer<float2> relocation_stats;

// Complete per-parameter FP32 gradient state.  SH uses the same block-reordered
// float4 layout as sh_coeffs; all other buffers are logical Gaussian-major.
layout(binding=11) RWStructuredBuffer<float> raw_grad_base_xyz_ws;
layout(binding=12) RWStructuredBuffer<float4> raw_grad_base_log_scales_opacity;
layout(binding=13) RWStructuredBuffer<float2> raw_grad_times_durations;
layout(binding=14) RWStructuredBuffer<float4> raw_grad_velocities_gates;
layout(binding=15) RWStructuredBuffer<float4> raw_grad_sh_coeffs;
layout(binding=16) RWStructuredBuffer<float4> raw_grad_rotations;
layout(binding=17) RWByteAddressBuffer loss_accum;
layout(binding=18) StructuredBuffer<TileWorkControl> work_control;

struct FTGSPPBackwardUniforms
{
    uint32_t step;
    uint32_t active_sh;
    uint32_t num_splats;
    uint32_t image_size;
    float fx;
    float fy;
    float cx;
    float cy;
    float4 dist_coeffs;
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
    float reserved0;
    float reserved1;
    float reserved2;
    float4x4 world_view_transform;
};

#define GROUP_SIZE SUBGROUP_SIZE

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uniform FTGSPPBackwardUniforms uniforms)
{
    if (tile_work_aborted(work_control[0]))
        return;
    const uint idx = tid.x;
#if LOCAL_BATCH_MEAN
    // The dispatch rounds N up to SUBGROUP_SIZE == SH_REORDER_SIZE. Only the
    // unused lanes of the final SH block need zeroing; valid lanes overwrite
    // their complete gradient below. This avoids a full SH-buffer clear.
    const bool reset_mean = uniforms.reserved2 > 0.5f;
    if (idx >= uniforms.num_splats && reset_mean) {
        const uint offset = idx % SH_REORDER_SIZE;
        const uint base = idx - offset;
        for (uint di = 0; di < 12u; ++di)
            raw_grad_sh_coeffs[12u * base + di * SH_REORDER_SIZE + offset] = float4(0);
    }
#endif
    if (idx >= uniforms.num_splats)
        return;

    const uint active_sh = uniforms.active_sh & 0xff;
    const uint camera_model = uniforms.active_sh >> 8;
    const bool zero_grad = tiles_touched[idx] == 0;
    const bool sparse_image_only = uniforms.reserved1 >= 0.5f;
    const float2 grad_xy = zero_grad ? float2(0) : v_output_xy_vs[idx];

    // Match ftgspp train/state.py anisotropic grad2d: |[gx*W/2, gy*H/2]|.
    const float W = float(uniforms.image_size & 0xffff);
    const float H = float(uniforms.image_size >> 16);
    const float2 scaled_grad = float2(
        grad_xy.x * (W * 0.5f), grad_xy.y * (H * 0.5f));
    relocation_stats[idx] += float2(
        length(scaled_grad), zero_grad ? 0.0f : 1.0f);

    DynamicGaussian4D p = load_ftgspp(
        idx, base_xyz_ws, base_log_scales_opacity,
        times_durations, velocities_gates);
    MaterializedGaussian mg = materialize_ftgspp(
        p, uniforms.timestamp, uniforms.max_duration,
        uniforms.marginal_threshold);

    Camera cam = load_camera(
        uniforms.world_view_transform,
        uniforms.fx, uniforms.fy, uniforms.cx, uniforms.cy,
        camera_model, uniforms.dist_coeffs,
        uniforms.image_size >> 16,
        uniforms.image_size & 0xffff);
    Gaussian_3D gauss = {
        mg.mean,
        read_spherical_harmonics_coeffs(idx, sh_coeffs, active_sh),
        rotations[idx],
        mg.scale,
        mg.opacity
    };

    Splat_2D_Vertex v_splat = {
        float3(grad_xy, 0.0f),
        zero_grad ? float3(0) : read_t3_float3(idx, v_output_rgb),
        float3(0),
        zero_grad ? float4(0) : v_output_inv_cov_vs[idx],
        0.0f
    };
    DifferentialPair<Gaussian_3D> v_gauss = diffPair(gauss);
    DifferentialPair<Camera> v_cam = diffPair(cam);
    bwd_diff(project_gaussian_to_camera)(v_gauss, v_cam, active_sh, v_splat);

    const float3 grad_mean_t = zero_grad ? float3(0) : v_gauss.d.xyz_ws;
    const float3 grad_scale = zero_grad ? float3(0) : v_gauss.d.scales;
    const float grad_opacity_t = zero_grad ? 0.0f : v_gauss.d.opacities;
    const float4 grad_rotation = zero_grad ? float4(0) : v_gauss.d.rotations;

    const float dt = uniforms.timestamp - p.time_center;
    const float temporal_scale = max(
        ftgspp_temporal_scale(p.duration_raw, uniforms.max_duration), 1e-8f);
    const float marginal = mg.marginal;
    const float gate = ftgspp_sigmoid(20.0f * p.gate_raw);
    const float base_opacity = ftgspp_sigmoid(p.opacity_logit);
    const bool active = mg.temporal_opacity > uniforms.marginal_threshold;

    // Scalar counterparts of the three regularizer gradients below. Each
    // dispatch is one sampled-view objective; synchronous training averages
    // these per-rank scalars exactly like its raw gradients.
    const bool collect_loss_scalar =
        (uniforms.reserved_layout & 0x80000000u) != 0u;
    if (collect_loss_scalar) {
        float3 regularizer_loss = float3(
            uniforms.reg_opacity * mg.temporal_opacity * base_opacity,
            uniforms.reg_scale * (mg.scale.x + mg.scale.y + mg.scale.z),
            uniforms.reg_gate * (1.0f - gate));
        regularizer_loss = WaveActiveSum(regularizer_loss);
        if (WaveIsFirstLane()) {
            loss_accum.InterlockedAddF32(8u, regularizer_loss.x);
            loss_accum.InterlockedAddF32(12u, regularizer_loss.y);
            loss_accum.InterlockedAddF32(16u, regularizer_loss.z);
        }
    }

    const float3 grad_mean = grad_mean_t;
    float3 grad_velocity = grad_mean_t * dt;
    float grad_time = -dot(grad_mean_t, p.velocity);
    float3 grad_log_scale = grad_scale * mg.scale;
    float grad_opacity_logit = 0.0f;
    float grad_duration = 0.0f;
    float grad_gate_raw = 0.0f;

    if (active) {
        grad_opacity_logit = grad_opacity_t * mg.temporal_opacity
            * base_opacity * (1.0f - base_opacity);
        const float grad_temporal = grad_opacity_t * base_opacity;
        const float grad_marginal = grad_temporal * (1.0f - gate);
        const float grad_temporal_scale = grad_marginal * marginal
            * dt * dt / (temporal_scale * temporal_scale * temporal_scale);
        grad_time += grad_marginal * marginal * dt
            / (temporal_scale * temporal_scale);

        if (uniforms.max_duration > 0.0f) {
            const float d = ftgspp_sigmoid(p.duration_raw);
            grad_duration = grad_temporal_scale
                * (2.0f * uniforms.max_duration * d * (1.0f - d));
        } else {
            grad_duration = grad_temporal_scale * temporal_scale;
        }
        grad_gate_raw = grad_temporal * (1.0f - marginal)
            * 20.0f * gate * (1.0f - gate);
    }

    // FTGSPP regularizers use the same parameterization as the old fused
    // implementation.  Averaging complete per-rank gradients keeps each
    // regularizer at its configured weight (rather than multiplying by P).
    if (!sparse_image_only) {
        grad_log_scale += uniforms.reg_scale * mg.scale;
        grad_opacity_logit += uniforms.reg_opacity * mg.temporal_opacity
            * base_opacity * (1.0f - base_opacity);
        grad_gate_raw -= uniforms.reg_gate * 20.0f * gate * (1.0f - gate);
    }

    float3 output_mean;
#if LOCAL_BATCH_MEAN
    if (reset_mean)
        output_mean = grad_mean * 0.25f;
    else
        output_mean = add_batch_gradient(
            read_t3_float3(idx, raw_grad_base_xyz_ws), grad_mean);
#else
    output_mean = grad_mean;
#endif
    write_t3_float3(output_mean, idx, raw_grad_base_xyz_ws);
    STORE_GRADIENT(raw_grad_base_log_scales_opacity[idx],
        float4(grad_log_scale, grad_opacity_logit))
    STORE_GRADIENT(raw_grad_times_durations[idx], float2(grad_time, grad_duration))
    STORE_GRADIENT(raw_grad_velocities_gates[idx], float4(grad_velocity, grad_gate_raw))
    STORE_GRADIENT(raw_grad_rotations[idx], grad_rotation)

    const uint reorder_offset = idx % SH_REORDER_SIZE;
    const uint reorder_base = idx - reorder_offset;
    const uint num_sh = get_num_sh(active_sh);
    for (int di = 0; di < 12; ++di) {
        const uint cid = 12 * idx + di;
        const uint bid = 12 * reorder_base + di * SH_REORDER_SIZE + reorder_offset;
        const float4 grad_sh =
            (zero_grad || is_sh_skipped(uniforms.step, cid, num_sh))
            ? float4(0) : get_sh_coeff_4(v_gauss.d.sh_coeffs, di);
        STORE_GRADIENT(raw_grad_sh_coeffs[bid], grad_sh)
    }
}
