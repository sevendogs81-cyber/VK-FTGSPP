#define NO_MAIN
#include "vertex_shader.slang"
#undef NO_MAIN

#include "ftgspp_dynamic.slang"
#include "ftgspp_work_control.slang"

import strategy_utils;
import spherical_harmonics;

// Single-device production path. It consumes raster backward outputs and
// updates explicit FTGSPP parameters/moments directly, without materializing
// the full raw-gradient ABI required by synchronous multi-device training.
layout(binding=0) RWStructuredBuffer<float> base_xyz_ws;
layout(binding=1) RWStructuredBuffer<float4> base_log_scales_opacity;
layout(binding=2) RWStructuredBuffer<float2> times_durations;
layout(binding=3) RWStructuredBuffer<float4> velocities_gates;
layout(binding=4) RWStructuredBuffer<float4> sh_coeffs;
layout(binding=5) RWStructuredBuffer<float4> rotations;

layout(binding=6) StructuredBuffer<int> tiles_touched;
layout(binding=7) StructuredBuffer<float2> v_output_xy_vs;
layout(binding=8) StructuredBuffer<float4> v_output_inv_cov_vs;
layout(binding=9) StructuredBuffer<float> v_output_rgb;
layout(binding=10) RWStructuredBuffer<float2> relocation_stats;

layout(binding=11) RWStructuredBuffer<float> g_base_xyz_ws;
layout(binding=12) RWStructuredBuffer<float4> g_base_log_scales_opacity;
layout(binding=13) RWStructuredBuffer<float2> g_times_durations;
layout(binding=14) RWStructuredBuffer<float4> g_velocities_gates;
layout(binding=15) RWStructuredBuffer<float4> g_sh_coeffs_1;
layout(binding=16) RWStructuredBuffer<float4> g_sh_coeffs_2;
layout(binding=17) RWStructuredBuffer<float4> g_rotations;
layout(binding=18) RWByteAddressBuffer loss_accum;
layout(binding=19) RWByteAddressBuffer grad_stats;
layout(binding=20) StructuredBuffer<TileWorkControl> work_control;

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

__generic<let dim: int>
void ftgspp_fused_adam(
    inout vector<float,dim> x,
    in vector<float,dim> grad,
    inout vector<float,dim> m,
    inout vector<float,dim> v,
    float step,
    uint optimizer_batch_size,
    vector<float,dim> lr)
{
    const float batch = float(max(optimizer_batch_size, 1u));
    const float beta1 = pow(0.9f, batch);
    const float beta2 = pow(0.999f, batch);
    const float eps = 1e-15f / sqrt(batch);
    m = lerp(m, grad, 1.0f - beta1);
    v = lerp(v, grad * grad, 1.0f - beta2);
    const vector<float,dim> m_hat = m / (1.0f - pow(beta1, step));
    const vector<float,dim> v_hat = v / (1.0f - pow(beta2, step));
    x -= lr * m_hat / (sqrt(v_hat) + eps);
}

#define GROUP_SIZE SUBGROUP_SIZE

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uniform FTGSPPBackwardUniforms uniforms)
{
    if (tile_work_aborted(work_control[0]))
        return;
    const uint idx = tid.x;
    if (idx >= uniforms.num_splats)
        return;

    const uint active_sh = uniforms.active_sh & 0xffu;
    const uint camera_model = uniforms.active_sh >> 8u;
    const uint num_sh = get_num_sh(active_sh);
    const uint optimizer_batch_size =
        max(uniforms.reserved_layout & 0xffffu, 1u);
    const bool collect_loss_scalar =
        (uniforms.reserved_layout & 0x80000000u) != 0u;
    const bool collect_grad_stats = uniforms.reserved2 > 0.5f;
    const bool zero_grad = tiles_touched[idx] == 0;
    const float2 grad_xy = zero_grad ? float2(0.0f) : v_output_xy_vs[idx];

    const float W = float(uniforms.image_size & 0xffffu);
    const float H = float(uniforms.image_size >> 16u);
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
        uniforms.image_size >> 16u,
        uniforms.image_size & 0xffffu);
    Gaussian_3D gauss = {
        mg.mean,
        read_spherical_harmonics_coeffs(idx, sh_coeffs, active_sh),
        rotations[idx],
        mg.scale,
        mg.opacity
    };

    Splat_2D_Vertex v_splat = {
        float3(grad_xy, 0.0f),
        zero_grad ? float3(0.0f) : read_t3_float3(idx, v_output_rgb),
        float3(0.0f),
        zero_grad ? float4(0.0f) : v_output_inv_cov_vs[idx],
        0.0f
    };
    DifferentialPair<Gaussian_3D> v_gauss = diffPair(gauss);
    DifferentialPair<Camera> v_cam = diffPair(cam);
    bwd_diff(project_gaussian_to_camera)(v_gauss, v_cam, active_sh, v_splat);

    const float3 grad_mean_t = zero_grad ? float3(0.0f) : v_gauss.d.xyz_ws;
    const float3 grad_scale = zero_grad ? float3(0.0f) : v_gauss.d.scales;
    const float grad_opacity_t = zero_grad ? 0.0f : v_gauss.d.opacities;
    const float4 grad_rotation =
        zero_grad ? float4(0.0f) : v_gauss.d.rotations;
    const float dt = uniforms.timestamp - p.time_center;
    const float temporal_scale = max(
        ftgspp_temporal_scale(p.duration_raw, uniforms.max_duration), 1e-8f);
    const float marginal = mg.marginal;
    const float gate = ftgspp_sigmoid(20.0f * p.gate_raw);
    const float base_opacity = ftgspp_sigmoid(p.opacity_logit);
    const bool active = mg.temporal_opacity > uniforms.marginal_threshold;

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
    grad_log_scale += uniforms.reg_scale * mg.scale;
    grad_opacity_logit += uniforms.reg_opacity * mg.temporal_opacity
        * base_opacity * (1.0f - base_opacity);
    grad_gate_raw -= uniforms.reg_gate * 20.0f * gate * (1.0f - gate);

    const float4 grad_scale_opacity =
        float4(grad_log_scale, grad_opacity_logit);
    const float2 grad_time_duration = float2(grad_time, grad_duration);
    const float4 grad_velocity_gate = float4(grad_velocity, grad_gate_raw);

    if (collect_grad_stats) {
        const float4 q = rotations[idx];
        const float q_norm_sq = max(dot(q, q), 1e-24f);
        const float4 q_hat = q * rsqrt(q_norm_sq);
        const float4 g_rot_t =
            grad_rotation - dot(grad_rotation, q_hat) * q_hat;
        const float rot_sum_sq = dot(g_rot_t, g_rot_t) * q_norm_sq;
        float4 grad_sum_sq = float4(
            dot(grad_mean, grad_mean),
            dot(grad_log_scale, grad_log_scale),
            grad_opacity_logit * grad_opacity_logit,
            rot_sum_sq);
        grad_sum_sq = WaveActiveSum(grad_sum_sq);
        if (WaveIsFirstLane()) {
            grad_stats.InterlockedAddF32(0u, grad_sum_sq.x);
            grad_stats.InterlockedAddF32(4u, grad_sum_sq.y);
            grad_stats.InterlockedAddF32(8u, grad_sum_sq.z);
            grad_stats.InterlockedAddF32(12u, grad_sum_sq.w);
        }
    }

    const uint m1 = 2u * idx;
    const uint m2 = m1 + 1u;
    {
        float3 x = p.mean;
        float3 m = read_mean(idx, g_base_xyz_ws, 2, 0);
        float3 v = read_mean(idx, g_base_xyz_ws, 2, 1);
        ftgspp_fused_adam(
            x, grad_mean, m, v, uniforms.step, optimizer_batch_size,
            uniforms.lr_means);
        write_t3_float3(x, idx, base_xyz_ws);
        write_mean(m, idx, g_base_xyz_ws, 2, 0);
        write_mean(v, idx, g_base_xyz_ws, 2, 1);
    }
    {
        float4 x = base_log_scales_opacity[idx];
        float4 m = g_base_log_scales_opacity[m1];
        float4 v = g_base_log_scales_opacity[m2];
        ftgspp_fused_adam(
            x, grad_scale_opacity, m, v, uniforms.step,
            optimizer_batch_size,
            float4(float3(uniforms.lr_scales), uniforms.lr_opacities));
        base_log_scales_opacity[idx] = x;
        g_base_log_scales_opacity[m1] = m;
        g_base_log_scales_opacity[m2] = v;
    }
    {
        float2 x = times_durations[idx];
        float2 m = g_times_durations[m1];
        float2 v = g_times_durations[m2];
        ftgspp_fused_adam(
            x, grad_time_duration, m, v, uniforms.step,
            optimizer_batch_size,
            float2(uniforms.lr_times, uniforms.lr_durations));
        times_durations[idx] = x;
        g_times_durations[m1] = m;
        g_times_durations[m2] = v;
    }
    {
        // Hard persistence freezes only the binary gate. Motion remains a
        // trainable, independent property in both persistence groups.
        const bool hard_phase = uniforms.reserved0 >= 0.5f;
        float4 x = velocities_gates[idx];
        const float fixed_gate = x.w;
        float4 m = g_velocities_gates[m1];
        float4 v = g_velocities_gates[m2];
        ftgspp_fused_adam(
            x,
            hard_phase ? float4(grad_velocity_gate.xyz, 0.0f) : grad_velocity_gate,
            m, v, uniforms.step, optimizer_batch_size,
            float4(
                float3(uniforms.lr_velocities),
                hard_phase ? 0.0f : uniforms.lr_gates));
        if (hard_phase) {
            x.w = fixed_gate;
            m.w = 0.0f;
            v.w = 0.0f;
        }
        velocities_gates[idx] = x;
        g_velocities_gates[m1] = m;
        g_velocities_gates[m2] = v;
    }
    {
        float4 x = rotations[idx];
        float4 m = g_rotations[m1];
        float4 v = g_rotations[m2];
        ftgspp_fused_adam(
            x, grad_rotation, m, v, uniforms.step, optimizer_batch_size,
            uniforms.lr_quats);
        rotations[idx] = bool(NORMALIZE_QUATS_IN_OPTIMIZER) ? normalize(x) : x;
        g_rotations[m1] = m;
        g_rotations[m2] = v;
    }

    const uint reorder_offset = idx % SH_REORDER_SIZE;
    const uint reorder_base = idx - reorder_offset;
    float2 sh_grad_sum_sq = float2(0.0f);
    for (int di = 0; di < 12; ++di) {
        const uint cid = 12u * idx + uint(di);
        const uint bid = 12u * reorder_base
            + uint(di) * SH_REORDER_SIZE + reorder_offset;
        const float4 grad =
            (zero_grad || is_sh_skipped(uniforms.step, cid, num_sh))
            ? float4(0.0f) : get_sh_coeff_4(v_gauss.d.sh_coeffs, di);
        if (collect_grad_stats) {
            if (di == 0) {
                sh_grad_sum_sq.x +=
                    grad.x * grad.x + grad.y * grad.y + grad.z * grad.z;
                sh_grad_sum_sq.y += grad.w * grad.w;
            } else {
                sh_grad_sum_sq.y += dot(grad, grad);
            }
        }
        if (!is_sh_skipped(uniforms.step, cid, num_sh)) {
            const float4 lr = cid % 12u == 0u
                ? float4(uniforms.lr_sh_dc, uniforms.lr_sh_dc,
                         uniforms.lr_sh_dc, uniforms.lr_sh_rest)
                : float4(uniforms.lr_sh_rest);
            float4 x = sh_coeffs[bid];
            float4 m = g_sh_coeffs_1[bid];
            float4 v = g_sh_coeffs_2[bid];
            ftgspp_fused_adam(
                x, grad, m, v, uniforms.step, optimizer_batch_size, lr);
            sh_coeffs[bid] = x;
            g_sh_coeffs_1[bid] = m;
            g_sh_coeffs_2[bid] = v;
        }
    }
    if (collect_grad_stats) {
        sh_grad_sum_sq = WaveActiveSum(sh_grad_sum_sq);
        if (WaveIsFirstLane()) {
            grad_stats.InterlockedAddF32(16u, sh_grad_sum_sq.x);
            grad_stats.InterlockedAddF32(20u, sh_grad_sum_sq.y);
        }
    }
}
