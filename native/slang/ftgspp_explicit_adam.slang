#include "ftgspp_dynamic.slang"
#include "ftgspp_work_control.slang"

import strategy_utils;
import spherical_harmonics;

// Persistent parameters.
layout(binding=0) RWStructuredBuffer<float> base_xyz_ws;
layout(binding=1) RWStructuredBuffer<float4> base_log_scales_opacity;
layout(binding=2) RWStructuredBuffer<float2> times_durations;
layout(binding=3) RWStructuredBuffer<float4> velocities_gates;
layout(binding=4) RWStructuredBuffer<float4> sh_coeffs;
layout(binding=5) RWStructuredBuffer<float4> rotations;

// Raw FP32 gradients.  These may have been replaced with a cross-device mean
// between the backward and optimizer dispatches.
layout(binding=6) RWStructuredBuffer<float> raw_grad_base_xyz_ws;
layout(binding=7) RWStructuredBuffer<float4> raw_grad_base_log_scales_opacity;
layout(binding=8) RWStructuredBuffer<float2> raw_grad_times_durations;
layout(binding=9) RWStructuredBuffer<float4> raw_grad_velocities_gates;
layout(binding=10) RWStructuredBuffer<float4> raw_grad_sh_coeffs;
layout(binding=11) RWStructuredBuffer<float4> raw_grad_rotations;

// Adam moments.
layout(binding=12) RWStructuredBuffer<float> g_base_xyz_ws;
layout(binding=13) RWStructuredBuffer<float4> g_base_log_scales_opacity;
layout(binding=14) RWStructuredBuffer<float2> g_times_durations;
layout(binding=15) RWStructuredBuffer<float4> g_velocities_gates;
layout(binding=16) RWStructuredBuffer<float4> g_sh_coeffs_1;
layout(binding=17) RWStructuredBuffer<float4> g_sh_coeffs_2;
layout(binding=18) RWStructuredBuffer<float4> g_rotations;

// Optional geometry / SH gradient scalars (sum of squares). Gated by
// reserved2; non-log steps never touch this buffer beyond the descriptor bind.
// Layout: xyz, log-scales, opacity-logit, rotations, sh_dc, sh_rest.
layout(binding=19) RWByteAddressBuffer grad_stats;
layout(binding=20) StructuredBuffer<TileWorkControl> work_control;

typedef vector<float,1> float1;

__generic<let dim: int>
void ftgspp_adam(
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

// This intentionally keeps the 192-byte push-constant ABI of the old fused
// pipeline.  It lets the public single-card entry point dispatch backward and
// Adam with one shared uniform construction.
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
    if (idx >= uniforms.num_splats)
        return;

    const uint m1 = 2 * idx;
    const uint m2 = m1 + 1;
    const uint optimizer_batch_size =
        max(uniforms.reserved_layout & 0xffffu, 1u);

    const float3 grad_mean = read_t3_float3(idx, raw_grad_base_xyz_ws);
    const float4 grad_scale_opacity = raw_grad_base_log_scales_opacity[idx];
    const float2 grad_time_duration = raw_grad_times_durations[idx];
    const float4 grad_velocity_gate = raw_grad_velocities_gates[idx];
    const float4 grad_rotation = raw_grad_rotations[idx];

    // reserved2 gates optional logging. Uniform-true for all lanes, so the
    // WaveActiveSum path is skipped entirely on ordinary training steps.
    // Rotation logging undoes normalize(q)'s 1/||q|| VJP scale and drops any
    // residual radial component; Adam below still consumes raw grad_rotation.
    if (uniforms.reserved2 > 0.5f) {
        const float4 q = rotations[idx];
        const float q_norm_sq = max(dot(q, q), 1e-24f);
        const float4 q_hat = q * rsqrt(q_norm_sq);
        const float4 g_rot_t =
            grad_rotation - dot(grad_rotation, q_hat) * q_hat;
        const float rot_sum_sq = dot(g_rot_t, g_rot_t) * q_norm_sq;
        float4 grad_sum_sq = float4(
            dot(grad_mean, grad_mean),
            dot(grad_scale_opacity.xyz, grad_scale_opacity.xyz),
            grad_scale_opacity.w * grad_scale_opacity.w,
            rot_sum_sq);
        grad_sum_sq = WaveActiveSum(grad_sum_sq);
        if (WaveIsFirstLane()) {
            grad_stats.InterlockedAddF32(0u, grad_sum_sq.x);
            grad_stats.InterlockedAddF32(4u, grad_sum_sq.y);
            grad_stats.InterlockedAddF32(8u, grad_sum_sq.z);
            grad_stats.InterlockedAddF32(12u, grad_sum_sq.w);
        }
    }

    {
        float3 x = read_t3_float3(idx, base_xyz_ws);
        float3 m = read_mean(idx, g_base_xyz_ws, 2, 0);
        float3 v = read_mean(idx, g_base_xyz_ws, 2, 1);
        ftgspp_adam(
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
        ftgspp_adam(
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
        ftgspp_adam(
            x, grad_time_duration, m, v, uniforms.step,
            optimizer_batch_size,
            float2(uniforms.lr_times, uniforms.lr_durations));
        times_durations[idx] = x;
        g_times_durations[m1] = m;
        g_times_durations[m2] = v;
    }
    {
        float4 x = velocities_gates[idx];
        float4 m = g_velocities_gates[m1];
        float4 v = g_velocities_gates[m2];
        ftgspp_adam(
            x, grad_velocity_gate, m, v, uniforms.step,
            optimizer_batch_size,
            float4(float3(uniforms.lr_velocities), uniforms.lr_gates));
        velocities_gates[idx] = x;
        g_velocities_gates[m1] = m;
        g_velocities_gates[m2] = v;
    }
    {
        float4 x = rotations[idx];
        float4 m = g_rotations[m1];
        float4 v = g_rotations[m2];
        ftgspp_adam(
            x, grad_rotation, m, v, uniforms.step, optimizer_batch_size,
            uniforms.lr_quats);
        rotations[idx] = bool(NORMALIZE_QUATS_IN_OPTIMIZER) ? normalize(x) : x;
        g_rotations[m1] = m;
        g_rotations[m2] = v;
    }

    const uint active_sh = uniforms.active_sh & 0xff;
    const uint num_sh = get_num_sh(active_sh);
    const uint reorder_offset = idx % SH_REORDER_SIZE;
    const uint reorder_base = idx - reorder_offset;
    // Accumulators stay zero on ordinary steps; reserved2 is uniform across
    // the wave, so the SH grad-stats path is skipped entirely then.
    float2 sh_grad_sum_sq = float2(0.0f, 0.0f);
    const bool collect_grad_stats = uniforms.reserved2 > 0.5f;
    for (int di = 0; di < 12; ++di) {
        const uint cid = 12 * idx + di;
        const uint bid = 12 * reorder_base + di * SH_REORDER_SIZE + reorder_offset;
        const float4 grad = raw_grad_sh_coeffs[bid];
        if (collect_grad_stats) {
            // First float4 packs SH DC RGB + the first rest channel.
            if (di == 0) {
                sh_grad_sum_sq.x += grad.x * grad.x + grad.y * grad.y
                    + grad.z * grad.z;
                sh_grad_sum_sq.y += grad.w * grad.w;
            } else {
                sh_grad_sum_sq.y += dot(grad, grad);
            }
        }
        if (!is_sh_skipped(uniforms.step, cid, num_sh)) {
            const float4 lr = cid % 12 == 0
                ? float4(uniforms.lr_sh_dc, uniforms.lr_sh_dc,
                         uniforms.lr_sh_dc, uniforms.lr_sh_rest)
                : float4(uniforms.lr_sh_rest);
            float4 x = sh_coeffs[bid];
            float4 m = g_sh_coeffs_1[bid];
            float4 v = g_sh_coeffs_2[bid];
            ftgspp_adam(
                x, grad, m, v, uniforms.step, optimizer_batch_size, lr);
            sh_coeffs[bid] = x;
            g_sh_coeffs_1[bid] = m;
            g_sh_coeffs_2[bid] = v;
        }
        raw_grad_sh_coeffs[bid] = float4(0);
    }
    if (collect_grad_stats) {
        sh_grad_sum_sq = WaveActiveSum(sh_grad_sum_sq);
        if (WaveIsFirstLane()) {
            grad_stats.InterlockedAddF32(16u, sh_grad_sum_sq.x);
            grad_stats.InterlockedAddF32(20u, sh_grad_sum_sq.y);
        }
    }

    // Gradients are single-use.  Clearing them here prevents an accidental
    // second optimizer_step from replaying the previous distributed batch.
    write_t3_float3(float3(0), idx, raw_grad_base_xyz_ws);
    raw_grad_base_log_scales_opacity[idx] = float4(0);
    raw_grad_times_durations[idx] = float2(0);
    raw_grad_velocities_gates[idx] = float4(0);
    raw_grad_rotations[idx] = float4(0);
}
