#include "config.slang"
#include "ftgspp_relocation_binoms.slang"

import utils;
import strategy_utils;

layout(binding=0) StructuredBuffer<int> relocation_counts;
layout(binding=1) RWStructuredBuffer<float4> base_log_scales_opacity;
layout(binding=2) RWStructuredBuffer<float> g_base_xyz_ws;
layout(binding=3) RWStructuredBuffer<float4> g_sh_coeffs_1;
layout(binding=4) RWStructuredBuffer<float4> g_sh_coeffs_2;
layout(binding=5) RWStructuredBuffer<float4> g_rotations;
layout(binding=6) RWStructuredBuffer<float4> g_base_log_scales_opacity;
layout(binding=7) RWStructuredBuffer<float2> g_times_durations;
layout(binding=8) RWStructuredBuffer<float4> g_velocities_gates;

struct RelocationUniforms
{
    uint32_t num_splats;
    uint32_t mode; // 0 partial_copy, 1 exact_copy, 2 3d_mcmc
    float min_opacity;
    uint32_t reserved;
};

static const float FLOAT32_EPSILON = 1.1920928955078125e-7f;

[shader("compute")]
[numthreads(SUBGROUP_SIZE, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uniform RelocationUniforms uniforms)
{
    uint idx = tid.x;
    if (idx >= uniforms.num_splats)
        return;
    int copies = relocation_counts[idx];
    if (copies <= 0)
        return;

    if (uniforms.mode == 2) {
        // gsplat==1.5.3 clamps ratios to binoms.shape[0] (51). Match its
        // float32 Equation (9) coefficient table and operation order.
        int ratio = min(copies + 1, int(FTGSPP_RELOCATION_MAX_RATIO));
        float4 raw = base_log_scales_opacity[idx];
        float opacity = 1.0 / (1.0 + exp(-raw.w));
        float new_opacity = 1.0 - pow(1.0 - opacity, 1.0 / ratio);
        float denom = 0.0;
        for (int i = 1; i <= ratio; ++i) {
            for (int k = 0; k <= i - 1; ++k) {
                const float sign = (k & 1) != 0 ? -1.0f : 1.0f;
                const float term = (sign / sqrt(float(k + 1)))
                    * pow(new_opacity, k + 1);
                denom += ftgspp_relocation_binom(uint(i - 1), uint(k)) * term;
            }
        }
        float coeff = opacity / denom;
        if (!isfinite(opacity) || !isfinite(new_opacity) ||
            !isfinite(denom) || denom <= 0.0f ||
            !isfinite(coeff) || coeff <= 0.0f)
            return;
        new_opacity = clamp(
            new_opacity, uniforms.min_opacity, 1.0f - FLOAT32_EPSILON);
        // Source order is exp(raw scale) -> coeff multiply -> torch.log.
        float3 new_log_scale = log(coeff * exp(raw.xyz));
        if (!isfinite(new_log_scale.x) || !isfinite(new_log_scale.y) ||
            !isfinite(new_log_scale.z))
            return;
        raw.xyz = new_log_scale;
        raw.w = log(new_opacity / (1.0 - new_opacity));
        base_log_scales_opacity[idx] = raw;
    }

    // Current FTGSPP MCMC resets donor moments. Copy modes reset only the
    // destination slot in the copy pass below.
    if (uniforms.mode != 2)
        return;
    write_mean(float3(0), idx, g_base_xyz_ws, 2, 0);
    write_mean(float3(0), idx, g_base_xyz_ws, 2, 1);
    for (int i = 0; i < 12; ++i) {
        g_sh_coeffs_1[shAt(idx, i)] = float4(0);
        g_sh_coeffs_2[shAt(idx, i)] = float4(0);
    }
    g_rotations[2 * idx] = float4(0);
    g_rotations[2 * idx + 1] = float4(0);
    g_base_log_scales_opacity[2 * idx] = float4(0);
    g_base_log_scales_opacity[2 * idx + 1] = float4(0);
    g_times_durations[2 * idx] = float2(0);
    g_times_durations[2 * idx + 1] = float2(0);
    g_velocities_gates[2 * idx] = float4(0);
    g_velocities_gates[2 * idx + 1] = float4(0);
}
