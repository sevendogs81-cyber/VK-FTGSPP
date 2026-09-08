#include "config.slang"

import utils;
import strategy_utils;

layout(binding=0) StructuredBuffer<int> relocation_map;
layout(binding=1) RWStructuredBuffer<float> base_xyz_ws;
layout(binding=2) RWStructuredBuffer<float4> base_log_scales_opacity;
layout(binding=3) RWStructuredBuffer<float2> times_durations;
layout(binding=4) RWStructuredBuffer<float4> velocities_gates;
layout(binding=5) RWStructuredBuffer<float4> sh_coeffs;
layout(binding=6) RWStructuredBuffer<float4> rotations;
layout(binding=7) RWStructuredBuffer<float> g_base_xyz_ws;
layout(binding=8) RWStructuredBuffer<float4> g_sh_coeffs_1;
layout(binding=9) RWStructuredBuffer<float4> g_sh_coeffs_2;
layout(binding=10) RWStructuredBuffer<float4> g_rotations;
layout(binding=11) RWStructuredBuffer<float4> g_base_log_scales_opacity;
layout(binding=12) RWStructuredBuffer<float2> g_times_durations;
layout(binding=13) RWStructuredBuffer<float4> g_velocities_gates;

struct RelocationUniforms
{
    uint32_t num_splats;
    uint32_t mode;
    float min_opacity;
    uint32_t reserved;
};

[shader("compute")]
[numthreads(SUBGROUP_SIZE, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uniform RelocationUniforms uniforms)
{
    uint dst = tid.x;
    if (dst >= uniforms.num_splats)
        return;
    int source = relocation_map[dst];
    if (source < 0 || uint(source) == dst)
        return;
    uint src = uint(source);

    write_mean(read_mean(src, base_xyz_ws), dst, base_xyz_ws);
    times_durations[dst] = times_durations[src];
    velocities_gates[dst] = velocities_gates[src];
    rotations[dst] = rotations[src];
    if (uniforms.mode != 0)
        base_log_scales_opacity[dst] = base_log_scales_opacity[src];
    for (int i = 0; i < 12; ++i)
        sh_coeffs[shAt(dst, i)] = sh_coeffs[shAt(src, i)];

    // partial_copy/exact_copy reset the dead destination.  Current FTGSPP
    // 3d_mcmc resets donor moments instead (done by the prepare pass).
    if (uniforms.mode == 2)
        return;
    write_mean(float3(0), dst, g_base_xyz_ws, 2, 0);
    write_mean(float3(0), dst, g_base_xyz_ws, 2, 1);
    for (int i = 0; i < 12; ++i) {
        g_sh_coeffs_1[shAt(dst, i)] = float4(0);
        g_sh_coeffs_2[shAt(dst, i)] = float4(0);
    }
    g_rotations[2 * dst] = float4(0);
    g_rotations[2 * dst + 1] = float4(0);
    g_base_log_scales_opacity[2 * dst] = float4(0);
    g_base_log_scales_opacity[2 * dst + 1] = float4(0);
    g_times_durations[2 * dst] = float2(0);
    g_times_durations[2 * dst + 1] = float2(0);
    g_velocities_gates[2 * dst] = float4(0);
    g_velocities_gates[2 * dst + 1] = float4(0);
}
