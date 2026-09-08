#include "ftgspp_dynamic.slang"

layout(binding=0) StructuredBuffer<float> base_xyz_ws;
layout(binding=1) StructuredBuffer<float4> base_log_scales_opacity;
layout(binding=2) StructuredBuffer<float2> times_durations;
layout(binding=3) StructuredBuffer<float4> velocities_gates;
layout(binding=4) RWStructuredBuffer<float> xyz_ws;
layout(binding=5) RWStructuredBuffer<float4> scales_opacs;

struct MaterializeUniforms
{
    uint32_t num_splats;
    float timestamp;
    float max_duration;
    float marginal_threshold;
};

[shader("compute")]
[numthreads(SUBGROUP_SIZE, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uniform MaterializeUniforms uniforms)
{
    uint idx = tid.x;
    if (idx >= uniforms.num_splats)
        return;

    DynamicGaussian4D p = load_ftgspp(
        idx, base_xyz_ws, base_log_scales_opacity,
        times_durations, velocities_gates);
    MaterializedGaussian g = materialize_ftgspp(
        p, uniforms.timestamp, uniforms.max_duration,
        uniforms.marginal_threshold);
    write_t3_float3(g.mean, idx, xyz_ws);
    scales_opacs[idx] = float4(g.scale, g.opacity);
}
