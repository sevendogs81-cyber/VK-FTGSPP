#ifndef SUBGROUP_SIZE
#include "config.slang"
#endif

import utils;

struct DynamicGaussian4D
{
    float3 mean;
    float3 log_scale;
    float opacity_logit;
    float time_center;
    float duration_raw;
    float3 velocity;
    float gate_raw;
};

struct MaterializedGaussian
{
    float3 mean;
    float3 scale;
    float opacity;
    float marginal;
    float temporal_opacity;
};

[ForceInline]
float ftgspp_sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

[ForceInline]
float ftgspp_temporal_scale(float duration_raw, float max_duration)
{
    // max_duration <= 0 is the on-device sentinel for the unbounded model.
    return max_duration > 0.0
        ? 2.0 * max_duration * ftgspp_sigmoid(duration_raw)
        : exp(duration_raw);
}

[ForceInline]
MaterializedGaussian materialize_ftgspp(
    DynamicGaussian4D p,
    float timestamp,
    float max_duration,
    float marginal_threshold)
{
    float dt = timestamp - p.time_center;
    float temporal_scale = max(ftgspp_temporal_scale(p.duration_raw, max_duration), 1e-8);
    float z = dt / temporal_scale;
    float marginal = exp(-0.5 * z * z);
    float gate = ftgspp_sigmoid(20.0 * p.gate_raw);
    float temporal_opacity = gate + (1.0 - gate) * marginal;
    float opacity = ftgspp_sigmoid(p.opacity_logit) * temporal_opacity;

    // FreeTimeGS++ gated marginalization: activity follows φ, not the raw
    // temporal Gaussian. Persistents (g→1) stay visible for the whole
    // sequence; transients still drop out when φ is numerically negligible.
    if (temporal_opacity <= marginal_threshold)
        opacity = 0.0;

    MaterializedGaussian result;
    result.mean = p.mean + dt * p.velocity;
    result.scale = exp(p.log_scale);
    result.opacity = opacity;
    result.marginal = marginal;
    result.temporal_opacity = temporal_opacity;
    return result;
}

[ForceInline]
DynamicGaussian4D load_ftgspp(
    uint idx,
    StructuredBuffer<float> base_xyz_ws,
    StructuredBuffer<float4> base_log_scales_opacity,
    StructuredBuffer<float2> times_durations,
    StructuredBuffer<float4> velocities_gates)
{
    float4 so = base_log_scales_opacity[idx];
    float2 td = times_durations[idx];
    float4 vg = velocities_gates[idx];
    DynamicGaussian4D p;
    p.mean = read_t3_float3(idx, base_xyz_ws);
    p.log_scale = so.xyz;
    p.opacity_logit = so.w;
    p.time_center = td.x;
    p.duration_raw = td.y;
    p.velocity = vg.xyz;
    p.gate_raw = vg.w;
    return p;
}

[ForceInline]
DynamicGaussian4D load_ftgspp(
    uint idx,
    RWStructuredBuffer<float> base_xyz_ws,
    RWStructuredBuffer<float4> base_log_scales_opacity,
    RWStructuredBuffer<float2> times_durations,
    RWStructuredBuffer<float4> velocities_gates)
{
    float4 so = base_log_scales_opacity[idx];
    float2 td = times_durations[idx];
    float4 vg = velocities_gates[idx];
    DynamicGaussian4D p;
    p.mean = read_t3_float3(idx, base_xyz_ws);
    p.log_scale = so.xyz;
    p.opacity_logit = so.w;
    p.time_center = td.x;
    p.duration_raw = td.y;
    p.velocity = vg.xyz;
    p.gate_raw = vg.w;
    return p;
}
