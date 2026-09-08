#include "ftgspp_dynamic.slang"

layout(binding=0) StructuredBuffer<float4> base_log_scales_opacity;
layout(binding=1) StructuredBuffer<float2> times_durations;
layout(binding=2) StructuredBuffer<float4> velocities_gates;
layout(binding=3) RWStructuredBuffer<float> grad_scale_opacity;
layout(binding=4) RWStructuredBuffer<float> grad_velocity_gate;
layout(binding=5) StructuredBuffer<float> timestamps;

struct RegularizerMeanUniforms
{
    uint num_splats;
    uint timestamp_count;
    float max_duration;
    float reg_scale;
    float reg_opacity;
    float reg_gate;
    uint reserved0;
    uint reserved1;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uniform RegularizerMeanUniforms u)
{
    const uint id = tid.x;
    if (id >= u.num_splats)
        return;
    const float4 so = base_log_scales_opacity[id];
    const float2 td = times_durations[id];
    const float4 vg = velocities_gates[id];
    const float temporal_scale = max(
        ftgspp_temporal_scale(td.y, u.max_duration), 1e-8f);
    const float gate = ftgspp_sigmoid(20.0f * vg.w);
    const float base_opacity = ftgspp_sigmoid(so.w);
    // Exact PK contract: average the analytic opacity regularizer over every
    // view timestamp that contributed to the local K-view and global P-rank mean.
    const uint count = max(u.timestamp_count, 1u);
    float temporal_sum = 0.0f;
    for (uint i = 0u; i < count; ++i) {
        const float dt = timestamps[i] - td.x;
        const float marginal = exp(
            -0.5f * dt * dt / (temporal_scale * temporal_scale));
        temporal_sum += gate + (1.0f - gate) * marginal;
    }
    const float opacity_grad = u.reg_opacity
        * (temporal_sum / float(count))
        * base_opacity * (1.0f - base_opacity);
    for (uint j = 0; j < 3; ++j)
        grad_scale_opacity[4u*id+j] += u.reg_scale * exp(so[j]);
    grad_scale_opacity[4u*id+3u] += opacity_grad;
    grad_velocity_gate[4u*id+3u] -=
        u.reg_gate * 20.0f * gate * (1.0f-gate);
}
