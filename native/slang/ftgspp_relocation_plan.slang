#include "config.slang"

#ifndef RELOCATION_PHASE
#define RELOCATION_PHASE 0
#endif

struct Uniforms {
    uint num_splats;
    uint include_gate;
    float opacity_threshold;
    uint seed;
};

bool relocation_uniform_fallback(float3 sum, uint include_gate) {
    // Match FTGSPP's whole-distribution fallback.  The reference constructs
    // every enabled normalized component first and replaces the complete
    // probability vector with ones if any component made it non-finite.  Do
    // not repair one degenerate component independently: that would retain
    // bias from the remaining opacity/gradient/gate components.
    return !isfinite(sum.x) || sum.x <= 0.0f ||
        !isfinite(sum.y) || sum.y <= 0.0f ||
        (include_gate != 0u && (!isfinite(sum.z) || sum.z <= 0.0f));
}

float relocation_donor_score(float4 raw, float3 sum, bool uniform_fallback, uint include_gate) {
    if (raw.w <= 0.5f)
        return 0.0f;
    if (uniform_fallback)
        return 1.0f;
    float score = raw.x / sum.x + raw.y / sum.y;
    if (include_gate != 0u)
        score += raw.z / sum.z;
    return score;
}

#if RELOCATION_PHASE == 0
layout(binding=0) StructuredBuffer<float4> base_log_scales_opacity;
layout(binding=1) StructuredBuffer<float4> velocities_gates;
layout(binding=2) StructuredBuffer<float2> relocation_stats;
layout(binding=3) RWStructuredBuffer<float4> raw_scores;
layout(binding=4) RWByteAddressBuffer totals;

[numthreads(SUBGROUP_SIZE,1,1)]
void main(uint3 tid : SV_DispatchThreadID, uniform Uniforms u) {
    const uint i = tid.x;
    if (i >= u.num_splats) return;
    const float opacity = 1.0f/(1.0f+exp(-base_log_scales_opacity[i].w));
    const float grad2d = relocation_stats[i].x/max(relocation_stats[i].y,1.0f);
    const float static_score = 1.0f-
        1.0f/(1.0f+exp(clamp(-20.0f*velocities_gates[i].w,-80.0f,80.0f)));
    const float alive = opacity > u.opacity_threshold ? 1.0f : 0.0f;
    raw_scores[i] = float4(opacity, grad2d, static_score, alive);
    totals.InterlockedAddF32(0, opacity);
    totals.InterlockedAddF32(4, grad2d);
    totals.InterlockedAddF32(8, static_score);
    if (alive > 0.5f)
        totals.InterlockedAdd(12, 1u);
}
#elif RELOCATION_PHASE == 1
layout(binding=0) StructuredBuffer<float4> raw_scores;
layout(binding=1) StructuredBuffer<float4> totals;
layout(binding=2) RWStructuredBuffer<float> donor_cdf;

[numthreads(SUBGROUP_SIZE,1,1)]
void main(uint3 tid : SV_DispatchThreadID, uniform Uniforms u) {
    const uint i = tid.x;
    if (i >= u.num_splats) return;
    const float3 sum = totals[0].xyz;
    const bool uniform_fallback = relocation_uniform_fallback(sum, u.include_gate);
    donor_cdf[i] = relocation_donor_score(
        raw_scores[i], sum, uniform_fallback, u.include_gate);
}
#elif RELOCATION_PHASE == 3
layout(binding=0) RWByteAddressBuffer totals;
layout(binding=1) RWStructuredBuffer<uint> status;

[numthreads(SUBGROUP_SIZE,1,1)]
void main(uint3 tid : SV_DispatchThreadID, uniform Uniforms u) {
    if (tid.x != 0u) return;
    const uint alive_count = totals.Load(12);
    const float3 sum = float3(
        asfloat(totals.Load(0)),
        asfloat(totals.Load(4)),
        asfloat(totals.Load(8)));
    const bool uniform_fallback = relocation_uniform_fallback(sum, u.include_gate);
    status[0] = u.num_splats - alive_count;
    status[1] = alive_count;
    status[2] = alive_count == 0u ? 1u : 0u;
    status[3] = uniform_fallback && alive_count != 0u ? 1u : 0u;
}
#else
layout(binding=0) StructuredBuffer<float4> raw_scores;
layout(binding=1) StructuredBuffer<float> donor_cdf;
layout(binding=2) RWStructuredBuffer<int> relocation_map;
layout(binding=3) RWByteAddressBuffer relocation_counts;
layout(binding=4) StructuredBuffer<uint> status;

uint pcg_hash(uint value) {
    uint state = value*747796405u+2891336453u;
    uint word = ((state >> ((state >> 28u)+4u))^state)*277803737u;
    return (word >> 22u)^word;
}

[numthreads(SUBGROUP_SIZE,1,1)]
void main(uint3 tid : SV_DispatchThreadID, uniform Uniforms u) {
    const uint i = tid.x;
    if (i >= u.num_splats) return;
    if (raw_scores[i].w > 0.5f || status[1] == 0u) {
        relocation_map[i] = -1;
        return;
    }
    // Search the unnormalized inclusive scan.  Multiplying the unit-interval
    // draw by cdf[N-1] is equivalent to dividing the CDF by that total, but
    // avoids a device-wide in-place normalize race on cdf[N-1].
    const float total = donor_cdf[u.num_splats - 1u];
    const float target = ((float(pcg_hash(u.seed^i))+0.5f)/4294967296.0f)*total;
    uint lo=0u, hi=u.num_splats-1u;
    while (lo < hi) {
        const uint mid = lo+(hi-lo)/2u;
        if (donor_cdf[mid] >= target) hi=mid; else lo=mid+1u;
    }
    relocation_map[i] = int(lo);
    relocation_counts.InterlockedAdd(lo*4u, 1u);
}
#endif
