#include "config.slang"

// Convert the learned continuous gate into two fixed groups.  Positive gate
// logits are the marginalised/static group; non-positive logits remain in the
// dynamic group.  The dynamic velocity is cleared for static splats and all
// velocity/gate Adam moments are reset before the dedicated phase starts.
layout(binding=0) RWStructuredBuffer<float4> velocities_gates;
layout(binding=1) RWStructuredBuffer<float4> velocity_gate_moments;
layout(binding=2) RWByteAddressBuffer status;

struct Uniforms {
    uint num_splats;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

[numthreads(SUBGROUP_SIZE,1,1)]
void main(uint3 tid : SV_DispatchThreadID, uniform Uniforms u) {
    const uint i = tid.x;
    if (i >= u.num_splats) return;
    float4 value = velocities_gates[i];
    const bool marginalize = value.w > 0.0f;
    if (marginalize) {
        value.xyz = float3(0.0f);
        value.w = 100.0f;
        status.InterlockedAdd(0, 1u);
    } else {
        value.w = -100.0f;
    }
    velocities_gates[i] = value;
    velocity_gate_moments[2u*i] = float4(0.0f);
    velocity_gate_moments[2u*i+1u] = float4(0.0f);
}
