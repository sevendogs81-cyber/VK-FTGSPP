#include "config.slang"

uint active_sh_float_count(uint active_sh) {
    const uint degree = min(active_sh, 3u);
    const uint coefficients = (degree + 1u) * (degree + 1u);
    return 3u * coefficients;
}

// Version 2 records carry only SH coefficients active at the current step.
// Full degree 3 remains 66 words for backwards capacity planning.
uint record_words(uint active_sh) {
    return 18u + active_sh_float_count(active_sh);
}

struct SparseGradientUniforms
{
    uint num_splats;
    uint peer_count;
    uint active_sh;
    uint reserved;
    uint world_size;
};

#if SPARSE_GRADIENT_PHASE == 0
layout(binding=0) StructuredBuffer<int> tiles_touched;
layout(binding=1) RWStructuredBuffer<int> active_flags;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uniform SparseGradientUniforms u)
{
    const uint id = tid.x;
    if (id < u.num_splats)
        active_flags[id] = tiles_touched[id] != 0 ? 1 : 0;
}
#elif SPARSE_GRADIENT_PHASE == 1
layout(binding=0) StructuredBuffer<int> active_flags;
layout(binding=1) StructuredBuffer<int> inclusive_offsets;
layout(binding=2) StructuredBuffer<float> grad_mean;
layout(binding=3) StructuredBuffer<float> grad_scale_opacity;
layout(binding=4) StructuredBuffer<float> grad_time_duration;
layout(binding=5) StructuredBuffer<float> grad_velocity_gate;
layout(binding=6) StructuredBuffer<float> grad_sh;
layout(binding=7) StructuredBuffer<float> grad_rotation;
layout(binding=8) RWByteAddressBuffer records;
layout(binding=9) RWByteAddressBuffer counter;

void store_float(uint word, float value) { records.Store(word * 4u, asuint(value)); }

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uniform SparseGradientUniforms u)
{
    const uint id = tid.x;
    if (id >= u.num_splats)
        return;
    const uint inclusive = uint(inclusive_offsets[id]);
    if (id + 1u == u.num_splats)
        counter.Store(0u, inclusive);
    if (active_flags[id] == 0)
        return;
    // The existing GPU cumsum is inclusive. Subtracting one gives the
    // exclusive active rank and guarantees records are strictly ID-sorted.
    uint word = (inclusive - 1u) * record_words(u.active_sh);
    records.Store(word++ * 4u, id);
    for (uint j = 0; j < 3; ++j) store_float(word++, grad_mean[3u*id+j]);
    for (uint j = 0; j < 4; ++j) store_float(word++, grad_scale_opacity[4u*id+j]);
    for (uint j = 0; j < 2; ++j) store_float(word++, grad_time_duration[2u*id+j]);
    for (uint j = 0; j < 4; ++j)
        store_float(word++, grad_velocity_gate[4u*id+j]);
    const uint lane = id % SH_REORDER_SIZE;
    const uint block = id - lane;
    const uint sh_floats = active_sh_float_count(u.active_sh);
    for (uint component = 0; component < sh_floats; ++component) {
        const uint float4_index = 12u*block
            + (component / 4u)*SH_REORDER_SIZE + lane;
        store_float(word++, grad_sh[4u*float4_index + component % 4u]);
    }
    for (uint j = 0; j < 4; ++j) store_float(word++, grad_rotation[4u*id+j]);
}
#else
layout(binding=0) ByteAddressBuffer records;
layout(binding=1) RWStructuredBuffer<float> grad_mean;
layout(binding=2) RWStructuredBuffer<float> grad_scale_opacity;
layout(binding=3) RWStructuredBuffer<float> grad_time_duration;
layout(binding=4) RWStructuredBuffer<float> grad_velocity_gate;
layout(binding=5) RWStructuredBuffer<float> grad_sh;
layout(binding=6) RWStructuredBuffer<float> grad_rotation;

float load_float(uint word) { return asfloat(records.Load(word * 4u)); }

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uniform SparseGradientUniforms u)
{
    const uint record = tid.x;
    if (record >= u.peer_count)
        return;
    uint word = record * record_words(u.active_sh);
    const uint id = records.Load(word++ * 4u);
    if (id >= u.num_splats)
        return;
    const float scale = 1.0f / float(max(u.world_size, 1u));
    // Split dest + (1/P)*peer so a later FMA cannot swallow the add. Pair
    // merge still scatters rank 0 then rank 1 so both replicas share one
    // operand order under -fp-mode fast.
    for (uint j = 0; j < 3; ++j) {
        precise float contrib = scale * load_float(word++);
        precise float dest = grad_mean[3u*id+j];
        grad_mean[3u*id+j] = dest + contrib;
    }
    for (uint j = 0; j < 4; ++j) {
        precise float contrib = scale * load_float(word++);
        precise float dest = grad_scale_opacity[4u*id+j];
        grad_scale_opacity[4u*id+j] = dest + contrib;
    }
    for (uint j = 0; j < 2; ++j) {
        precise float contrib = scale * load_float(word++);
        precise float dest = grad_time_duration[2u*id+j];
        grad_time_duration[2u*id+j] = dest + contrib;
    }
    for (uint j = 0; j < 4; ++j) {
        precise float contrib = scale * load_float(word++);
        precise float dest = grad_velocity_gate[4u*id+j];
        grad_velocity_gate[4u*id+j] = dest + contrib;
    }
    const uint lane = id % SH_REORDER_SIZE;
    const uint block = id - lane;
    const uint sh_floats = active_sh_float_count(u.active_sh);
    for (uint component = 0; component < sh_floats; ++component) {
        const uint float4_index = 12u*block
            + (component / 4u)*SH_REORDER_SIZE + lane;
        precise float contrib = scale * load_float(word++);
        const uint idx = 4u*float4_index + component % 4u;
        precise float dest = grad_sh[idx];
        grad_sh[idx] = dest + contrib;
    }
    for (uint j = 0; j < 4; ++j) {
        precise float contrib = scale * load_float(word++);
        precise float dest = grad_rotation[4u*id+j];
        grad_rotation[4u*id+j] = dest + contrib;
    }
}
#endif
