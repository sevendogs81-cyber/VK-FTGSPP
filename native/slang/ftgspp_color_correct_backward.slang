#include "config.slang"
#include "ftgspp_work_control.slang"
import utils;

// Backward of c' = c @ (mat+I) + vec:
//   dc = dc' @ (mat+I)^T
//   dmat += c_outer_dc'
//   dvec += dc'
// Writes dc into v_pixel_state RGB and accumulates one color_grad atomic per
// wave, preserving the affine gradient while avoiding per-pixel atomic storms.

layout(binding=0) StructuredBuffer<float> pixel_state;         // corrected? use pre-correct saved buffer
layout(binding=1) RWStructuredBuffer<float> v_pixel_state;
layout(binding=2) StructuredBuffer<float> color_params;
layout(binding=3) RWByteAddressBuffer color_grad;
layout(binding=4) StructuredBuffer<float> raw_rgb;             // RGB before correction
layout(binding=5) StructuredBuffer<TileWorkControl> work_control;

struct ColorCorrectUniforms {
    uint width;
    uint height;
    uint camera_index;
    uint enabled;
};

[ForceInline]
void add_grad(uint i, float v) {
    const float total = WaveActiveSum(v);
    if (WaveIsFirstLane() && total != 0.0f)
        color_grad.InterlockedAddF32(i * 4u, total);
}

[numthreads(SUBGROUP_SIZE, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uniform ColorCorrectUniforms u) {
    if (tile_work_aborted(work_control[0]))
        return;
    const uint n = u.width * u.height;
    if (u.enabled == 0u)
        return;
    const uint base = u.camera_index * 12u;
    float3x3 mat = float3x3(
        color_params[base+0]+1.0f, color_params[base+1],       color_params[base+2],
        color_params[base+3],      color_params[base+4]+1.0f,  color_params[base+5],
        color_params[base+6],      color_params[base+7],       color_params[base+8]+1.0f);
    const uint pix = tid.x * 4u;
    const bool valid = tid.x < n;
    float3 g = float3(0.0f);
    float3 raw_c = float3(0.0f);
    if (valid) {
        g = float3(
            v_pixel_state[pix], v_pixel_state[pix+1u],
            v_pixel_state[pix+2u]);
        raw_c = float3(raw_rgb[pix], raw_rgb[pix+1u], raw_rgb[pix+2u]);
    }
    const float3 c = clamp(raw_c, float3(0.0f), float3(1.0f));
    // dc = g @ M^T
    float3 dc = mul(g, transpose(mat));
    // Chain through the renderer's [0,1] clamp. torch.clamp propagates at
    // exactly 0 and 1, hence the inclusive comparisons.
    const float3 clamp_derivative = float3(
        raw_c.x >= 0.0f && raw_c.x <= 1.0f ? 1.0f : 0.0f,
        raw_c.y >= 0.0f && raw_c.y <= 1.0f ? 1.0f : 0.0f,
        raw_c.z >= 0.0f && raw_c.z <= 1.0f ? 1.0f : 0.0f);
    dc *= clamp_derivative;
    if (valid) {
        v_pixel_state[pix] = dc.x;
        v_pixel_state[pix+1u] = dc.y;
        v_pixel_state[pix+2u] = dc.z;
    }
    // enabled=2 is hard separation's input-Jacobian-only path. The affine
    // transform remains in the chain, but its frozen parameters have no
    // consumer, so avoid all global color-gradient atomics.
    if (u.enabled == 1u) {
        // dmat[i,j] += c[i] * g[j]  (c' = c @ M, dL/dM = c^T g)
        add_grad(base+0u, c.x*g.x); add_grad(base+1u, c.x*g.y); add_grad(base+2u, c.x*g.z);
        add_grad(base+3u, c.y*g.x); add_grad(base+4u, c.y*g.y); add_grad(base+5u, c.y*g.z);
        add_grad(base+6u, c.z*g.x); add_grad(base+7u, c.z*g.y); add_grad(base+8u, c.z*g.z);
        add_grad(base+9u, g.x); add_grad(base+10u, g.y); add_grad(base+11u, g.z);
    }
}
