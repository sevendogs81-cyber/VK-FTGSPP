#include "config.slang"
#include "ftgspp_work_control.slang"
#include "ftgspp_image_objective.slang"
import utils;

// Per-camera affine color correction: c' = c @ (mat + I) + vec.
// Params layout for camera i: 9 floats of mat (row-major) + 3 floats of vec.

layout(binding=0) RWStructuredBuffer<float> pixel_state;       // H*W*4 RGBA
layout(binding=1) StructuredBuffer<float> color_params;        // N_cam * 12
layout(binding=2) RWStructuredBuffer<float> raw_rgb;           // pre-correct H*W*4 RGBA
layout(binding=3) StructuredBuffer<TileWorkControl> work_control;

struct ColorCorrectUniforms {
    uint width;
    uint height;
    uint camera_index;
    uint enabled;
    uint objective_mode;
};

[numthreads(SUBGROUP_SIZE, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uniform ColorCorrectUniforms u) {
    if (tile_work_aborted(work_control[0]))
        return;
    const uint n = u.width * u.height;
    if (tid.x >= n || u.enabled == 0u)
        return;
    const uint base = u.camera_index * 12u;
    float3x3 mat = float3x3(
        color_params[base+0]+1.0f, color_params[base+1],       color_params[base+2],
        color_params[base+3],      color_params[base+4]+1.0f,  color_params[base+5],
        color_params[base+6],      color_params[base+7],       color_params[base+8]+1.0f);
    float3 bias = float3(color_params[base+9], color_params[base+10], color_params[base+11]);
    const uint pix = tid.x * 4u;
    const float4 raw = float4(
        pixel_state[pix], pixel_state[pix+1u],
        pixel_state[pix+2u], pixel_state[pix+3u]);
    raw_rgb[pix] = raw.x;
    raw_rgb[pix+1u] = raw.y;
    raw_rgb[pix+2u] = raw.z;
    raw_rgb[pix+3u] = raw.w;
    // composite_v1 extends the affine transform over premultiplied foreground:
    // H=F@W+(1-T)b. scene_rgb_v1 corrects full-frame rendered RGB.
    float3 rgb = ftgspp_is_composite_objective(u.objective_mode)
        ? raw.xyz : clamp(raw.xyz, float3(0.0f), float3(1.0f));
    float bias_scale = ftgspp_is_composite_objective(u.objective_mode)
        ? (1.0f-raw.w) : 1.0f;
    // row-vector @ matrix  ==  mul(rgb, mat) in HLSL when mat columns are basis
    float3 out_rgb = mul(rgb, mat) + bias_scale * bias;
    pixel_state[pix] = out_rgb.x;
    pixel_state[pix+1u] = out_rgb.y;
    pixel_state[pix+2u] = out_rgb.z;
}
