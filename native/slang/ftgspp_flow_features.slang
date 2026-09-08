#include "config.slang"

import utils;

layout(binding=0) StructuredBuffer<float> end_xyz_ws;
layout(binding=1) StructuredBuffer<float2> start_xy_vs;
layout(binding=2) RWStructuredBuffer<float> flow_features;
layout(binding=3) StructuredBuffer<int> tiles_touched;

[shader("compute")]
[numthreads(SUBGROUP_SIZE, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uniform Uniforms uniforms)
{
    uint idx = tid.x;
    if (idx >= uniforms.num_splats)
        return;
    Camera cam = load_camera(
        uniforms.world_view_transform,
        uniforms.fx, uniforms.fy, uniforms.cx, uniforms.cy,
        uniforms.camera_model, uniforms.dist_coeffs,
        uniforms.image_height, uniforms.image_width);
    float3 end_vs = project_point(read_t3_float3(idx, end_xyz_ws), cam);
    float2 delta = tiles_touched[idx] == 0 || end_vs.z <= NEAR_CLIP
        ? float2(0)
        : (end_vs.xy - 0.5) - start_xy_vs[idx];
    write_t3_float3(float3(delta, 0.0), idx, flow_features);
}
