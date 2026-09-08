#include "config.slang"

import utils;
import spherical_harmonics;

#ifndef FTGSPP_DYNAMIC_PROJECTION
#define FTGSPP_DYNAMIC_PROJECTION 0
#endif

#if FTGSPP_DYNAMIC_PROJECTION
#include "ftgspp_dynamic.slang"
#endif


struct Gaussian_3D : IDifferentiable
{
    float3 xyz_ws;
    SpherHarmCoeffs sh_coeffs;
    float4 rotations;
    float3 scales;
    float opacities;
};

Gaussian_3D load_gaussian(
    int32_t g_idx,
    StructuredBuffer<float> xyz_ws,
    StructuredBuffer<float4> sh_coeffs,
    StructuredBuffer<float4> rotations,
    StructuredBuffer<float4> scales_opacs,
    uint active_sh
) {
    float3 g_xyz_ws = read_t3_float3(g_idx, xyz_ws);

    SpherHarmCoeffs g_sh_coeffs = read_spherical_harmonics_coeffs(g_idx, sh_coeffs, active_sh);

    float4 g_rotations = rotations[g_idx];
    float4 g_scales_opacs = scales_opacs[g_idx];
    float3 g_scales = g_scales_opacs.xyz;
    float g_opacities = g_scales_opacs.w;

    return { g_xyz_ws, g_sh_coeffs, g_rotations, g_scales, g_opacities };
}

[ForceInline]
Gaussian_3D load_zero_gaussian() {
    float3 g_xyz_ws = float3(0);
    float4 g_rotations = float4(1, 0, 0, 0);
    float3 g_scales = float3(1, 1, 1);
    float g_opacities = 0.0;
    float3 z = float3(0);
    SpherHarmCoeffs g_sh_coeffs = { z, z,z,z, z,z,z,z,z, z,z,z,z,z,z,z };
    return { g_xyz_ws, g_sh_coeffs, g_rotations, g_scales, g_opacities };
}

[Differentiable]
Splat_2D_Vertex project_gaussian_to_camera(Gaussian_3D g, Camera cam, uint active_sh) {
    float3 xyz_vs = project_point(g.xyz_ws, cam);
    if (xyz_vs.z <= NEAR_CLIP) {
        return { float3(0.0), float3(0.0), float3(0.0), float4(0.0), 0.0 };
    }
    xyz_vs.xy -= 0.5;  // pixel offset

    float4x3 mat = transpose(cam.world_view_transform);
    float3 cam_pos = -mul(float3x3(mat), mat[3]);
    float3 dir = float3(normalize(g.xyz_ws - cam_pos));
    float3 rgb = compute_color_from_sh_coeffs(g.sh_coeffs, dir, active_sh);

    float3x3 cov_ws = get_covariance_from_quat_scales(
        normalize(g.rotations), g.scales
    );
    float4 cov_vs = covariance_3d_to_2d(cam, g.xyz_ws, cov_ws);

    float det = compute_det(cov_vs.xyz);
    float3 inv_cov_vs = compute_inverse(cov_vs.xyz);
    float opac = g.opacities * cov_vs.w;

    return { xyz_vs, rgb, cov_vs.xyz, float4(inv_cov_vs, opac), det };
}


rectangle get_rectangle_tile_space(
    float2 ndc_xy,
    float2 radii,
    uint grid_height,
    uint grid_width) {

    rectangle rect_tile_space;

    rect_tile_space.min_x = uint16_t(floor(clamp((ndc_xy.x - radii.x) / TILE_WIDTH, 0, grid_width)));
    rect_tile_space.min_y = uint16_t(floor(clamp((ndc_xy.y - radii.y) / TILE_HEIGHT, 0, grid_height)));
    // Include an exact max-boundary tangent: raster accepts alpha equal to
    // ALPHA_THRESHOLD, and that sample belongs to the tile on the right/bottom.
    rect_tile_space.max_x = uint16_t(clamp(
        floor((ndc_xy.x + radii.x) / TILE_WIDTH) + 1,
        0, grid_width));
    rect_tile_space.max_y = uint16_t(clamp(
        floor((ndc_xy.y + radii.y) / TILE_HEIGHT) + 1,
        0, grid_height));

    return rect_tile_space;
}


// #define EARLY_RETURN_ON_ZERO_GRAD 0
#define EARLY_RETURN_ON_ZERO_GRAD 0


#ifndef NO_MAIN

#if EXPORT_MODE == EXPORT_MODE_VULKAN_FORWARD

// Input buffers
#if FTGSPP_DYNAMIC_PROJECTION
layout(binding=0) StructuredBuffer<float> base_xyz_ws;
layout(binding=1) StructuredBuffer<float4> base_log_scales_opacity;
layout(binding=2) StructuredBuffer<float2> times_durations;
layout(binding=3) StructuredBuffer<float4> velocities_gates;
layout(binding=4) StructuredBuffer<float4> sh_coeffs;
layout(binding=5) StructuredBuffer<float4> rotations;
#else
layout(binding=0) StructuredBuffer<float> xyz_ws;  // 3
layout(binding=1) StructuredBuffer<float4> sh_coeffs;  // 16*3
layout(binding=2) StructuredBuffer<float4> rotations;  // 4
layout(binding=3) StructuredBuffer<float4> scales_opacs;  // 4
#endif

// Output buffers
#if FTGSPP_DYNAMIC_PROJECTION
layout(binding=6) RWStructuredBuffer<int> out_tiles_touched;
#if USE_EMULATED_INT64
layout(binding=7) RWStructuredBuffer<int32_t> out_rect_tile_space;
#else
layout(binding=7) RWStructuredBuffer<int64_t> out_rect_tile_space;
#endif
layout(binding=8) RWStructuredBuffer<int> out_radii;
layout(binding=9) RWStructuredBuffer<float2> out_xy_vs;
layout(binding=10) RWStructuredBuffer<float> out_depths;
layout(binding=11) RWStructuredBuffer<float4> out_inv_cov_vs;
layout(binding=12) RWStructuredBuffer<float> out_rgb;
layout(binding=13) RWStructuredBuffer<uint> out_cull_reason;
#else
layout(binding=4) RWStructuredBuffer<int> out_tiles_touched;
#if USE_EMULATED_INT64
layout(binding=5) RWStructuredBuffer<int32_t> out_rect_tile_space;
#else
layout(binding=5) RWStructuredBuffer<int64_t> out_rect_tile_space;
#endif
layout(binding=6) RWStructuredBuffer<int> out_radii;
layout(binding=7) RWStructuredBuffer<float2> out_xy_vs;
layout(binding=8) RWStructuredBuffer<float> out_depths;
layout(binding=9) RWStructuredBuffer<float4> out_inv_cov_vs;
layout(binding=10) RWStructuredBuffer<float> out_rgb;
layout(binding=11) RWStructuredBuffer<uint> out_cull_reason;
#endif

[ForceInline]
void clear_projected_outputs(uint g_idx) {
    out_tiles_touched[g_idx] = 0;
  #if USE_EMULATED_INT64
    out_rect_tile_space[2*g_idx+0] = 0;
    out_rect_tile_space[2*g_idx+1] = 0;
  #else
    out_rect_tile_space[g_idx] = int64_t(0);
  #endif
    out_radii[g_idx] = 0;
    out_xy_vs[g_idx] = float2(0.0f);
    out_depths[g_idx] = 0.0f;
    out_inv_cov_vs[g_idx] = float4(0.0f);
    write_t3_float3(float3(0.0f), g_idx, out_rgb);
}

#if FTGSPP_DYNAMIC_PROJECTION
struct DynamicProjectionUniforms {
    uint image_height;
    uint image_width;
    uint grid_height;
    uint grid_width;
    uint num_splats;
    uint active_sh;
    uint step;
    uint camera_model;
    float fx;
    float fy;
    float cx;
    float cy;
    float4 dist_coeffs;
    float timestamp;
    float max_duration;
    float marginal_threshold;
    float reserved;
    float4x4 world_view_transform;
};
#endif

#define no_diff

[shader("compute")]
[numthreads(SUBGROUP_SIZE, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
#if FTGSPP_DYNAMIC_PROJECTION
    uniform DynamicProjectionUniforms uniforms
#else
    uniform Uniforms uniforms
#endif
) {
    const uint active_sh = uniforms.active_sh;
    const uint image_height = uniforms.image_height;
    const uint image_width = uniforms.image_width;
    const uint grid_height = uniforms.grid_height;
    const uint grid_width = uniforms.grid_width;
    const uint camera_model = uniforms.camera_model;
    const float4x4 world_view_transform = (uniforms.world_view_transform);

    uint g_idx = dispatchThreadID.x;
    if (g_idx >= uniforms.num_splats)
        return;

    // bit 1 near clip, bit 2 opacity, bit 3 determinant,
    // bit 4 empty image rectangle, bit 5 zero ellipse/tile overlap.
    out_cull_reason[g_idx] = 0u;

    Camera cam = no_diff load_camera(world_view_transform,
        uniforms.fx, uniforms.fy, uniforms.cx, uniforms.cy,
        camera_model, uniforms.dist_coeffs,
        image_height, image_width);
  #if FTGSPP_DYNAMIC_PROJECTION
    DynamicGaussian4D dynamic_gauss = load_ftgspp(
        g_idx, base_xyz_ws, base_log_scales_opacity,
        times_durations, velocities_gates);
    MaterializedGaussian materialized = materialize_ftgspp(
        dynamic_gauss, uniforms.timestamp, uniforms.max_duration,
        uniforms.marginal_threshold);
    Gaussian_3D gauss = {
        materialized.mean,
        read_spherical_harmonics_coeffs(g_idx, sh_coeffs, active_sh),
        rotations[g_idx],
        materialized.scale,
        materialized.opacity
    };
  #else
    Gaussian_3D gauss = load_gaussian(
        g_idx, xyz_ws, sh_coeffs, rotations,
        scales_opacs,
        active_sh
    );
  #endif

    Splat_2D_Vertex splat = project_gaussian_to_camera(gauss, cam, active_sh);

    uint cull_reason = 0u;
    const float diagnostic_depth = project_point(gauss.xyz_ws, cam).z;
    if (diagnostic_depth <= NEAR_CLIP) {
        cull_reason |= (1u << 1);
    } else {
        if (splat.inv_cov_vs_opac.w <= ALPHA_THRESHOLD)
            cull_reason |= (1u << 2);
        if (splat.det <= 0.0)
            cull_reason |= (1u << 3);
    }
    if (cull_reason != 0u) {
        out_cull_reason[g_idx] = cull_reason;
        clear_projected_outputs(g_idx);
        return;
    }

    Splat_2D_Vertex splat_dist = splat;

    float radius = splat_radius(splat_dist.cov_vs.xyz, splat_dist.det);

  #if TILE_CULLING_MODE == 0
    float2 radii = float2(radius);
  #else
    float extend = no_diff extend_from_opacity(gauss.opacities);
    float2 radii = sqrt(splat_dist.cov_vs.xz) * extend;
  #endif

    const float2 visibility_radii = ceil(radii);
    const float2 visibility_xy = splat_dist.xyz_vs.xy + 0.5;
    if (visibility_xy.x + visibility_radii.x <= 0.0 ||
        visibility_xy.x - visibility_radii.x >= float(image_width) ||
        visibility_xy.y + visibility_radii.y <= 0.0 ||
        visibility_xy.y - visibility_radii.y >= float(image_height)
    ) {
        out_cull_reason[g_idx] = (1u << 4);
        clear_projected_outputs(g_idx);
        return;
    }

    rectangle rect_tile_space = get_rectangle_tile_space(
        visibility_xy, visibility_radii, grid_height, grid_width);

    int32_t n_tiles = int32_t(rect_tile_space.max_x - rect_tile_space.min_x) * int32_t(rect_tile_space.max_y - rect_tile_space.min_y);
    if (n_tiles == 0) {
        out_cull_reason[g_idx] = (1u << 4);
        clear_projected_outputs(g_idx);
        return;
    }

  #if TILE_CULLING_MODE == 2
    n_tiles = no_diff count_ellipse_grid_overlaps(
        splat_dist.xyz_vs.xy, splat_dist.inv_cov_vs_opac.xyz/(extend*extend),
        rect_tile_space
    );
  #endif

    if (n_tiles == 0) {
        out_cull_reason[g_idx] = (1u << 5);
        clear_projected_outputs(g_idx);
        return;
    }

    out_radii[g_idx] = (uint32_t)ceil(max(radii.x, radii.y));
    out_tiles_touched[g_idx] = n_tiles;
  #if USE_EMULATED_INT64
    out_rect_tile_space[2*g_idx+0] = (int32_t)rect_tile_space.min_x | ((int32_t)rect_tile_space.min_y << 16);
    out_rect_tile_space[2*g_idx+1] = (int32_t)rect_tile_space.max_x | ((int32_t)rect_tile_space.max_y << 16);
  #else
    out_rect_tile_space[g_idx] = reinterpret<int64_t>(rect_tile_space);
  #endif

    out_xy_vs[g_idx] = splat.xyz_vs.xy;
    out_depths[g_idx] = splat.xyz_vs.z;
    out_inv_cov_vs[g_idx] = splat.inv_cov_vs_opac;
    write_t3_float3(splat.rgb, g_idx, out_rgb);
}

#endif  // EXPORT_MODE == EXPORT_MODE_VULKAN_FORWARD

#endif  // NO_MAIN
