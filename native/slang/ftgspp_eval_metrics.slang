#include "config.slang"
#include "ftgspp_work_control.slang"
#include "ftgspp_image_objective.slang"

// Hold-out ranking reduce.  SSIM_2 (data_range=2) lives in ssim_map.x;
// this kernel only sums MSE / SSIM / object-MSE over the already gated
// validity bits.  Training loss_reduce is unchanged.
layout(binding=0) StructuredBuffer<float4> pixel_state;
layout(binding=1) StructuredBuffer<uint32_t> reference_rgba;
layout(binding=2) StructuredBuffer<uint32_t> validity_flags;
layout(binding=3) StructuredBuffer<float4> ssim_map;
layout(binding=4) RWByteAddressBuffer eval_accum;
layout(binding=5) StructuredBuffer<TileWorkControl> work_control;

struct Uniforms {
    uint image_width;
    uint image_height;
    uint roi_y0;
    uint roi_y1;
    uint prediction_clamp_mode;
    uint objective_mode;
    float background_r;
    float background_g;
    float background_b;
    uint roi_x0;
    uint roi_x1;
};

groupshared float se_values[256];
groupshared float ssim_values[256];
groupshared float fg_se_values[256];
groupshared float fg_count_values[256];
groupshared float pixel_count_values[256];
groupshared float window_count_values[256];
groupshared float roi_ssim_values[256];
groupshared float roi_window_count_values[256];
groupshared float foreground_ssim_values[256];
groupshared float foreground_window_count_values[256];

[numthreads(256, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uniform Uniforms u
) {
    if (tile_work_aborted(work_control[0]))
        return;
    const uint pixel = dispatchThreadID.x;
    const uint count = u.image_width * u.image_height;
    float se = 0.0f;
    float ssim = 0.0f;
    float fg_se = 0.0f;
    float fg_count = 0.0f;
    float pixel_count = 0.0f;
    float window_count = 0.0f;
    float roi_ssim = 0.0f;
    float roi_window_count = 0.0f;
    float foreground_ssim = 0.0f;
    float foreground_window_count = 0.0f;
    if (pixel < count) {
        const uint packed = reference_rgba[pixel];
        const FTGSPPObjectivePixel objective = ftgspp_objective_pixel(
            pixel_state[pixel], packed, u.objective_mode,
            u.prediction_clamp_mode,
            float3(u.background_r, u.background_g, u.background_b));
        if ((validity_flags[pixel] & 1u) != 0u) {
            const float3 diff = objective.prediction - objective.target;
            se = dot(diff, diff);
            pixel_count = 1.0f;
            const float mask = ftgspp_decode_mask(packed);
            if (mask >= 0.5f) {
                const float3 reference = ftgspp_decode_rgb(packed);
                const float3 clipped = clamp(
                    pixel_state[pixel].rgb, float3(0.0f), float3(1.0f));
                const float3 object_diff = clipped - reference * mask;
                fg_se = dot(object_diff, object_diff);
                fg_count = 1.0f;
            }
        }
        if ((validity_flags[pixel] & 2u) != 0u) {
            float ssim_sum = 0.0f;
            [ForceUnroll]
            for (uint channel = 0; channel < 3u; ++channel)
                ssim_sum += ssim_map[pixel*3u+channel].x;
            ssim = ssim_sum;
            window_count = 1.0f;
            const uint y = pixel / u.image_width;
            const uint x = pixel - y * u.image_width;
            if (
                u.roi_y1 > u.roi_y0 + 10u
                && u.roi_x1 > u.roi_x0 + 10u
            ) {
                // Slang 2026.10 can miscompile one long unsigned comparison
                // chain here; preserve the separately named axis predicates.
                const bool inside_y =
                    y >= u.roi_y0 + 5u && y + 5u < u.roi_y1;
                const bool inside_x =
                    x >= u.roi_x0 + 5u && x + 5u < u.roi_x1;
                if (inside_y && inside_x) {
                    roi_ssim = ssim_sum;
                    roi_window_count = 1.0f;
                }
            }
            // Bit 2 marks an 11x11 window fully inside the reference mask.
            // The SSIM map must be the black-background objective so this
            // matches foreground PSNR's clamp(F) vs I*M target.
            if ((validity_flags[pixel] & 4u) != 0u) {
                foreground_ssim = ssim_sum;
                foreground_window_count = 1.0f;
            }
        }
    }

    const uint lane = groupThreadID.x;
    se_values[lane] = se;
    ssim_values[lane] = ssim;
    fg_se_values[lane] = fg_se;
    fg_count_values[lane] = fg_count;
    pixel_count_values[lane] = pixel_count;
    window_count_values[lane] = window_count;
    roi_ssim_values[lane] = roi_ssim;
    roi_window_count_values[lane] = roi_window_count;
    foreground_ssim_values[lane] = foreground_ssim;
    foreground_window_count_values[lane] = foreground_window_count;
    GroupMemoryBarrierWithGroupSync();
    [ForceUnroll]
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            se_values[lane] += se_values[lane + stride];
            ssim_values[lane] += ssim_values[lane + stride];
            fg_se_values[lane] += fg_se_values[lane + stride];
            fg_count_values[lane] += fg_count_values[lane + stride];
            pixel_count_values[lane] += pixel_count_values[lane + stride];
            window_count_values[lane] += window_count_values[lane + stride];
            roi_ssim_values[lane] += roi_ssim_values[lane + stride];
            roi_window_count_values[lane] +=
                roi_window_count_values[lane + stride];
            foreground_ssim_values[lane] +=
                foreground_ssim_values[lane + stride];
            foreground_window_count_values[lane] +=
                foreground_window_count_values[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u) {
        eval_accum.InterlockedAddF32(0, se_values[0]);
        eval_accum.InterlockedAddF32(4, ssim_values[0]);
        eval_accum.InterlockedAddF32(8, fg_se_values[0]);
        eval_accum.InterlockedAddF32(12, fg_count_values[0]);
        eval_accum.InterlockedAddF32(16, pixel_count_values[0]);
        eval_accum.InterlockedAddF32(20, window_count_values[0]);
        eval_accum.InterlockedAddF32(24, roi_ssim_values[0]);
        eval_accum.InterlockedAddF32(28, roi_window_count_values[0]);
        eval_accum.InterlockedAddF32(32, foreground_ssim_values[0]);
        eval_accum.InterlockedAddF32(36, foreground_window_count_values[0]);
    }
}
