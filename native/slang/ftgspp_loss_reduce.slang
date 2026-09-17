#include "config.slang"
#include "ftgspp_work_control.slang"
#include "ftgspp_image_objective.slang"

// Scalar counterpart of ssim.slang.  The first two outputs are already
// weighted exactly like ssimBackward's L1 and SSIM gradients, so their sum is
// the scalar objective whose derivative is consumed by the native backward.
layout(binding=0) StructuredBuffer<float4> pixel_state;
layout(binding=1) StructuredBuffer<uint32_t> reference_rgba;
layout(binding=2) StructuredBuffer<uint32_t> validity_flags;
layout(binding=3) StructuredBuffer<float4> ssim_map;
layout(binding=4) RWByteAddressBuffer loss_accum;
layout(binding=5) StructuredBuffer<TileWorkControl> work_control;

struct Uniforms {
    uint image_width;
    uint image_height;
    float l1_weight;
    float ssim_grad_weight;
    // Matches ssim.slang: 0 none, 1 legacy lower clamp, 2 FTGSPP [0,1].
    uint prediction_clamp_mode;
    uint objective_mode;
    float background_r;
    float background_g;
    float background_b;
    uint valid_pixel_count;
    uint valid_window_count;
};

groupshared float l1_values[256];
groupshared float ssim_values[256];

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
    float l1 = 0.0f;
    float ssim = 0.0f;
    if (pixel < count) {
        const uint packed = reference_rgba[pixel];
        const FTGSPPObjectivePixel objective = ftgspp_objective_pixel(
            pixel_state[pixel], packed, u.objective_mode,
            u.prediction_clamp_mode,
            float3(u.background_r, u.background_g, u.background_b));
        if ((validity_flags[pixel] & 1u) != 0u) {
            l1 = (
                abs(objective.prediction.x-objective.target.x) +
                abs(objective.prediction.y-objective.target.y) +
                abs(objective.prediction.z-objective.target.z)
            ) * u.l1_weight;
        }

        const uint x = pixel % u.image_width;
        const uint y = pixel / u.image_width;
        if ((validity_flags[pixel] & 2u) != 0u) {
            float ssim_sum = 0.0f;
            [ForceUnroll]
            for (uint channel = 0; channel < 3u; ++channel)
                ssim_sum += 1.0f - ssim_map[pixel*3u+channel].x;
            // The backward constant is negative because it differentiates
            // SSIM; the scalar loss is +lambda*(1-SSIM).
            ssim = ssim_sum * (-u.ssim_grad_weight);
        }
    }

    const uint lane = groupThreadID.x;
    l1_values[lane] = l1;
    ssim_values[lane] = ssim;
    GroupMemoryBarrierWithGroupSync();
    [ForceUnroll]
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            l1_values[lane] += l1_values[lane + stride];
            ssim_values[lane] += ssim_values[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u) {
        loss_accum.InterlockedAddF32(0, l1_values[0]);
        loss_accum.InterlockedAddF32(4, ssim_values[0]);
    }
}
