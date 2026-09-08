#include "config.slang"
#include "ftgspp_work_control.slang"

// Scalar counterpart of ssim.slang.  The first two outputs are already
// weighted exactly like ssimBackward's L1 and SSIM gradients, so their sum is
// the scalar objective whose derivative is consumed by the native backward.
layout(binding=0) StructuredBuffer<float4> pixel_state;
layout(binding=1) StructuredBuffer<uint32_t> reference_rgba;
layout(binding=2) StructuredBuffer<float4> ssim_map;
layout(binding=3) RWByteAddressBuffer loss_accum;
layout(binding=4) StructuredBuffer<TileWorkControl> work_control;

struct Uniforms {
    uint image_width;
    uint image_height;
    float l1_weight;
    float ssim_grad_weight;
    // Matches ssim.slang: 0 none, 1 legacy lower clamp, 2 FTGSPP [0,1].
    uint prediction_clamp_mode;
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
        const float alpha = float(packed >> 24) / 255.0f;
        const float3 reference = float3(
            packed & 255u, (packed >> 8) & 255u, (packed >> 16) & 255u
        ) / 255.0f;
        const float3 raw_prediction = pixel_state[pixel].rgb;
        const float3 prediction = u.prediction_clamp_mode == 1u
            ? max(raw_prediction, float3(0.0f))
            : (u.prediction_clamp_mode == 2u
                ? clamp(raw_prediction, float3(0.0f), float3(1.0f))
                : raw_prediction);
        l1 = alpha * (
            abs(prediction.x-reference.x) +
            abs(prediction.y-reference.y) +
            abs(prediction.z-reference.z)
        ) * u.l1_weight;

        const uint x = pixel % u.image_width;
        const uint y = pixel / u.image_width;
        if (x >= 5u && y >= 5u &&
            x + 5u < u.image_width && y + 5u < u.image_height
        ) {
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
