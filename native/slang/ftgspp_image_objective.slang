// Shared image-objective contract. Raster state is (F.rgb, T), where T is
// transmittance and A=1-T. Reference RGBA stores straight RGB plus the
// foreground mask M; scene_rgb_v1 deliberately ignores M.

static const uint FTGSPP_OBJECTIVE_SCENE_RGB_V1 = 0u;
static const uint FTGSPP_OBJECTIVE_COMPOSITE_V1 = 1u;

[ForceInline]
bool ftgspp_is_composite_objective(uint mode) {
    return mode == FTGSPP_OBJECTIVE_COMPOSITE_V1;
}

struct FTGSPPObjectivePixel {
    float3 raw_prediction;
    float3 prediction;
    float3 target;
};

[ForceInline]
float3 ftgspp_decode_rgb(uint packed) {
    return float3(
        packed & 255u, (packed >> 8u) & 255u, (packed >> 16u) & 255u
    ) / 255.0f;
}

[ForceInline]
float ftgspp_decode_mask(uint packed) {
    return float(packed >> 24u) / 255.0f;
}

[ForceInline]
float3 ftgspp_apply_prediction_clamp(float3 value, uint mode) {
    if (mode == 1u)
        return max(value, float3(0.0f));
    if (mode == 2u)
        return clamp(value, float3(0.0f), float3(1.0f));
    return value;
}

[ForceInline]
float ftgspp_clamp_derivative(float value, uint mode) {
    if (mode == 1u)
        return value > 0.0f ? 1.0f : 0.0f;
    if (mode == 2u)
        // Match torch.clamp at exact endpoints. Finite differences avoid them.
        return value >= 0.0f && value <= 1.0f ? 1.0f : 0.0f;
    return 1.0f;
}

[ForceInline]
FTGSPPObjectivePixel ftgspp_objective_pixel(
    float4 state,
    uint packed_reference,
    uint objective_mode,
    uint prediction_clamp_mode,
    float3 background
) {
    FTGSPPObjectivePixel result;
    const float3 reference = ftgspp_decode_rgb(packed_reference);
    if (ftgspp_is_composite_objective(objective_mode)) {
        const float mask = ftgspp_decode_mask(packed_reference);
        result.raw_prediction = state.rgb + background * state.w;
        result.prediction = ftgspp_apply_prediction_clamp(
            result.raw_prediction, prediction_clamp_mode);
        result.target = reference * mask + background * (1.0f-mask);
    } else {
        result.raw_prediction = state.rgb;
        result.prediction = ftgspp_apply_prediction_clamp(
            result.raw_prediction, prediction_clamp_mode);
        result.target = reference;
    }
    return result;
}
