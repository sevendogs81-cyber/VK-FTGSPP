#include "config.slang"
#include "ftgspp_work_control.slang"

// Per-parameter Adam for color_params (N_cam * 12).  FTGSPP-main defines
// L_cc = reg * sum(params^2), so its parameter gradient is 2 * reg * params.

struct ColorAdamUniforms {
    uint count;
    uint step;
    float lr;
    float reg;
    uint update_parameters;
};

layout(binding=0) RWStructuredBuffer<float> color_params;
layout(binding=1) RWStructuredBuffer<float> color_grad;
layout(binding=2) RWStructuredBuffer<float> color_m;
layout(binding=3) RWStructuredBuffer<float> color_v;
layout(binding=4) RWByteAddressBuffer loss_accum;
layout(binding=5) StructuredBuffer<TileWorkControl> work_control;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uniform ColorAdamUniforms u) {
    if (tile_work_aborted(work_control[0]))
        return;
    const uint i = tid.x;
    if (i >= u.count)
        return;
    if ((u.update_parameters & 2u) != 0u) {
        float color_loss = WaveActiveSum(
            u.reg * color_params[i] * color_params[i]);
        if (WaveIsFirstLane())
            loss_accum.InterlockedAddF32(20u, color_loss);
    }
    // Retain an input-Jacobian-only/diagnostic ABI mode without touching a
    // dummy gradient buffer.  Native hard separation does not dispatch this
    // shader because that phase has neither color optimization nor L_cc.
    if ((u.update_parameters & 1u) == 0u)
        return;
    const float g = color_grad[i] + 2.0f * u.reg * color_params[i];
    float m = color_m[i];
    float v = color_v[i];
    m = lerp(m, g, 0.1f);
    v = lerp(v, g * g, 0.001f);
    const float t = max(float(u.step), 1.0f);
    const float mhat = m / (1.0f - pow(0.9f, t));
    const float vhat = v / (1.0f - pow(0.999f, t));
    color_params[i] -= u.lr * mhat / (sqrt(vhat) + 1e-15f);
    color_m[i] = m;
    color_v[i] = v;
    color_grad[i] = 0.0f;
}
