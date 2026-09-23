// Headless display conversion. Original FP32 F/T is never modified.
struct PackParams {
    uint width; uint height; uint mode; uint flags;
    float3 background; float exposure_scale;
};
[[vk::push_constant]] ConstantBuffer<PackParams> p;
[[vk::binding(0, 0)]] StructuredBuffer<float4> pixel_ft;
// One uint per RGBA pixel, followed by a non-finite-input status word.
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> rgba_status;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x, count = p.width * p.height;
    if (i >= count) return;
    float4 ft = pixel_ft[i];
    if (!all(isfinite(ft))) {
        InterlockedOr(rgba_status[count], 1u);
        rgba_status[i] = 0;
        return;
    }
    float t = saturate(ft.w), alpha = 1 - t;
    float3 rgb;
    bool transparent = (p.flags & 2u) != 0;
    if (transparent) {
        // PNG straight alpha: match the CPU reference's tiny-alpha cutoff.
        rgb = alpha > 1e-6 ? ft.xyz / alpha : float3(0);
        rgb *= p.exposure_scale;
    } else if (p.mode == 1) rgb = float3(alpha);
    else if (p.mode == 2) rgb = float3(t);
    else {
        float3 bg = p.background;
        if ((p.flags & 1u) != 0) {
            uint x = i % p.width, y = i / p.width;
            bg = float3(((x / 16 + y / 16) & 1u) != 0 ? 0.65 : 0.35);
        }
        rgb = (ft.xyz + t * bg) * p.exposure_scale;
    }
    uint4 bytes = uint4(round(saturate(float4(rgb, transparent ? alpha : 1)) * 255));
    rgba_status[i] = bytes.x | (bytes.y << 8) | (bytes.z << 16) | (bytes.w << 24);
}
