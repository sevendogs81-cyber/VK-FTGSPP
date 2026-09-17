// GPU-side pair reduction used by the host-mailbox transport.  Host memory is
// only a cross-device mailbox; it never owns the arithmetic reduction.
layout(binding=0) RWStructuredBuffer<float> local_gradient;
layout(binding=1) StructuredBuffer<float> peer_gradient;

struct PeerMeanUniforms
{
    uint count;
    uint local_offset;
    uint peer_offset;
    uint reset;
    float local_weight;
    float peer_weight;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uniform PeerMeanUniforms u)
{
    const uint i = tid.x;
    if (i >= u.count)
        return;
    const float b = peer_gradient[u.peer_offset + i];
    // Keep the production reduction branch-free and bandwidth-bound. A
    // non-finite input propagates to the loss/replica validation boundary;
    // numerical test references perform eager finite checks.
    if (u.reset != 0u) {
        local_gradient[u.local_offset + i] = b * u.peer_weight;
    } else {
        const float a = local_gradient[u.local_offset + i];
        // Rank 0 computes 0.5*local+0.5*peer; rank 1 swaps the operands.
        // FMA contraction of `a*w + b*w` is not commutative and lets replica
        // opacities drift across the relocation death threshold.
        if (u.local_weight == u.peer_weight) {
            precise float sum = a + b;
            local_gradient[u.local_offset + i] = sum * u.local_weight;
        } else {
            precise float wa = a * u.local_weight;
            precise float wb = b * u.peer_weight;
            local_gradient[u.local_offset + i] = wa + wb;
        }
    }
}
