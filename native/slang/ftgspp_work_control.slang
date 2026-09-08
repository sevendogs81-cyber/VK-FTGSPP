// Device ABI matching native/src/tile_work_control.h.  Explicit uint fields
// only; do not add padding-sensitive nested structs.
struct TileWorkControl {
    uint exact_count;
    uint capacity;
    uint overflow;
    uint invariant_bits;
    uint abort_code;
    uint step_generation;
    uint emit_x;
    uint emit_y;
    uint emit_z;
    uint sort_upsweep_x;
    uint sort_upsweep_y;
    uint sort_upsweep_z;
    uint range_x;
    uint range_y;
    uint range_z;
};

struct PipelineAbortLatch {
    uint failed_step;
    uint error_code;
    uint pad0;
    uint pad1;
};

static const uint TILE_ABORT_NONE = 0u;
static const uint TILE_ABORT_OVERFLOW = 1u;
static const uint TILE_ABORT_INVARIANT = 2u;
static const uint TILE_ABORT_PREFIX = 3u;
static const uint TILE_ABORT_LATCHED = 4u;

[ForceInline]
bool tile_work_aborted(TileWorkControl wc) {
    return wc.abort_code != TILE_ABORT_NONE;
}
