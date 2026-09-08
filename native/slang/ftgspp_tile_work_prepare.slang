#include "config.slang"
#include "ftgspp_work_control.slang"

// After the inclusive prefix of tiles_touched, write exact tile-pair count,
// overflow/abort flags, and indirect dispatch arguments.  Host never reads
// the last prefix element on the ordinary training path.

layout(binding=0) StructuredBuffer<int> index_buffer_offset;
layout(binding=1) RWStructuredBuffer<TileWorkControl> work_control;
layout(binding=2) RWStructuredBuffer<PipelineAbortLatch> abort_latch;
layout(binding=3) RWStructuredBuffer<int> tile_ranges;

struct PrepareUniforms {
    uint num_splats;
    uint capacity;
    uint partition_size;
    uint emit_group_size;
    uint range_group_size;
    uint num_tiles;
    uint step_generation;
    uint pad;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uniform PrepareUniforms u) {
    TileWorkControl wc;
    wc.exact_count = 0u;
    wc.capacity = u.capacity;
    wc.overflow = 0u;
    wc.invariant_bits = 0u;
    wc.abort_code = TILE_ABORT_NONE;
    wc.step_generation = u.step_generation;
    wc.emit_x = 0u; wc.emit_y = 1u; wc.emit_z = 1u;
    wc.sort_upsweep_x = 0u; wc.sort_upsweep_y = 1u; wc.sort_upsweep_z = 1u;
    wc.range_x = 0u; wc.range_y = 1u; wc.range_z = 1u;

    PipelineAbortLatch latch = abort_latch[0];
    if (latch.error_code != 0u && latch.failed_step != 0u &&
        latch.failed_step < u.step_generation) {
        wc.abort_code = TILE_ABORT_LATCHED;
        work_control[0] = wc;
        return;
    }

    if (u.num_splats == 0u) {
        wc.range_x = (u.num_tiles + u.range_group_size) / u.range_group_size;
        work_control[0] = wc;
        return;
    }

    const int last = index_buffer_offset[u.num_splats - 1u];
    if (last < 0) {
        wc.abort_code = TILE_ABORT_PREFIX;
        wc.overflow = 1u;
        // Eval uses step_generation 0 so a failed val view cannot latch
        // later training steps (failed_step < train_step).
        if (u.step_generation != 0u) {
            abort_latch[0].failed_step = u.step_generation;
            abort_latch[0].error_code = TILE_ABORT_PREFIX;
        }
        work_control[0] = wc;
        return;
    }

    const uint exact = (uint)last;
    wc.exact_count = exact;
    if (exact > u.capacity) {
        wc.overflow = 1u;
        wc.abort_code = TILE_ABORT_OVERFLOW;
        if (u.step_generation != 0u) {
            abort_latch[0].failed_step = u.step_generation;
            abort_latch[0].error_code = TILE_ABORT_OVERFLOW;
        }
        work_control[0] = wc;
        return;
    }

    const uint emit_groups = u.emit_group_size == 0u ? 0u :
        (u.num_splats + u.emit_group_size - 1u) / u.emit_group_size;
    const uint sort_groups = (u.partition_size == 0u || exact == 0u) ? 0u :
        (exact + u.partition_size - 1u) / u.partition_size;
    const uint range_groups = u.range_group_size == 0u ? 0u :
        (exact + 1u + u.range_group_size - 1u) / u.range_group_size;
    wc.emit_x = emit_groups;
    wc.sort_upsweep_x = sort_groups;
    wc.range_x = range_groups;
    work_control[0] = wc;
}
