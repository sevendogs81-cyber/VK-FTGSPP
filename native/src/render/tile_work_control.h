#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stddef.h>

// Host/device ABI for device-driven tile workload.  Layout is an explicit
// uint32 array so Slang and C++ agree without relying on C++ struct padding.
// Conceptual reference: LichtFeld FastGS keeps n_instances device-side and
// only reads it after the submission retires (forward.cu, instance count).
namespace ftgspp {

enum TileAbortCode : uint32_t {
    kTileAbortNone = 0,
    kTileAbortOverflow = 1,
    kTileAbortInvariant = 2,
    kTileAbortPrefix = 3,
    kTileAbortLatched = 4,
};

struct TileWorkControl {
    uint32_t exact_count;
    uint32_t capacity;
    uint32_t overflow;
    uint32_t invariant_bits;
    uint32_t abort_code;
    uint32_t step_generation;
    uint32_t emit_x, emit_y, emit_z;
    uint32_t sort_upsweep_x, sort_upsweep_y, sort_upsweep_z;
    uint32_t range_x, range_y, range_z;
};

struct PipelineAbortLatch {
    uint32_t failed_step;
    uint32_t error_code;
    uint32_t pad0;
    uint32_t pad1;
};

static constexpr uint32_t kTileWorkControlU32Count =
    sizeof(TileWorkControl) / sizeof(uint32_t);
static constexpr uint32_t kTileWorkControlBytes = sizeof(TileWorkControl);

static constexpr size_t kTileWorkExactCountOffset = 0;
static constexpr size_t kTileWorkCapacityOffset = 4;
static constexpr size_t kTileWorkOverflowOffset = 8;
static constexpr size_t kTileWorkInvariantOffset = 12;
static constexpr size_t kTileWorkAbortOffset = 16;
static constexpr size_t kTileWorkStepOffset = 20;
static constexpr size_t kTileWorkEmitOffset = 24;
static constexpr size_t kTileWorkSortOffset = 36;
static constexpr size_t kTileWorkRangeOffset = 48;

static_assert(sizeof(TileWorkControl) == 15 * 4, "TileWorkControl ABI size");
static_assert(offsetof(TileWorkControl, exact_count) == kTileWorkExactCountOffset, "");
static_assert(offsetof(TileWorkControl, capacity) == kTileWorkCapacityOffset, "");
static_assert(offsetof(TileWorkControl, overflow) == kTileWorkOverflowOffset, "");
static_assert(offsetof(TileWorkControl, invariant_bits) == kTileWorkInvariantOffset, "");
static_assert(offsetof(TileWorkControl, abort_code) == kTileWorkAbortOffset, "");
static_assert(offsetof(TileWorkControl, step_generation) == kTileWorkStepOffset, "");
static_assert(offsetof(TileWorkControl, emit_x) == kTileWorkEmitOffset, "");
static_assert(offsetof(TileWorkControl, sort_upsweep_x) == kTileWorkSortOffset, "");
static_assert(offsetof(TileWorkControl, range_x) == kTileWorkRangeOffset, "");
static_assert(sizeof(PipelineAbortLatch) == 16, "PipelineAbortLatch ABI size");

// Orig sized sort buffers to the exact pair count (~3-4 pairs/splat on corgi).
// 8x live splats is ~71 MiB keys+indices and fills the 4090 72 MiB L2.
// 4x is too tight after relocation: live N drops while pairs/live-splat rise
// (corgi peaked at 4.3), so eval views overflowed, latched abort, and poisoned
// raster-backward density.  6x leaves ~20% headroom under L2.
inline size_t bootstrapTilePairCapacity(size_t num_splats) {
    return std::max(num_splats * 6, size_t(65536));
}

inline size_t growTilePairCapacity(size_t current, uint32_t exact_count) {
    const size_t exact = static_cast<size_t>(exact_count);
    const size_t need = exact + 1;
    const size_t slack = std::max(exact / 4, size_t(65536));
    size_t next = need + slack;
    if (next < current + slack)
        next = current + slack;
    if (next < need)
        next = need;
    return next;
}

}  // namespace ftgspp
