// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "config.slang"
#include "ftgspp_work_control.slang"

import utils;

#define _ENTRY_generate_keys 1
#define _ENTRY_compute_tile_ranges 2

#ifndef ENTRY
#define ENTRY 0
#endif

static const uint32_t FLOAT32_FRACTION_BITS = 23;


#if ENTRY == _ENTRY_generate_keys

layout(binding=0) StructuredBuffer<float> xy_vs;
layout(binding=1) StructuredBuffer<float> inv_cov_vs_opacity;
layout(binding=2) StructuredBuffer<float> depths;
#if USE_EMULATED_INT64
layout(binding=3) StructuredBuffer<int32_t> rect_tile_space;
#else
layout(binding=3) StructuredBuffer<int64_t> rect_tile_space;
#endif
layout(binding=4) StructuredBuffer<int32_t> index_buffer_offset;

layout(binding=5) RWStructuredBuffer<sortingKey_t> out_unsorted_keys;
layout(binding=6) RWStructuredBuffer<int32_t> out_unsorted_gauss_idx;
layout(binding=7) RWByteAddressBuffer out_tile_invariant_status;
layout(binding=8) StructuredBuffer<TileWorkControl> work_control;

#define BLOCK_SIZE 64
#define NUM_SUBGROUPS (BLOCK_SIZE / SUBGROUP_SIZE)

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uniform Uniforms uniforms
) {
    const uint image_height = uniforms.image_height;
    const uint image_width = uniforms.image_width;
    const uint grid_height = uniforms.grid_height;
    const uint grid_width = uniforms.grid_width;

    int32_t globalIdx = dispatchThreadID.x;
    const TileWorkControl wc = work_control[0];
    if (tile_work_aborted(wc))
        return;
    if (globalIdx >= uniforms.num_splats)
        return;
    const int capacity = (int)wc.capacity;

    uint32_t DEPTH_BITS = min(uint32_t(31.99-log2(grid_height*grid_width)), FLOAT32_FRACTION_BITS);

#if TILE_CULLING_MODE > 0
    float2 xy = read_t2_float2(globalIdx, xy_vs);
    float4 inv_cov_opac = read_t4_float4(globalIdx, inv_cov_vs_opacity);

    float extend = extend_from_opacity(inv_cov_opac.w);
    float3 inv_cov = inv_cov_opac.xyz / (extend*extend);
#endif  // TILE_CULLING_MODE > 0

    float zf = read_t1_float(globalIdx, depths);
    zf = (2.0*zf+1.0)/(zf+1.0);  // between 1 and 2, make float bits easier
    uint32_t z = (reinterpret<uint32_t>(zf) & ((1u<<FLOAT32_FRACTION_BITS)-1u)) >> (FLOAT32_FRACTION_BITS-DEPTH_BITS);

  #if USE_EMULATED_INT64
    rectangle rect;
    rect.min_x = (uint16_t)rect_tile_space[2*globalIdx+0];
    rect.min_y = (uint16_t)(rect_tile_space[2*globalIdx+0] >> 16);
    rect.max_x = (uint16_t)rect_tile_space[2*globalIdx+1];
    rect.max_y = (uint16_t)(rect_tile_space[2*globalIdx+1] >> 16);
  #else
    rectangle rect = reinterpret<rectangle>(rect_tile_space[globalIdx]);
  #endif

    int offset = globalIdx == 0 ? 0 : index_buffer_offset[globalIdx-1];
    int offset_max = index_buffer_offset[globalIdx];
    uint invariant_bits = 0u;

#if TILE_CULLING_MODE == 0 || TILE_CULLING_MODE == 1

    for (uint16_t y = rect.min_y; y < rect.max_y; y++) {
        for (uint16_t x = rect.min_x; x < rect.max_x; x++) {
            uint32_t key = uint32_t(y) * grid_width + uint32_t(x);
            if (key >= grid_width * grid_height)
                invariant_bits |= 2u;
            if (offset < offset_max && offset < capacity && key < grid_width * grid_height) {
                out_unsorted_keys[offset] = sortingKey_t((key << DEPTH_BITS) | z);
                out_unsorted_gauss_idx[offset] = globalIdx;
            }
            offset++;
        }
    }

#endif  // TILE_CULLING_MODE == 0 || TILE_CULLING_MODE == 1

#if TILE_CULLING_MODE == 2

    const bool scan_rows = ellipse_scan_rows(rect);
    if (scan_rows) {
        for (uint16_t y = rect.min_y; y < rect.max_y; y++) {
            const EllipseTileSpan span = ellipse_scanline_tile_span(
                xy, inv_cov, rect, scan_rows, y);
            for (uint16_t x = span.begin; x < span.end; x++) {
                uint32_t key = uint32_t(y) * grid_width + uint32_t(x);
                if (key >= grid_width * grid_height)
                    invariant_bits |= 2u;
                if (offset < offset_max && offset < capacity && key < grid_width * grid_height) {
                    out_unsorted_keys[offset] =
                        sortingKey_t((key << DEPTH_BITS) | z);
                    out_unsorted_gauss_idx[offset] = globalIdx;
                    offset++;
                }
            }
        }
    } else {
        for (uint16_t x = rect.min_x; x < rect.max_x; x++) {
            const EllipseTileSpan span = ellipse_scanline_tile_span(
                xy, inv_cov, rect, scan_rows, x);
            for (uint16_t y = span.begin; y < span.end; y++) {
                uint32_t key = uint32_t(y) * grid_width + uint32_t(x);
                if (key >= grid_width * grid_height)
                    invariant_bits |= 2u;
                if (offset < offset_max && offset < capacity && key < grid_width * grid_height) {
                    out_unsorted_keys[offset] =
                        sortingKey_t((key << DEPTH_BITS) | z);
                    out_unsorted_gauss_idx[offset] = globalIdx;
                    offset++;
                }
            }
        }
    }

#endif  // TILE_CULLING_MODE == 2

    if (offset != offset_max)
        invariant_bits |= 1u;
    if (invariant_bits != 0u)
        out_tile_invariant_status.InterlockedOr(0u, invariant_bits);
}

#endif  // ENTRY == _ENTRY_generate_keys



#if ENTRY == _ENTRY_compute_tile_ranges

layout(binding=0) StructuredBuffer<sortingKey_t> sorted_keys;
layout(binding=1) RWStructuredBuffer<int32_t> out_tile_ranges;
layout(binding=2) StructuredBuffer<TileWorkControl> work_control;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    int32_t globalIdx = dispatchThreadID.x;
    const TileWorkControl wc = work_control[0];
    if (tile_work_aborted(wc))
        return;

    const uint grid_height = uniforms.grid_height;
    const uint grid_width = uniforms.grid_width;
    uint32_t DEPTH_BITS = min(uint32_t(31.99-log2(grid_height*grid_width)), FLOAT32_FRACTION_BITS);

    uint32_t num_isects = wc.exact_count;
    if (globalIdx > (int)num_isects)
        return;

    // Read tile ID from key. Update start/end of tile range if at limit.
    sortingKey_t key = sorted_keys[globalIdx];
    uint32_t currtile = globalIdx == num_isects ?
        grid_height*grid_width : (uint32_t)(key) >> DEPTH_BITS;
    uint32_t prevtile = globalIdx == 0 ? (uint32_t)-1 :
        (uint32_t)sorted_keys[globalIdx-1] >> DEPTH_BITS;
    for (uint32_t i = prevtile+1; i <= currtile; ++i) {
        out_tile_ranges[i] = globalIdx;
    }
}

#endif  // ENTRY == _ENTRY_compute_tile_ranges
