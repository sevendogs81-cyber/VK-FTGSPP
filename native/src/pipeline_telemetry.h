#pragma once

#include <cstdint>
#include <map>
#include <string>

// Per-step and cumulative counters for the single-GPU Vulkan pipeline.
// JSONL rows are assembled on the host with explicit step/submission_id.
struct PipelineTelemetry {
    uint64_t step = 0;
    uint64_t submission_id = 0;
    uint32_t submits = 0;
    uint32_t fence_waits = 0;
    uint32_t host_guards = 0;
    uint32_t descriptor_updates = 0;
    uint32_t buffer_allocations = 0;
    uint32_t buffer_reallocations = 0;
    uint32_t vk_allocate_memory = 0;
    uint32_t vk_create_buffer = 0;
    uint32_t host_guard_image_upload = 0;
    uint32_t host_guard_descriptor_update = 0;
    uint32_t host_guard_buffer_growth = 0;
    uint32_t exact_tile_count = 0;
    uint32_t tile_capacity = 0;
    uint32_t tile_overflow = 0;
    uint32_t tile_replay = 0;
    uint32_t sync_abort = 0;
    uint64_t pcie_d2h_bytes = 0;
    uint64_t pcie_h2d_bytes = 0;
    uint32_t query_harvests = 0;
    uint32_t in_flight_high_water = 0;
    uint64_t uploaded_bytes = 0;
    uint64_t h2d_bytes = 0;
    double cpu_h2d_seconds = 0.0;
    double cpu_submit_seconds = 0.0;
    double cpu_slot_wait_seconds = 0.0;

    void reset_step() {
        submits = 0;
        fence_waits = 0;
        host_guards = 0;
        descriptor_updates = 0;
        buffer_allocations = 0;
        buffer_reallocations = 0;
        vk_allocate_memory = 0;
        vk_create_buffer = 0;
        host_guard_image_upload = 0;
        host_guard_descriptor_update = 0;
        host_guard_buffer_growth = 0;
        tile_overflow = 0;
        tile_replay = 0;
        sync_abort = 0;
        pcie_d2h_bytes = 0;
        pcie_h2d_bytes = 0;
        query_harvests = 0;
        uploaded_bytes = 0;
        h2d_bytes = 0;
        cpu_h2d_seconds = 0.0;
        cpu_submit_seconds = 0.0;
        cpu_slot_wait_seconds = 0.0;
    }

    std::map<std::string, double> as_map() const {
        return {
            {"step", (double)step},
            {"submission_id", (double)submission_id},
            {"submits_per_step", (double)submits},
            {"fence_waits_per_step", (double)fence_waits},
            {"host_guards_per_step", (double)host_guards},
            {"descriptor_updates", (double)descriptor_updates},
            {"buffer_allocations", (double)buffer_allocations},
            {"buffer_reallocations", (double)buffer_reallocations},
            {"vk_allocate_memory", (double)vk_allocate_memory},
            {"vk_create_buffer", (double)vk_create_buffer},
            {"host_guard_image_upload", (double)host_guard_image_upload},
            {"host_guard_descriptor_update", (double)host_guard_descriptor_update},
            {"host_guard_buffer_growth", (double)host_guard_buffer_growth},
            {"exact_tile_count", (double)exact_tile_count},
            {"tile_capacity", (double)tile_capacity},
            {"tile_overflow", (double)tile_overflow},
            {"tile_replay", (double)tile_replay},
            {"sync_abort", (double)sync_abort},
            {"pcie_d2h_bytes", (double)pcie_d2h_bytes},
            {"pcie_h2d_bytes", (double)pcie_h2d_bytes},
            {"query_harvests", (double)query_harvests},
            {"in_flight_high_water", (double)in_flight_high_water},
            {"uploaded_bytes", (double)uploaded_bytes},
            {"h2d_bytes", (double)h2d_bytes},
            {"cpu_h2d_seconds", cpu_h2d_seconds},
            {"cpu_submit_seconds", cpu_submit_seconds},
            {"cpu_slot_wait_seconds", cpu_slot_wait_seconds},
        };
    }
};
