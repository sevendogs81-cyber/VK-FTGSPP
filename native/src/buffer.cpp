#include "gs_renderer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <limits>

namespace {
std::atomic<uint64_t> next_buffer_generation{1};

double elapsed_seconds(
    const std::chrono::steady_clock::time_point& begin
) {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
}
}


size_t VulkanGSPipelineBuffers::getTotalAllocSize() {
    size_t total = 0;
    #define _(name) { \
        total += this->name.deviceBuffer.allocSize; \
    }
    _(xyz_ws)_(sh_coeffs)_(rotations)_(scales_opacs)
    _(base_xyz_ws)_(base_log_scales_opacity)_(times_durations)_(velocities_gates)
    _(flow_end_xyz)
    _(grad_base_xyz_ws)_(grad_base_log_scales_opacity)_(grad_times_durations)
    _(grad_velocities_gates)_(grad_sh_coeffs)_(grad_rotations)
    _(peer_grad_base_xyz_ws)_(peer_grad_base_log_scales_opacity)
    _(peer_grad_times_durations)_(peer_grad_velocities_gates)
    _(peer_grad_sh_coeffs)_(peer_grad_rotations)
    _(color_params)_(color_grad)_(color_m)_(color_v)_(peer_color_grad)_(color_raw_rgb)
    _(sparse_gradient_records)_(peer_sparse_gradient_records)_(sparse_gradient_count)
    _(sparse_active_flags)_(sparse_active_offsets)_(regularizer_timestamps)_(peer_relocation_stats)
    _(tiles_touched)_(rect_tile_space)_(radii)_(xy_vs)_(depths)_(inv_cov_vs_opacity)_(rgb)
    _(projection_cull_reason)
    _(index_buffer_offset)_(sorting_keys_1)_(sorting_keys_2)_(sorting_gauss_idx_1)_(sorting_gauss_idx_2)_(tile_ranges)
    _(tile_invariant_status)_(tile_work_control)_(pipeline_abort_latch)
    _(pixel_state)_(n_contributors)_(ssim_map)_(v_pixel_state)_(ref_image)_(loss_accum)_(grad_stats)
    _(v_xy_vs)_(v_depths)_(v_inv_cov_vs_opacity)_(v_rgb)
    _(g_xyz_ws)_(g_sh_coeffs_1)_(g_sh_coeffs_2)_(g_rotations)_(g_scales_opacs)
    _(g_times_durations)_(g_velocities_gates)_(relocation_map)_(relocation_counts)
    _(relocation_raw)_(relocation_cdf)_(relocation_totals)_(relocation_status)
    _(relocation_stats)_(relocation_stats_generation)_(_temp_gauss_attr)
    _(_cumsum_blockSums)_(_cumsum_blockSums2)_(_cumsum_f32_blockSums)_(_cumsum_f32_blockSums2)_(_sorting_histogram)_(_sorting_histogram_cumsum)
    #undef _
    return total;
}

void VulkanGSPipelineBuffers::updateTotalAllocSize() {
    size_t sz = getTotalAllocSize();
    if (sz > maxTotalAllocSize) 
        maxTotalAllocSize = sz;
}

std::map<std::string, size_t> VulkanGSPipelineBuffers::getVramBreakdown() {
    std::map<std::string, size_t> breakdown;
    #define _(name) { \
        breakdown[#name] += this->name.deviceBuffer.allocSize; \
    }
    _(xyz_ws)_(sh_coeffs)_(rotations)_(scales_opacs)
    _(base_xyz_ws)_(base_log_scales_opacity)_(times_durations)_(velocities_gates)
    _(flow_end_xyz)
    _(grad_base_xyz_ws)_(grad_base_log_scales_opacity)_(grad_times_durations)
    _(grad_velocities_gates)_(grad_sh_coeffs)_(grad_rotations)
    _(peer_grad_base_xyz_ws)_(peer_grad_base_log_scales_opacity)
    _(peer_grad_times_durations)_(peer_grad_velocities_gates)
    _(peer_grad_sh_coeffs)_(peer_grad_rotations)
    _(color_params)_(color_grad)_(color_m)_(color_v)_(peer_color_grad)_(color_raw_rgb)
    _(sparse_gradient_records)_(peer_sparse_gradient_records)_(sparse_gradient_count)
    _(sparse_active_flags)_(sparse_active_offsets)_(regularizer_timestamps)_(peer_relocation_stats)
    _(tiles_touched)_(rect_tile_space)_(radii)_(xy_vs)_(depths)_(inv_cov_vs_opacity)_(rgb)
    _(projection_cull_reason)
    _(index_buffer_offset)_(sorting_keys_1)_(sorting_keys_2)_(sorting_gauss_idx_1)_(sorting_gauss_idx_2)_(tile_ranges)
    _(tile_invariant_status)_(tile_work_control)_(pipeline_abort_latch)
    _(pixel_state)_(n_contributors)_(ssim_map)_(v_pixel_state)_(ref_image)_(loss_accum)_(grad_stats)
    _(v_xy_vs)_(v_depths)_(v_inv_cov_vs_opacity)_(v_rgb)
    _(g_xyz_ws)_(g_sh_coeffs_1)_(g_sh_coeffs_2)_(g_rotations)_(g_scales_opacs)
    _(g_times_durations)_(g_velocities_gates)_(relocation_map)_(relocation_counts)
    _(relocation_raw)_(relocation_cdf)_(relocation_totals)_(relocation_status)
    _(relocation_stats)_(relocation_stats_generation)_(_temp_gauss_attr)
    _(_cumsum_blockSums)_(_cumsum_blockSums2)_(_cumsum_f32_blockSums)_(_cumsum_f32_blockSums2)_(_sorting_histogram)_(_sorting_histogram_cumsum)
    #undef _
    return breakdown;
}

uint32_t findMemoryType(VkPhysicalDevice physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);
    
    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    
    return UINT32_MAX;
}


void VulkanGSPipeline::destroyStager(_Stager& target) {
    if (target.mapped != nullptr && target.memory != VK_NULL_HANDLE &&
        device != VK_NULL_HANDLE) {
        vkUnmapMemory(device, target.memory);
    }
    target.reset_mapping_fields();
    if (target.buffer != VK_NULL_HANDLE && device != VK_NULL_HANDLE)
        vkDestroyBuffer(device, target.buffer, nullptr);
    if (target.memory != VK_NULL_HANDLE && device != VK_NULL_HANDLE)
        vkFreeMemory(device, target.memory, nullptr);
    target.buffer = VK_NULL_HANDLE;
    target.memory = VK_NULL_HANDLE;
    target.allocSize = 0;
}


void* VulkanGSPipeline::ensureStagerMapped(_Stager& target) {
    if (target.mapped != nullptr)
        return target.mapped;
    if (target.memory == VK_NULL_HANDLE)
        _THROW_ERROR("Cannot map an empty staging buffer");
    void* mapped = nullptr;
    if (vkMapMemory(device, target.memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS)
        _THROW_ERROR("Failed to persistently map staging buffer memory");
    target.mapped = mapped;
    return target.mapped;
}


void VulkanGSPipeline::invalidateStager(_Stager& target, size_t size) {
    if (target.coherent || target.mapped == nullptr || size == 0)
        return;
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = target.memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    if (vkInvalidateMappedMemoryRanges(device, 1, &range) != VK_SUCCESS)
        _THROW_ERROR("Failed to invalidate staging mapped memory");
}


void VulkanGSPipeline::flushStager(_Stager& target, size_t size) {
    if (target.coherent || target.mapped == nullptr || size == 0)
        return;
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = target.memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    if (vkFlushMappedMemoryRanges(device, 1, &range) != VK_SUCCESS)
        _THROW_ERROR("Failed to flush staging mapped memory");
}


void VulkanGSPipeline::allocStagingBufferFor(
    _Stager& target, size_t size, StagingPurpose purpose
) {
    if (size == 0)
        _THROW_ERROR("Attempted to allocate a zero-byte staging buffer");

    std::lock_guard<std::mutex> lock(target.mutex);
    if (target.buffer != VK_NULL_HANDLE && target.allocSize >= size) {
        ensureStagerMapped(target);
        return;
    }

    HOST_GUARD;

    if (target.buffer != VK_NULL_HANDLE || target.memory != VK_NULL_HANDLE)
        destroyStager(target);

    auto release_partial = [&]() {
        destroyStager(target);
    };

    const auto stagingUsage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VkBufferCreateInfo staging_info = {};
    staging_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    staging_info.size = size;
    staging_info.usage = stagingUsage;
    staging_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &staging_info, nullptr, &target.buffer) != VK_SUCCESS) {
        target.buffer = VK_NULL_HANDLE;
        _THROW_ERROR("Failed to create staging buffer");
    }

    VkMemoryRequirements staging_mem_req;
    vkGetBufferMemoryRequirements(device, target.buffer, &staging_mem_req);

    // Preference order depends on transfer direction:
    // - Readback / CachedUpload: cached memory makes large host memcpy
    //   dramatically faster (pair-exchange copies peer readback into the
    //   local upload stager).
    // - Upload: coherent write-combine is preferred for small CPU fills.
    // - General: try cached+coherent first, then coherent-only.
    std::vector<VkMemoryPropertyFlags> candidates;
    if (purpose == StagingPurpose::Readback) {
        candidates = {
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        };
    } else if (purpose == StagingPurpose::Upload) {
        candidates = {
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        };
    } else if (purpose == StagingPurpose::CachedUpload) {
        candidates = {
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        };
    } else {
        candidates = {
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        };
    }

    uint32_t stagingMemTypeIndex = UINT32_MAX;
    VkMemoryPropertyFlags chosen = 0;
    for (VkMemoryPropertyFlags props : candidates) {
        stagingMemTypeIndex = findMemoryType(
            physical_device, staging_mem_req.memoryTypeBits, props);
        if (stagingMemTypeIndex != UINT32_MAX) {
            chosen = props;
            break;
        }
    }
    if (stagingMemTypeIndex == UINT32_MAX) {
        release_partial();
        _THROW_ERROR("Failed to find staging buffer memory type");
    }

    VkMemoryAllocateInfo staging_alloc = {};
    staging_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    staging_alloc.allocationSize = staging_mem_req.size;
    staging_alloc.memoryTypeIndex = stagingMemTypeIndex;

    if (vkAllocateMemory(device, &staging_alloc, nullptr, &target.memory) != VK_SUCCESS) {
        release_partial();
        throw std::runtime_error(
            "Failed to allocate staging buffer memory. You are likely running out of RAM.");
    }

    if (vkBindBufferMemory(device, target.buffer, target.memory, 0) != VK_SUCCESS) {
        release_partial();
        _THROW_ERROR("Failed to bind staging buffer memory");
    }

    target.allocSize = size;
    target.memory_flags = chosen;
    target.coherent =
        (chosen & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    target.cached =
        (chosen & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
    ensureStagerMapped(target);
}


void VulkanGSPipeline::allocStagingBuffer(size_t size) {
    allocStagingBufferFor(stager, size, StagingPurpose::General);
}


void VulkanGSPipeline::createBuffer(size_t size, _VulkanBuffer& buffer) {
    if (size == 0)
        _THROW_ERROR("Attempted to allocate a zero-byte Vulkan buffer");
    if (buffer.buffer != VK_NULL_HANDLE || buffer.memory != VK_NULL_HANDLE)
        _THROW_ERROR("createBuffer target already owns a Vulkan allocation");
    buffer.allocSize = size;
    buffer.size = size;
    auto release_partial = [&]() {
        if (buffer.buffer != VK_NULL_HANDLE)
            vkDestroyBuffer(device, buffer.buffer, nullptr);
        if (buffer.memory != VK_NULL_HANDLE)
            vkFreeMemory(device, buffer.memory, nullptr);
        buffer = _VulkanBuffer();
    };
    
    // Try to create device-local buffer first (optimal for GPU operations)
    const auto deviceUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    const auto deviceProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    
    // Create main device-local buffer
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = deviceUsage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateBuffer(device, &buffer_info, nullptr, &buffer.buffer) != VK_SUCCESS) {
        buffer = _VulkanBuffer();
        _THROW_ERROR("Failed to create buffer");
    }
    telemetry_.vk_create_buffer += 1;
    telemetry_.buffer_allocations += 1;
    
    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(device, buffer.buffer, &mem_requirements);
    
    uint32_t memoryTypeIndex = findMemoryType(physical_device, mem_requirements.memoryTypeBits, deviceProperties);
    
    if (memoryTypeIndex == UINT32_MAX) {
        release_partial();
        _THROW_ERROR("Device-local memory is not available");
    }
    
    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = memoryTypeIndex;
    
    if (vkAllocateMemory(device, &alloc_info, nullptr, &buffer.memory) != VK_SUCCESS) {
        release_partial();
        throw std::runtime_error("Failed to allocate buffer memory. You are likely running out of VRAM.");
    }
    telemetry_.vk_allocate_memory += 1;
    if (vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0) != VK_SUCCESS) {
        release_partial();
        _THROW_ERROR("Failed to bind device-local buffer memory");
    }
    buffer.generation = next_buffer_generation.fetch_add(1, std::memory_order_relaxed);
    
    current_vram += size;
    if (current_vram > peak_vram)
        peak_vram = current_vram;
}


void VulkanGSPipeline::destroyBuffer(_VulkanBuffer &buffer) {
    if (commandBatchInProgress)
        _THROW_ERROR("destroyBuffer called when command batch in progress");
    if (buffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer.buffer, nullptr);
        if (buffer.memory != VK_NULL_HANDLE)
            vkFreeMemory(device, buffer.memory, nullptr);
        buffer.buffer = VK_NULL_HANDLE;
        buffer.memory = VK_NULL_HANDLE;
        if (current_vram < buffer.allocSize) {
            fprintf(stderr, "Vulkan VRAM accounting underflow during buffer destruction\n");
            current_vram = 0;
        } else {
            current_vram -= buffer.allocSize;
        }
    }
    buffer.allocSize = 0;
    buffer.size = 0;
    buffer.generation = 0;
}


void VulkanGSPipeline::resizeDeviceBuffer(_VulkanBuffer& deviceBuffer, size_t new_byte_size, bool no_shrink) {
    if (new_byte_size == 0) {
        if (deviceBuffer.buffer != VK_NULL_HANDLE || deviceBuffer.memory != VK_NULL_HANDLE) {
            HOST_GUARD;
            destroyBuffer(deviceBuffer);
        }
        deviceBuffer = _VulkanBuffer();
        return;
    }
    if (deviceBuffer.allocSize >= new_byte_size &&
        (no_shrink || deviceBuffer.allocSize == new_byte_size)) {
        deviceBuffer.size = new_byte_size;
        return;
    }
    if (deviceBuffer.allocSize < new_byte_size || (!no_shrink && deviceBuffer.allocSize > new_byte_size)) {
        HOST_GUARD;
        telemetry_.host_guard_buffer_growth += 1;
        telemetry_.buffer_reallocations += 1;
        size_t alloc = new_byte_size;
        if (no_shrink && deviceBuffer.allocSize > 0) {
            size_t grown = deviceBuffer.allocSize > (64ull << 20)
                ? deviceBuffer.allocSize + deviceBuffer.allocSize / 4
                : deviceBuffer.allocSize + deviceBuffer.allocSize / 2;
            if (grown > alloc)
                alloc = grown;
        }
        destroyBuffer(deviceBuffer);
        try {
            createBuffer(alloc, deviceBuffer);
            deviceBuffer.size = new_byte_size;
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". createBuffer failed inside resizeDeviceBuffer");
        }
        return;
    }
    deviceBuffer.size = new_byte_size;
}

template<typename T>
_VulkanBuffer& VulkanGSPipeline::resizeDeviceBuffer(Buffer<T>& buffer, size_t new_size, bool no_shrink) {
    auto& deviceBuffer = buffer.deviceBuffer;
    size_t new_byte_size = new_size * sizeof(T);
    resizeDeviceBuffer(deviceBuffer, new_byte_size, no_shrink);
    return deviceBuffer;
}


template<typename T>
_VulkanBuffer& VulkanGSPipeline::clearDeviceBuffer(Buffer<T>& buffer, size_t new_size) {
    auto& deviceBuffer = buffer.deviceBuffer;
    if (new_size == 0) {
        resizeDeviceBuffer(buffer, 0);
        return deviceBuffer;
    }
    if (deviceBuffer.size != new_size * sizeof(T)) {
        HOST_GUARD;
        resizeDeviceBuffer(buffer, new_size);
    }

    {
        DEVICE_GUARD;
        // A reused allocation may still be the target of an earlier compute
        // or transfer write in this command batch.  vkCmdFillBuffer is a new
        // transfer write and needs an execution/memory dependency for WAW as
        // well as the post-fill visibility dependency below.
        bufferMemoryBarrier({
            {deviceBuffer, TRANSFER_COMPUTE_SHADER_READ_WRITE},
        }, TRANSFER_WRITE);
        vkCmdFillBuffer(command_buffer, deviceBuffer.buffer, 0, deviceBuffer.size, 0);
        bufferMemoryBarrier({
            {deviceBuffer, TRANSFER_WRITE},
        }, TRANSFER_COMPUTE_SHADER_READ_WRITE);
    }

    return deviceBuffer;
}


template<typename T>
_VulkanBuffer& VulkanGSPipeline::resizeAndCopyDeviceBuffer(
    Buffer<T>& buffer,
    size_t new_size,
    bool clear
) {
    auto& deviceBuffer = buffer.deviceBuffer;

    size_t new_byte_size = new_size * sizeof(T);
    size_t old_byte_size = deviceBuffer.size;

    if (new_size == 0) {
        resizeDeviceBuffer(deviceBuffer, 0);
        return deviceBuffer;
    }

    if (new_size <= deviceBuffer.allocSize / sizeof(T)) {
        deviceBuffer.size = new_byte_size;

        if (clear && new_byte_size > old_byte_size) {
            VkDeviceSize offset = old_byte_size;
            VkDeviceSize size   = new_byte_size - old_byte_size;

            VkDeviceSize alignedOffset = (offset + 3) & ~3ULL;
            VkDeviceSize prefix = alignedOffset - offset;
            if (prefix < size) {
                offset = alignedOffset;
                size -= prefix;
                DEVICE_GUARD;
                vkCmdFillBuffer(command_buffer, deviceBuffer.buffer, offset, size, 0u);
                HOST_GUARD;  // will apply fence
            }
        }

        return deviceBuffer;
    }

    _VulkanBuffer newBuffer;
    try {
        createBuffer(new_byte_size, newBuffer);
    } catch (const std::runtime_error& err) {
        _THROW_ERROR(std::string(err.what()) +
            ". createBuffer failed inside resizeAndCopyDeviceBuffer");
    }

    {
    DEVICE_GUARD;

    if (deviceBuffer.buffer != VK_NULL_HANDLE && old_byte_size > 0) {
        bufferMemoryBarrier({
            {deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
        }, TRANSFER_READ);
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = old_byte_size;

        vkCmdCopyBuffer(
            command_buffer,
            deviceBuffer.buffer,
            newBuffer.buffer,
            1,
            &copyRegion
        );
    }

    if (clear && old_byte_size < new_byte_size) {
        VkDeviceSize offset = old_byte_size;
        VkDeviceSize size   = new_byte_size - old_byte_size;

        VkDeviceSize alignedOffset = (offset + 3) & ~3ULL;
        VkDeviceSize prefix = alignedOffset - offset;
        if (prefix < size) {
            offset = alignedOffset;
            size -= prefix;

            vkCmdFillBuffer(
                command_buffer,
                newBuffer.buffer,
                offset,
                size,
                0u
            );
        }
    }

    bufferMemoryBarrier({
        {newBuffer, TRANSFER_WRITE},
    }, TRANSFER_COMPUTE_SHADER_READ_WRITE);

    }

    HOST_GUARD;
    destroyBuffer(deviceBuffer);
    deviceBuffer = newBuffer;
    deviceBuffer.size = new_byte_size;

    return deviceBuffer;
}


template<typename T>
_VulkanBuffer& VulkanGSPipeline::copyToDevice(Buffer<T>& buffer) {

    resizeDeviceBuffer(buffer, buffer.size());
    auto& deviceBuffer = buffer.deviceBuffer;
    if (buffer.empty())
        return deviceBuffer;

    allocStagingBuffer(buffer.byteLength());
    {
        std::lock_guard<std::mutex> lock(stager.mutex);

        // Use staging buffer for device-local memory
    {
        HOST_GUARD;
        void* temp = ensureStagerMapped(stager);
        memcpy(temp, buffer.data(), buffer.byteLength());
        flushStager(stager, deviceBuffer.size);
    }

        // Copy from staging to device buffer
        DEVICE_GUARD;
        VkBufferCopy copy_region = {};
        copy_region.size = deviceBuffer.size;
        vkCmdCopyBuffer(command_buffer, stager.buffer, deviceBuffer.buffer, 1, &copy_region);
        bufferMemoryBarrier({
            {deviceBuffer, TRANSFER_WRITE},
        }, TRANSFER_COMPUTE_SHADER_READ);
    }
    
    return deviceBuffer;
}


template<typename T>
void VulkanGSPipeline::copyFromDevice(Buffer<T>& buffer) {

    auto& deviceBuffer = buffer.deviceBuffer;
    buffer.resize(deviceBuffer.size / sizeof(T));
    if (buffer.empty())
        return;
    
    allocStagingBuffer(buffer.byteLength());
    {
        std::lock_guard<std::mutex> lock(stager.mutex);

        // Copy from device to staging buffer

    {
        DEVICE_GUARD;
        bufferMemoryBarrier({
            {deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
        }, TRANSFER_READ);
        VkBufferCopy copy_region = {};
        copy_region.size = deviceBuffer.size;
        vkCmdCopyBuffer(command_buffer, deviceBuffer.buffer, stager.buffer, 1, &copy_region);
    }
        HOST_GUARD;  // will apply fence

        // Read from staging buffer
        void* temp = ensureStagerMapped(stager);
        invalidateStager(stager, deviceBuffer.size);
        memcpy(reinterpret_cast<void*>(buffer.data()), temp, deviceBuffer.size);
    }
}


void VulkanGSPipeline::copyBuffersFromDevicePacked(
    const std::vector<PackedBufferRegion>& regions,
    void* destination,
    size_t packedSize
) {
    if (destination == nullptr || packedSize == 0)
        _THROW_ERROR("Packed device readback requires non-empty destination storage");
    if (regions.empty())
        _THROW_ERROR("Packed device readback requires at least one region");
    for (const auto& region : regions) {
        if (region.buffer.buffer == VK_NULL_HANDLE || region.buffer.size == 0)
            _THROW_ERROR("Packed device readback contains an empty source buffer");
        if (region.packedOffset % 4 != 0 || region.buffer.size % 4 != 0 ||
            region.packedOffset > packedSize ||
            region.buffer.size > packedSize - region.packedOffset)
            _THROW_ERROR("Packed device readback region is out of range or unaligned");
    }

    PackedTransferTiming timing{};
    // If a caller already owns a command batch, suspend it for the complete
    // packed transfer. The nested DeviceGuard below must own a submission and
    // fence before host mapping; HostGuard restores the caller's batch at exit.
    HOST_GUARD;
    allocStagingBufferFor(readback_stager, packedSize, StagingPurpose::Readback);
    std::lock_guard<std::mutex> lock(readback_stager.mutex);
    timing.used_cached_memory = readback_stager.cached;
    timing.used_persistent_map = readback_stager.mapped != nullptr;
    {
        const auto gpu_begin = std::chrono::steady_clock::now();
        DEVICE_GUARD;
        std::vector<std::pair<_VulkanBuffer, BarrierMask>> barriers;
        barriers.reserve(regions.size());
        for (const auto& region : regions)
            barriers.push_back({region.buffer, TRANSFER_COMPUTE_SHADER_WRITE});
        bufferMemoryBarrier(barriers, TRANSFER_READ);
        for (const auto& region : regions) {
            VkBufferCopy copy{};
            copy.srcOffset = 0;
            copy.dstOffset = region.packedOffset;
            copy.size = region.buffer.size;
            vkCmdCopyBuffer(
                command_buffer, region.buffer.buffer, readback_stager.buffer, 1, &copy);
        }
        timing.gpu_copy_seconds = elapsed_seconds(gpu_begin);
    }

    const auto map_begin = std::chrono::steady_clock::now();
    void* mapped = ensureStagerMapped(readback_stager);
    timing.map_seconds = elapsed_seconds(map_begin);
    const auto invalidate_begin = std::chrono::steady_clock::now();
    invalidateStager(readback_stager, packedSize);
    timing.invalidate_seconds = elapsed_seconds(invalidate_begin);
    const auto memcpy_begin = std::chrono::steady_clock::now();
    memcpy(destination, mapped, packedSize);
    timing.host_memcpy_seconds = elapsed_seconds(memcpy_begin);
    last_packed_download_timing_ = timing;
}


void VulkanGSPipeline::copyBuffersToDevicePacked(
    const void* source,
    size_t packedSize,
    const std::vector<PackedBufferRegion>& regions
) {
    if (source == nullptr || packedSize == 0)
        _THROW_ERROR("Packed device upload requires non-empty source storage");
    if (regions.empty())
        _THROW_ERROR("Packed device upload requires at least one region");
    for (const auto& region : regions) {
        if (region.buffer.buffer == VK_NULL_HANDLE || region.buffer.size == 0)
            _THROW_ERROR("Packed device upload contains an empty destination buffer");
        if (region.packedOffset % 4 != 0 || region.buffer.size % 4 != 0 ||
            region.packedOffset > packedSize ||
            region.buffer.size > packedSize - region.packedOffset)
            _THROW_ERROR("Packed device upload region is out of range or unaligned");
    }

    PackedTransferTiming timing{};
    // See readback above: guarantee that the copy batch is submitted and
    // fenced even when this helper is reached from an existing device batch.
    HOST_GUARD;
    allocStagingBufferFor(upload_stager, packedSize, StagingPurpose::Upload);
    std::lock_guard<std::mutex> lock(upload_stager.mutex);
    timing.used_cached_memory = upload_stager.cached;
    timing.used_persistent_map = upload_stager.mapped != nullptr;
    const auto map_begin = std::chrono::steady_clock::now();
    void* mapped = ensureStagerMapped(upload_stager);
    timing.map_seconds = elapsed_seconds(map_begin);
    const auto memcpy_begin = std::chrono::steady_clock::now();
    memcpy(mapped, source, packedSize);
    timing.host_memcpy_seconds = elapsed_seconds(memcpy_begin);
    const auto flush_begin = std::chrono::steady_clock::now();
    flushStager(upload_stager, packedSize);
    timing.invalidate_seconds = elapsed_seconds(flush_begin);
    {
        const auto gpu_begin = std::chrono::steady_clock::now();
        DEVICE_GUARD;
        for (const auto& region : regions) {
            VkBufferCopy copy{};
            copy.srcOffset = region.packedOffset;
            copy.dstOffset = 0;
            copy.size = region.buffer.size;
            vkCmdCopyBuffer(
                command_buffer, upload_stager.buffer, region.buffer.buffer, 1, &copy);
        }
        std::vector<std::pair<_VulkanBuffer, BarrierMask>> barriers;
        barriers.reserve(regions.size());
        for (const auto& region : regions)
            barriers.push_back({region.buffer, TRANSFER_WRITE});
        bufferMemoryBarrier(barriers, TRANSFER_COMPUTE_SHADER_READ_WRITE);
        timing.gpu_copy_seconds = elapsed_seconds(gpu_begin);
    }
    last_packed_upload_timing_ = timing;
}


namespace {
struct PackedRangeIntersection {
    _VulkanBuffer buffer;
    size_t buffer_offset = 0;
    size_t range_offset = 0;
    size_t bytes = 0;
};

std::vector<PackedRangeIntersection> intersect_packed_range(
    const std::vector<VulkanGSPipeline::PackedBufferRegion>& regions,
    size_t packed_offset,
    size_t range_size
) {
    if (range_size == 0 ||
        packed_offset > std::numeric_limits<size_t>::max() - range_size)
        throw std::runtime_error("packed transfer range is empty or overflows");
    const size_t range_end = packed_offset + range_size;
    std::vector<PackedRangeIntersection> result;
    for (const auto& region : regions) {
        if (region.buffer.buffer == VK_NULL_HANDLE || region.buffer.size == 0)
            throw std::runtime_error("packed transfer range contains an empty buffer");
        if (region.packedOffset >
            std::numeric_limits<size_t>::max() - region.buffer.size)
            throw std::runtime_error("packed transfer region overflows");
        const size_t region_end = region.packedOffset + region.buffer.size;
        const size_t begin = std::max(region.packedOffset, packed_offset);
        const size_t end = std::min(region_end, range_end);
        if (begin >= end)
            continue;
        result.push_back({
            region.buffer,
            begin - region.packedOffset,
            begin - packed_offset,
            end - begin,
        });
    }
    if (result.empty())
        throw std::runtime_error("packed transfer range intersects no ABI region");
    std::sort(
        result.begin(), result.end(),
        [](const PackedRangeIntersection& lhs,
           const PackedRangeIntersection& rhs) {
            return lhs.range_offset < rhs.range_offset;
        });
    size_t covered = 0;
    for (const auto& slice : result) {
        if (slice.range_offset != covered)
            throw std::runtime_error(
                "packed transfer range has a gap or overlapping ABI region");
        covered += slice.bytes;
    }
    if (covered != range_size)
        throw std::runtime_error(
            "packed transfer range is not fully covered by ABI regions");
    return result;
}
}


VulkanGSPipeline::PackedTransferTiming
VulkanGSPipeline::waitPackedTransferSlot(
    _PackedTransferSlot& slot, uint64_t generation
) {
    if (generation == 0 || slot.generation != generation)
        _THROW_ERROR("Packed-transfer ticket is stale or belongs to another slot");
    if (slot.in_flight) {
        const VkResult result = vkWaitForFences(
            device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
        if (result != VK_SUCCESS)
            _THROW_ERROR("Failed while waiting for packed-transfer fence");
        slot.in_flight = false;
        slot.completed_generation = generation;
        slot.timing.gpu_copy_seconds = elapsed_seconds(slot.submitted_at);
    }
    if (slot.completed_generation != generation)
        _THROW_ERROR("Packed-transfer ticket completed out of generation order");
    return slot.timing;
}


VulkanGSPipeline::PackedTransferTicket
VulkanGSPipeline::submitBuffersFromDevicePackedRange(
    const std::vector<PackedBufferRegion>& regions,
    size_t packedOffset,
    size_t rangeSize
) {
    if (commandBatchInProgress)
        _THROW_ERROR("Async packed readback requires an idle primary command batch");
    const auto slices = intersect_packed_range(regions, packedOffset, rangeSize);
    const uint32_t slot_index =
        packed_readback_cursor_++ % packed_transfer_ring_depth;
    auto& slot = packed_readback_ring_[slot_index];
    if (slot.in_flight)
        _THROW_ERROR("Packed readback ring exhausted before its oldest ticket completed");

    allocStagingBufferFor(slot.stager, rangeSize, StagingPurpose::Readback);
    slot.timing = PackedTransferTiming{};
    slot.timing.used_cached_memory = slot.stager.cached;
    slot.timing.used_persistent_map = slot.stager.mapped != nullptr;
    slot.range_size = rangeSize;
    slot.generation = ++packed_transfer_generation_;
    slot.completed_generation = 0;

    if (vkResetFences(device, 1, &slot.fence) != VK_SUCCESS)
        _THROW_ERROR("Failed to reset packed readback fence");
    if (vkResetCommandBuffer(slot.command_buffer, 0) != VK_SUCCESS)
        _THROW_ERROR("Failed to reset packed readback command buffer");
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(slot.command_buffer, &begin_info) != VK_SUCCESS)
        _THROW_ERROR("Failed to begin packed readback command buffer");

    std::vector<std::pair<_VulkanBuffer, BarrierMask>> barriers;
    barriers.reserve(slices.size());
    for (const auto& slice : slices)
        barriers.push_back({slice.buffer, TRANSFER_COMPUTE_SHADER_WRITE});
    bufferMemoryBarrier(slot.command_buffer, barriers, TRANSFER_READ);
    for (const auto& slice : slices) {
        VkBufferCopy copy{};
        copy.srcOffset = slice.buffer_offset;
        copy.dstOffset = slice.range_offset;
        copy.size = slice.bytes;
        vkCmdCopyBuffer(
            slot.command_buffer, slice.buffer.buffer, slot.stager.buffer, 1, &copy);
    }
    _VulkanBuffer dst;
    dst.buffer = slot.stager.buffer;
    dst.memory = slot.stager.memory;
    dst.allocSize = slot.stager.allocSize;
    dst.size = rangeSize;
    bufferMemoryBarrier(slot.command_buffer, {{dst, TRANSFER_WRITE}}, HOST_READ);
    if (vkEndCommandBuffer(slot.command_buffer) != VK_SUCCESS)
        _THROW_ERROR("Failed to end packed readback command buffer");
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &slot.command_buffer;
    slot.submitted_at = std::chrono::steady_clock::now();
    VkTimelineSemaphoreSubmitInfo timeline_info{};
    uint64_t wait_value = 0;
    uint64_t signal_value = 0;
    attachComputeTimeline(submit_info, timeline_info, wait_value, signal_value);
    if (vkQueueSubmit(command_queue, 1, &submit_info, slot.fence) != VK_SUCCESS)
        _THROW_ERROR("Failed to submit packed readback command buffer");
    slot.in_flight = true;
    telemetry_.pcie_d2h_bytes += rangeSize;
    return {slot_index, slot.generation, rangeSize, true};
}


VulkanGSPipeline::PackedTransferTicket
VulkanGSPipeline::submitComputeReadback(const _VulkanBuffer& src, size_t bytes) {
    if (commandBatchInProgress)
        _THROW_ERROR("Compute readback requires an idle primary command batch");
    if (src.buffer == VK_NULL_HANDLE || bytes == 0)
        _THROW_ERROR("Compute readback requires a non-empty source buffer");
    if (bytes > src.size)
        _THROW_ERROR("Compute readback size exceeds the source buffer");
    if (bytes % 4 != 0)
        _THROW_ERROR("Compute readback size must be 4-byte aligned");

    PackedBufferRegion region;
    region.buffer = src;
    region.packedOffset = 0;
    const auto slices = intersect_packed_range({region}, 0, bytes);
    const uint32_t slot_index =
        packed_readback_cursor_++ % packed_transfer_ring_depth;
    auto& slot = packed_readback_ring_[slot_index];
    if (slot.in_flight)
        _THROW_ERROR("Packed readback ring exhausted before its oldest ticket completed");

    allocStagingBufferFor(slot.stager, bytes, StagingPurpose::Readback);
    slot.timing = PackedTransferTiming{};
    slot.timing.used_cached_memory = slot.stager.cached;
    slot.timing.used_persistent_map = slot.stager.mapped != nullptr;
    slot.range_size = bytes;
    slot.generation = ++packed_transfer_generation_;
    slot.completed_generation = 0;

    if (vkResetFences(device, 1, &slot.fence) != VK_SUCCESS)
        _THROW_ERROR("Failed to reset compute readback fence");
    if (vkResetCommandBuffer(slot.command_buffer, 0) != VK_SUCCESS)
        _THROW_ERROR("Failed to reset compute readback command buffer");
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(slot.command_buffer, &begin_info) != VK_SUCCESS)
        _THROW_ERROR("Failed to begin compute readback command buffer");

    std::vector<std::pair<_VulkanBuffer, BarrierMask>> barriers;
    barriers.reserve(slices.size());
    for (const auto& slice : slices)
        barriers.push_back({slice.buffer, TRANSFER_COMPUTE_SHADER_WRITE});
    bufferMemoryBarrier(slot.command_buffer, barriers, TRANSFER_READ);
    for (const auto& slice : slices) {
        VkBufferCopy copy{};
        copy.srcOffset = slice.buffer_offset;
        copy.dstOffset = slice.range_offset;
        copy.size = slice.bytes;
        vkCmdCopyBuffer(
            slot.command_buffer, slice.buffer.buffer, slot.stager.buffer, 1, &copy);
    }
    _VulkanBuffer dst;
    dst.buffer = slot.stager.buffer;
    dst.memory = slot.stager.memory;
    dst.allocSize = slot.stager.allocSize;
    dst.size = bytes;
    bufferMemoryBarrier(slot.command_buffer, {{dst, TRANSFER_WRITE}}, HOST_READ);
    if (vkEndCommandBuffer(slot.command_buffer) != VK_SUCCESS)
        _THROW_ERROR("Failed to end compute readback command buffer");

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &slot.command_buffer;
    VkTimelineSemaphoreSubmitInfo timeline_info{};
    uint64_t wait_value = 0;
    uint64_t signal_value = 0;
    attachComputeTimeline(submit_info, timeline_info, wait_value, signal_value);
    slot.submitted_at = std::chrono::steady_clock::now();
    if (vkQueueSubmit(command_queue, 1, &submit_info, slot.fence) != VK_SUCCESS)
        _THROW_ERROR("Failed to submit compute readback command buffer");
    slot.in_flight = true;
    return {slot_index, slot.generation, bytes, true};
}


VulkanGSPipeline::PackedTransferTiming
VulkanGSPipeline::completeBuffersFromDevicePackedRange(
    const PackedTransferTicket& ticket,
    void* destination
) {
    if (!ticket.valid() || !ticket.readback || destination == nullptr)
        _THROW_ERROR("Packed readback completion received an invalid ticket");
    const PackedMappedReadback mapped =
        completeBuffersFromDevicePackedRangeMapped(ticket);
    PackedTransferTiming timing = mapped.timing;
    const auto memcpy_begin = std::chrono::steady_clock::now();
    memcpy(destination, mapped.data, ticket.range_size);
    timing.host_memcpy_seconds += elapsed_seconds(memcpy_begin);
    auto& slot = packed_readback_ring_[ticket.slot];
    slot.timing = timing;
    last_packed_download_timing_ = timing;
    return timing;
}


VulkanGSPipeline::PackedMappedReadback
VulkanGSPipeline::completeBuffersFromDevicePackedRangeMapped(
    const PackedTransferTicket& ticket
) {
    if (!ticket.valid() || !ticket.readback)
        _THROW_ERROR("Mapped packed readback received an invalid ticket");
    auto& slot = packed_readback_ring_[ticket.slot];
    PackedTransferTiming timing = waitPackedTransferSlot(slot, ticket.generation);
    const auto invalidate_begin = std::chrono::steady_clock::now();
    invalidateStager(slot.stager, ticket.range_size);
    timing.invalidate_seconds += elapsed_seconds(invalidate_begin);
    const void* mapped = ensureStagerMapped(slot.stager);
    slot.timing = timing;
    last_packed_download_timing_ = timing;
    return {mapped, timing};
}


VulkanGSPipeline::PackedTransferTicket
VulkanGSPipeline::submitBuffersToDevicePackedRange(
    const void* source,
    size_t packedOffset,
    size_t rangeSize,
    const std::vector<PackedBufferRegion>& regions
) {
    if (commandBatchInProgress)
        _THROW_ERROR("Async packed upload requires an idle primary command batch");
    if (source == nullptr)
        _THROW_ERROR("Async packed upload requires non-empty source storage");
    const auto slices = intersect_packed_range(regions, packedOffset, rangeSize);
    const uint32_t slot_index = packed_upload_cursor_++ % packed_transfer_ring_depth;
    auto& slot = packed_upload_ring_[slot_index];
    if (slot.in_flight)
        _THROW_ERROR("Packed upload ring exhausted before its oldest ticket completed");

    allocStagingBufferFor(slot.stager, rangeSize, StagingPurpose::CachedUpload);
    slot.timing = PackedTransferTiming{};
    slot.timing.used_cached_memory = slot.stager.cached;
    slot.timing.used_persistent_map = slot.stager.mapped != nullptr;
    slot.range_size = rangeSize;
    slot.generation = ++packed_transfer_generation_;
    slot.completed_generation = 0;
    const auto memcpy_begin = std::chrono::steady_clock::now();
    memcpy(ensureStagerMapped(slot.stager), source, rangeSize);
    slot.timing.host_memcpy_seconds = elapsed_seconds(memcpy_begin);
    const auto flush_begin = std::chrono::steady_clock::now();
    flushStager(slot.stager, rangeSize);
    slot.timing.invalidate_seconds = elapsed_seconds(flush_begin);

    if (vkResetFences(device, 1, &slot.fence) != VK_SUCCESS)
        _THROW_ERROR("Failed to reset packed upload fence");
    if (vkResetCommandBuffer(slot.command_buffer, 0) != VK_SUCCESS)
        _THROW_ERROR("Failed to reset packed upload command buffer");
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(slot.command_buffer, &begin_info) != VK_SUCCESS)
        _THROW_ERROR("Failed to begin packed upload command buffer");
    for (const auto& slice : slices) {
        VkBufferCopy copy{};
        copy.srcOffset = slice.range_offset;
        copy.dstOffset = slice.buffer_offset;
        copy.size = slice.bytes;
        vkCmdCopyBuffer(
            slot.command_buffer, slot.stager.buffer, slice.buffer.buffer, 1, &copy);
    }
    std::vector<std::pair<_VulkanBuffer, BarrierMask>> barriers;
    barriers.reserve(slices.size());
    for (const auto& slice : slices)
        barriers.push_back({slice.buffer, TRANSFER_WRITE});
    bufferMemoryBarrier(
        slot.command_buffer, barriers, TRANSFER_COMPUTE_SHADER_READ_WRITE);
    if (vkEndCommandBuffer(slot.command_buffer) != VK_SUCCESS)
        _THROW_ERROR("Failed to end packed upload command buffer");
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &slot.command_buffer;
    slot.submitted_at = std::chrono::steady_clock::now();
    VkTimelineSemaphoreSubmitInfo timeline_info{};
    uint64_t wait_value = 0;
    uint64_t signal_value = 0;
    attachComputeTimeline(submit_info, timeline_info, wait_value, signal_value);
    if (vkQueueSubmit(command_queue, 1, &submit_info, slot.fence) != VK_SUCCESS)
        _THROW_ERROR("Failed to submit packed upload command buffer");
    slot.in_flight = true;
    telemetry_.pcie_h2d_bytes += rangeSize;
    telemetry_.h2d_bytes += rangeSize;
    return {slot_index, slot.generation, rangeSize, false};
}


VulkanGSPipeline::PackedTransferTiming
VulkanGSPipeline::completeBuffersToDevicePackedRange(
    const PackedTransferTicket& ticket
) {
    if (!ticket.valid() || ticket.readback)
        _THROW_ERROR("Packed upload completion received an invalid ticket");
    auto& slot = packed_upload_ring_[ticket.slot];
    PackedTransferTiming timing = waitPackedTransferSlot(slot, ticket.generation);
    slot.timing = timing;
    last_packed_upload_timing_ = timing;
    return timing;
}


void VulkanGSPipeline::copyFromDeviceToDevice(const _VulkanBuffer& srcBuffer, _VulkanBuffer& dstBuffer) {
    
    if (srcBuffer.buffer == VK_NULL_HANDLE || srcBuffer.size == 0)
        _THROW_ERROR("Attempt to copy from empty buffer");
    
    const size_t elementCount = srcBuffer.size;
    resizeDeviceBuffer(dstBuffer, elementCount);
    if (dstBuffer.buffer == VK_NULL_HANDLE)
        _THROW_ERROR("Attempt to copy to empty buffer");

    DEVICE_GUARD;

    bufferMemoryBarrier({
        {srcBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
    }, TRANSFER_READ);

    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = srcBuffer.size;
    
    vkCmdCopyBuffer(command_buffer, srcBuffer.buffer, dstBuffer.buffer, 1, &copyRegion);
    bufferMemoryBarrier({
        {dstBuffer, TRANSFER_WRITE},
    }, TRANSFER_COMPUTE_SHADER_READ);
}

template<typename T>
void VulkanGSPipeline::copyFromDeviceToDevice(const Buffer<T>& src, Buffer<T>& dst) {
    const auto& srcBuffer = src.deviceBuffer;
    
    if (srcBuffer.buffer == VK_NULL_HANDLE || srcBuffer.size == 0)
        _THROW_ERROR("Attempt to copy from empty buffer");
    
    const size_t elementCount = srcBuffer.size / sizeof(T);
    resizeDeviceBuffer(dst, elementCount);
    auto& dstBuffer = dst.deviceBuffer;
    if (dstBuffer.buffer == VK_NULL_HANDLE)
        _THROW_ERROR("Attempt to copy to empty buffer");

    DEVICE_GUARD;

    bufferMemoryBarrier({
        {srcBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
    }, TRANSFER_READ);

    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = srcBuffer.size;
    
    vkCmdCopyBuffer(command_buffer, srcBuffer.buffer, dstBuffer.buffer, 1, &copyRegion);
    bufferMemoryBarrier({
        {dstBuffer, TRANSFER_WRITE},
    }, TRANSFER_COMPUTE_SHADER_READ);
}


template<typename T>
T VulkanGSPipeline::readElement(const _VulkanBuffer& buffer, size_t index) {

    const size_t elementSize = sizeof(T);
    const size_t offset = index * elementSize;
    
    // Validate bounds
    if (offset + elementSize > buffer.size)
        _THROW_ERROR("Index out of bound while reading buffer element");
    
    T outValue;

    allocStagingBuffer(buffer.size);
    {
        std::lock_guard<std::mutex> lock(stager.mutex);
    {
        DEVICE_GUARD;

        bufferMemoryBarrier({
            {buffer, TRANSFER_COMPUTE_SHADER_WRITE},
        }, TRANSFER_READ);

        // Copy only the specific element from device buffer to staging buffer
        VkBufferCopy copyRegion = {};
        copyRegion.srcOffset = offset;
        copyRegion.dstOffset = offset; // Keep same offset in staging buffer
        copyRegion.size = elementSize;
        
        vkCmdCopyBuffer(command_buffer, buffer.buffer, stager.buffer, 1, &copyRegion);
    }
        HOST_GUARD;  // will apply fence

        // Map the staging buffer and read the specific element
        void* data = static_cast<char*>(ensureStagerMapped(stager)) + offset;
        invalidateStager(stager, buffer.size);
        
        // Copy the data
        memcpy(&outValue, data, elementSize);
    }
    
    return outValue;
}


template<typename T>
void VulkanGSPipeline::dumpRawBytesToFile(Buffer<T>& buffer, std::string filename) {
    copyFromDevice(buffer);
    FILE* fp = fopen(filename.c_str(), "wb");
    if (!fp) throw std::runtime_error("Error opening file `" + filename + "`");
    fwrite(buffer.data(), sizeof(T), buffer.size(), fp);
    fclose(fp);
}



#ifdef WIN32
void VulkanGSPipeline::_displayImage(Buffer<float>& pixels, int width, bool reverse_alpha) {
    _THROW_ERROR("Displaying image in terminal is not supported on Windows");
}
#else
#include <sys/ioctl.h>
#include <unistd.h>
void VulkanGSPipeline::_displayImage(Buffer<float>& pixels, int width, bool reverse_alpha) {

    copyFromDevice(pixels);

    const int aspect = 2;

    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    int window_width = std::max((int)std::min((int)w.ws_col, (int)w.ws_row*aspect) - 1, 0);
    window_width = std::min(window_width, width);
    int height = (int)pixels.size() / (4*width);
    int window_height = (window_width * height) / (width*aspect);
    printf("%d x %d (display: %d x %d)\n", width, height, window_width, window_height);

    auto f2b = [](float x) -> uint8_t {
        return (uint8_t)(255.0 * fmin(fmax(x, 0.0f), 1.0f));
    };

    char buffer[256];
    std::string image;

    for (int y = 0; y < window_height; y++) {
        for (int x = 0; x < window_width; x++) {
            int x1 = (x * width) / window_width;
            int y1 = (y * height) / window_height;
            int idx = 4 * (y1 * width + x1);
            float *rgba = pixels.data() + idx;
            uint8_t alpha = reverse_alpha ? f2b(1.0f-rgba[3]) : f2b(rgba[3]);
            // printf("%d,", (int)alpha);
            sprintf(buffer,
                "\033[38;2;%d;%d;%d;48;2;%d;%d;%dm%c\033[0m",
                alpha, alpha, alpha,
                f2b(rgba[0]), f2b(rgba[1]), f2b(rgba[2]),
                std::isfinite(rgba[0]+rgba[1]+rgba[2]+rgba[3]) ? ' ' : 'X'
            );
            image += buffer;
        }
        image += "\n";
    }
    // printf("\n");

    printf("%s", image.data());
    fflush(stdout);
}
#endif  // WIN32, display image in terminal


float halfToFloat(uint16_t h) {
    // Extract sign, exponent, and fraction from half-precision
    uint16_t sign = (h >> 15) & 0x0001;
    uint16_t exp  = (h >> 10) & 0x001F;
    uint16_t frac =  h        & 0x03FF;

    uint32_t sign32 = sign << 31;
    uint32_t exp32;
    uint32_t frac32;

    if (exp == 0) {
        if (frac == 0) {
            // Zero (±0)
            exp32  = 0;
            frac32 = 0;
        } else {
            // Subnormal half -> normalize it
            int shift = 0;
            while ((frac & 0x0400) == 0) { // until leading 1 appears
                frac <<= 1;
                shift++;
            }
            frac &= 0x03FF; // remove the leading 1
            exp32 = 127 - 15 - shift + 1; // bias adjust (float32 bias=127, half bias=15)
            frac32 = frac << 13;
        }
    } else if (exp == 0x1F) {
        // Inf or NaN
        exp32  = 0xFF;
        frac32 = frac << 13;
        if (frac32) {
            frac32 |= 0x00000001; // make sure NaNs are quiet NaNs
        }
    } else {
        // Normalized number
        exp32  = exp - 15 + 127; // bias adjust
        frac32 = frac << 13;
    }

    uint32_t f32 = sign32 | (exp32 << 23) | frac32;
    float result;
    std::memcpy(&result, &f32, sizeof(result));
    return result;
}

uint16_t floatToHalf(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));

    uint32_t sign = (bits >> 31) & 0x0001;
    int32_t  exp  = (bits >> 23) & 0x00FF;
    uint32_t frac =  bits        & 0x007FFFFF;

    uint16_t sign16 = sign << 15;
    uint16_t exp16;
    uint16_t frac16;

    if (exp == 0xFF) {
        // Infinity or NaN
        exp16  = 0x1F;
        frac16 = (frac ? (frac >> 13) | 0x0001 : 0); // preserve signaling as quiet NaN
    }
    else if (exp == 0) {
        // Zero or subnormal in float32 → zero in half (no gradual underflow from subnormals in float32 to half)
        exp16  = 0;
        frac16 = 0;
    }
    else {
        // Normalized float32 → may map to normalized or subnormal half
        int32_t newExp = exp - 127 + 15;

        if (newExp >= 0x1F) {
            // Overflow → Inf
            exp16  = 0x1F;
            frac16 = 0;
        }
        else if (newExp <= 0) {
            // Underflow → subnormal half or zero
            if (newExp < -10) {
                // Too small → zero
                exp16  = 0;
                frac16 = 0;
            } else {
                // Subnormal half (shift mantissa)
                frac = (frac | 0x00800000) >> (1 - newExp);
                frac16 = frac >> 13;
                exp16 = 0;
            }
        }
        else {
            // Normal case
            exp16  = newExp;
            frac16 = frac >> 13;
        }
    }

    return sign16 | (exp16 << 10) | frac16;
}



template<typename T>
void VulkanGSPipelineBuffers::reorderSH(Buffer<T> &coeffs) {
    if (SH_REORDER_SIZE <= 1)
        return;

    static constexpr size_t SH_DIM = 12;

    coeffs.resize(_CEIL_ROUND(coeffs.size(),4*SH_DIM*SH_REORDER_SIZE), T(0.0));

    auto forwardIndex = [=](size_t i) {
        size_t group_idx = i / (SH_DIM * SH_REORDER_SIZE);
        size_t gauss_idx = (i / SH_DIM) % SH_REORDER_SIZE;
        size_t sh_idx = i % SH_DIM;
        return (group_idx * SH_DIM + sh_idx) * SH_REORDER_SIZE + gauss_idx;
    };

    typedef struct { T _[4]; } __m128;
    __m128* sh = reinterpret_cast<__m128*>(coeffs.data());

    size_t n = coeffs.size() / 4;

    // TODO: do this in O(1) additional memory
    std::vector<__m128> sh_copy(sh, sh+n);
    for (size_t i = 0; i < n; i++) {
        sh[forwardIndex(i)] = sh_copy[i];
    }

}

template<typename T>
void VulkanGSPipelineBuffers::undoReorderSH(Buffer<T> &coeffs, size_t num_splats) {
    if (SH_REORDER_SIZE <= 1)
        return;

    static constexpr size_t SH_DIM = 12;

    coeffs.resize(4*SH_DIM*_CEIL_ROUND(num_splats,SH_REORDER_SIZE), T(0.0));

    auto forwardIndex = [=](size_t i) {
        size_t group_idx = i / (SH_DIM * SH_REORDER_SIZE);
        size_t gauss_idx = (i / SH_DIM) % SH_REORDER_SIZE;
        size_t sh_idx = i % SH_DIM;
        return (group_idx * SH_DIM + sh_idx) * SH_REORDER_SIZE + gauss_idx;
    };

    typedef struct { T _[4]; } __m128;
    __m128* sh = reinterpret_cast<__m128*>(coeffs.data());

    size_t n = coeffs.size() / 4;

    // TODO: do this in O(1) additional memory
    std::vector<__m128> sh_copy(sh, sh+n);
    for (size_t i = 0; i < n; i++) {
        sh[i] = sh_copy[forwardIndex(i)];
    }

    coeffs.resize(4*SH_DIM*num_splats);
}

void VulkanGSPipelineBuffers::assignScalesOpacs(
    Buffer<float> &scales_opacs,
    size_t n, const float* scales, const float* opacs
) {
    scales_opacs.resize(4*n);
    for (size_t i = 0; i < n; i++) {
        float* so = &scales_opacs[4*i];
        so[0] = scales[3*i];
        so[1] = scales[3*i+1];
        so[2] = scales[3*i+2];
        so[3] = opacs[i];
    }
}



#define _INSTANTIATE_BUFFER(dtype) \
    template _VulkanBuffer& VulkanGSPipeline::resizeDeviceBuffer(Buffer<dtype>& buffer, size_t new_size, bool no_shrink); \
    template _VulkanBuffer& VulkanGSPipeline::clearDeviceBuffer(Buffer<dtype>& buffer, size_t new_size); \
    template _VulkanBuffer& VulkanGSPipeline::resizeAndCopyDeviceBuffer(Buffer<dtype>& buffer, size_t new_size, bool clear); \
    template _VulkanBuffer& VulkanGSPipeline::copyToDevice(Buffer<dtype>& buffer); \
    template void VulkanGSPipeline::copyFromDevice(Buffer<dtype>& buffer); \
    template void VulkanGSPipeline::copyFromDeviceToDevice(const Buffer<dtype>& src, Buffer<dtype>& dst); \
    template dtype VulkanGSPipeline::readElement(const _VulkanBuffer& buffer, size_t index); \
    template void VulkanGSPipeline::dumpRawBytesToFile(Buffer<dtype>& buffer, std::string filename); \
    template void VulkanGSPipelineBuffers::reorderSH(Buffer<dtype> &coeffs); \
    template void VulkanGSPipelineBuffers::undoReorderSH(Buffer<dtype> &coeffs, size_t num_splats);

_INSTANTIATE_BUFFER(uint8_t)
_INSTANTIATE_BUFFER(float)
_INSTANTIATE_BUFFER(int32_t)
_INSTANTIATE_BUFFER(uint32_t)
_INSTANTIATE_BUFFER(int64_t)

#undef _INSTANTIATE_BUFFER
