#include "device/gs_pipeline.h"
#include "device/perf_timer.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>

namespace {

VkPipelineStageFlags kAllCommands = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

}  // namespace

void VulkanGSPipeline::createTimelineSemaphore() {
    has_timeline_semaphore_ = false;
    compute_timeline_ = VK_NULL_HANDLE;
    compute_timeline_value_ = 0;
    last_compute_signal_ = 0;
    pending_upload_wait_value_ = 0;

    VkPhysicalDeviceVulkan12Features v12q{};
    v12q.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &v12q;
    vkGetPhysicalDeviceFeatures2(physical_device, &features2);
    if (v12q.timelineSemaphore != VK_TRUE)
        return;

    VkSemaphoreTypeCreateInfo type_info{};
    type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    type_info.initialValue = 0;
    VkSemaphoreCreateInfo sem_info{};
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sem_info.pNext = &type_info;
    if (vkCreateSemaphore(device, &sem_info, nullptr, &compute_timeline_) != VK_SUCCESS)
        return;
    has_timeline_semaphore_ = true;
}

void VulkanGSPipeline::createTransferResources() {
    retired_tile_overflow_ = RetiredTileOverflow{};
    // Upload on the same compute queue. Signaling one timeline from two
    // queues can race (a later upload value can land before the previous
    // compute signal) and hang the device.

    for (uint32_t i = 0; i < kReferenceRingDepth; ++i) {
        auto& slot = reference_upload_ring_[i];
        VkCommandBufferAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc.commandPool = command_pool;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device, &alloc, &slot.transfer_cmd) != VK_SUCCESS)
            _THROW_ERROR("Failed to allocate reference-upload command buffer");
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(device, &fence_info, nullptr, &slot.fallback_fence) != VK_SUCCESS)
            _THROW_ERROR("Failed to create reference-upload fence");
    }
    for (auto& slot : compute_slots_) {
        allocStagingBufferFor(
            slot.wc_stager, 256, StagingPurpose::Readback);
        slot.wc_valid = false;
        slot.wc_copy_recorded = false;
    }
}

void VulkanGSPipeline::destroyAsyncResources() noexcept {
    if (device == VK_NULL_HANDLE)
        return;
    for (auto& slot : compute_slots_) {
        if (slot.fence != VK_NULL_HANDLE) {
            vkDestroyFence(device, slot.fence, nullptr);
            slot.fence = VK_NULL_HANDLE;
        }
        if (slot.query_pool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device, slot.query_pool, nullptr);
            slot.query_pool = VK_NULL_HANDLE;
        }
        if (slot.command_buffer != VK_NULL_HANDLE && command_pool != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device, command_pool, 1, &slot.command_buffer);
            slot.command_buffer = VK_NULL_HANDLE;
        }
        if (slot.wc_stager.buffer != VK_NULL_HANDLE ||
            slot.wc_stager.memory != VK_NULL_HANDLE || slot.wc_stager.mapped != nullptr) {
            try {
                destroyStager(slot.wc_stager);
            } catch (...) {
            }
        }
        slot.in_flight = false;
        slot.wc_valid = false;
        slot.wc_copy_recorded = false;
        slot.markers.clear();
    }
    command_buffer = VK_NULL_HANDLE;
    fence = VK_NULL_HANDLE;
    timestamp_query_pool = VK_NULL_HANDLE;

    for (auto& slot : reference_upload_ring_) {
        if (slot.stager.buffer != VK_NULL_HANDLE ||
            slot.stager.memory != VK_NULL_HANDLE || slot.stager.mapped != nullptr) {
            try {
                destroyStager(slot.stager);
            } catch (...) {
            }
        }
        if (slot.device_reference.buffer != VK_NULL_HANDLE) {
            try {
                destroyBuffer(slot.device_reference);
            } catch (...) {
            }
        }
        if (slot.fallback_fence != VK_NULL_HANDLE) {
            vkDestroyFence(device, slot.fallback_fence, nullptr);
            slot.fallback_fence = VK_NULL_HANDLE;
        }
        if (slot.transfer_cmd != VK_NULL_HANDLE && command_pool != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device, command_pool, 1, &slot.transfer_cmd);
            slot.transfer_cmd = VK_NULL_HANDLE;
        }
        slot.in_flight = false;
        slot.generation = 0;
    }
    if (compute_timeline_ != VK_NULL_HANDLE) {
        vkDestroySemaphore(device, compute_timeline_, nullptr);
        compute_timeline_ = VK_NULL_HANDLE;
    }
    has_timeline_semaphore_ = false;
}

void VulkanGSPipeline::harvestWorkControl(ComputeSubmissionSlot& slot) {
    if (!slot.wc_copy_recorded || slot.wc_stager.mapped == nullptr)
        return;
    invalidateStager(slot.wc_stager, ftgspp::kTileWorkControlBytes);
    std::memcpy(&slot.wc, slot.wc_stager.mapped, sizeof(slot.wc));
    slot.wc_copy_recorded = false;
    slot.wc_valid = true;
    auto& retired = retired_tile_overflow_;
    if (slot.wc.abort_code == ftgspp::kTileAbortPrefix ||
        slot.wc.abort_code == ftgspp::kTileAbortInvariant) {
        retired.fatal = true;
        retired.abort_code = slot.wc.abort_code;
    } else if (slot.wc.overflow != 0) {
        retired.overflow = true;
        retired.exact_count = std::max(retired.exact_count, slot.wc.exact_count);
        if (!retired.fatal)
            retired.abort_code = std::max(retired.abort_code, slot.wc.abort_code);
        if (slot.is_train) {
            if (!retired.is_train || retired.step < 0 ||
                (slot.step >= 0 && slot.step < retired.step))
                retired.step = slot.step;
            retired.is_train = true;
        }
    }
    // Eval forwards also copy TileWorkControl.  Their exact_count / overflow
    // must not drive train-step density or abort replay.
    if (slot.is_train) {
        telemetry_.exact_tile_count = slot.wc.exact_count;
        if (slot.wc.overflow != 0 &&
            slot.wc.abort_code != ftgspp::kTileAbortPrefix &&
            slot.wc.abort_code != ftgspp::kTileAbortInvariant)
            telemetry_.tile_overflow = 1;
    }
}

void VulkanGSPipeline::harvestComputeSlot(uint32_t index) {
    auto& slot = compute_slots_[index];
    harvestWorkControl(slot);
    if (!slot.is_train) {
        slot.timestamp_count = 0;
        slot.markers.clear();
        return;
    }
    if (slot.timestamp_count == 0 || slot.query_pool == VK_NULL_HANDLE)
        return;
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physical_device, &deviceProperties);
    const double timestampPeriod = deviceProperties.limits.timestampPeriod;
    std::vector<uint64_t> timestamps(slot.timestamp_count);
    const VkResult query_result = vkGetQueryPoolResults(
        device, slot.query_pool,
        0, slot.timestamp_count,
        sizeof(uint64_t) * slot.timestamp_count,
        timestamps.data(), sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT);
    telemetry_.query_harvests += 1;
    if (query_result != VK_SUCCESS)
        return;
    std::vector<double> times(slot.timestamp_count);
    for (uint32_t i = 0; i < slot.timestamp_count; i++)
        times[i] = 1e-9 * double(timestamps[i] - timestamps[0]) * timestampPeriod;
    auto time_updates = PerfTimer::update(times, slot.markers);
    for (auto& callback : timerCallbacks)
        callback(time_updates);
    slot.timestamp_count = 0;
    slot.markers.clear();
}

void VulkanGSPipeline::waitComputeSlot(uint32_t index, const char* reason) {
    (void)reason;
    auto& slot = compute_slots_[index];
    if (!slot.in_flight)
        return;
    const auto wait_begin = std::chrono::steady_clock::now();
    if (has_timeline_semaphore_ && slot.timeline_value > 0) {
        VkSemaphoreWaitInfo wait{};
        wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wait.semaphoreCount = 1;
        wait.pSemaphores = &compute_timeline_;
        wait.pValues = &slot.timeline_value;
        const VkResult wait_result = vkWaitSemaphores(device, &wait, UINT64_MAX);
        if (wait_result != VK_SUCCESS)
            _THROW_ERROR("Failed while waiting for the compute timeline");
    } else if (slot.fence != VK_NULL_HANDLE) {
        const VkResult wait_result =
            vkWaitForFences(device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
        if (wait_result != VK_SUCCESS)
            _THROW_ERROR("Failed while waiting for the command batch fence");
    }
    if (slot.fence != VK_NULL_HANDLE)
        vkResetFences(device, 1, &slot.fence);
    telemetry_.fence_waits += 1;
    telemetry_.cpu_slot_wait_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wait_begin).count();
    harvestComputeSlot(index);
    slot.in_flight = false;
    if (in_flight_count_ > 0)
        in_flight_count_ -= 1;
}

void VulkanGSPipeline::drain(const char* reason) {
    if (commandBatchInProgress)
        endCommandBatch(true);
    waitAllComputeSlots(reason);
    drainPackedTransfers();
    for (auto& slot : reference_upload_ring_) {
        if (!slot.in_flight || slot.fallback_fence == VK_NULL_HANDLE)
            continue;
        vkWaitForFences(device, 1, &slot.fallback_fence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &slot.fallback_fence);
        slot.in_flight = false;
    }
}

void VulkanGSPipeline::waitAllComputeSlots(const char* reason) {
    for (uint32_t i = 0; i < kComputeRingDepth; ++i)
        waitComputeSlot(i, reason);
}

void VulkanGSPipeline::drainPackedTransfers() {
    for (auto& slot : packed_readback_ring_) {
        if (!slot.in_flight)
            continue;
        waitPackedTransferSlot(slot, slot.generation);
    }
    for (auto& slot : packed_upload_ring_) {
        if (!slot.in_flight)
            continue;
        waitPackedTransferSlot(slot, slot.generation);
    }
}

void VulkanGSPipeline::attachComputeTimeline(
    VkSubmitInfo& submit,
    VkTimelineSemaphoreSubmitInfo& timeline_info,
    uint64_t& wait_value,
    uint64_t& signal_value
) {
    if (!has_timeline_semaphore_ || compute_timeline_ == VK_NULL_HANDLE)
        return;
    wait_value = last_compute_signal_;
    signal_value = compute_timeline_value_ + 1;
    compute_timeline_value_ = signal_value;
    timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    if (wait_value > 0) {
        timeline_info.waitSemaphoreValueCount = 1;
        timeline_info.pWaitSemaphoreValues = &wait_value;
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &compute_timeline_;
        submit.pWaitDstStageMask = &kAllCommands;
    }
    timeline_info.signalSemaphoreValueCount = 1;
    timeline_info.pSignalSemaphoreValues = &signal_value;
    submit.pNext = &timeline_info;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &compute_timeline_;
    last_compute_signal_ = signal_value;
}

void VulkanGSPipeline::beginCommandBatch() {
    if (commandBatchInProgress)
        _THROW_ERROR("Command batch already in progress");

    const uint32_t idx = compute_ring_cursor_ % kComputeRingDepth;
    waitComputeSlot(idx, "acquire_slot");
    current_slot_index_ = idx;
    current_descriptor_slot_ = idx;
    compute_ring_cursor_ += 1;
    auto& slot = compute_slots_[idx];
    command_buffer = slot.command_buffer;
    timestamp_query_pool = slot.query_pool;
    fence = slot.fence;
    for (_ComputePipeline* pipeline : all_compute_pipelines)
        pipeline->used_in_batch[idx] = 0;

    commandBatchInProgress = true;
    timestampNumWritten = 0;
    timestampStackDepth = 0;

    PerfTimer::hostToc();

    if (vkResetCommandBuffer(command_buffer, 0) != VK_SUCCESS)
        _THROW_ERROR("Failed to reset command buffer for batch");

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS)
        _THROW_ERROR("Failed to begin command buffer for batch");

    vkCmdResetQueryPool(command_buffer, timestamp_query_pool, 0, MAX_TIMESTAMP_QUERY_COUNT);
    PerfTimer::popMarkers(this);
}

void VulkanGSPipeline::endCommandBatch(bool use_fence, bool ignore_sync_env) {
    if (!commandBatchInProgress)
        _THROW_ERROR("No command batch in progress");

    if (timestampNumWritten > 0) {
        while (timestampStackDepth > 0)
            PerfTimer::pushMarker(this);
    }

    auto& slot = compute_slots_[current_slot_index_];
    if (tile_work_readback_src_.buffer != VK_NULL_HANDLE &&
        slot.wc_stager.buffer != VK_NULL_HANDLE) {
        bufferMemoryBarrier({
            {tile_work_readback_src_, COMPUTE_SHADER_WRITE},
        }, TRANSFER_READ);
        VkBufferCopy copy{};
        copy.size = ftgspp::kTileWorkControlBytes;
        vkCmdCopyBuffer(
            command_buffer,
            tile_work_readback_src_.buffer,
            slot.wc_stager.buffer,
            1, &copy);
        _VulkanBuffer dst;
        dst.buffer = slot.wc_stager.buffer;
        dst.memory = slot.wc_stager.memory;
        dst.allocSize = slot.wc_stager.allocSize;
        dst.size = ftgspp::kTileWorkControlBytes;
        bufferMemoryBarrier({{dst, TRANSFER_WRITE}}, HOST_READ);
        slot.wc_copy_recorded = true;
        tile_work_readback_src_ = _VulkanBuffer();
    }

    if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS)
        _THROW_ERROR("Failed to end command buffer for batch");

    slot.markers = PerfTimer::takeMarks();
    slot.timestamp_count = timestampNumWritten;
    timestampNumWritten = 0;
    slot.step = pending_submission_step_;
    slot.is_train = pending_submission_is_train_;

    const auto submit_begin = std::chrono::steady_clock::now();
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    VkTimelineSemaphoreSubmitInfo timeline_info{};
    uint64_t wait_value = 0;
    uint64_t signal_value = 0;
    VkSemaphore wait_sem = compute_timeline_;
    VkSemaphore signal_sem = compute_timeline_;
    if (has_timeline_semaphore_ && compute_timeline_ != VK_NULL_HANDLE) {
        const uint64_t wait_on = std::max(
            last_compute_signal_, pending_upload_wait_value_);
        signal_value = compute_timeline_value_ + 1;
        compute_timeline_value_ = signal_value;
        timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        if (wait_on > 0) {
            wait_value = wait_on;
            timeline_info.waitSemaphoreValueCount = 1;
            timeline_info.pWaitSemaphoreValues = &wait_value;
            submit_info.waitSemaphoreCount = 1;
            submit_info.pWaitSemaphores = &wait_sem;
            submit_info.pWaitDstStageMask = &kAllCommands;
        }
        timeline_info.signalSemaphoreValueCount = 1;
        timeline_info.pSignalSemaphoreValues = &signal_value;
        submit_info.pNext = &timeline_info;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &signal_sem;
        slot.timeline_value = signal_value;
        last_compute_signal_ = signal_value;
        if (pending_reference_ticket_.valid()) {
            auto& ref = reference_upload_ring_[pending_reference_ticket_.slot];
            if (ref.generation == pending_reference_ticket_.generation)
                ref.consumed_value = signal_value;
        }
    } else {
        slot.timeline_value = 0;
    }

    if (vkQueueSubmit(command_queue, 1, &submit_info, slot.fence) != VK_SUCCESS)
        _THROW_ERROR("Failed to submit batch");

    telemetry_.submits += 1;
    telemetry_.cpu_submit_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - submit_begin).count();
    telemetry_.submission_id += 1;
    slot.submission_id = telemetry_.submission_id;
    slot.in_flight = true;
    in_flight_count_ += 1;
    if (in_flight_count_ > telemetry_.in_flight_high_water)
        telemetry_.in_flight_high_water = in_flight_count_;

    commandBatchInProgress = false;
    pending_upload_wait_value_ = 0;

    const bool wait_host = !ignore_sync_env && (use_fence || !async_submit_);
    if (wait_host) {
        waitComputeSlot(current_slot_index_, "endCommandBatch");
        PerfTimer::hostTic();
    } else {
        PerfTimer::hostTic();
    }
}

VulkanGSPipeline::ReferenceUploadTicket VulkanGSPipeline::enqueueReferenceRgba(
    const uint8_t* rgba, size_t bytes, uint32_t width, uint32_t height
) {
    if (rgba == nullptr || bytes == 0)
        _THROW_ERROR("enqueueReferenceRgba requires a non-empty RGBA buffer");
    const uint32_t idx = reference_upload_cursor_ % kReferenceRingDepth;
    reference_upload_cursor_ += 1;
    auto& slot = reference_upload_ring_[idx];
    if (slot.consumed_value > 0 && has_timeline_semaphore_) {
        VkSemaphoreWaitInfo wait{};
        wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wait.semaphoreCount = 1;
        wait.pSemaphores = &compute_timeline_;
        wait.pValues = &slot.consumed_value;
        vkWaitSemaphores(device, &wait, UINT64_MAX);
        slot.consumed_value = 0;
    }
    if (slot.in_flight && slot.fallback_fence != VK_NULL_HANDLE) {
        vkWaitForFences(device, 1, &slot.fallback_fence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &slot.fallback_fence);
        slot.in_flight = false;
    }

    const auto h2d_begin = std::chrono::steady_clock::now();
    allocStagingBufferFor(slot.stager, bytes, StagingPurpose::Upload);
    void* mapped = ensureStagerMapped(slot.stager);
    memcpy(mapped, rgba, bytes);
    flushStager(slot.stager, bytes);

    if (slot.device_reference.allocSize < bytes) {
        if (commandBatchInProgress) {
            HOST_GUARD;
            telemetry_.host_guard_image_upload += 1;
        }
        if (slot.device_reference.buffer != VK_NULL_HANDLE)
            destroyBuffer(slot.device_reference);
        size_t alloc = bytes;
        if (slot.device_reference.allocSize > 0) {
            size_t grown = slot.device_reference.allocSize +
                slot.device_reference.allocSize / 2;
            if (grown > alloc)
                alloc = grown;
        }
        createBuffer(alloc, slot.device_reference);
        slot.device_reference.size = bytes;
    } else {
        slot.device_reference.size = bytes;
    }

    if (vkResetCommandBuffer(slot.transfer_cmd, 0) != VK_SUCCESS)
        _THROW_ERROR("Failed to reset reference-upload command buffer");
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(slot.transfer_cmd, &begin_info) != VK_SUCCESS)
        _THROW_ERROR("Failed to begin reference-upload command buffer");
    VkBufferCopy copy{};
    copy.size = bytes;
    vkCmdCopyBuffer(
        slot.transfer_cmd, slot.stager.buffer, slot.device_reference.buffer, 1, &copy);
    bufferMemoryBarrier(
        slot.transfer_cmd,
        {{slot.device_reference, TRANSFER_WRITE}},
        COMPUTE_SHADER_READ);
    if (vkEndCommandBuffer(slot.transfer_cmd) != VK_SUCCESS)
        _THROW_ERROR("Failed to end reference-upload command buffer");

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &slot.transfer_cmd;
    VkTimelineSemaphoreSubmitInfo timeline_info{};
    uint64_t signal_value = 0;
    if (has_timeline_semaphore_ && compute_timeline_ != VK_NULL_HANDLE) {
        signal_value = compute_timeline_value_ + 1;
        compute_timeline_value_ = signal_value;
        timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_info.signalSemaphoreValueCount = 1;
        timeline_info.pSignalSemaphoreValues = &signal_value;
        submit.pNext = &timeline_info;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &compute_timeline_;
        slot.ready_value = signal_value;
        pending_upload_wait_value_ = signal_value;
    } else {
        slot.ready_value = 0;
        pending_upload_wait_value_ = 0;
    }
    if (vkQueueSubmit(command_queue, 1, &submit, slot.fallback_fence) != VK_SUCCESS)
        _THROW_ERROR("Failed to submit reference upload");
    slot.in_flight = true;
    slot.width = width;
    slot.height = height;
    slot.generation = ++reference_upload_generation_;
    telemetry_.uploaded_bytes += bytes;
    telemetry_.h2d_bytes += bytes;
    telemetry_.cpu_h2d_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - h2d_begin).count();
    telemetry_.submits += 1;

    ReferenceUploadTicket ticket;
    ticket.slot = idx;
    ticket.generation = slot.generation;
    pending_reference_ticket_ = ticket;
    pending_reference_slot_ = (int32_t)idx;
    return ticket;
}

void VulkanGSPipeline::ensureDeviceBufferBytes(_VulkanBuffer& dest, size_t bytes) {
    if (bytes == 0)
        _THROW_ERROR("ensureDeviceBufferBytes requires a positive size");
    if (dest.allocSize >= bytes) {
        dest.size = bytes;
        return;
    }
    if (commandBatchInProgress)
        _THROW_ERROR("cannot realloc a device image while a command batch is open");
    if (dest.buffer != VK_NULL_HANDLE) {
        destroyBuffer(dest);
        dest = _VulkanBuffer();
    }
    createBuffer(bytes, dest);
    dest.size = bytes;
}

VulkanGSPipeline::ReferenceUploadTicket VulkanGSPipeline::enqueueRgbaToDeviceBuffer(
    _VulkanBuffer& dest,
    const uint8_t* rgba, size_t bytes, uint32_t width, uint32_t height
) {
    if (rgba == nullptr || bytes == 0)
        _THROW_ERROR("enqueueRgbaToDeviceBuffer requires a non-empty RGBA buffer");
    const uint32_t idx = reference_upload_cursor_ % kReferenceRingDepth;
    reference_upload_cursor_ += 1;
    auto& slot = reference_upload_ring_[idx];
    // Only wait the previous H2D on this staging slot.  Do not wait for
    // compute to finish consuming a ring device_reference: dest is a
    // persistent per-view buffer, so overwriting the stager is safe after
    // the prior copy fence.
    if (slot.in_flight && slot.fallback_fence != VK_NULL_HANDLE) {
        vkWaitForFences(device, 1, &slot.fallback_fence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &slot.fallback_fence);
        slot.in_flight = false;
    }

    const auto h2d_begin = std::chrono::steady_clock::now();
    allocStagingBufferFor(slot.stager, bytes, StagingPurpose::Upload);
    void* mapped = ensureStagerMapped(slot.stager);
    memcpy(mapped, rgba, bytes);
    flushStager(slot.stager, bytes);
    ensureDeviceBufferBytes(dest, bytes);

    if (vkResetCommandBuffer(slot.transfer_cmd, 0) != VK_SUCCESS)
        _THROW_ERROR("Failed to reset reference-upload command buffer");
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(slot.transfer_cmd, &begin_info) != VK_SUCCESS)
        _THROW_ERROR("Failed to begin reference-upload command buffer");
    VkBufferCopy copy{};
    copy.size = bytes;
    vkCmdCopyBuffer(
        slot.transfer_cmd, slot.stager.buffer, dest.buffer, 1, &copy);
    bufferMemoryBarrier(
        slot.transfer_cmd,
        {{dest, TRANSFER_WRITE}},
        COMPUTE_SHADER_READ);
    if (vkEndCommandBuffer(slot.transfer_cmd) != VK_SUCCESS)
        _THROW_ERROR("Failed to end reference-upload command buffer");

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &slot.transfer_cmd;
    VkTimelineSemaphoreSubmitInfo timeline_info{};
    uint64_t signal_value = 0;
    if (has_timeline_semaphore_ && compute_timeline_ != VK_NULL_HANDLE) {
        signal_value = compute_timeline_value_ + 1;
        compute_timeline_value_ = signal_value;
        timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_info.signalSemaphoreValueCount = 1;
        timeline_info.pSignalSemaphoreValues = &signal_value;
        submit.pNext = &timeline_info;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &compute_timeline_;
        slot.ready_value = signal_value;
        pending_upload_wait_value_ = signal_value;
    } else {
        slot.ready_value = 0;
        pending_upload_wait_value_ = 0;
    }
    if (vkQueueSubmit(command_queue, 1, &submit, slot.fallback_fence) != VK_SUCCESS)
        _THROW_ERROR("Failed to submit reference image upload");
    slot.in_flight = true;
    slot.width = width;
    slot.height = height;
    slot.generation = ++reference_upload_generation_;
    telemetry_.uploaded_bytes += bytes;
    telemetry_.h2d_bytes += bytes;
    telemetry_.cpu_h2d_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - h2d_begin).count();
    telemetry_.submits += 1;

    ReferenceUploadTicket ticket;
    ticket.slot = idx;
    ticket.generation = slot.generation;
    return ticket;
}

void VulkanGSPipeline::waitReferenceUpload(const ReferenceUploadTicket& ticket) {
    if (!ticket.valid())
        return;
    auto& slot = reference_upload_ring_[ticket.slot];
    // Ring depth is 3; a later enqueue to the same slot waits the fence and
    // bumps generation.  That means this ticket already completed.
    if (slot.generation != ticket.generation)
        return;
    if (has_timeline_semaphore_ && slot.ready_value > 0 &&
        compute_timeline_ != VK_NULL_HANDLE) {
        VkSemaphoreWaitInfo wait{};
        wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wait.semaphoreCount = 1;
        wait.pSemaphores = &compute_timeline_;
        wait.pValues = &slot.ready_value;
        if (vkWaitSemaphores(device, &wait, UINT64_MAX) != VK_SUCCESS)
            _THROW_ERROR("Failed while waiting for reference image upload");
    }
    if (slot.in_flight && slot.fallback_fence != VK_NULL_HANDLE) {
        if (vkWaitForFences(device, 1, &slot.fallback_fence, VK_TRUE, UINT64_MAX)
            != VK_SUCCESS)
            _THROW_ERROR("Failed while waiting for reference image-upload fence");
        vkResetFences(device, 1, &slot.fallback_fence);
        slot.in_flight = false;
    }
}

void VulkanGSPipeline::waitAllReferenceUploads() {
    for (auto& slot : reference_upload_ring_) {
        if (!slot.in_flight)
            continue;
        if (has_timeline_semaphore_ && slot.ready_value > 0 &&
            compute_timeline_ != VK_NULL_HANDLE) {
            VkSemaphoreWaitInfo wait{};
            wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
            wait.semaphoreCount = 1;
            wait.pSemaphores = &compute_timeline_;
            wait.pValues = &slot.ready_value;
            if (vkWaitSemaphores(device, &wait, UINT64_MAX) != VK_SUCCESS)
            _THROW_ERROR("Failed while waiting for reference image uploads");
        }
        if (slot.fallback_fence != VK_NULL_HANDLE) {
            if (vkWaitForFences(device, 1, &slot.fallback_fence, VK_TRUE, UINT64_MAX)
                != VK_SUCCESS)
            _THROW_ERROR("Failed while waiting for reference image-upload fence");
            vkResetFences(device, 1, &slot.fallback_fence);
        }
        slot.in_flight = false;
    }
}

_VulkanBuffer VulkanGSPipeline::referenceBufferForTicket(
    const ReferenceUploadTicket& ticket
) const {
    if (!ticket.valid())
        return _VulkanBuffer();
    const auto& slot = reference_upload_ring_[ticket.slot];
    if (slot.generation != ticket.generation)
        return _VulkanBuffer();
    return slot.device_reference;
}

void VulkanGSPipeline::bindPendingReferenceTicket(const ReferenceUploadTicket& ticket) {
    pending_reference_ticket_ = ticket;
    pending_reference_slot_ = ticket.valid() ? (int32_t)ticket.slot : -1;
}

void VulkanGSPipeline::writeDescriptorSet(
    VkDescriptorSet set,
    const std::vector<int>& layouts,
    const std::vector<_VulkanBuffer>& data_buffers
) {
    const size_t num_buffers = layouts.size();
    if (num_buffers == 0)
        return;
    std::vector<VkWriteDescriptorSet> descriptor_writes(num_buffers);
    std::vector<VkDescriptorBufferInfo> buffer_infos(num_buffers);
    int idx = 0;
    for (int i : layouts) {
        if (data_buffers[i].buffer == VK_NULL_HANDLE)
            _THROW_ERROR("Buffer " + std::to_string(i) + " is NULL");
        buffer_infos[idx].buffer = data_buffers[i].buffer;
        buffer_infos[idx].offset = 0;
        buffer_infos[idx].range = data_buffers[i].allocSize;
        descriptor_writes[idx].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[idx].dstSet = set;
        descriptor_writes[idx].dstBinding = i;
        descriptor_writes[idx].dstArrayElement = 0;
        descriptor_writes[idx].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptor_writes[idx].descriptorCount = 1;
        descriptor_writes[idx].pBufferInfo = &buffer_infos[idx];
        idx++;
    }
    vkUpdateDescriptorSets(
        device, static_cast<uint32_t>(descriptor_writes.size()),
        descriptor_writes.data(), 0, nullptr);
}

void VulkanGSPipeline::harvestNextComputeSlot() {
    // Training steps mutate shared geometry/adam state.  Keep at most one
    // compute submission in flight by waiting the previous slot.  Overflow
    // is still read from that slot's mapped TileWorkControl copy.
    if (compute_ring_cursor_ == 0)
        return;
    const uint32_t last = (compute_ring_cursor_ - 1) % kComputeRingDepth;
    waitComputeSlot(last, "prev_train");
}

VulkanGSPipeline::RetiredTileOverflow VulkanGSPipeline::collectRetiredTileOverflow() const {
    return retired_tile_overflow_;
}

VulkanGSPipeline::RetiredTileOverflow
VulkanGSPipeline::collectLastHarvestedOverflow() const {
    RetiredTileOverflow out;
    if (compute_ring_cursor_ == 0)
        return out;
    const uint32_t last = (compute_ring_cursor_ - 1) % kComputeRingDepth;
    const auto& slot = compute_slots_[last];
    if (!slot.wc_valid)
        return out;
    const bool aborted =
        slot.wc.abort_code == ftgspp::kTileAbortOverflow ||
        slot.wc.abort_code == ftgspp::kTileAbortLatched ||
        slot.wc.abort_code == ftgspp::kTileAbortPrefix ||
        slot.wc.abort_code == ftgspp::kTileAbortInvariant;
    if (slot.wc.overflow == 0 && !aborted)
        return out;
    if (slot.wc.abort_code == ftgspp::kTileAbortPrefix ||
        slot.wc.abort_code == ftgspp::kTileAbortInvariant)
        return out;
    out.overflow = true;
    out.exact_count = slot.wc.exact_count;
    out.abort_code = slot.wc.abort_code;
    out.is_train = slot.is_train;
    out.step = slot.step;
    return out;
}

void VulkanGSPipeline::clearRetiredTileOverflow() {
    if (!retired_tile_overflow_.fatal)
        retired_tile_overflow_ = RetiredTileOverflow{};
    for (auto& slot : compute_slots_) {
        slot.wc.overflow = 0;
        if (slot.wc.abort_code == ftgspp::kTileAbortOverflow ||
            slot.wc.abort_code == ftgspp::kTileAbortLatched)
            slot.wc.abort_code = ftgspp::kTileAbortNone;
    }
}

void VulkanGSPipeline::setSubmissionMeta(int64_t step, bool is_train) {
    pending_submission_step_ = step;
    pending_submission_is_train_ = is_train;
}

void VulkanGSPipeline::recordTileWorkReadback(const _VulkanBuffer& src) {
    tile_work_readback_src_ = src;
}

uint32_t VulkanGSPipeline::lastForwardTileCount() const {
    if(!compute_ring_cursor_)return 0;
    const auto& slot=compute_slots_[(compute_ring_cursor_-1)%kComputeRingDepth];
    return slot.wc_valid?slot.wc.exact_count:0;
}
