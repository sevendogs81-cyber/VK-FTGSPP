#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <map>
#include <algorithm>  // std::sort
#include <cstring>  // memcpy
#include <functional>
#include <array>
#include <variant>
#include <exception>
#include <chrono>


#include <cassert>

#include "device/buffer.h"
#include "device/pipeline_telemetry.h"
#include "render/tile_work_control.h"

static constexpr uint32_t MAX_TIMESTAMP_QUERY_COUNT = 48;

union Uniform32_t {
    uint32_t u;
    float f;
};


struct DeviceInitOptions {
    bool require_graphics = false;
    std::vector<const char*> instance_extensions;
    std::function<VkSurfaceKHR(VkInstance)> create_surface;
};

class VulkanGSPipeline {
public:
    VulkanGSPipeline();
    ~VulkanGSPipeline();

    void initialize(int device_id, const DeviceInitOptions& options = {});
    struct GraphicsContext {
        VkInstance instance; VkPhysicalDevice physical_device; VkDevice device;
        VkQueue queue; uint32_t queue_family; VkSurfaceKHR surface;
    };
    GraphicsContext graphicsContext() const {
        return {instance, physical_device, device, command_queue, queue_family_index, surface_};
    }
    void cleanup();
    void cleanupBuffers(VulkanGSPipelineBuffers& buffers);

    void createBuffer(size_t size, _VulkanBuffer& buffer);
    void destroyBuffer(_VulkanBuffer &buffer);
    void resizeDeviceBuffer(_VulkanBuffer& deviceBuffer, size_t new_byte_size, bool no_shrink=true);
    template<typename T> _VulkanBuffer& resizeDeviceBuffer(Buffer<T>& buffer, size_t new_size, bool no_shrink=true);
    template<typename T> _VulkanBuffer& clearDeviceBuffer(Buffer<T>& buffer, size_t new_size);
    template<typename T> _VulkanBuffer& resizeAndCopyDeviceBuffer(Buffer<T>& buffer, size_t new_size, bool clear);
    template<typename T> _VulkanBuffer& copyToDevice(Buffer<T>& buffer);
    template<typename T> void copyFromDevice(Buffer<T>& buffer);
    // Contiguous multi-buffer transport used by synchronous data parallelism.
    // Every region is expressed in bytes in one reusable host-visible staging
    // allocation.  A whole gradient payload therefore needs one queue submit
    // and one fence instead of one submit/fence per logical tensor.
    struct PackedBufferRegion {
        _VulkanBuffer buffer;
        size_t packedOffset = 0;
    };
    struct PackedTransferTiming;
    void copyBuffersFromDevicePacked(
        const std::vector<PackedBufferRegion>& regions,
        void* destination, size_t packedSize);
    void copyBuffersToDevicePacked(
        const void* source, size_t packedSize,
        const std::vector<PackedBufferRegion>& regions);
    // Non-blocking bucket transfers used by pair-exchange.  A ticket owns
    // one staging/command-buffer/fence slot until it is completed.  The ring
    // permits later D2H work and earlier H2D work to remain queued at the same
    // time instead of waiting one fence after every bucket.
    static constexpr uint32_t packed_transfer_ring_depth = 8;
    struct PackedTransferTicket {
        uint32_t slot = UINT32_MAX;
        uint64_t generation = 0;
        size_t range_size = 0;
        bool readback = false;

        bool valid() const {
            return slot < packed_transfer_ring_depth && generation != 0;
        }
    };
    PackedTransferTicket submitBuffersFromDevicePackedRange(
        const std::vector<PackedBufferRegion>& regions,
        size_t packedOffset, size_t rangeSize);
    PackedTransferTiming completeBuffersFromDevicePackedRange(
        const PackedTransferTicket& ticket, void* destination);
    PackedTransferTicket submitBuffersToDevicePackedRange(
        const void* source, size_t packedOffset, size_t rangeSize,
        const std::vector<PackedBufferRegion>& regions);
    PackedTransferTiming completeBuffersToDevicePackedRange(
        const PackedTransferTicket& ticket);
    // Last packed D2H / H2D timing breakdown (seconds). Reset each call.
    struct PackedTransferTiming {
        double gpu_copy_seconds = 0.0;
        double host_memcpy_seconds = 0.0;
        double map_seconds = 0.0;
        double invalidate_seconds = 0.0;
        bool used_cached_memory = false;
        bool used_persistent_map = false;
    };
    struct PackedMappedReadback {
        const void* data = nullptr;
        PackedTransferTiming timing{};
    };
    PackedMappedReadback completeBuffersFromDevicePackedRangeMapped(
        const PackedTransferTicket& ticket);
    // Same-queue D2H that waits/signals the compute timeline so a later
    // compute submission cannot overwrite ``src`` while the copy is pending.
    PackedTransferTicket submitComputeReadback(
        const _VulkanBuffer& src, size_t bytes);
    PackedTransferTiming lastPackedDownloadTiming() const {
        return last_packed_download_timing_;
    }
    PackedTransferTiming lastPackedUploadTiming() const {
        return last_packed_upload_timing_;
    }
    void copyFromDeviceToDevice(const _VulkanBuffer& src, _VulkanBuffer& dst);
    template<typename T> void copyFromDeviceToDevice(const Buffer<T>& src, Buffer<T>& dst);
    template<typename T> T readElement(const _VulkanBuffer& buffer, size_t index);

    void beginCommandBatch();
    void endCommandBatch(bool use_fence = true, bool ignore_sync_env = false);
    void computeToComputeBarrier() {
        memoryBarrier(COMPUTE_SHADER_READ_WRITE, COMPUTE_SHADER_READ_WRITE);
    }
    void drain(const char* reason);
    bool asyncSubmitEnabled() const { return async_submit_; }
    bool stageTimestampsEnabled() const { return stage_timestamps_all_; }
    PipelineTelemetry snapshotTelemetry() const { return telemetry_; }
    PipelineTelemetry& telemetry() { return telemetry_; }
    bool isCommandBatchInProgress() const {
        return commandBatchInProgress;
    }
    bool writeTimestamp(int delta);
    bool writeTimestampNoExcept(int delta);

    size_t getPeakAllocSize() const
        { return peak_vram; }

    enum BarrierMask {
        TRANSFER_READ,
        TRANSFER_WRITE,
        TRANSFER_READ_WRITE,
        COMPUTE_SHADER_READ,
        COMPUTE_SHADER_WRITE,
        COMPUTE_SHADER_READ_WRITE,
        TRANSFER_COMPUTE_SHADER_READ,
        TRANSFER_COMPUTE_SHADER_WRITE,
        TRANSFER_COMPUTE_SHADER_READ_WRITE,
        HOST_READ,
        HOST_WRITE,
        HOST_READ_WRITE,
        INDIRECT_COMMAND_READ,
        COMPUTE_AND_INDIRECT_READ,
    };

    struct RetiredTileOverflow {
        bool overflow = false;
        bool fatal = false;
        bool is_train = false;
        uint32_t exact_count = 0;
        uint32_t abort_code = 0;
        int64_t step = -1;
    };
    void harvestNextComputeSlot();
    void waitAllComputeSlots(const char* reason);
    void drainPackedTransfers();
    RetiredTileOverflow collectRetiredTileOverflow() const;
    uint32_t lastForwardTileCount() const;
    RetiredTileOverflow collectLastHarvestedOverflow() const;
    void clearRetiredTileOverflow();
    void setSubmissionMeta(int64_t step, bool is_train);

    std::map<std::string, std::variant<uint32_t, std::vector<uint32_t>, bool, std::string>> get_device_info() const;

protected:

    struct DeviceRequirement {
        std::array<uint32_t, 3> minMaxGroups;
        std::array<uint32_t, 3> minMaxThreads;
        uint32_t minSharedMemory;
    };

    virtual DeviceRequirement getDeviceRequirement() = 0;

    bool enableValidationLayer = false;
    DeviceInitOptions init_options_;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;

    bool commandBatchInProgress = false;
    uint32_t timestampNumWritten = 0;
    uint32_t timestampStackDepth = 0;
    std::vector<std::function<void(const std::vector<std::pair<size_t, double>>&)>> timerCallbacks;

    void memoryBarrier(BarrierMask srcMask, BarrierMask dstMask);
    void bufferMemoryBarrier(const std::vector<std::pair<_VulkanBuffer, BarrierMask>> &buffers, BarrierMask dstMask);
    void bufferMemoryBarrier(
        VkCommandBuffer target,
        const std::vector<std::pair<_VulkanBuffer, BarrierMask>>& buffers,
        BarrierMask dstMask);

    size_t current_vram = 0;
    size_t peak_vram = 0;

    // Vulkan objects
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue command_queue;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkFence fence;
    VkQueryPool timestamp_query_pool;

    enum class DeviceVendor {
        Unknown,
        NVIDIA, AMD, Intel_R_, ARM, Qualcomm, 
    };
    struct DeviceInfo {
        uint32_t subgroupSize;
        uint32_t sharedSize;
        uint32_t maxGroupsX;
        uint32_t maxGroupsY;
        uint32_t maxGroupsZ;
        uint32_t maxThreadsX;
        uint32_t maxThreadsY;
        uint32_t maxThreadsZ;
        bool hasInt16;
        bool hasInt64;
        bool hasFloat32AtomicAdd;
        DeviceVendor vendor;
        uint32_t vendorId;
        std::string name;
    } deviceInfo;
    
    // Compute pipeline
    struct _ComputePipeline {
        VkShaderModule shader;
        VkDescriptorPool descriptor_pool;
        VkDescriptorSetLayout descriptor_set_layout;
        VkDescriptorSet descriptor_set;
        VkPipelineLayout pipeline_layout;
        VkPipeline pipeline;
        std::vector<int> buffer_layouts;
        static constexpr uint32_t kDescriptorSlotCount = 3;
        std::array<VkDescriptorSet, kDescriptorSlotCount> descriptor_sets{};
        std::array<std::vector<_VulkanBuffer>, kDescriptorSlotCount> slot_buffers{};
        std::array<uint8_t, kDescriptorSlotCount> used_in_batch{};

        _ComputePipeline(
            std::vector<int> buffer_layouts
        ): shader(VK_NULL_HANDLE), descriptor_pool(VK_NULL_HANDLE),
            descriptor_set_layout(VK_NULL_HANDLE), descriptor_set(VK_NULL_HANDLE),
            pipeline_layout(VK_NULL_HANDLE), pipeline(VK_NULL_HANDLE),
            buffer_layouts(buffer_layouts) {}

        _ComputePipeline(int num_buffers)
        : shader(VK_NULL_HANDLE), descriptor_pool(VK_NULL_HANDLE),
            descriptor_set_layout(VK_NULL_HANDLE), descriptor_set(VK_NULL_HANDLE),
            pipeline_layout(VK_NULL_HANDLE), pipeline(VK_NULL_HANDLE)
        {
            buffer_layouts.resize(num_buffers);
            for (int i = 0; i < num_buffers; i++)
                buffer_layouts[i] = i;
        }

        std::vector<_VulkanBuffer> buffers;
    };

    struct _ComputePipelinePair {
        _ComputePipeline _cp0, _cp1;
        _ComputePipelinePair(int num_buffers)
            : _cp0(num_buffers), _cp1(num_buffers) {}
        _ComputePipeline& operator[](bool b)
            { return b ? _cp0 : _cp1; }
    };

    std::vector<_ComputePipeline*> all_compute_pipelines;

    uint32_t queue_family_index;
    bool hasSubgroupSizeControl = false;
    bool hasAtomicFloatExtension = false;

    // For CPU-GPU transfers. Packed gradient paths prefer HOST_CACHED for
    // both readback and the pair-exchange upload ring: the remaining host
    // copy is borrowed-readback → private upload stager, and write-combine
    // destinations make that memcpy the step's critical path. Both keep a
    // persistent mapping.
    struct _Stager {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        size_t allocSize = 0;
        void* mapped = nullptr;
        VkMemoryPropertyFlags memory_flags = 0;
        bool coherent = false;
        bool cached = false;
        std::mutex mutex;

        void reset_mapping_fields() {
            mapped = nullptr;
            memory_flags = 0;
            coherent = false;
            cached = false;
        }
    };
    _Stager stager;
    _Stager readback_stager;
    _Stager upload_stager;
    struct _PackedTransferSlot {
        _Stager stager;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        uint64_t generation = 0;
        uint64_t completed_generation = 0;
        size_t range_size = 0;
        bool in_flight = false;
        std::chrono::steady_clock::time_point submitted_at{};
        PackedTransferTiming timing{};
    };
    std::array<_PackedTransferSlot, packed_transfer_ring_depth>
        packed_readback_ring_{};
    std::array<_PackedTransferSlot, packed_transfer_ring_depth>
        packed_upload_ring_{};
    uint32_t packed_readback_cursor_ = 0;
    uint32_t packed_upload_cursor_ = 0;
    uint64_t packed_transfer_generation_ = 0;
    PackedTransferTiming last_packed_download_timing_{};
    PackedTransferTiming last_packed_upload_timing_{};
    PipelineTelemetry telemetry_{};
    bool async_submit_ = true;
    bool stage_timestamps_all_ = false;
    bool has_timeline_semaphore_ = false;
    VkSemaphore compute_timeline_ = VK_NULL_HANDLE;
    uint64_t compute_timeline_value_ = 0;
    uint32_t current_descriptor_slot_ = 0;
    uint32_t compute_ring_cursor_ = 0;
    static constexpr uint32_t kComputeRingDepth = 3;
    struct ComputeSubmissionSlot {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkQueryPool query_pool = VK_NULL_HANDLE;
        uint64_t timeline_value = 0;
        uint64_t submission_id = 0;
        int64_t step = -1;
        uint32_t timestamp_count = 0;
        bool in_flight = false;
        bool is_train = false;
        bool wc_valid = false;
        bool wc_copy_recorded = false;
        ftgspp::TileWorkControl wc{};
        _Stager wc_stager;
        std::vector<std::pair<int, int>> markers;
    };
    std::array<ComputeSubmissionSlot, kComputeRingDepth> compute_slots_{};
    uint32_t current_slot_index_ = 0;

    struct ReferenceUploadSlot {
        _Stager stager;
        _VulkanBuffer device_reference;
        VkCommandBuffer transfer_cmd = VK_NULL_HANDLE;
        VkFence fallback_fence = VK_NULL_HANDLE;
        uint64_t ready_value = 0;
        uint64_t consumed_value = 0;
        uint64_t generation = 0;
        bool in_flight = false;
        uint32_t width = 0;
        uint32_t height = 0;
    };
    static constexpr uint32_t kReferenceRingDepth = 3;
    std::array<ReferenceUploadSlot, kReferenceRingDepth> reference_upload_ring_{};
    uint32_t reference_upload_cursor_ = 0;
    uint64_t reference_upload_generation_ = 0;
    int32_t pending_reference_slot_ = -1;
    VkQueue transfer_queue_ = VK_NULL_HANDLE;
    uint32_t transfer_queue_family_ = UINT32_MAX;
    VkCommandPool transfer_command_pool_ = VK_NULL_HANDLE;
    bool transfer_shares_compute_family_ = true;
    uint64_t pending_upload_wait_value_ = 0;
    uint64_t last_compute_signal_ = 0;
    uint32_t in_flight_count_ = 0;
    int64_t pending_submission_step_ = -1;
    bool pending_submission_is_train_ = false;
    _VulkanBuffer tile_work_readback_src_{};

    void waitComputeSlot(uint32_t index, const char* reason);
    void harvestComputeSlot(uint32_t index);
    void harvestWorkControl(ComputeSubmissionSlot& slot);
    void recordTileWorkReadback(const _VulkanBuffer& src);
    void createTimelineSemaphore();
    void createTransferResources();
    void destroyAsyncResources() noexcept;
    void writeDescriptorSet(
        VkDescriptorSet set,
        const std::vector<int>& layouts,
        const std::vector<_VulkanBuffer>& data_buffers);

public:
    struct ReferenceUploadTicket {
        uint32_t slot = UINT32_MAX;
        uint64_t generation = 0;
        bool valid() const {
            return slot < kReferenceRingDepth && generation != 0;
        }
    };
    ReferenceUploadTicket pending_reference_ticket_{};
    ReferenceUploadTicket enqueueReferenceRgba(
        const uint8_t* rgba, size_t bytes, uint32_t width, uint32_t height);
    // Async H2D into a persistent per-view buffer.  Does not drain compute
    // (unlike copyToDevice) and does not occupy the 3-deep SSIM ring.
    ReferenceUploadTicket enqueueRgbaToDeviceBuffer(
        _VulkanBuffer& dest,
        const uint8_t* rgba, size_t bytes, uint32_t width, uint32_t height);
    void waitReferenceUpload(const ReferenceUploadTicket& ticket);
    void waitAllReferenceUploads();
    void ensureDeviceBufferBytes(_VulkanBuffer& dest, size_t bytes);
    _VulkanBuffer referenceBufferForTicket(const ReferenceUploadTicket& ticket) const;
    void bindPendingReferenceTicket(const ReferenceUploadTicket& ticket);

    enum class StagingPurpose {
        General,
        Readback,
        Upload,
        CachedUpload,
    };
    void allocStagingBuffer(size_t size);
    void allocStagingBufferFor(_Stager& target, size_t size, StagingPurpose purpose);
    void destroyStager(_Stager& target);
    void* ensureStagerMapped(_Stager& target);
    void invalidateStager(_Stager& target, size_t size);
    void flushStager(_Stager& target, size_t size);
    
    void createInstance();
    void selectPhysicalDevice(int device_id);
    void createDevice();
    void createCommandPool();
    void createFence();
    void createPackedTransferRing();
    void destroyPackedTransferRing() noexcept;
    PackedTransferTiming waitPackedTransferSlot(
        _PackedTransferSlot& slot, uint64_t generation);
    void attachComputeTimeline(
        VkSubmitInfo& submit,
        VkTimelineSemaphoreSubmitInfo& timeline_info,
        uint64_t& wait_value,
        uint64_t& signal_value);
    void createQueryPools();
    void createShaderModule(const std::vector<uint32_t>& spirv_code, VkShaderModule *pShaderModule);

    void createComputeDescriptorSetLayout(_ComputePipeline &pipeline);
    void createComputeDescriptorPool(_ComputePipeline &pipeline);
    void updateComputeDescriptorSet(_ComputePipeline &pipeline, const std::vector<_VulkanBuffer> &buffers);
    void createComputePipeline(_ComputePipeline &pipeline, const std::string& spirv_path, uint32_t min_shared_memory=0, bool compatible_subgroup_size=true);
    void executeCompute(
        std::vector<std::pair<size_t, size_t>> dims,
        const void* uniformsPtr, size_t uniformSize,
        _ComputePipeline &pipeline,
        const std::vector<_VulkanBuffer> &buffers
    );
    void executeComputeIndirect(
        const _VulkanBuffer& indirect_buffer,
        VkDeviceSize indirect_offset,
        const void* uniformsPtr, size_t uniformSize,
        _ComputePipeline &pipeline,
        const std::vector<_VulkanBuffer> &buffers
    );


private:
    void bindComputePipeline(
        const void* uniformsPtr, size_t uniformSize,
        _ComputePipeline &pipeline,
        const std::vector<_VulkanBuffer> &buffers
    );
    void destroyComputePipeline(_ComputePipeline &pipeline);
};


class [[nodiscard]] DeviceGuard {
    VulkanGSPipeline* pipeline;
    bool cbip;
    const char* debugInfo1 = nullptr;
    int debugInfo2 = -1;
public:
    DeviceGuard(VulkanGSPipeline* pipeline, const char* debugInfo1 = nullptr, const int debugInfo2 = -1) {
        // printf("DeviceGuard constructor\n");
        this->pipeline = pipeline;
        cbip = pipeline->isCommandBatchInProgress();
        if (!cbip) {
            pipeline->beginCommandBatch();
            if (debugInfo1) {
                this->debugInfo1 = debugInfo1;
                this->debugInfo2 = debugInfo2;
                printf("DeviceGuard created: %s:%d\n", debugInfo1, debugInfo2);
            }
        }
    }
    ~DeviceGuard() noexcept(false) {
        // printf("DeviceGuard destructor\n");
        if (!cbip) {
            // Never replace an exception from the guarded operation with a
            // second exception raised while its stack is being unwound.
            if (std::uncaught_exceptions() > 0) {
                try {
                    pipeline->endCommandBatch();
                }
                catch (const std::exception& error) {
                    fprintf(stderr, "DeviceGuard cleanup failed: %s\n", error.what());
                }
                catch (...) {
                    fprintf(stderr, "DeviceGuard cleanup failed with an unknown exception\n");
                }
            }
            else {
                pipeline->endCommandBatch(!pipeline->asyncSubmitEnabled());
            }
            if (debugInfo1) {
                printf("DeviceGuard freed: %s:%d\n", debugInfo1, debugInfo2);
            }
        }
        else if (cbip != pipeline->isCommandBatchInProgress()) {
            fprintf(stderr, "commandBatchInProgress changed during DeviceGuard (originally %d)\n", (int)cbip);
            std::terminate();
        }

    }
};

class [[nodiscard]] HostGuard {
    VulkanGSPipeline* pipeline;
    bool cbip;
    const char* debugInfo1 = nullptr;
    int debugInfo2 = -1;
public:
    HostGuard(VulkanGSPipeline* pipeline, const char* debugInfo1 = nullptr, const int debugInfo2 = -1) {
        // printf("HostGuard constructor\n");
        this->pipeline = pipeline;
        cbip = pipeline->isCommandBatchInProgress();
        if (cbip) {
            pipeline->endCommandBatch(true);
            if (debugInfo1) {
                this->debugInfo1 = debugInfo1;
                this->debugInfo2 = debugInfo2;
                printf("HostGuard created: %s:%d\n", debugInfo1, debugInfo2);
            }
        } else {
            pipeline->drain("HOST_GUARD");
        }
        pipeline->telemetry().host_guards += 1;
    }
    ~HostGuard() noexcept(false) {
        // printf("HostGuard destructor\n");
        if (cbip) {
            if (std::uncaught_exceptions() > 0) {
                try {
                    pipeline->beginCommandBatch();
                }
                catch (const std::exception& error) {
                    fprintf(stderr, "HostGuard cleanup failed: %s\n", error.what());
                }
                catch (...) {
                    fprintf(stderr, "HostGuard cleanup failed with an unknown exception\n");
                }
            }
            else {
                pipeline->beginCommandBatch();
            }
            if (debugInfo1) {
                printf("HostGuard freed: %s:%d\n", debugInfo1, debugInfo2);
            }
        }
        else if (cbip != pipeline->isCommandBatchInProgress()) {
            fprintf(stderr, "commandBatchInProgress changed during HostGuard (originally %d)\n", (int)cbip);
            std::terminate();
        }

    }
};

// #define DeviceGuard(args) DeviceGuard(args, __FILE__, __LINE__)
// #define HostGuard(args) HostGuard(args, __FILE__, __LINE__)

#define DEVICE_GUARD auto deviceGuard = DeviceGuard(this)
#define HOST_GUARD auto hostGuard = HostGuard(this)
