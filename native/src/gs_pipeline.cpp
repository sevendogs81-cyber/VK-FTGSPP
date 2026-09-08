#include "gs_pipeline.h"
#include "perf_timer.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstdlib>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

static const size_t MAX_UNIFORM_SIZE = 192;

#if defined(__SSE2__) || defined(_MSC_VER)
#define SSE2_AVAILABLE 1
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif


std::vector<uint32_t> loadSpirv(std::string spirv_path) {
    // Load the SPIR-V file
    #ifdef WIN32
    // replace "/" with "\\"
    size_t start_pos = 0;
    while((start_pos = spirv_path.find("/", start_pos)) != std::string::npos) {
        spirv_path.replace(start_pos, 1, "\\");
        start_pos += 1;
    }
    #endif

    std::ifstream file(spirv_path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Failed to open file: " + spirv_path);

    std::streamsize fileSize = file.tellg();
    if (fileSize <= 0 || fileSize % (std::streamsize)sizeof(uint32_t) != 0)
        throw std::runtime_error("Invalid SPIR-V byte length: " + spirv_path);
    file.seekg(0, std::ios::beg);

    std::vector<uint32_t> spirv_code(fileSize / sizeof(uint32_t));
    if (!file.read(reinterpret_cast<char*>(spirv_code.data()), fileSize))
        throw std::runtime_error("Failed to read file: " + spirv_path);
    
    return spirv_code;
}


VulkanGSPipeline::VulkanGSPipeline() :
    instance(VK_NULL_HANDLE),
    physical_device(VK_NULL_HANDLE),
    device(VK_NULL_HANDLE),
    command_queue(VK_NULL_HANDLE),
    command_pool(VK_NULL_HANDLE),
    command_buffer(VK_NULL_HANDLE),
    fence(VK_NULL_HANDLE),
    timestamp_query_pool(VK_NULL_HANDLE),
    queue_family_index(UINT32_MAX) {
}

VulkanGSPipeline::~VulkanGSPipeline() {
    try {
        cleanup();
    } catch (const std::exception& error) {
        fprintf(stderr, "Vulkan cleanup failed in destructor: %s\n", error.what());
    } catch (...) {
        fprintf(stderr, "Vulkan cleanup failed in destructor\n");
    }
}

void VulkanGSPipeline::initialize(int device_id) {

    if (device != VK_NULL_HANDLE || instance != VK_NULL_HANDLE)
        _THROW_ERROR("VulkanGSPipeline is already initialized");

#if ENABLE_VULKAN_VALIDATION_LAYER
    do {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        enableValidationLayer = false;
        for (const auto& layerProperties : availableLayers) {
            if (strcmp("VK_LAYER_KHRONOS_validation", layerProperties.layerName) == 0)
                { enableValidationLayer = true; break; }
        }

        if (!enableValidationLayer)
            fprintf(stderr, "WARNING: Vulkan validation layer not available");
    } while(0);
#endif

    createInstance();
    selectPhysicalDevice(device_id);
    createDevice();
    createCommandPool();
    createFence();
    createPackedTransferRing();
    createQueryPools();
    createTimelineSemaphore();
    createTransferResources();

    const char* sync_submit = std::getenv("FTGSPP_VK_SYNC_SUBMIT");
    async_submit_ = !(sync_submit != nullptr && sync_submit[0] == '1');
    const char* stage_timers = std::getenv("FTGSPP_VK_STAGE_TIMERS");
    stage_timestamps_all_ = stage_timers != nullptr && stage_timers[0] == '1';

    commandBatchInProgress = false;
}


void VulkanGSPipeline::cleanupBuffers(VulkanGSPipelineBuffers& buffers) {
    HOST_GUARD;
    #define _(name) { \
        destroyBuffer(buffers.name.deviceBuffer); \
        buffers.name.clear(); \
        buffers.name.shrink_to_fit(); \
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
}

void VulkanGSPipeline::cleanup() {
    // Cleanup must never let HostGuard restart a command buffer after the
    // command pool/device has been destroyed.
    if (commandBatchInProgress)
        endCommandBatch(false);

    if (device != VK_NULL_HANDLE) {
        // The stager and every descriptor-owned buffer may still be referenced
        // by work submitted outside the batched helper.  Make the device idle
        // before destroying the first device child object.
        vkDeviceWaitIdle(device);

        if (stager.buffer != VK_NULL_HANDLE || stager.memory != VK_NULL_HANDLE ||
            stager.mapped != nullptr)
            destroyStager(stager);
        if (readback_stager.buffer != VK_NULL_HANDLE ||
            readback_stager.memory != VK_NULL_HANDLE ||
            readback_stager.mapped != nullptr)
            destroyStager(readback_stager);
        if (upload_stager.buffer != VK_NULL_HANDLE ||
            upload_stager.memory != VK_NULL_HANDLE ||
            upload_stager.mapped != nullptr)
            destroyStager(upload_stager);
        destroyPackedTransferRing();
        destroyAsyncResources();
        
        for (_ComputePipeline* pipeline : all_compute_pipelines)
            destroyComputePipeline(*pipeline);
        all_compute_pipelines.clear();
        
        if (fence != VK_NULL_HANDLE) {
            vkDestroyFence(device, fence, nullptr);
            fence = VK_NULL_HANDLE;
        }
        if (timestamp_query_pool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device, timestamp_query_pool, nullptr);
            timestamp_query_pool = VK_NULL_HANDLE;
        }
        
        if (command_buffer != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
            command_buffer = VK_NULL_HANDLE;
        }
        if (command_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, command_pool, nullptr);
            command_pool = VK_NULL_HANDLE;
        }
        
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
        physical_device = VK_NULL_HANDLE;
        command_queue = VK_NULL_HANDLE;
        current_vram = 0;
    }
    else {
        // A partially initialized instance cannot own valid device memory.
        stager.buffer = VK_NULL_HANDLE;
        stager.memory = VK_NULL_HANDLE;
        stager.allocSize = 0;
        stager.reset_mapping_fields();
        readback_stager.buffer = VK_NULL_HANDLE;
        readback_stager.memory = VK_NULL_HANDLE;
        readback_stager.allocSize = 0;
        readback_stager.reset_mapping_fields();
        upload_stager.buffer = VK_NULL_HANDLE;
        upload_stager.memory = VK_NULL_HANDLE;
        upload_stager.allocSize = 0;
        upload_stager.reset_mapping_fields();
        for (auto& slot : packed_readback_ring_) {
            slot.command_buffer = VK_NULL_HANDLE;
            slot.fence = VK_NULL_HANDLE;
            slot.in_flight = false;
            slot.stager.buffer = VK_NULL_HANDLE;
            slot.stager.memory = VK_NULL_HANDLE;
            slot.stager.allocSize = 0;
            slot.stager.reset_mapping_fields();
        }
        for (auto& slot : packed_upload_ring_) {
            slot.command_buffer = VK_NULL_HANDLE;
            slot.fence = VK_NULL_HANDLE;
            slot.in_flight = false;
            slot.stager.buffer = VK_NULL_HANDLE;
            slot.stager.memory = VK_NULL_HANDLE;
            slot.stager.allocSize = 0;
            slot.stager.reset_mapping_fields();
        }
    }
    
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }

}


void VulkanGSPipeline::createInstance() {
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pNext = VK_NULL_HANDLE;
    app_info.pApplicationName = "Vulkan Gaussian Rasterization";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "No Engine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;
    
    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pNext = VK_NULL_HANDLE;
    create_info.pApplicationInfo = &app_info;

    if (enableValidationLayer) {
        static const char* validation_layer_name = "VK_LAYER_KHRONOS_validation";
        create_info.enabledLayerCount = 1u;
        create_info.ppEnabledLayerNames = &validation_layer_name;
    }
    
    if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS)
        _THROW_ERROR("Failed to create instance");
}

void VulkanGSPipeline::selectPhysicalDevice(int device_id) {
    static constexpr int kANSIDefault = 0;
    static constexpr int kANSIRed = 91;
    static constexpr int kANSIGreen = 32;
    static constexpr int kANSIOrange = 93;

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    if (device_count == 0) {
        const char* message = "Could not find any physical device.";
        printf("\033[%dm%s\033[m\n", kANSIRed, message);
        throw std::runtime_error(message);
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

    struct SelectedDevice {
        int idx = -1;
        VkPhysicalDevice device = VK_NULL_HANDLE;
        uint32_t queueFamilyIdx = UINT32_MAX;
        DeviceInfo deviceInfo;
    };
    std::vector<SelectedDevice> viableDevices;
    std::vector<SelectedDevice> softViableDevices;

    const auto& [
        minMaxGroups, minMaxThreads, minSharedMemory
    ] = getDeviceRequirement();

    printf(
        "Device Requirement: subgroup>=%d, maxGroups>=[%u %u %u], maxThreads>=[%u %u %u], maxShared>=%u, I16|I64|F32Atomic \n",
        (int)SUBGROUP_SIZE,
        minMaxGroups[0], minMaxGroups[1], minMaxGroups[2],
        minMaxThreads[0], minMaxThreads[1], minMaxThreads[2],
        minSharedMemory
    );
    fflush(stdout);

    for (size_t i = 0; i < devices.size(); i++) {
        VkPhysicalDevice& device = devices[i];

        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);
        auto& limits = deviceProperties.limits;
        uint32_t maxGroupsX = limits.maxComputeWorkGroupCount[0];
        uint32_t maxGroupsY = limits.maxComputeWorkGroupCount[1];
        uint32_t maxGroupsZ = limits.maxComputeWorkGroupCount[2];
        uint32_t maxThreadsX = limits.maxComputeWorkGroupSize[0];
        uint32_t maxThreadsY = limits.maxComputeWorkGroupSize[1];
        uint32_t maxThreadsZ = limits.maxComputeWorkGroupSize[2];
        bool validGroupCount[4] = {
            maxGroupsX >= minMaxGroups[0],
            maxGroupsY >= minMaxGroups[1],
            maxGroupsZ >= minMaxGroups[2],
        };
        validGroupCount[3] = (validGroupCount[0] && validGroupCount[1] && validGroupCount[2]);
        bool validGroupSize[4] = {
            maxThreadsX >= minMaxThreads[0],
            maxThreadsY >= minMaxThreads[1],
            maxThreadsZ >= minMaxThreads[2],
        };
        validGroupSize[3] = (validGroupSize[0] && validGroupSize[1] && validGroupSize[2]);
        bool validSharedSize = limits.maxComputeSharedMemorySize >= minSharedMemory;
        bool validPushConstants = limits.maxPushConstantsSize >= MAX_UNIFORM_SIZE;

        // check subgroup size support
        VkPhysicalDeviceSubgroupProperties subgroupProperties{};
        subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        subgroupProperties.pNext = VK_NULL_HANDLE;
        VkPhysicalDeviceProperties2KHR deviceProperties2{};
        deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        deviceProperties2.pNext = &subgroupProperties;
        vkGetPhysicalDeviceProperties2(device, &deviceProperties2);
        bool validSubgroupSize = subgroupProperties.subgroupSize >= SUBGROUP_SIZE;

        // check compute pipeline.support
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());
        uint32_t queueFamilyIdx = (uint32_t)(-1);
        for (uint32_t i = 0; i < queue_families.size(); i++) {
            if ((queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                (queue_families[i].timestampValidBits != 0)
            ) queueFamilyIdx = i;
        }
        bool validQueueFamily = ((int32_t)queueFamilyIdx != -1);

        // check feature support
        bool hasInt16 = true, hasInt64 = true, hasFloat32AtomicAdd = true;
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features{};
        atomic_float_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        atomic_float_features.pNext = VK_NULL_HANDLE;
        VkPhysicalDeviceFeatures2 deviceFeatures2 = {};
        deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        deviceFeatures2.pNext = &atomic_float_features;
        vkGetPhysicalDeviceFeatures2(device, &deviceFeatures2);
        if (deviceFeatures2.features.shaderInt16 == VK_FALSE)
            hasInt16 = false;
        if (deviceFeatures2.features.shaderInt64 == VK_FALSE)
            hasInt64 = false;
        if (atomic_float_features.shaderBufferFloat32AtomicAdd == VK_FALSE)
            hasFloat32AtomicAdd = false;

        DeviceVendor vendor = DeviceVendor::Unknown;
        if (deviceProperties.vendorID == 0x10DE)
            vendor = DeviceVendor::NVIDIA;
        else if (deviceProperties.vendorID == 0x1002 || deviceProperties.vendorID == 0x1022)
            vendor = DeviceVendor::AMD;
        else if (deviceProperties.vendorID == 0x8086)
            vendor = DeviceVendor::Intel_R_;
        else if (deviceProperties.vendorID == 0x13B5)
            vendor = DeviceVendor::ARM;
        else if (deviceProperties.vendorID == 0x5143)
            vendor = DeviceVendor::Qualcomm;

        bool softViable = validSubgroupSize && validGroupSize[3] && validQueueFamily &&
            hasInt16 && validPushConstants;
        bool viable = softViable && validGroupCount[3] && validSharedSize && hasInt64 && hasFloat32AtomicAdd;
        SelectedDevice deviceInfo{
            (int)i, device, queueFamilyIdx,
            {
                subgroupProperties.subgroupSize,
                limits.maxComputeSharedMemorySize,
                maxGroupsX, maxGroupsY, maxGroupsZ,
                maxThreadsX, maxThreadsY, maxThreadsZ,
                hasInt16, hasInt64, hasFloat32AtomicAdd,
                vendor, deviceProperties.vendorID,
                deviceProperties.deviceName
            }
        };
        bool is_excluded_device = (device_id >= 0 && device_id != (int)i);
        if (viable && !is_excluded_device)
            viableDevices.push_back(deviceInfo);
        else if (softViable && !is_excluded_device)
            softViableDevices.push_back(deviceInfo);

        printf(
            "[%d] %s - \033[%dm%s\033[m\n "
            " subgroup=\033[%dm%u\033[m, "
            "maxGroups=[\033[%dm%u\033[m \033[%dm%u\033[m \033[%dm%u\033[m], "
            "maxThreads=[\033[%dm%u\033[m \033[%dm%u\033[m \033[%dm%u\033[m], "
            "maxShared=\033[%dm%u\033[m, "
            "maxPush=\033[%dm%u\033[m, "
            "\033[%dmI16\033[m|\033[%dmI64\033[m|\033[%dmF32Atomic\033[m\n",
            (int)i, deviceProperties.deviceName,
            viable ? kANSIGreen : softViable ? kANSIOrange : kANSIRed,
            viable ? "VIABLE" : softViable ? "POSSIBLY VIABLE" : "NOT VIABLE",
            validSubgroupSize ? kANSIDefault : kANSIRed, subgroupProperties.subgroupSize,
            validGroupCount[0] ? kANSIDefault : kANSIOrange, maxGroupsX,
            validGroupCount[1] ? kANSIDefault : kANSIOrange, maxGroupsY,
            validGroupCount[2] ? kANSIDefault : kANSIOrange, maxGroupsZ,
            validGroupSize[0] ? kANSIDefault : kANSIRed, maxThreadsX,
            validGroupSize[1] ? kANSIDefault : kANSIRed, maxThreadsY,
            validGroupSize[2] ? kANSIDefault : kANSIRed, maxThreadsZ,
            validSharedSize ? kANSIDefault : kANSIOrange, limits.maxComputeSharedMemorySize,
            validPushConstants ? kANSIDefault : kANSIRed, limits.maxPushConstantsSize,
            hasInt16 ? kANSIDefault : kANSIRed,
            hasInt64 ? kANSIDefault : kANSIOrange,
            hasFloat32AtomicAdd ? kANSIDefault : kANSIOrange
        );
        if (softViable) {
            if (!hasInt64)
                printf("  \033[%dm%s\033[m\n", kANSIOrange, "WARNING: To use this device, shaders must be compiled with USE_EMULATED_INT64=1.");
            if (!hasFloat32AtomicAdd)
                printf(
                    "  \033[%dm%s\033[m\n", kANSIOrange,
                    "WARNING: USE_EMULATED_F32_ATOMIC=1 only covers inherited "
                    "VkSplat shaders; FTGSPP native training is unavailable "
                    "without shaderBufferFloat32AtomicAdd.");
            if (!validGroupCount[3])
                printf("  \033[%dm%s\033[m\n", kANSIOrange, "WARNING: This device may not work if you want to train a large scene.");
            if (!validSharedSize)
                printf("  \033[%dm%s\033[m\n", kANSIOrange, "WARNING: This device does not have sufficient shared memory, which can involve undefined behavior.");
        }
        fflush(stdout);
    }
    
    SelectedDevice device;
    if (!viableDevices.empty())
        device = viableDevices[0];
    else if (!softViableDevices.empty())
        device = softViableDevices[0];
    else {
        const char* message = "Could not find a viable physical device.";
        printf("\033[%dm%s\033[m\n", kANSIRed, message);
        if (device_id >= 0)
            printf("\033[%dmNote: Device [%d] is requested, but it %s.\033[m\n",
                kANSIOrange, device_id, device_id >= (int)devices.size() ? "does not exist" : "is not viable");
        throw std::runtime_error(message);
    }

    this->physical_device = device.device;
    this->queue_family_index = device.queueFamilyIdx;
    this->deviceInfo = device.deviceInfo;
    printf("Using device [\033[%dm%d\033[m]%s\n",
        viableDevices.empty() ? kANSIOrange : kANSIDefault,
        device.idx,
        viableDevices.empty() ? " (\033[93mPOSSIBLY VIABLE\033[m)" : ""
    );
    if (!deviceInfo.hasFloat32AtomicAdd)
        printf(
            "\033[%dm%s\033[m\n", kANSIOrange,
            "WARNING: Float32AtomicAdd is not available. "
            "USE_EMULATED_F32_ATOMIC=1 can cover inherited VkSplat shaders, "
            "but FTGSPP native training requires shaderBufferFloat32AtomicAdd.");
    if (!deviceInfo.hasInt64)
        printf("\033[%dm%s\033[m\n", kANSIOrange, "WARNING: Int64 is not available. Make sure shaders are compiled with USE_EMULATED_INT64=1.");
    printf("\n");
    fflush(stdout);
}

std::map<std::string, std::variant<uint32_t, std::vector<uint32_t>, bool, std::string>>
VulkanGSPipeline::get_device_info() const {
    std::map<std::string, std::variant<uint32_t, std::vector<uint32_t>, bool, std::string>> result;
    uint32_t extension_count = 0;
    vkEnumerateDeviceExtensionProperties(
        physical_device, nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(
        physical_device, nullptr, &extension_count, extensions.data());
    const auto has_extension = [&](const char* name) {
        return std::any_of(
            extensions.begin(), extensions.end(),
            [&](const VkExtensionProperties& extension) {
                return std::strcmp(extension.extensionName, name) == 0;
            });
    };

    VkPhysicalDeviceIDProperties ids{};
    ids.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 properties2{};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &ids;
#ifdef VK_EXT_pci_bus_info
    VkPhysicalDevicePCIBusInfoPropertiesEXT pci{};
    pci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT;
    const bool has_pci_bus_info =
        has_extension(VK_EXT_PCI_BUS_INFO_EXTENSION_NAME);
    if (has_pci_bus_info)
        ids.pNext = &pci;
#else
    const bool has_pci_bus_info = false;
#endif
    vkGetPhysicalDeviceProperties2(physical_device, &properties2);
    const VkPhysicalDeviceProperties& properties = properties2.properties;

    std::ostringstream uuid;
    uuid << std::hex << std::setfill('0');
    for (uint32_t index = 0; index < VK_UUID_SIZE; ++index)
        uuid << std::setw(2) << static_cast<unsigned>(ids.deviceUUID[index]);
    result["device_uuid"] = uuid.str();
    result["pci_bus_info_available"] = has_pci_bus_info;
#ifdef VK_EXT_pci_bus_info
    if (has_pci_bus_info) {
        result["pci_domain"] = pci.pciDomain;
        result["pci_bus"] = pci.pciBus;
        result["pci_device"] = pci.pciDevice;
        result["pci_function"] = pci.pciFunction;
        std::ostringstream bdf;
        bdf << std::hex << std::setfill('0')
            << std::setw(4) << pci.pciDomain << ':'
            << std::setw(2) << pci.pciBus << ':'
            << std::setw(2) << pci.pciDevice << '.'
            << pci.pciFunction;
        result["pci_bdf"] = bdf.str();
    } else {
        result["pci_bdf"] = std::string("");
    }
#else
    result["pci_bdf"] = std::string("");
#endif
    result["subgroup_size"] = deviceInfo.subgroupSize;
    result["shared_size"] = deviceInfo.sharedSize;
    result["max_groups"] = std::vector<uint32_t>{deviceInfo.maxGroupsX, deviceInfo.maxGroupsY, deviceInfo.maxGroupsZ};
    result["max_threads"] = std::vector<uint32_t>{deviceInfo.maxThreadsX, deviceInfo.maxThreadsY, deviceInfo.maxThreadsZ};
    result["has_int16"] = deviceInfo.hasInt16;
    result["has_int64"] = deviceInfo.hasInt64;
    result["has_float32_atomic_add"] = deviceInfo.hasFloat32AtomicAdd;
    result["vendor"] = properties.vendorID;
    result["vendor_id"] = properties.vendorID;
    result["device_id_pci"] = properties.deviceID;
    result["driver_version"] = properties.driverVersion;
    result["api_version_raw"] = properties.apiVersion;
    result["api_version"] = std::to_string(VK_VERSION_MAJOR(properties.apiVersion))
        + "." + std::to_string(VK_VERSION_MINOR(properties.apiVersion))
        + "." + std::to_string(VK_VERSION_PATCH(properties.apiVersion));
    result["device_type"] = static_cast<uint32_t>(properties.deviceType);
    result["name"] = std::string(properties.deviceName);
    return result;
}


void VulkanGSPipeline::createDevice() {
    uint32_t extension_count = 0;
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> extension_properties(extension_count);
    vkEnumerateDeviceExtensionProperties(
        physical_device, nullptr, &extension_count, extension_properties.data());
    auto has_extension = [&](const char* name) {
        return std::any_of(
            extension_properties.begin(), extension_properties.end(),
            [&](const VkExtensionProperties& item) {
                return strcmp(item.extensionName, name) == 0;
            });
    };
    hasSubgroupSizeControl = has_extension(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    hasAtomicFloatExtension = has_extension(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    if (hasSubgroupSizeControl) {
        VkPhysicalDeviceSubgroupSizeControlFeaturesEXT supported{};
        supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
        VkPhysicalDeviceFeatures2 supported_features{};
        supported_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        supported_features.pNext = &supported;
        vkGetPhysicalDeviceFeatures2(physical_device, &supported_features);
        hasSubgroupSizeControl = supported.subgroupSizeControl == VK_TRUE &&
            supported.computeFullSubgroups == VK_TRUE;
    }
    VkPhysicalDeviceVulkan12Features vulkan12_supported = {};
    vulkan12_supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    VkPhysicalDeviceFeatures2 supported_v12{};
    supported_v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    supported_v12.pNext = &vulkan12_supported;
    vkGetPhysicalDeviceFeatures2(physical_device, &supported_v12);

    VkPhysicalDeviceVulkan12Features vulkan12_features = {};
    vulkan12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12_features.timelineSemaphore =
        vulkan12_supported.timelineSemaphore;
    vulkan12_features.pNext = VK_NULL_HANDLE;

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info = {};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.pNext = VK_NULL_HANDLE;
    queue_create_info.queueFamilyIndex = queue_family_index;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;
    
    VkPhysicalDeviceFeatures enabledFeatures = {};
    enabledFeatures.shaderInt16 = deviceInfo.hasInt16 ? VK_TRUE : VK_FALSE;
    if (deviceInfo.hasInt64)
        enabledFeatures.shaderInt64 = VK_TRUE;

    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features = {};
    atomic_float_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    atomic_float_features.shaderBufferFloat32AtomicAdd = VK_TRUE;
    atomic_float_features.pNext = VK_NULL_HANDLE;

    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroupSizeControlFeatures = {};
    subgroupSizeControlFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
    subgroupSizeControlFeatures.subgroupSizeControl = VK_TRUE;
    subgroupSizeControlFeatures.computeFullSubgroups = VK_TRUE;
    subgroupSizeControlFeatures.pNext = VK_NULL_HANDLE;
    if (deviceInfo.hasFloat32AtomicAdd && hasAtomicFloatExtension)
        subgroupSizeControlFeatures.pNext = &atomic_float_features;

    void* feature_chain = nullptr;
    if (hasSubgroupSizeControl)
        feature_chain = &subgroupSizeControlFeatures;
    else if (deviceInfo.hasFloat32AtomicAdd && hasAtomicFloatExtension)
        feature_chain = &atomic_float_features;
    vulkan12_features.pNext = feature_chain;

    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.pQueueCreateInfos = &queue_create_info;
    create_info.queueCreateInfoCount = 1;
    create_info.pEnabledFeatures = &enabledFeatures;

    std::vector<const char*> device_extensions;
    if (hasSubgroupSizeControl)
        device_extensions.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    if (hasAtomicFloatExtension)
        device_extensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    create_info.enabledExtensionCount = (uint32_t)device_extensions.size();
    create_info.ppEnabledExtensionNames = device_extensions.data();
    create_info.pNext = &vulkan12_features;

    if (vkCreateDevice(physical_device, &create_info, nullptr, &device) != VK_SUCCESS) {
        _THROW_ERROR("Failed to create device");
    }
    
    vkGetDeviceQueue(device, queue_family_index, 0, &command_queue);
    transfer_queue_ = command_queue;
    transfer_queue_family_ = queue_family_index;
    transfer_shares_compute_family_ = true;
}

void VulkanGSPipeline::createCommandPool() {
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = queue_family_index;
    
    if (vkCreateCommandPool(device, &pool_info, nullptr, &command_pool) != VK_SUCCESS)
        _THROW_ERROR("Failed to create command pool");

    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = command_pool;
    alloc_info.commandBufferCount = kComputeRingDepth;

    VkCommandBuffer command_buffers[kComputeRingDepth];
    if (vkAllocateCommandBuffers(device, &alloc_info, command_buffers) != VK_SUCCESS)
        _THROW_ERROR("Failed to allocate command buffer");
    for (uint32_t i = 0; i < kComputeRingDepth; ++i)
        compute_slots_[i].command_buffer = command_buffers[i];
    command_buffer = compute_slots_[0].command_buffer;
}

void VulkanGSPipeline::createFence() {
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = 0;

    for (uint32_t i = 0; i < kComputeRingDepth; ++i) {
        VkResult result = vkCreateFence(
            device, &fenceInfo, nullptr, &compute_slots_[i].fence);
        if (result != VK_SUCCESS)
            _THROW_ERROR("Failed to create fence");
    }
    fence = compute_slots_[0].fence;
}

void VulkanGSPipeline::createPackedTransferRing() {
    auto create_slot = [&](auto& slot) {
        VkCommandBufferAllocateInfo command_info{};
        command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        command_info.commandPool = command_pool;
        command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_info.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(
                device, &command_info, &slot.command_buffer) != VK_SUCCESS)
            _THROW_ERROR("Failed to allocate packed-transfer command buffer");

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(device, &fence_info, nullptr, &slot.fence) != VK_SUCCESS)
            _THROW_ERROR("Failed to create packed-transfer fence");
    };
    try {
        for (auto& slot : packed_readback_ring_)
            create_slot(slot);
        for (auto& slot : packed_upload_ring_)
            create_slot(slot);
    } catch (...) {
        destroyPackedTransferRing();
        throw;
    }
}

void VulkanGSPipeline::destroyPackedTransferRing() noexcept {
    if (device == VK_NULL_HANDLE)
        return;
    auto destroy_slot = [&](auto& slot) {
        if (slot.stager.buffer != VK_NULL_HANDLE ||
            slot.stager.memory != VK_NULL_HANDLE || slot.stager.mapped != nullptr) {
            try {
                destroyStager(slot.stager);
            } catch (...) {
                // cleanup() has already made the device idle; continue releasing
                // the remaining device children even if a driver reports a
                // malformed staging allocation.
            }
        }
        if (slot.fence != VK_NULL_HANDLE) {
            vkDestroyFence(device, slot.fence, nullptr);
            slot.fence = VK_NULL_HANDLE;
        }
        if (slot.command_buffer != VK_NULL_HANDLE && command_pool != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device, command_pool, 1, &slot.command_buffer);
            slot.command_buffer = VK_NULL_HANDLE;
        }
        slot.in_flight = false;
        slot.generation = 0;
        slot.completed_generation = 0;
    };
    for (auto& slot : packed_readback_ring_)
        destroy_slot(slot);
    for (auto& slot : packed_upload_ring_)
        destroy_slot(slot);
}

void VulkanGSPipeline::createQueryPools() {
    VkQueryPoolCreateInfo queryPoolCreateInfo = {};
    queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolCreateInfo.queryCount = MAX_TIMESTAMP_QUERY_COUNT;
    for (uint32_t i = 0; i < kComputeRingDepth; ++i) {
        if (vkCreateQueryPool(
                device, &queryPoolCreateInfo, nullptr,
                &compute_slots_[i].query_pool) != VK_SUCCESS)
            _THROW_ERROR("Failed to create timestamp query pool");
    }
    timestamp_query_pool = compute_slots_[0].query_pool;
}

void VulkanGSPipeline::createShaderModule(const std::vector<uint32_t>& spirv_code, VkShaderModule *pShaderModule) {
    VkShaderModuleCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = spirv_code.size() * sizeof(uint32_t);
    create_info.pCode = spirv_code.data();
    
    if (vkCreateShaderModule(device, &create_info, nullptr, pShaderModule) != VK_SUCCESS)
        throw std::runtime_error("Failed to create shader module");
}


bool VulkanGSPipeline::writeTimestamp(int delta) {
    if (!commandBatchInProgress)
        _THROW_ERROR("writeTimestamp requires command batch in progress");
    if (timestampNumWritten >= MAX_TIMESTAMP_QUERY_COUNT)
        _THROW_ERROR("Too many timestamps written");
    if (delta != 1 && delta != -1)
        _THROW_ERROR("delta in writeTimestamp must be 1 or -1");
    if (delta == -1 && timestampStackDepth == 0)
        _THROW_ERROR("attempt to write exit timestamp while stack is empty");
    vkCmdWriteTimestamp(
        command_buffer,
        // delta == 1 ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        timestamp_query_pool, timestampNumWritten
    );
    timestampNumWritten += 1;
    timestampStackDepth += delta;
    return true;
}

bool VulkanGSPipeline::writeTimestampNoExcept(int delta) {
    if (!commandBatchInProgress)
        return false;
    if (timestampNumWritten >= MAX_TIMESTAMP_QUERY_COUNT)
        return false;
    if (delta != 1 && delta != -1)
        return false;
    if (delta == -1 && timestampStackDepth == 0)
        return false;
    vkCmdWriteTimestamp(
        command_buffer,
        // delta == 1 ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        timestamp_query_pool, timestampNumWritten
    );
    timestampNumWritten += 1;
    timestampStackDepth += delta;
    return true;
}


VkAccessFlags toAccessMask(VulkanGSPipeline::BarrierMask barrierMask) {
    VkAccessFlags result = (VkAccessFlags)0;
    if (barrierMask == VulkanGSPipeline::TRANSFER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE
    ) result |= VK_ACCESS_TRANSFER_READ_BIT;
    if (barrierMask == VulkanGSPipeline::TRANSFER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE
    ) result |= VK_ACCESS_TRANSFER_WRITE_BIT;
    if (barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE
    ) result |= VK_ACCESS_SHADER_READ_BIT;
    if (barrierMask == VulkanGSPipeline::COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE
    ) result |= VK_ACCESS_SHADER_WRITE_BIT;
    if (barrierMask == VulkanGSPipeline::HOST_READ ||
        barrierMask == VulkanGSPipeline::HOST_READ_WRITE
    ) result |= VK_ACCESS_HOST_READ_BIT;
    if (barrierMask == VulkanGSPipeline::HOST_WRITE ||
        barrierMask == VulkanGSPipeline::HOST_READ_WRITE
    ) result |= VK_ACCESS_HOST_WRITE_BIT;
    if (barrierMask == VulkanGSPipeline::INDIRECT_COMMAND_READ)
        result |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    if (barrierMask == VulkanGSPipeline::COMPUTE_AND_INDIRECT_READ) {
        result |= VK_ACCESS_SHADER_READ_BIT;
        result |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    }
    return result;
}

VkPipelineStageFlags toStageMask(VulkanGSPipeline::BarrierMask barrierMask) {
    VkPipelineStageFlags result = (VkPipelineStageFlags)0;
    if (barrierMask == VulkanGSPipeline::TRANSFER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE
    ) result |= VK_PIPELINE_STAGE_TRANSFER_BIT;
    if (barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::COMPUTE_SHADER_READ_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE ||
        barrierMask == VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_READ_WRITE
    ) result |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (barrierMask == VulkanGSPipeline::HOST_READ ||
        barrierMask == VulkanGSPipeline::HOST_WRITE ||
        barrierMask == VulkanGSPipeline::HOST_READ_WRITE
    ) result |= VK_PIPELINE_STAGE_HOST_BIT;
    if (barrierMask == VulkanGSPipeline::INDIRECT_COMMAND_READ)
        result |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
    if (barrierMask == VulkanGSPipeline::COMPUTE_AND_INDIRECT_READ) {
        result |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        result |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
    }
    return result;
}

void VulkanGSPipeline::memoryBarrier(
    VulkanGSPipeline::BarrierMask srcMask,
    VulkanGSPipeline::BarrierMask dstMask
) {
    if (!commandBatchInProgress)
        return;

    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = toAccessMask(srcMask);
    barrier.dstAccessMask = toAccessMask(dstMask);
    
    vkCmdPipelineBarrier(
        command_buffer,
        toStageMask(srcMask), toStageMask(dstMask),
        0, // dependencyFlags
        1, &barrier, // memory barriers
        0, nullptr, // buffer barriers
        0, nullptr // image barriers
    );
}

void VulkanGSPipeline::bufferMemoryBarrier(
    const std::vector<std::pair<_VulkanBuffer, VulkanGSPipeline::BarrierMask>> &buffers,
    VulkanGSPipeline::BarrierMask dstMask
) {
    if (!commandBatchInProgress)
        return;

    bufferMemoryBarrier(command_buffer, buffers, dstMask);
}

void VulkanGSPipeline::bufferMemoryBarrier(
    VkCommandBuffer target,
    const std::vector<std::pair<_VulkanBuffer, VulkanGSPipeline::BarrierMask>>& buffers,
    VulkanGSPipeline::BarrierMask dstMask
) {
    if (target == VK_NULL_HANDLE)
        return;

    std::vector<VkBufferMemoryBarrier> barriers;
    barriers.reserve(buffers.size());
    VkPipelineStageFlags srcStageFlags = (VkPipelineStageFlags)0;
    for (auto& [buffer, srcMask] : buffers) {
        if (buffer.buffer == VK_NULL_HANDLE)
            continue;
        VkBufferMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.pNext = nullptr;
        barrier.srcAccessMask = toAccessMask(srcMask);
        barrier.dstAccessMask = toAccessMask(dstMask);
        // This is a visibility dependency within one queue family, not an
        // ownership transfer.
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = buffer.buffer;
        barrier.offset = 0;
        barrier.size = buffer.size;
        barriers.push_back(barrier);
        srcStageFlags |= toStageMask(srcMask);
    }
    if (barriers.empty())
        return;
    
    vkCmdPipelineBarrier(
        target,
        srcStageFlags, toStageMask(dstMask),
        0, // dependencyFlags
        0, nullptr, // memory barriers
        (uint32_t)barriers.size(), barriers.data(), // buffer barriers
        0, nullptr // image barriers
    );
}


// Compute pipeline

void VulkanGSPipeline::createComputeDescriptorSetLayout(_ComputePipeline &pipeline) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.reserve(pipeline.buffer_layouts.size());
    
    for (int i : pipeline.buffer_layouts) {
        VkDescriptorSetLayoutBinding binding;
        binding.binding = i;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        binding.pImmutableSamplers = nullptr;
        bindings.push_back(binding);
    }
    
    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();
    
    if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &pipeline.descriptor_set_layout) != VK_SUCCESS)
        _THROW_ERROR("Failed to create descriptor set layout");
}

void VulkanGSPipeline::createComputeDescriptorPool(_ComputePipeline &pipeline) {
    const uint32_t slot_count = _ComputePipeline::kDescriptorSlotCount;
    const uint32_t binding_count = (uint32_t)pipeline.buffer_layouts.size();
    VkDescriptorPoolSize pool_size = {};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = std::max(binding_count, 1u) * slot_count;
    
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    pool_info.maxSets = slot_count;
    
    if (vkCreateDescriptorPool(device, &pool_info, nullptr, &pipeline.descriptor_pool) != VK_SUCCESS)
        _THROW_ERROR("Failed to create descriptor pool");

    std::array<VkDescriptorSetLayout, _ComputePipeline::kDescriptorSlotCount> layouts;
    layouts.fill(pipeline.descriptor_set_layout);
    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = pipeline.descriptor_pool;
    alloc_info.descriptorSetCount = slot_count;
    alloc_info.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(
            device, &alloc_info, pipeline.descriptor_sets.data()) != VK_SUCCESS)
        _THROW_ERROR("Failed to allocate descriptor sets while creating pool");
    pipeline.descriptor_set = pipeline.descriptor_sets[0];
}

void VulkanGSPipeline::updateComputeDescriptorSet(_ComputePipeline &pipeline, const std::vector<_VulkanBuffer> &data_buffers) {
    const uint32_t slot = current_slot_index_;
    VkDescriptorSet set = pipeline.descriptor_sets[slot];
    if (set == VK_NULL_HANDLE)
        _THROW_ERROR("Descriptor set is NULL while updating");
    writeDescriptorSet(set, pipeline.buffer_layouts, data_buffers);
    pipeline.descriptor_set = set;
    pipeline.slot_buffers[slot] = data_buffers;
    telemetry_.descriptor_updates += 1;
}

void VulkanGSPipeline::createComputePipeline(_ComputePipeline &pipeline, const std::string& spirv_path, uint32_t min_shared_memory, bool compatible_subgroup_size) {

    if (pipeline.pipeline != VK_NULL_HANDLE || pipeline.shader != VK_NULL_HANDLE)
        _THROW_ERROR("compute pipeline initialized twice: " + spirv_path);

    if (min_shared_memory > this->deviceInfo.sharedSize) {
        pipeline.shader = VK_NULL_HANDLE;
        return;
    }

    try {
        createShaderModule(loadSpirv(spirv_path), &pipeline.shader);
        createComputeDescriptorSetLayout(pipeline);

    // Create push constant range for uniforms
    VkPushConstantRange push_constant_range = {};
    push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant_range.offset = 0;
    push_constant_range.size = (uint32_t)MAX_UNIFORM_SIZE;
    
    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &pipeline.descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_constant_range;
    
        if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline.pipeline_layout) != VK_SUCCESS)
            _THROW_ERROR("Failed to create pipeline set layout");

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT req = {};
    req.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT;
    req.requiredSubgroupSize = SUBGROUP_SIZE;  // 32

    VkPipelineShaderStageCreateInfo compute_shader_stage_info = {};
    compute_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compute_shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compute_shader_stage_info.module = pipeline.shader;
    compute_shader_stage_info.pName = "main";
    if (hasSubgroupSizeControl && compatible_subgroup_size && (
        deviceInfo.subgroupSize != SUBGROUP_SIZE ||
        deviceInfo.vendor == DeviceVendor::Intel_R_
    ))
        compute_shader_stage_info.pNext = &req;

    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.layout = pipeline.pipeline_layout;
    pipeline_info.stage = compute_shader_stage_info;
    
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline.pipeline) != VK_SUCCESS)
            _THROW_ERROR("Failed to create compute pipeline");

        createComputeDescriptorPool(pipeline);
    } catch (...) {
        destroyComputePipeline(pipeline);
        throw;
    }

    all_compute_pipelines.push_back(&pipeline);
}

void VulkanGSPipeline::bindComputePipeline(
    const void* uniformsPtr, size_t uniformSize,
    _ComputePipeline &pipeline,
    const std::vector<_VulkanBuffer> &buffers
) {
    if (uniformSize > MAX_UNIFORM_SIZE)
        _THROW_ERROR("Maximum uniform size exceeded");
    if (pipeline.pipeline == VK_NULL_HANDLE || pipeline.pipeline_layout == VK_NULL_HANDLE)
        _THROW_ERROR("Attempted to dispatch an unavailable compute pipeline");
    if (!pipeline.buffer_layouts.empty() && buffers.size() <= (size_t)*std::max_element(
            pipeline.buffer_layouts.begin(), pipeline.buffer_layouts.end()))
        _THROW_ERROR("Compute descriptor buffer count does not match pipeline layout");

    uint32_t slot = current_slot_index_;
    if (pipeline.slot_buffers[slot] != buffers) {
        if (pipeline.used_in_batch[slot]) {
            HOST_GUARD;
            telemetry_.host_guard_descriptor_update += 1;
            slot = current_slot_index_;
        }
        if (pipeline.slot_buffers[slot] != buffers)
            updateComputeDescriptorSet(pipeline, buffers);
    }
    pipeline.buffers = buffers;

    DEVICE_GUARD;

    slot = current_slot_index_;
    VkDescriptorSet set = pipeline.descriptor_sets[slot];
    if (set == VK_NULL_HANDLE)
        set = pipeline.descriptor_set;
    pipeline.used_in_batch[slot] = 1;

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline_layout, 0, 1, &set, 0, nullptr);

    if (uniformsPtr) {
        vkCmdPushConstants(
            command_buffer,
            pipeline.pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0, (uint32_t)uniformSize, uniformsPtr
        );
    }
}

void VulkanGSPipeline::executeCompute(
    std::vector<std::pair<size_t, size_t>> dims,
    const void* uniformsPtr, size_t uniformSize,
    _ComputePipeline &pipeline,
    const std::vector<_VulkanBuffer> &buffers
) {
    while (dims.size() < 3)
        dims.push_back({1, 1});
    for (const auto& dim : dims) {
        if (dim.second == 0)
            _THROW_ERROR("Compute local dimension must be non-zero");
        if (dim.first == 0)
            return;
    }

    bindComputePipeline(uniformsPtr, uniformSize, pipeline, buffers);

    uint32_t nGroupsX = (uint32_t)_CEIL_DIV(dims[0].first, dims[0].second);
    uint32_t nGroupsY = (uint32_t)_CEIL_DIV(dims[1].first, dims[1].second);
    uint32_t nGroupsZ = (uint32_t)_CEIL_DIV(dims[2].first, dims[2].second);
    if (nGroupsX > deviceInfo.maxGroupsX ||
        nGroupsY > deviceInfo.maxGroupsY ||
        nGroupsZ > deviceInfo.maxGroupsZ
    ) _THROW_ERROR("Cannot launch compute kernel, too many groups: [" +
            std::to_string(nGroupsX) + " " +
            std::to_string(nGroupsY) + " " +
            std::to_string(nGroupsZ) + "] > [" +
            std::to_string(deviceInfo.maxGroupsX) + " " +
            std::to_string(deviceInfo.maxGroupsY) + " " +
            std::to_string(deviceInfo.maxGroupsZ) + "]"
        );
    vkCmdDispatch(command_buffer, nGroupsX, nGroupsY, nGroupsZ);
}

void VulkanGSPipeline::executeComputeIndirect(
    const _VulkanBuffer& indirect_buffer,
    VkDeviceSize indirect_offset,
    const void* uniformsPtr, size_t uniformSize,
    _ComputePipeline &pipeline,
    const std::vector<_VulkanBuffer> &buffers
) {
    if (indirect_buffer.buffer == VK_NULL_HANDLE)
        _THROW_ERROR("Indirect dispatch requires a valid TileWorkControl buffer");
    if (indirect_offset + 3 * sizeof(uint32_t) > indirect_buffer.allocSize)
        _THROW_ERROR("Indirect dispatch offset is outside the command buffer");

    bindComputePipeline(uniformsPtr, uniformSize, pipeline, buffers);
    vkCmdDispatchIndirect(command_buffer, indirect_buffer.buffer, indirect_offset);
}


void VulkanGSPipeline::destroyComputePipeline(_ComputePipeline &pipeline) {
    if (pipeline.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, pipeline.pipeline, nullptr);
        pipeline.pipeline = VK_NULL_HANDLE;
    }
    if (pipeline.pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, pipeline.pipeline_layout, nullptr);
        pipeline.pipeline_layout = VK_NULL_HANDLE;
    }
    if (pipeline.descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, pipeline.descriptor_pool, nullptr);
        pipeline.descriptor_pool = VK_NULL_HANDLE;
        pipeline.descriptor_set = VK_NULL_HANDLE;
        pipeline.descriptor_sets.fill(VK_NULL_HANDLE);
        for (auto& cached : pipeline.slot_buffers)
            cached.clear();
        pipeline.used_in_batch.fill(0);
    }
    if (pipeline.descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, pipeline.descriptor_set_layout, nullptr);
        pipeline.descriptor_set_layout = VK_NULL_HANDLE;
    }
    if (pipeline.shader != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, pipeline.shader, nullptr);
        pipeline.shader = VK_NULL_HANDLE;
    }
    pipeline.buffers.clear();
}
