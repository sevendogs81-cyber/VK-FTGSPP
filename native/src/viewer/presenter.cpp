#include "viewer/presenter.h"
#include <array>
#include <backends/imgui_impl_glfw.h>
#include <fstream>
#include <stdexcept>
namespace ftgspp {
static void check(VkResult r) {
  if (r != VK_SUCCESS)
    throw std::runtime_error("Viewer Vulkan error " + std::to_string(r));
}
static VkShaderModule shader(VkDevice d, const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("Missing viewer shader: " + path);
  size_t n = f.tellg();
  if (!n || n % 4)
    throw std::runtime_error("Invalid viewer SPIR-V");
  std::vector<uint32_t> code(n / 4);
  f.seekg(0);
  f.read(reinterpret_cast<char *>(code.data()), n);
  VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  ci.codeSize = n;
  ci.pCode = code.data();
  VkShaderModule m;
  check(vkCreateShaderModule(d, &ci, nullptr, &m));
  return m;
}
Presenter::Presenter(GLFWwindow *win, ViewRenderer &r,
                     const std::string &shaders, bool sync)
    : vsync(sync), renderer(r), ctx(r.graphicsContext()), shader_dir(shaders) {
  try {
    wd.Surface = ctx.surface;
    VkFormat formats[] = {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM};
    wd.SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
        ctx.physical_device, wd.Surface, formats, 2,
        VK_COLOR_SPACE_SRGB_NONLINEAR_KHR);
    if (wd.SurfaceFormat.format != formats[0] &&
        wd.SurfaceFormat.format != formats[1])
      throw std::runtime_error(
          "Viewer requires an RGBA/BGRA UNORM display surface");
    VkPresentModeKHR modes[] = {VK_PRESENT_MODE_MAILBOX_KHR,
                                VK_PRESENT_MODE_FIFO_KHR};
    wd.PresentMode = vsync ? VK_PRESENT_MODE_FIFO_KHR
                           : ImGui_ImplVulkanH_SelectPresentMode(
                                 ctx.physical_device, wd.Surface, modes, 2);
    int w, h;
    glfwGetFramebufferSize(win, &w, &h);
    ImGui_ImplVulkanH_CreateOrResizeWindow(ctx.instance, ctx.physical_device,
                                           ctx.device, &wd, ctx.queue_family,
                                           nullptr, w, h, 2);
    VkDescriptorPoolSize sizes[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}};
    VkDescriptorPoolCreateInfo pi{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pi.maxSets = 18;
    pi.poolSizeCount = 2;
    pi.pPoolSizes = sizes;
    check(vkCreateDescriptorPool(ctx.device, &pi, nullptr, &pool));
    VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                         1, VK_SHADER_STAGE_FRAGMENT_BIT,
                                         nullptr};
    VkDescriptorSetLayoutCreateInfo li{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 1;
    li.pBindings = &binding;
    check(vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &layout));
    VkDescriptorSetAllocateInfo ai{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &layout;
    check(vkAllocateDescriptorSets(ctx.device, &ai, &set));
    VkPushConstantRange push{VK_SHADER_STAGE_FRAGMENT_BIT, 0, 64};
    VkPipelineLayoutCreateInfo pci{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pci.setLayoutCount = 1;
    pci.pSetLayouts = &layout;
    pci.pushConstantRangeCount = 1;
    pci.pPushConstantRanges = &push;
    check(vkCreatePipelineLayout(ctx.device, &pci, nullptr, &pipeline_layout));
    create_pipeline();
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    check(vkCreateSemaphore(ctx.device, &si, nullptr, &acquired));
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = 2;
    check(vkCreateQueryPool(ctx.device, &qi, nullptr, &queries));
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx.physical_device, &props);
    timestamp_period = props.limits.timestampPeriod;
    if (!ImGui_ImplGlfw_InitForVulkan(win, true))
      throw std::runtime_error("ImGui GLFW initialization failed");
    glfw_ready = true;
    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion = VK_API_VERSION_1_2;
    info.Instance = ctx.instance;
    info.PhysicalDevice = ctx.physical_device;
    info.Device = ctx.device;
    info.QueueFamily = ctx.queue_family;
    info.Queue = ctx.queue;
    info.DescriptorPool = pool;
    info.RenderPass = wd.RenderPass;
    info.MinImageCount = 2;
    info.ImageCount = wd.ImageCount;
    info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    info.CheckVkResultFn = check;
    if (!ImGui_ImplVulkan_Init(&info))
      throw std::runtime_error("ImGui Vulkan initialization failed");
    backend_ready = true;
  } catch (...) {
    cleanup();
    throw;
  }
}
void Presenter::create_pipeline() {
  pipeline = create_scene_pipeline(ctx.device, wd.RenderPass, pipeline_layout,
                                   shader_dir);
}
VkPipeline create_scene_pipeline(VkDevice device, VkRenderPass render_pass,
                                 VkPipelineLayout pipeline_layout,
                                 const std::string &shader_dir) {
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkShaderModule vs = shader(device,
                             shader_dir + "/generated/viewer_present_vs.spv"),
                 fs = VK_NULL_HANDLE;
  try {
    fs = shader(device, shader_dir + "/generated/viewer_present_fs.spv");
    VkPipelineShaderStageCreateInfo stages[2]{};
    for (auto &s : stages) {
      s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      s.pName = "main";
    }
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs;
    VkPipelineVertexInputStateCreateInfo vi{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo ms{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState ba{};
    ba.colorWriteMask = 15;
    VkPipelineColorBlendStateCreateInfo bs{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    bs.attachmentCount = 1;
    bs.pAttachments = &ba;
    VkDynamicState states[] = {VK_DYNAMIC_STATE_VIEWPORT,
                               VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    ds.dynamicStateCount = 2;
    ds.pDynamicStates = states;
    VkGraphicsPipelineCreateInfo ci{
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.stageCount = 2;
    ci.pStages = stages;
    ci.pVertexInputState = &vi;
    ci.pInputAssemblyState = &ia;
    ci.pViewportState = &vp;
    ci.pRasterizationState = &rs;
    ci.pMultisampleState = &ms;
    ci.pColorBlendState = &bs;
    ci.pDynamicState = &ds;
    ci.layout = pipeline_layout;
    ci.renderPass = render_pass;
    check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr,
                                    &pipeline));
  } catch (...) {
    vkDestroyShaderModule(device, vs, nullptr);
    if (fs)
      vkDestroyShaderModule(device, fs, nullptr);
    throw;
  }
  vkDestroyShaderModule(device, vs, nullptr);
  vkDestroyShaderModule(device, fs, nullptr);
  return pipeline;
}
void Presenter::wait() {
  if (last_fence) {
    check(vkWaitForFences(ctx.device, 1, &last_fence, VK_TRUE, UINT64_MAX));
    last_fence = VK_NULL_HANDLE;
  }
  if (query_pending) {
    uint64_t t[2]{};
    check(vkGetQueryPoolResults(ctx.device, queries, 0, 2, sizeof(t), t,
                                sizeof(uint64_t), VK_QUERY_RESULT_64_BIT));
    display_ms = (t[1] - t[0]) * timestamp_period / 1e6;
    query_pending = false;
  }
}
void Presenter::resize(int w, int h) {
  if (w <= 0 || h <= 0)
    return;
  if (!rebuild && w == wd.Width && h == wd.Height)
    return;
  wait();
  check(vkDeviceWaitIdle(ctx.device));
  ImGui_ImplVulkan_SetMinImageCount(2);
  ImGui_ImplVulkanH_CreateOrResizeWindow(ctx.instance, ctx.physical_device,
                                         ctx.device, &wd, ctx.queue_family,
                                         nullptr, w, h, 2);
  rebuild = false;
}
void Presenter::draw(const Viewport &v, const DisplaySettings &d) {
  wait();
  VkResult result =
      vkAcquireNextImageKHR(ctx.device, wd.Swapchain, UINT64_MAX, acquired,
                            VK_NULL_HANDLE, &wd.FrameIndex);
  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    rebuild = true;
    return;
  }
  if (result == VK_SUBOPTIMAL_KHR)
    rebuild = true;
  else
    check(result);
  auto &frame = wd.Frames[wd.FrameIndex];
  check(vkWaitForFences(ctx.device, 1, &frame.Fence, VK_TRUE, UINT64_MAX));
  check(vkResetFences(ctx.device, 1, &frame.Fence));
  check(vkResetCommandPool(ctx.device, frame.CommandPool, 0));
  VkCommandBuffer cmd = frame.CommandBuffer;
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check(vkBeginCommandBuffer(cmd, &begin));
  vkCmdResetQueryPool(cmd, queries, 0, 2);
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
  if (renderer.valid) {
    const auto &b = renderer.buffers.pixel_state.deviceBuffer;
    VkDeviceSize bytes = VkDeviceSize(renderer.uniforms.image_width) *
                         renderer.uniforms.image_height * 16;
    if (buffer_generation != b.generation || buffer_bytes != bytes) {
      VkDescriptorBufferInfo bi{b.buffer, 0, bytes};
      VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      write.dstSet = set;
      write.dstBinding = 0;
      write.descriptorCount = 1;
      write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      write.pBufferInfo = &bi;
      vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
      buffer_generation = b.generation;
      buffer_bytes = bytes;
    }
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask =
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex =
        VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = b.buffer;
    barrier.size = bytes;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                         1, &barrier, 0, nullptr);
  }
  VkClearValue clear{};
  clear.color = {{0.035f, 0.045f, 0.06f, 1}};
  VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  rp.renderPass = wd.RenderPass;
  rp.framebuffer = frame.Framebuffer;
  rp.renderArea.extent = {(uint32_t)wd.Width, (uint32_t)wd.Height};
  rp.clearValueCount = 1;
  rp.pClearValues = &clear;
  vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
  if (renderer.valid && v.w > 0 && v.h > 0) {
    struct Push {
      uint32_t w, h, mode, checker;
      float x, y, vw, vh;
      float bg[4];
      float exposure;
      uint32_t linear;
      float pad[2];
    } p{renderer.uniforms.image_width,
        renderer.uniforms.image_height,
        (uint32_t)d.mode,
        (uint32_t)d.checker,
        v.x,
        v.y,
        v.w,
        v.h,
        {d.background.x, d.background.y, d.background.z, 1},
        d.exposure,
        0,
        {0, 0}};
    static_assert(sizeof(Push) == 64);
    VkViewport vp{v.x, v.y, v.w, v.h, 0, 1};
    VkRect2D sc{{(int32_t)v.x, (int32_t)v.y}, {(uint32_t)v.w, (uint32_t)v.h}};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline_layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(p), &p);
    vkCmdDraw(cmd, 3, 1, 0, 0);
  }
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
  vkCmdEndRenderPass(cmd);
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
  check(vkEndCommandBuffer(cmd));
  // Per-swapchain-image semaphore: reacquiring the image proves its prior
  // present has consumed it.
  VkSemaphore finished =
      wd.FrameSemaphores[wd.FrameIndex].RenderCompleteSemaphore;
  VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.waitSemaphoreCount = 1;
  submit.pWaitSemaphores = &acquired;
  submit.pWaitDstStageMask = &stage;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  submit.signalSemaphoreCount = 1;
  submit.pSignalSemaphores = &finished;
  check(vkQueueSubmit(ctx.queue, 1, &submit, frame.Fence));
  last_fence = frame.Fence;
  query_pending = true;
  VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  present.waitSemaphoreCount = 1;
  present.pWaitSemaphores = &finished;
  present.swapchainCount = 1;
  present.pSwapchains = &wd.Swapchain;
  present.pImageIndices = &wd.FrameIndex;
  result = vkQueuePresentKHR(ctx.queue, &present);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
    rebuild = true;
  else
    check(result);
}
Presenter::~Presenter() { cleanup(); }
void Presenter::cleanup() noexcept {
  vkDeviceWaitIdle(ctx.device);
  if (backend_ready)
    ImGui_ImplVulkan_Shutdown();
  if (glfw_ready)
    ImGui_ImplGlfw_Shutdown();
  if (pipeline)
    vkDestroyPipeline(ctx.device, pipeline, nullptr);
  if (pipeline_layout)
    vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
  if (layout)
    vkDestroyDescriptorSetLayout(ctx.device, layout, nullptr);
  if (pool)
    vkDestroyDescriptorPool(ctx.device, pool, nullptr);
  if (acquired)
    vkDestroySemaphore(ctx.device, acquired, nullptr);
  if (queries)
    vkDestroyQueryPool(ctx.device, queries, nullptr);
  // The compute context owns the surface and destroys it after the swapchain.
  wd.Surface = VK_NULL_HANDLE;
  ImGui_ImplVulkanH_DestroyWindow(ctx.instance, ctx.device, &wd, nullptr);
}
} // namespace ftgspp
