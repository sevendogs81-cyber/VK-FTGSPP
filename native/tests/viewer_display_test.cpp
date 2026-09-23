// Exercise the production display pipeline on a GPU without requiring a WSI
// display. The color image and readback below exist only in this acceptance
// test.
#include "viewer/presenter.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace ftgspp;
static void check(VkResult result) {
  if (result != VK_SUCCESS)
    throw std::runtime_error("Vulkan error " + std::to_string(result));
}
static uint32_t memory_type(VkPhysicalDevice gpu, uint32_t mask,
                            VkMemoryPropertyFlags flags) {
  VkPhysicalDeviceMemoryProperties props;
  vkGetPhysicalDeviceMemoryProperties(gpu, &props);
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
    if ((mask & (1u << i)) &&
        (props.memoryTypes[i].propertyFlags & flags) == flags)
      return i;
  throw std::runtime_error("No matching memory type");
}
int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: viewer_display_test SHADERS VULKAN_DEVICE\n";
    return 2;
  }
  try {
    DeviceInitOptions init;
    init.require_graphics = true;
    ViewRenderer r(argv[1], std::stoi(argv[2]), init);
    auto ctx = r.graphicsContext();
    auto d = ctx.device;
    constexpr int W = 64, H = 48, OW = 128, OH = 96;
    r.buffers.pixel_state.resize(W * H * 4);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        float t = .2f + .8f * x / (W - 1);
        int i = (y * W + x) * 4;
        r.buffers.pixel_state[i] = float(x) / (W - 1) * (1 - t);
        r.buffers.pixel_state[i + 1] = float(y) / (H - 1) * (1 - t);
        r.buffers.pixel_state[i + 2] = .2f * (1 - t);
        r.buffers.pixel_state[i + 3] = t;
      }
    r.copyToDevice(r.buffers.pixel_state);
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = {OW, OH, 1};
    ici.mipLevels = ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VkImage img;
    check(vkCreateImage(d, &ici, nullptr, &img));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(d, img, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = memory_type(ctx.physical_device, req.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory mem;
    check(vkAllocateMemory(d, &mai, nullptr, &mem));
    check(vkBindImageMemory(d, img, mem, 0));
    VkImageViewCreateInfo iv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    iv.image = img;
    iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
    iv.format = ici.format;
    iv.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView view;
    check(vkCreateImageView(d, &iv, nullptr, &view));
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = OW * OH * 4;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer out;
    check(vkCreateBuffer(d, &bci, nullptr, &out));
    vkGetBufferMemoryRequirements(d, out, &req);
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = memory_type(ctx.physical_device, req.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory host;
    check(vkAllocateMemory(d, &mai, nullptr, &host));
    check(vkBindBufferMemory(d, out, host, 0));
    void *mapped;
    check(vkMapMemory(d, host, 0, VK_WHOLE_SIZE, 0, &mapped));
    VkAttachmentDescription attachment{};
    attachment.format = ici.format;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &ref;
    VkSubpassDependency dep{};
    dep.srcSubpass = 0;
    dep.dstSubpass = VK_SUBPASS_EXTERNAL;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dep.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rp.attachmentCount = 1;
    rp.pAttachments = &attachment;
    rp.subpassCount = 1;
    rp.pSubpasses = &sub;
    rp.dependencyCount = 1;
    rp.pDependencies = &dep;
    VkRenderPass pass;
    check(vkCreateRenderPass(d, &rp, nullptr, &pass));
    VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fb.renderPass = pass;
    fb.attachmentCount = 1;
    fb.pAttachments = &view;
    fb.width = OW;
    fb.height = OH;
    fb.layers = 1;
    VkFramebuffer frame;
    check(vkCreateFramebuffer(d, &fb, nullptr, &frame));
    VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                         1, VK_SHADER_STAGE_FRAGMENT_BIT,
                                         nullptr};
    VkDescriptorSetLayoutCreateInfo dl{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount = 1;
    dl.pBindings = &binding;
    VkDescriptorSetLayout layout;
    check(vkCreateDescriptorSetLayout(d, &dl, nullptr, &layout));
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo pc{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pc.maxSets = 1;
    pc.poolSizeCount = 1;
    pc.pPoolSizes = &ps;
    VkDescriptorPool pool;
    check(vkCreateDescriptorPool(d, &pc, nullptr, &pool));
    VkDescriptorSetAllocateInfo da{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    da.descriptorPool = pool;
    da.descriptorSetCount = 1;
    da.pSetLayouts = &layout;
    VkDescriptorSet set;
    check(vkAllocateDescriptorSets(d, &da, &set));
    VkDescriptorBufferInfo bi{r.buffers.pixel_state.deviceBuffer.buffer, 0,
                              W * H * 16};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &bi;
    vkUpdateDescriptorSets(d, 1, &write, 0, nullptr);
    VkPushConstantRange push{VK_SHADER_STAGE_FRAGMENT_BIT, 0, 64};
    VkPipelineLayoutCreateInfo pl{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &layout;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &push;
    VkPipelineLayout pipeline_layout;
    check(vkCreatePipelineLayout(d, &pl, nullptr, &pipeline_layout));
    auto pipeline = create_scene_pipeline(d, pass, pipeline_layout, argv[1]);
    VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cp.queueFamilyIndex = ctx.queue_family;
    VkCommandPool cmdpool;
    check(vkCreateCommandPool(d, &cp, nullptr, &cmdpool));
    VkCommandBufferAllocateInfo ca{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = cmdpool;
    ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ca.commandBufferCount = 1;
    VkCommandBuffer cmd;
    check(vkAllocateCommandBuffers(d, &ca, &cmd));
    int worst = 0, cases = 0;
    for (int scale : {1, 2})
      for (int mode : {0, 1, 2})
        for (int checker : {0, 1}) {
          int width = W * scale, height = H * scale;
          check(vkResetCommandPool(d, cmdpool, 0));
          VkCommandBufferBeginInfo begin{
              VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
          check(vkBeginCommandBuffer(cmd, &begin));
          VkBufferMemoryBarrier barrier{
              VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
          barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
          barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
          barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex =
              VK_QUEUE_FAMILY_IGNORED;
          barrier.buffer = bi.buffer;
          barrier.size = bi.range;
          vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                               nullptr, 1, &barrier, 0, nullptr);
          VkClearValue clear{};
          VkRenderPassBeginInfo rb{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
          rb.renderPass = pass;
          rb.framebuffer = frame;
          rb.renderArea.extent = {OW, OH};
          rb.clearValueCount = 1;
          rb.pClearValues = &clear;
          vkCmdBeginRenderPass(cmd, &rb, VK_SUBPASS_CONTENTS_INLINE);
          VkViewport viewport{0, 0, float(width), float(height), 0, 1};
          VkRect2D scissor{{0, 0}, {uint32_t(width), uint32_t(height)}};
          vkCmdSetViewport(cmd, 0, 1, &viewport);
          vkCmdSetScissor(cmd, 0, 1, &scissor);
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  pipeline_layout, 0, 1, &set, 0, nullptr);
          struct Push {
            uint32_t w, h, mode, checker;
            float x, y, vw, vh;
            float bg[4];
            float ev;
            uint32_t linear;
            float pad[2];
          } p{W,
              H,
              uint32_t(mode),
              uint32_t(checker),
              0,
              0,
              float(width),
              float(height),
              {.15, .25, .4, 1},
              .5,
              0,
              {0, 0}};
          static_assert(sizeof(p) == 64);
          vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
                             0, 64, &p);
          vkCmdDraw(cmd, 3, 1, 0, 0);
          vkCmdEndRenderPass(cmd);
          VkBufferImageCopy copy{};
          copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
          copy.imageExtent = {OW, OH, 1};
          vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                 out, 1, &copy);
          barrier.buffer = out;
          barrier.size = OW * OH * 4;
          barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
          barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
          vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                               &barrier, 0, nullptr);
          check(vkEndCommandBuffer(cmd));
          VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
          submit.commandBufferCount = 1;
          submit.pCommandBuffers = &cmd;
          check(vkQueueSubmit(ctx.queue, 1, &submit, VK_NULL_HANDLE));
          check(vkQueueWaitIdle(ctx.queue));
          auto *pixels = static_cast<unsigned char *>(mapped);
          auto fetch = [&](int x, int y, int k) {
            x = std::clamp(x, 0, W - 1);
            y = std::clamp(y, 0, H - 1);
            return r.buffers.pixel_state[(y * W + x) * 4 + k];
          };
          for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
              float u = (x + .5f) / scale - .5f, v = (y + .5f) / scale - .5f;
              int ix = std::floor(u), iy = std::floor(v);
              float a = u - ix, b = v - iy, ft[4];
              for (int k = 0; k < 4; ++k)
                ft[k] =
                    ((1 - a) * fetch(ix, iy, k) + a * fetch(ix + 1, iy, k)) *
                        (1 - b) +
                    ((1 - a) * fetch(ix, iy + 1, k) +
                     a * fetch(ix + 1, iy + 1, k)) *
                        b;
              for (int k = 0; k < 3; ++k) {
                float bg =
                    checker ? ((x / 16 + y / 16) % 2 ? .65f : .35f) : p.bg[k];
                float value = mode == 1 ? 1 - ft[3]
                              : mode == 2
                                  ? ft[3]
                                  : (ft[k] + ft[3] * bg) * std::sqrt(2.f);
                int expected = std::lround(std::clamp(value, 0.f, 1.f) * 255);
                worst =
                    std::max(worst, std::abs(int(pixels[(y * OW + x) * 4 + k]) -
                                             expected));
              }
            }
          ++cases;
        }
    vkDestroyCommandPool(d, cmdpool, nullptr);
    vkDestroyPipeline(d, pipeline, nullptr);
    vkDestroyPipelineLayout(d, pipeline_layout, nullptr);
    vkDestroyDescriptorPool(d, pool, nullptr);
    vkDestroyDescriptorSetLayout(d, layout, nullptr);
    vkDestroyFramebuffer(d, frame, nullptr);
    vkDestroyRenderPass(d, pass, nullptr);
    vkDestroyImageView(d, view, nullptr);
    vkDestroyImage(d, img, nullptr);
    vkFreeMemory(d, mem, nullptr);
    vkUnmapMemory(d, host);
    vkDestroyBuffer(d, out, nullptr);
    vkFreeMemory(d, host, nullptr);
    std::cout << "Display cases=" << cases << ", worst UNORM error=" << worst
              << "/255\n";
    return worst <= 1 ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
