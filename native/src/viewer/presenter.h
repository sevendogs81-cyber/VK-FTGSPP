#pragma once
#include "render/view_renderer.h"
#include <backends/imgui_impl_vulkan.h>
#include <imgui.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
namespace ftgspp {
struct DisplaySettings {
  int mode = 0;
  bool checker = true;
  glm::vec3 background{0};
  float exposure = 0;
};
struct Viewport {
  float x = 0, y = 0, w = 1, h = 1;
};
VkPipeline create_scene_pipeline(VkDevice device, VkRenderPass render_pass,
                                 VkPipelineLayout pipeline_layout,
                                 const std::string &shader_dir);
class Presenter {
public:
  Presenter(GLFWwindow *, ViewRenderer &, const std::string &, bool vsync);
  ~Presenter();
  void wait();
  void resize(int w, int h);
  void draw(const Viewport &, const DisplaySettings &);
  double display_ms = 0;
  bool vsync;
  int width() const { return wd.Width; }
  int height() const { return wd.Height; }

private:
  ViewRenderer &renderer;
  VulkanGSPipeline::GraphicsContext ctx;
  ImGui_ImplVulkanH_Window wd;
  VkDescriptorPool pool = VK_NULL_HANDLE;
  VkDescriptorSetLayout layout = VK_NULL_HANDLE;
  VkDescriptorSet set = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkSemaphore acquired = VK_NULL_HANDLE;
  VkFence last_fence = VK_NULL_HANDLE;
  VkQueryPool queries = VK_NULL_HANDLE;
  float timestamp_period = 1;
  bool query_pending = false, backend_ready = false, glfw_ready = false,
       rebuild = false;
  uint64_t buffer_generation = UINT64_MAX;
  VkDeviceSize buffer_bytes = 0;
  std::string shader_dir;
  void create_pipeline();
  void cleanup() noexcept;
};
} // namespace ftgspp
