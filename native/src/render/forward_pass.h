#pragma once
#include "render/gs_renderer.h"
#include <glm/glm.hpp>
#include <string>

namespace ftgspp {
struct ViewCamera {
  uint32_t width = 1280, height = 720;
  float fx = 1000, fy = 1000, cx = 640, cy = 360;
  glm::mat4 w2c{1.0f};
  float timestamp = 0;
  std::string name;
};
void camera_uniforms(const ViewCamera &, VulkanGSRendererUniforms &);
void forward_tiles(VulkanGSRenderer &, VulkanGSRendererUniforms &,
                   VulkanGSPipelineBuffers &);
void forward_dynamic(VulkanGSRenderer &, VulkanGSRendererUniforms &,
                     VulkanGSPipelineBuffers &, float timestamp,
                     float max_duration, float marginal_threshold);
} // namespace ftgspp
