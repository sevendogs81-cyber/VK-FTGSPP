#pragma once
#include "render/forward_pass.h"
#include <map>
namespace ftgspp {
using ModelArrays = std::map<std::string, std::vector<float>>;
struct ViewDisplayOptions {
  uint32_t mode = 0;
  std::array<float, 3> background{0, 0, 0};
  float exposure = 0;
  bool checker = false;
  bool transparent = false;
};
class ViewRenderer : public VulkanGSRenderer {
public:
  VulkanGSPipelineBuffers buffers;
  VulkanGSRendererUniforms uniforms{};
  float max_duration = -1;
  bool valid = false;
  double render_ms = 0;
  double readback_ms = 0;
  uint64_t pixel_readbacks = 0;
  double rgba8_ms = 0;
  uint64_t rgba8_readbacks = 0;
  size_t rgba8_readback_bytes = 0;
  uint64_t rendered_frames = 0;
  uint32_t replay_count = 0;
  ViewRenderer(const std::string &shaders, int device,
               const DeviceInitOptions &options = {});
  ~ViewRenderer();
  void upload(const ModelArrays &, float duration);
  void render(const ViewCamera &, float timestamp, uint32_t sh_degree,
              float threshold);
  std::vector<float> readback();
  void readback_into(float *destination, size_t count);
  void readback_rgba8_into(uint8_t *destination, size_t bytes,
                          const ViewDisplayOptions &options = {});
  size_t rgba8_alloc_bytes() const { return rgba8_buffer_.allocSize; }
  void close();

private:
  bool closed_ = false;
  bool uploaded_ = false;
  std::string pack_shader_;
  _ComputePipeline pack_pipeline_{2};
  _VulkanBuffer rgba8_buffer_;
};
} // namespace ftgspp
