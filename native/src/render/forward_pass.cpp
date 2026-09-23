#include "render/forward_pass.h"
#include <cmath>
#include <stdexcept>
namespace ftgspp {
void camera_uniforms(const ViewCamera &c, VulkanGSRendererUniforms &u) {
  if (!c.width || !c.height || c.width > 16384 || c.height > 16384 ||
      !std::isfinite(c.fx) || !std::isfinite(c.fy) || c.fx <= 0 || c.fy <= 0 ||
      !std::isfinite(c.cx) || !std::isfinite(c.cy))
    throw std::invalid_argument("Invalid viewer camera extent or intrinsic");
  u.image_width = c.width;
  u.image_height = c.height;
  u.grid_width = _CEIL_DIV(c.width, TILE_WIDTH);
  u.grid_height = _CEIL_DIV(c.height, TILE_HEIGHT);
  u.camera_model = 0;
  u.fx = c.fx;
  u.fy = c.fy;
  u.cx = c.cx;
  u.cy = c.cy;
  for (int i = 0; i < 4; ++i)
    u.dist_coeffs[i] = 0;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      if (!std::isfinite(c.w2c[i][j]))
        throw std::invalid_argument("Non-finite camera pose");
      u.world_view_transform[4 * i + j] = c.w2c[i][j];
    }
}
void forward_tiles(VulkanGSRenderer &r, VulkanGSRendererUniforms &u,
                   VulkanGSPipelineBuffers &b) {
  DeviceGuard guard(&r);
  r.executeCalculateIndexBufferOffset(b);
  r.executeTileWorkPrepare(u, b);
  r.executeGenerateKeys(u, b);
  r.executeSort(u, b, -1);
  r.executeComputeTileRanges(u, b);
}
void forward_dynamic(VulkanGSRenderer &r, VulkanGSRendererUniforms &u,
                     VulkanGSPipelineBuffers &b, float t, float duration,
                     float threshold) {
  DeviceGuard guard(&r);
  u.num_splats = (uint32_t)b.num_splats;
  r.executeDynamicProjectionForward(u, b, t, duration, threshold, 0);
  forward_tiles(r, u, b);
  r.executeRasterizeForward(u, b);
}
} // namespace ftgspp
