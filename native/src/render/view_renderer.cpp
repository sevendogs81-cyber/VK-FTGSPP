#include "render/view_renderer.h"
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace ftgspp {
ViewRenderer::ViewRenderer(const std::string &shaders, int device,
                           const DeviceInitOptions &options)
    : pack_shader_(shaders + "/generated/viewer_pack.spv") {
  std::map<std::string, std::string> paths;
  for (auto n : {"projection_forward",
                 "projection_forward_sh0",
                 "projection_forward_sh1",
                 "projection_forward_sh2",
                 "ftgspp_dynamic_projection",
                 "ftgspp_dynamic_projection_sh0",
                 "ftgspp_dynamic_projection_sh1",
                 "ftgspp_dynamic_projection_sh2",
                 "generate_keys",
                 "ftgspp_tile_work_prepare",
                 "compute_tile_ranges",
                 "rasterize_forward",
                 "cumsum_single_pass",
                 "cumsum_block_scan",
                 "cumsum_scan_block_sums",
                 "cumsum_add_block_offsets",
                 "cumsum_f32_single_pass",
                 "cumsum_f32_block_scan",
                 "cumsum_f32_scan_block_sums",
                 "cumsum_f32_add_block_offsets"})
    paths[n] = shaders + "/generated/" + n + ".spv";
  for (auto n : {"upsweep", "spine", "downsweep"})
    paths[std::string("radix_sort/") + n] =
        shaders + "/radix_sort/" + n + ".spv";
  initialize(paths, device, true, options);
}
ViewRenderer::~ViewRenderer() {
  try {
    close();
  } catch (...) {
  }
}
void ViewRenderer::close() {
  if (closed_)
    return;
  drain("viewer_close");
  destroyBuffer(rgba8_buffer_);
  cleanupBuffers(buffers);
  cleanup();
  closed_ = true;
  valid = false;
}
void ViewRenderer::upload(const ModelArrays &a, float duration) {
  if (closed_)
    throw std::runtime_error("Renderer is closed");
  if (std::isnan(duration) || duration <= 0)
    throw std::invalid_argument(
        "Model max_duration must be positive or infinity");
  const auto &means = a.at("means");
  if (means.size() % 3 ||
      means.size() / 3 > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("Invalid model count");
  size_t n = means.size() / 3;
  for (auto item : std::map<std::string, size_t>{{"means", 3},
                                                 {"log_scales", 3},
                                                 {"quaternions", 4},
                                                 {"opacity_logits", 1},
                                                 {"sh_coefficients", 48},
                                                 {"times", 1},
                                                 {"duration_raw", 1},
                                                 {"velocities", 3},
                                                 {"gate_raw", 1}}) {
    const auto &v = a.at(item.first);
    if (v.size() != n * item.second)
      throw std::invalid_argument("Invalid model array: " + item.first);
    for (float x : v)
      if (!std::isfinite(x))
        throw std::invalid_argument("Non-finite model array: " + item.first);
  }
  uploaded_ = false;
  drain("viewer_upload");
  destroyBuffer(rgba8_buffer_);
  cleanupBuffers(buffers);
  buffers = VulkanGSPipelineBuffers{};
  valid = false;
  max_duration = std::isinf(duration) ? -1 : duration;
  buffers.num_splats = n;
  if (!n) {
    uploaded_ = true;
    return;
  }
  buffers.base_xyz_ws.assign(means.begin(), means.end());
  buffers.rotations.assign(a.at("quaternions").begin(),
                           a.at("quaternions").end());
  buffers.sh_coeffs.assign(a.at("sh_coefficients").begin(),
                           a.at("sh_coefficients").end());
  buffers.base_log_scales_opacity.resize(4 * n);
  buffers.times_durations.resize(2 * n);
  buffers.velocities_gates.resize(4 * n);
  for (size_t i = 0; i < n; ++i) {
    for (int j = 0; j < 3; ++j) {
      buffers.base_log_scales_opacity[4 * i + j] =
          a.at("log_scales")[3 * i + j];
      buffers.velocities_gates[4 * i + j] = a.at("velocities")[3 * i + j];
    }
    buffers.base_log_scales_opacity[4 * i + 3] = a.at("opacity_logits")[i];
    buffers.velocities_gates[4 * i + 3] = a.at("gate_raw")[i];
    buffers.times_durations[2 * i] = a.at("times")[i];
    buffers.times_durations[2 * i + 1] = a.at("duration_raw")[i];
  }
  buffers.reorderSH(buffers.sh_coeffs);
  copyToDevice(buffers.base_xyz_ws);
  copyToDevice(buffers.rotations);
  copyToDevice(buffers.sh_coeffs);
  copyToDevice(buffers.base_log_scales_opacity);
  copyToDevice(buffers.times_durations);
  copyToDevice(buffers.velocities_gates);
  for (auto *b : {&buffers.base_xyz_ws, &buffers.rotations, &buffers.sh_coeffs,
                  &buffers.base_log_scales_opacity, &buffers.times_durations,
                  &buffers.velocities_gates}) {
    b->clear();
    b->shrink_to_fit();
  }
  uploaded_ = true;
}
void ViewRenderer::render(const ViewCamera &c, float t, uint32_t degree,
                          float threshold) {
  if (!uploaded_)
    throw std::runtime_error("No uploaded model");
  if (closed_)
    throw std::runtime_error("Renderer is closed");
  if (!std::isfinite(t) || !std::isfinite(threshold) || threshold < 0 ||
      threshold > 1 || degree > 3)
    throw std::invalid_argument("Invalid render time/SH/threshold");
  valid = false;
  camera_uniforms(c, uniforms);
  uniforms.active_sh = degree;
  uniforms.step = 0;
  auto start = std::chrono::steady_clock::now();
  const size_t pixels = size_t(c.width) * c.height;
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(graphicsContext().physical_device, &props);
  if (pixels * 16 > props.limits.maxStorageBufferRange)
    throw std::invalid_argument("Image exceeds GPU maxStorageBufferRange");
  if (!buffers.num_splats) {
    buffers.pixel_state.assign(4 * pixels, 0);
    for (size_t i = 0; i < pixels; ++i)
      buffers.pixel_state[4 * i + 3] = 1;
    copyToDevice(buffers.pixel_state);
  } else {
    for (int retry = 0; retry < 4; ++retry) {
      setSubmissionMeta(-1, false);
      forward_dynamic(*this, uniforms, buffers, t, max_duration, threshold);
      waitAllComputeSlots("viewer_forward");
      auto status = collectRetiredTileOverflow();
      if (status.fatal)
        throw std::runtime_error("Viewer tile invariant/prefix failure");
      if (!status.overflow)
        break;
      if (retry == 3)
        throw std::runtime_error("Viewer tile overflow after four retries");
      buffers.tile_pair_capacity =
          growTilePairCapacity(buffers.tile_pair_capacity, status.exact_count);
      clearDeviceBuffer(buffers.pipeline_abort_latch, 4);
      clearRetiredTileOverflow();
      ++replay_count;
    }
  }
  render_ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - start)
                  .count();
  valid = true;
  ++rendered_frames;
}
std::vector<float> ViewRenderer::readback() {
  size_t count = size_t(uniforms.image_width) * uniforms.image_height * 4;
  if (!valid)
    throw std::runtime_error("No valid viewer frame");
  std::vector<float> result(count);
  readback_into(result.data(), result.size());
  return result;
}
void ViewRenderer::readback_into(float *destination, size_t count) {
  if (!valid)
    throw std::runtime_error("No valid viewer frame");
  size_t expected = size_t(uniforms.image_width) * uniforms.image_height * 4;
  if (!destination || count != expected)
    throw std::invalid_argument("Readback destination must match the current HxWx4 frame");
  auto source = buffers.pixel_state.deviceBuffer;
  if (expected * sizeof(float) > source.size)
    throw std::runtime_error("Viewer pixel buffer is smaller than its extent");
  // Restrict the copy to this frame, not the allocation high-water mark. Copy
  // cached, persistently mapped staging memory directly into caller storage.
  source.size = expected * sizeof(float);
  auto start = std::chrono::steady_clock::now();
  copyBuffersFromDevicePacked({{source, 0}}, destination, source.size);
  readback_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
  ++pixel_readbacks;
}
void ViewRenderer::readback_rgba8_into(uint8_t *destination, size_t bytes,
                                     const ViewDisplayOptions &options) {
  if (!valid)
    throw std::runtime_error("No valid viewer frame");
  const size_t expected = size_t(uniforms.image_width) * uniforms.image_height * 4;
  if (!destination || bytes != expected)
    throw std::invalid_argument("RGBA8 destination must match the current HxWx4 frame");
  if (options.mode > 2 || (options.transparent && options.mode != 0) ||
      !std::isfinite(options.exposure) || options.exposure < -20 || options.exposure > 20)
    throw std::invalid_argument("Invalid display mode / transparent alpha / exposure");
  for (float c : options.background)
    if (!std::isfinite(c) || c < 0 || c > 1)
      throw std::invalid_argument("Background must be finite RGB in [0,1]");
  if (expected * sizeof(float) > buffers.pixel_state.deviceBuffer.size)
    throw std::runtime_error("Viewer pixel buffer is smaller than its extent");
  const auto start = std::chrono::steady_clock::now();
  if (pack_pipeline_.pipeline == VK_NULL_HANDLE)
    createComputePipeline(pack_pipeline_, pack_shader_);
  // Four bytes per pixel and one status word; neither host nor device storage
  // grows with the number of frames. Local presentation never allocates these.
  const size_t transfer_bytes = expected + sizeof(uint32_t);
  resizeDeviceBuffer(rgba8_buffer_, transfer_bytes);
  allocStagingBufferFor(readback_stager, transfer_bytes, StagingPurpose::Readback);
  std::lock_guard<std::mutex> lock(readback_stager.mutex);
  struct PackParams {
    uint32_t width, height, mode, flags;
    float background[3], exposure_scale;
  } p{uniforms.image_width, uniforms.image_height, options.mode,
      uint32_t(options.checker) | (uint32_t(options.transparent) << 1),
      {options.background[0], options.background[1], options.background[2]},
      std::exp2(options.exposure)};
  static_assert(sizeof(PackParams) == 32, "Match viewer_pack.slang");
  _VulkanBuffer staging;
  staging.buffer = readback_stager.buffer;
  staging.size = transfer_bytes;
  {
    DeviceGuard guard(this);
    bufferMemoryBarrier({{rgba8_buffer_, TRANSFER_COMPUTE_SHADER_READ_WRITE}}, TRANSFER_WRITE);
    vkCmdFillBuffer(command_buffer, rgba8_buffer_.buffer, expected, 4, 0);
    bufferMemoryBarrier({{buffers.pixel_state.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                         {rgba8_buffer_, TRANSFER_COMPUTE_SHADER_READ_WRITE}},
                        COMPUTE_SHADER_READ_WRITE);
    executeCompute({{expected / 4, 256}}, &p, sizeof(p), pack_pipeline_,
                   {buffers.pixel_state.deviceBuffer, rgba8_buffer_});
    bufferMemoryBarrier({{rgba8_buffer_, COMPUTE_SHADER_WRITE}}, TRANSFER_READ);
    bufferMemoryBarrier({{staging, TRANSFER_WRITE}}, TRANSFER_WRITE);
    VkBufferCopy region{0, 0, transfer_bytes};
    vkCmdCopyBuffer(command_buffer, rgba8_buffer_.buffer, staging.buffer, 1, &region);
    bufferMemoryBarrier({{staging, TRANSFER_WRITE}}, HOST_READ);
  }
  // Compute composition + D2H share one submission and one completion wait.
  drain("viewer_rgba8");
  auto *mapped = static_cast<const uint8_t *>(ensureStagerMapped(readback_stager));
  invalidateStager(readback_stager, transfer_bytes);
  uint32_t invalid;
  std::memcpy(&invalid, mapped + expected, sizeof(invalid));
  if (invalid)
    throw std::runtime_error("Non-finite viewer foreground/transmittance");
  std::memcpy(destination, mapped, expected);
  rgba8_ms = std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - start).count();
  rgba8_readback_bytes = transfer_bytes;
  ++rgba8_readbacks;
}
} // namespace ftgspp
