// Arbitrary F/T fixtures exercise display conversion independently of splatting.
// This test requires only Vulkan compute, not a window or a graphics queue.
#include "render/view_renderer.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace ftgspp;

static void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
template <class F> static void rejects(F fn) {
  try { fn(); } catch (const std::exception &) { return; }
  throw std::runtime_error("Invalid input was accepted");
}
static uint8_t unorm(float x) {
  return uint8_t(std::nearbyint(std::clamp(x, 0.f, 1.f) * 255.f));
}
int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: viewer_pack_test SHADERS VULKAN_DEVICE\n";
    return 2;
  }
  try {
    ViewRenderer r(argv[1], std::stoi(argv[2]));
    unsigned cases = 0, worst = 0;
    for (auto extent : {std::pair{37u, 19u}, {257u, 129u}, {7u, 3u}, {1u, 1u}}) {
      const auto [width, height] = extent;
      const size_t count = size_t(width) * height;
      r.uniforms.image_width = width;
      r.uniforms.image_height = height;
      auto &ft = r.buffers.pixel_state;
      ft.resize(count * 4);
      const float ts[] = {-0.2f, 0, .1f, .5f, .9f, 1 - 2e-6f, 1 - 5e-7f, 1, 1.2f};
      for (size_t i = 0; i < count; ++i) {
        ft[4*i] = float(i % 71) / 47.f - .1f;
        ft[4*i+1] = float(i % 13) / 16.f;
        ft[4*i+2] = float(i % 257) / 255.f;
        ft[4*i+3] = ts[i % std::size(ts)];
      }
      r.copyToDevice(ft);
      r.valid = true;
      const std::vector<float> original(ft.begin(), ft.end());
      std::vector<uint8_t> output(count * 4);
      for (uint32_t mode = 0; mode < 3; ++mode)
        for (bool transparent : {false, true}) {
          if (transparent && mode) continue;
          for (int background = 0; background < 4; ++background)
            for (float exposure : {-20.f, -1.25f, 0.f, .75f, 20.f}) {
              ViewDisplayOptions o;
              o.mode = mode;
              o.transparent = transparent;
              o.checker = background == 3;
              o.background = background == 1 ? std::array<float,3>{1,1,1}
                           : background == 2 ? std::array<float,3>{.15f,.4f,.8f}
                                             : std::array<float,3>{0,0,0};
              o.exposure = exposure;
              r.readback_rgba8_into(output.data(), output.size(), o);
              require(r.rgba8_readback_bytes == count*4+4, "Incorrect transfer size");
              for (size_t i = 0; i < count; ++i) {
                const float t = std::clamp(ft[4*i+3], 0.f, 1.f), alpha = 1-t;
                for (int channel = 0; channel < 4; ++channel) {
                  float value = transparent ? alpha : 1.f;
                  if (channel < 3) {
                    float bg = o.checker ? (((i % width / 16 + i / width / 16) & 1) ? .65f : .35f)
                                         : o.background[channel];
                    value = transparent ? (alpha > 1e-6f ? ft[4*i+channel]/alpha : 0.f)
                          : mode == 1 ? alpha : mode == 2 ? t : ft[4*i+channel] + t*bg;
                    if (mode == 0) value *= std::exp2(exposure);
                  }
                  unsigned error = std::abs(int(output[4*i+channel]) - int(unorm(value)));
                  worst = std::max(worst, error);
                  require(error <= 1, "RGBA8 differs from CPU reference by more than 1/255");
                }
              }
              ++cases;
            }
        }
      require(r.readback() == original, "Packing changed FP32 F/T");
      const auto capacity = r.rgba8_alloc_bytes();
      for (int i = 0; i < 8; ++i) r.readback_rgba8_into(output.data(), output.size());
      require(capacity == r.rgba8_alloc_bytes(), "Repeated readback reallocates");
      rejects([&] { r.readback_rgba8_into(output.data(), output.size()-1); });
      ViewDisplayOptions bad;
      bad.mode = 3;
      rejects([&] { r.readback_rgba8_into(output.data(), output.size(), bad); });
      bad.mode = 1; bad.transparent = true;
      rejects([&] { r.readback_rgba8_into(output.data(), output.size(), bad); });
      bad = {}; bad.background[0] = -1;
      rejects([&] { r.readback_rgba8_into(output.data(), output.size(), bad); });
      bad = {}; bad.exposure = std::numeric_limits<float>::quiet_NaN();
      rejects([&] { r.readback_rgba8_into(output.data(), output.size(), bad); });
      // GPU reduction flag preserves the old finite check without full F/T D2H.
      for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity()}) {
        ft[0] = invalid;
        r.copyToDevice(ft);
        rejects([&] { r.readback_rgba8_into(output.data(), output.size()); });
        ft[0] = original[0];
        r.copyToDevice(ft);
        r.readback_rgba8_into(output.data(), output.size());
      }
    }
    r.close();
    uint8_t pixel[4];
    rejects([&] { r.readback_rgba8_into(pixel, 4); });
    r.close();
    std::cout << "RGBA8 cases=" << cases << ", worst UNORM error=" << worst
              << "/255; non-finite, resize, reuse, close: PASS\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
