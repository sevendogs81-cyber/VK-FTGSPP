#pragma once
#include "render/view_renderer.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace ftgspp {
namespace py = pybind11;
inline ViewCamera camera_from_dict(const py::dict &d) {
  ViewCamera c;
  c.width = py::cast<uint32_t>(d["width"]);
  c.height = py::cast<uint32_t>(d["height"]);
  py::array_t<float, py::array::c_style | py::array::forcecast> k(d["K"]),
      w(d["w2c"]);
  if (k.ndim() != 2 || k.shape(0) != 3 || k.shape(1) != 3 || w.ndim() != 2 ||
      w.shape(0) != 4 || w.shape(1) != 4)
    throw std::invalid_argument("Camera requires K[3,3] and w2c[4,4]");
  auto kp = k.unchecked<2>();
  auto wp = w.unchecked<2>();
  c.fx = kp(0, 0);
  c.fy = kp(1, 1);
  c.cx = kp(0, 2);
  c.cy = kp(1, 2);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      c.w2c[j][i] = wp(i, j);
  if (d.contains("name"))
    c.name = py::cast<std::string>(d["name"]);
  if (d.contains("timestamp"))
    c.timestamp = py::cast<float>(d["timestamp"]);
  VulkanGSRendererUniforms u{};
  camera_uniforms(c, u);
  return c;
}
inline py::dict camera_to_dict(const ViewCamera &c) {
  py::array_t<float> k({3, 3}), w({4, 4});
  auto kp = k.mutable_unchecked<2>();
  auto wp = w.mutable_unchecked<2>();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      kp(i, j) = i == j ? 1 : 0;
  kp(0, 0) = c.fx;
  kp(1, 1) = c.fy;
  kp(0, 2) = c.cx;
  kp(1, 2) = c.cy;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      wp(i, j) = c.w2c[j][i];
  py::dict d;
  d["K"] = k;
  d["w2c"] = w;
  d["width"] = c.width;
  d["height"] = c.height;
  d["name"] = c.name;
  d["timestamp"] = c.timestamp;
  return d;
}
inline ModelArrays model_from_dict(const py::dict &d) {
  ModelArrays a;
  for (auto key :
       {"means", "log_scales", "quaternions", "opacity_logits",
        "sh_coefficients", "times", "duration_raw", "velocities", "gate_raw"}) {
    py::array_t<float, py::array::c_style | py::array::forcecast> v(d[key]);
    a[key] = std::vector<float>(v.data(), v.data() + v.size());
  }
  return a;
}
inline py::array_t<float> readback_array(ViewRenderer &r) {
  if (!r.valid)
    throw std::runtime_error("No valid viewer frame");
  py::array_t<float> out({(py::ssize_t)r.uniforms.image_height,
                          (py::ssize_t)r.uniforms.image_width, py::ssize_t(4)});
  auto *destination = out.mutable_data();
  auto count = out.size();
  {
    py::gil_scoped_release release;
    r.readback_into(destination, count);
  }
  return out;
}
inline void readback_rgba8_array(ViewRenderer &r, py::array out, uint32_t mode,
                                const std::array<float, 3> &background,
                                bool checker, float exposure, bool transparent) {
  if (!out.dtype().is(py::dtype::of<uint8_t>()) || !out.writeable() ||
      !(out.flags() & py::array::c_style) || out.ndim() != 3 ||
      out.shape(0) != r.uniforms.image_height ||
      out.shape(1) != r.uniforms.image_width || out.shape(2) != 4)
    throw std::invalid_argument("Expected a writable contiguous uint8 HxWx4 array matching the frame");
  ViewDisplayOptions options;
  options.mode = mode;
  options.background = background;
  options.checker = checker;
  options.exposure = exposure;
  options.transparent = transparent;
  auto *destination = static_cast<uint8_t *>(out.mutable_data());
  auto count = out.size();
  py::gil_scoped_release release;
  r.readback_rgba8_into(destination, count, options);
}
inline void bind_view_renderer(py::module_ &m) {
  py::class_<ViewRenderer>(m, "ViewRenderer", py::module_local())
      .def(py::init([](const std::string &s, int d) {
             return std::make_unique<ViewRenderer>(s, d);
           }),
           py::arg("shaders"), py::arg("device") = 0)
      .def("upload",
           [](ViewRenderer &r, py::dict d) {
             auto a = model_from_dict(d);
             float t = py::cast<float>(d["max_duration"]);
             py::gil_scoped_release release;
             r.upload(a, t);
           })
      .def(
          "render",
          [](ViewRenderer &r, py::dict d, float t, uint32_t sh,
             float threshold) {
            auto c = camera_from_dict(d);
            py::gil_scoped_release release;
            r.render(c, t, sh, threshold);
          },
          py::arg("camera"), py::arg("timestamp"), py::arg("sh_degree") = 3,
          py::arg("marginal_threshold") = 0.05f)
      .def("readback", &readback_array)
      .def("readback_into", [](ViewRenderer &r, py::array out) {
        if (!out.dtype().is(py::dtype::of<float>()) || !out.writeable() ||
            !(out.flags() & py::array::c_style) || out.ndim() != 3 ||
            out.shape(0) != r.uniforms.image_height ||
            out.shape(1) != r.uniforms.image_width || out.shape(2) != 4)
          throw std::invalid_argument("Expected a writable contiguous float32 HxWx4 array matching the frame");
        auto *destination = static_cast<float *>(out.mutable_data());
        auto count = out.size();
        py::gil_scoped_release release;
        r.readback_into(destination, count);
      }, py::arg("out").noconvert())
      .def("readback_rgba8_into", &readback_rgba8_array,
           py::arg("out").noconvert(), py::kw_only(), py::arg("mode") = 0,
           py::arg("background") = std::array<float, 3>{0, 0, 0},
           py::arg("checker") = false, py::arg("exposure") = 0,
           py::arg("transparent") = false)
      .def("close", &ViewRenderer::close,
           py::call_guard<py::gil_scoped_release>())
      .def("stats",
           [](ViewRenderer &r) {
             py::dict d;
             d["render_ms"] = r.render_ms;
             d["readback_ms"] = r.readback_ms;
             d["readbacks"] = r.pixel_readbacks;
             d["rgba8_ms"] = r.rgba8_ms;
             d["rgba8_readbacks"] = r.rgba8_readbacks;
             d["rgba8_readback_bytes"] = r.rgba8_readback_bytes;
             d["frames"] = r.rendered_frames;
             d["replays"] = r.replay_count;
             d["gaussians"] = r.buffers.num_splats;
             d["vram_bytes"] = r.buffers.getTotalAllocSize() + r.rgba8_alloc_bytes();
             d["tile_pairs"] = r.lastForwardTileCount();
             return d;
           })
      .def("_set_tile_capacity",
           [](ViewRenderer &r, size_t n) { r.buffers.tile_pair_capacity = n; });
}
} // namespace ftgspp
