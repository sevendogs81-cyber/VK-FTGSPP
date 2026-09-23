#include "bindings/view_renderer_binding.h"
#include "viewer/viewer_app.h"
namespace ftgspp {
static glm::vec3 vec3(const py::handle &v) {
  auto a = py::cast<std::vector<float>>(v);
  if (a.size() != 3)
    throw std::invalid_argument("Expected vector of length 3");
  return {a[0], a[1], a[2]};
}
static py::list cameras(const std::vector<ViewCamera> &cs) {
  py::list l;
  for (const auto &c : cs)
    l.append(camera_to_dict(c));
  return l;
}
static std::vector<ViewCamera> parse_cameras(const py::handle &cs) {
  std::vector<ViewCamera> out;
  for (auto c : py::reinterpret_borrow<py::iterable>(cs))
    out.push_back(camera_from_dict(py::cast<py::dict>(c)));
  return out;
}
PYBIND11_MODULE(_ftgspp_viewer, m) {
  py::class_<ViewerApp>(m, "Desktop")
      .def(py::init<const std::string &, int, int, int, bool>(),
           py::arg("shaders"), py::arg("device") = 0, py::arg("width") = 1280,
           py::arg("height") = 800, py::arg("vsync") = true)
      .def("load_scene",
           [](ViewerApp &a, py::dict model, py::dict meta) {
             auto arrays = model_from_dict(model);
             float duration = py::cast<float>(model["max_duration"]);
             ViewerScene s;
             s.initial =
                 camera_from_dict(py::cast<py::dict>(meta["initial_camera"]));
             s.cameras = parse_cameras(meta["cameras"]);
             s.sample_times =
                 py::cast<std::vector<float>>(meta["sample_times"]);
             s.center = vec3(meta["center"]);
             s.radius = py::cast<float>(meta["radius"]);
             s.start = py::cast<float>(meta["time_start"]);
             s.end = py::cast<float>(meta["time_end"]);
             s.fps = py::cast<float>(meta["fps"]);
             s.sh_degree = py::cast<int>(meta["sh_degree"]);
             s.threshold = py::cast<float>(meta["marginal_threshold"]);
             s.label = py::cast<std::string>(meta["label"]);
             if (meta.contains("world_up"))
               s.world_up = vec3(meta["world_up"]);
             py::gil_scoped_release release;
             a.load(arrays, duration, s);
           })
      .def("frame",
           [](ViewerApp &a) {
             ViewerEvents ev;
             {
               py::gil_scoped_release release;
               ev = a.frame();
             }
             py::dict d;
             d["open"] = ev.open;
             d["screenshot"] = ev.screenshot;
             d["save_session"] = ev.save_session;
             d["save_path"] = ev.save_path;
             d["load_path"] = ev.load_path;
             d["export_path"] = ev.export_path;
             return d;
           })
      .def("close", &ViewerApp::close, py::call_guard<py::gil_scoped_release>())
      .def("should_close", &ViewerApp::should_close)
      .def("resize_window", &ViewerApp::resize_window)
      .def_readwrite("status", &ViewerApp::status)
      .def("state",
           [](ViewerApp &a) {
             py::dict d;
             d["camera"] = camera_to_dict(a.camera);
             d["time"] = a.time;
             d["effective_time"] = a.effective_time();
             d["playing"] = a.playing;
             d["loop"] = a.loop;
             d["snap"] = a.snap;
             d["speed"] = a.speed;
             d["interactive"] = a.interactive;
             d["target_fps"] = a.target_fps;
             d["transparent"] = a.transparent;
             d["mode"] = a.display.mode;
             d["checker"] = a.display.checker;
             d["background"] = std::vector<float>{a.display.background.x,
                                                  a.display.background.y,
                                                  a.display.background.z};
             d["exposure"] = a.display.exposure;
             d["sh_degree"] = a.scene.sh_degree;
             d["marginal_threshold"] = a.scene.threshold;
             d["bookmarks"] = cameras(a.bookmarks);
             d["path_times"] = a.path_times;
             d["world_up"] = std::vector<float>{
                 a.scene.world_up.x, a.scene.world_up.y, a.scene.world_up.z};
             d["lock_camera"] = a.lock_camera;
             d["selected_camera"] = a.selected_camera;
             d["axes"] = a.axes;
             d["bounds"] = a.bounds;
             d["frustums"] = a.frustums;
             d["pixel_readbacks"] = a.renderer().pixel_readbacks;
             d["render_scale"] = a.render_scale;
             d["render_ms"] = a.renderer().render_ms;
             d["display_ms"] = a.display_ms();
             d["frames"] = a.frame_count;
             d["rendered_frames"] = a.renderer().rendered_frames;
             d["vram_bytes"] = a.renderer().buffers.getTotalAllocSize();
             return d;
           })
      .def("restore_state",
           [](ViewerApp &a, py::dict d) {
             if (d.contains("camera"))
               a.set_camera(camera_from_dict(py::cast<py::dict>(d["camera"])));
             if (d.contains("time"))
               a.time = py::cast<float>(d["time"]);
             if (d.contains("playing"))
               a.playing = py::cast<bool>(d["playing"]);
             if (d.contains("loop"))
               a.loop = py::cast<bool>(d["loop"]);
             if (d.contains("snap"))
               a.snap = py::cast<bool>(d["snap"]);
             if (d.contains("speed"))
               a.speed = py::cast<float>(d["speed"]);
             if (d.contains("interactive"))
               a.interactive = py::cast<bool>(d["interactive"]);
             if (d.contains("target_fps"))
               a.target_fps = py::cast<float>(d["target_fps"]);
             if (d.contains("transparent"))
               a.transparent = py::cast<bool>(d["transparent"]);
             if (d.contains("mode"))
               a.display.mode = py::cast<int>(d["mode"]);
             if (d.contains("checker"))
               a.display.checker = py::cast<bool>(d["checker"]);
             if (d.contains("exposure"))
               a.display.exposure = py::cast<float>(d["exposure"]);
             if (d.contains("sh_degree"))
               a.scene.sh_degree = py::cast<int>(d["sh_degree"]);
             if (d.contains("marginal_threshold"))
               a.scene.threshold = py::cast<float>(d["marginal_threshold"]);
             // Old sessions may have lock_camera=false; aspect is now fixed.
             if (d.contains("selected_camera"))
               a.selected_camera = py::cast<int>(d["selected_camera"]);
             if (d.contains("axes"))
               a.axes = py::cast<bool>(d["axes"]);
             if (d.contains("bounds"))
               a.bounds = py::cast<bool>(d["bounds"]);
             if (d.contains("frustums"))
               a.frustums = py::cast<bool>(d["frustums"]);
             if (d.contains("background"))
               a.display.background = vec3(d["background"]);
             if (d.contains("world_up"))
               a.scene.world_up = vec3(d["world_up"]);
             if (d.contains("bookmarks")) {
               auto cs = parse_cameras(d["bookmarks"]);
               auto ts = py::cast<std::vector<float>>(d["path_times"]);
               if (cs.size() != ts.size())
                 throw std::invalid_argument(
                     "Path time count does not match cameras");
               a.bookmarks = std::move(cs);
               a.path_times = std::move(ts);
               a.selected_bookmark = -1;
             }
             if (a.selected_camera < 0 ||
                 size_t(a.selected_camera) >= a.scene.cameras.size())
               a.selected_camera = -1;
             for (auto key :
                  {"camera", "time", "sh_degree", "marginal_threshold", "snap",
                   "lock_camera", "interactive"})
               if (d.contains(key)) {
                 a.invalidate();
                 break;
               }
           })
      .def(
          "output_camera",
          [](ViewerApp &a, int w, int h) {
            return camera_to_dict(a.output_camera(w, h));
          },
          py::arg("width") = 0, py::arg("height") = 0)
      .def(
          "capture",
          [](ViewerApp &a, int w, int h) {
            std::vector<float> data;
            {
              py::gil_scoped_release release;
              data = a.capture(w, h);
            }
            auto &u = a.renderer().uniforms;
            py::array_t<float> out({py::ssize_t(u.image_height),
                                    py::ssize_t(u.image_width),
                                    py::ssize_t(4)});
            std::memcpy(out.mutable_data(), data.data(),
                        data.size() * sizeof(float));
            return out;
          },
          py::arg("width") = 0, py::arg("height") = 0);
}
} // namespace ftgspp
