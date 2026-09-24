#pragma once
#include "viewer/presenter.h"
#include <memory>
#include <stdexcept>
namespace ftgspp {
struct ViewerScene {
  ViewCamera initial;
  std::vector<ViewCamera> cameras;
  std::vector<float> sample_times;
  glm::vec3 center{0}, world_up{0, -1, 0};
  float radius = 1, start = 0, end = 1, fps = 30;
  int sh_degree = 3;
  float threshold = 0.05f;
  std::string label;
};
struct ViewerEvents {
  bool screenshot = false, save_session = false, save_path = false,
       load_path = false, export_path = false;
  std::string open;
};
class ViewerApp {
public:
  ViewerApp(const std::string &, int, int, int, bool);
  ~ViewerApp();
  void load(const ModelArrays &, float, const ViewerScene &);
  ViewerEvents frame();
  void close();
  bool should_close() const;
  void resize_window(int w, int h);
  void set_camera(const ViewCamera &c);
  float effective_time() const;
  void invalidate() { dirty = true; }
  std::vector<float> capture(int width = 0, int height = 0);
  ViewCamera output_camera(int w = 0, int h = 0) const;
  ViewRenderer &renderer() {
    if (!render)
      throw std::runtime_error("Viewer is closed");
    return *render;
  }
  ViewerScene scene;
  ViewCamera camera;
  std::vector<ViewCamera> bookmarks;
  std::vector<float> path_times;
  DisplaySettings display;
  std::string status;
  bool playing = false, loop = true, snap = false, interactive = false,
       transparent = false;
  float time = 0, speed = 1, target_fps = 30;
  int selected_camera = -1, selected_bookmark = -1;
  // Kept in serialized state for compatibility; navigation never unlocks aspect.
  static constexpr bool lock_camera = true;
  bool axes = false, bounds = false, frustums = false;
  bool path_playing = false;
  float path_position = 0;
  uint64_t frame_count = 0;
  float render_scale = 1;
  double present_fps = 0;
  double display_ms() const { return present ? present->display_ms : 0; }

private:
  GLFWwindow *window = nullptr;
  std::unique_ptr<ViewRenderer> render;
  std::unique_ptr<Presenter> present;
  bool imgui_context = false, loaded = false, dirty = true, panel = true,
       closed = false, fullscreen = false, fly = false;
  int old_x = 0, old_y = 0, old_w = 1280, old_h = 720, quality_level = 0,
      slow_frames = 0, fast_frames = 0;
  double previous = 0, last_motion = 0, last_quality = 0, last_frame = 0;
  glm::vec3 pivot{0};
  Viewport viewport;
  void fit();
  void roll_camera(float radians);
  void input(float dt);
  void draw_ui(ViewerEvents &);
  void sample_path(float t);
  void draw_overlays();
  void update_quality(double now, bool rendered);
};
} // namespace ftgspp
