#include "viewer/viewer_app.h"
#include <algorithm>
#include <backends/imgui_impl_glfw.h>
#include <cmath>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <stdexcept>
namespace ftgspp {
static glm::mat4 cv_pose(glm::vec3 eye, glm::vec3 right, glm::vec3 down,
                         glm::vec3 front) {
  front = glm::normalize(front);
  right = glm::normalize(right - glm::dot(right, front) * front);
  glm::vec3 rebuilt_down = glm::normalize(glm::cross(front, right));
  if (glm::dot(rebuilt_down, down) < 0) {
    right = -right;
    rebuilt_down = -rebuilt_down;
  }
  glm::mat4 c2w(1);
  c2w[0] = glm::vec4(right, 0);
  c2w[1] = glm::vec4(rebuilt_down, 0);
  c2w[2] = glm::vec4(front, 0);
  c2w[3] = glm::vec4(eye, 1);
  return glm::inverse(c2w);
}
ViewerApp::ViewerApp(const std::string &shaders, int device, int w, int h,
                     bool vsync) {
  if (ImGui::GetCurrentContext())
    throw std::runtime_error("Only one desktop viewer may be open per process");
  try {
    if (!glfwInit())
      throw std::runtime_error(
          "GLFW could not open a display. Use a desktop/X11 display, or ftgspp "
          "render for headless export.");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(w, h, "VK-FTGSPP Viewer", nullptr, nullptr);
    if (window)
      glfwSetWindowSizeLimits(window, 640, 360, GLFW_DONT_CARE, GLFW_DONT_CARE);
    if (!window)
      throw std::runtime_error("Could not create viewer window");
    uint32_t count = 0;
    const char **exts = glfwGetRequiredInstanceExtensions(&count);
    if (!exts || !count)
      throw std::runtime_error("Display has no Vulkan surface support");
    DeviceInitOptions options;
    options.instance_extensions.assign(exts, exts + count);
    options.create_surface = [this](VkInstance i) {
      VkSurfaceKHR s = VK_NULL_HANDLE;
      if (glfwCreateWindowSurface(i, window, nullptr, &s) != VK_SUCCESS)
        throw std::runtime_error("Could not create Vulkan display surface");
      return s;
    };
    render = std::make_unique<ViewRenderer>(shaders, device, options);
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    imgui_context = true;
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::StyleColorsDark();
    auto &style = ImGui::GetStyle();
    style.WindowRounding = 0;
    style.FrameRounding = 4;
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.055f, 0.065f, 0.085f, 1);
    present = std::make_unique<Presenter>(window, *render, shaders, vsync);
    previous = glfwGetTime();
  } catch (...) {
    close();
    throw;
  }
}
ViewerApp::~ViewerApp() {
  try {
    close();
  } catch (...) {
  }
}
void ViewerApp::close() {
  if (closed)
    return;
  present.reset();
  render.reset();
  if (imgui_context) {
    ImGui::DestroyContext();
    imgui_context = false;
  }
  if (window) {
    glfwDestroyWindow(window);
    window = nullptr;
  }
  glfwTerminate();
  closed = true;
}
bool ViewerApp::should_close() const {
  return closed || glfwWindowShouldClose(window);
}
void ViewerApp::resize_window(int w, int h) {
  if (closed)
    throw std::runtime_error("Viewer is closed");
  if (w < 640 || h < 360)
    throw std::invalid_argument("Window must be at least 640x360");
  glfwSetWindowSize(window, w, h);
}
void ViewerApp::load(const ModelArrays &model, float duration,
                     const ViewerScene &data) {
  if (closed)
    throw std::runtime_error("Viewer is closed");
  present->wait();
  loaded = false;
  playing = false;
  path_playing = false;
  render->valid = false;
  render->upload(model, duration);
  scene = data;
  camera = data.initial;
  time = data.start;
  pivot = data.center;
  playing = false;
  selected_camera = data.cameras.empty() ? -1 : 0;
  selected_bookmark = -1;
  bookmarks.clear();
  path_times.clear();
  quality_level = 0;
  render_scale = 1;
  dirty = true;
  loaded = true;
  status = "Loaded " + data.label;
}
void ViewerApp::set_camera(const ViewCamera &c) {
  if (closed)
    throw std::runtime_error("Viewer is closed");
  path_playing = false;
  camera = c;
  time = c.timestamp;
  selected_camera = -1;
  dirty = true;
}
void ViewerApp::fit() {
  glm::mat4 c2w = glm::inverse(camera.w2c);
  glm::vec3 right(c2w[0]), down(c2w[1]), front(c2w[2]);
  front = glm::normalize(front);
  float half_fov = std::atan(0.5f * camera.height / camera.fy);
  float distance = std::max(scene.radius, 0.01f) /
                   std::max(std::sin(half_fov), 0.05f) * 1.2f;
  pivot = scene.center;
  camera.w2c = cv_pose(pivot - front * distance, right, down, front);
  dirty = true;
}
void ViewerApp::roll_camera(float radians) {
  glm::mat4 c2w = glm::inverse(camera.w2c);
  glm::vec3 eye(c2w[3]), right(c2w[0]), down(c2w[1]), front(c2w[2]);
  glm::quat rotation = glm::angleAxis(radians, glm::normalize(front));
  camera.w2c = cv_pose(eye, rotation * right, rotation * down, front);
  path_playing = false;
  selected_camera = -1;
  dirty = true;
  last_motion = glfwGetTime();
}
ViewCamera ViewerApp::output_camera(int w, int h) const {
  ViewCamera c = camera;
  uint32_t width = w > 0 ? w : std::max(1, int(viewport.w)),
           height = h > 0 ? h : std::max(1, int(viewport.h));
  if (w > 0 && h <= 0)
    height = std::max(
        1u, (uint32_t)std::lround(width * float(camera.height) / camera.width));
  if (h > 0 && w <= 0)
    width = std::max(
        1u, (uint32_t)std::lround(height * float(camera.width) / camera.height));
  const double scale = std::min(double(width) / camera.width,
                                double(height) / camera.height);
  width = std::max(1u, uint32_t(std::lround(camera.width * scale)));
  height = std::max(1u, uint32_t(std::lround(camera.height * scale)));
  float sx = float(width) / camera.width, sy = float(height) / camera.height;
  c.width = width;
  c.height = height;
  c.fx *= sx;
  c.cx *= sx;
  c.fy *= sy;
  c.cy *= sy;
  c.timestamp = effective_time();
  return c;
}
void ViewerApp::input(float dt) {
  auto &io = ImGui::GetIO();
  if (!glfwGetWindowAttrib(window, GLFW_FOCUSED))
    return;
  float sx = io.DisplayFramebufferScale.x, sy = io.DisplayFramebufferScale.y;
  bool inside = io.MousePos.x * sx >= viewport.x &&
                io.MousePos.x * sx < viewport.x + viewport.w &&
                io.MousePos.y * sy >= viewport.y &&
                io.MousePos.y * sy < viewport.y + viewport.h;
  bool changed = false;
  glm::mat4 c2w = glm::inverse(camera.w2c);
  glm::vec3 eye(c2w[3]), right(c2w[0]), down(c2w[1]), front(c2w[2]);
  right = glm::normalize(right);
  down = glm::normalize(down);
  front = glm::normalize(front);
  float distance = std::max(glm::length(eye - pivot), scene.radius * 0.05f);
  if (inside && !io.WantCaptureMouse) {
    float dx = io.MouseDelta.x, dy = io.MouseDelta.y;
    if ((io.MouseDown[0] || io.MouseDown[1] || io.MouseDown[2]) && (dx || dy)) {
      if (io.MouseDown[0] && io.KeyCtrl) {
        glm::quat rotation = glm::angleAxis(dx * 0.004f, front);
        right = rotation * right;
        down = rotation * down;
      } else if (io.MouseDown[2] || (io.MouseDown[0] && io.KeyShift)) {
        glm::vec3 shift =
            (-dx * right - dy * down) * (distance / std::max(camera.fy, 1.0f));
        eye += shift;
        pivot += shift;
      } else if (io.MouseDown[1] || fly) {
        glm::quat yaw = glm::angleAxis(-dx * 0.004f, scene.world_up);
        right = yaw * right;
        down = yaw * down;
        front = yaw * front;
        glm::quat pitch = glm::angleAxis(-dy * 0.004f, right);
        right = pitch * right;
        down = pitch * down;
        front = pitch * front;
        pivot = eye + front * distance;
      } else {
        glm::vec3 offset = eye - pivot;
        glm::quat yaw = glm::angleAxis(-dx * 0.004f, scene.world_up);
        offset = yaw * offset;
        right = yaw * right;
        down = yaw * down;
        front = yaw * front;
        glm::quat pitch = glm::angleAxis(-dy * 0.004f, right);
        offset = pitch * offset;
        right = pitch * right;
        down = pitch * down;
        front = pitch * front;
        eye = pivot + offset;
      }
      camera.w2c = cv_pose(eye, right, down, front);
      changed = true;
    }
    if (io.MouseWheel != 0) {
      eye = pivot + (eye - pivot) * std::exp(-io.MouseWheel * 0.12f);
      camera.w2c = cv_pose(eye, right, down, front);
      changed = true;
    }
    if ((io.MouseDown[1] || fly) && !io.WantCaptureKeyboard) {
      glm::vec3 delta(0);
      if (ImGui::IsKeyDown(ImGuiKey_W))
        delta += front;
      if (ImGui::IsKeyDown(ImGuiKey_S))
        delta -= front;
      if (ImGui::IsKeyDown(ImGuiKey_A))
        delta -= right;
      if (ImGui::IsKeyDown(ImGuiKey_D))
        delta += right;
      if (ImGui::IsKeyDown(ImGuiKey_Q))
        delta -= scene.world_up;
      if (ImGui::IsKeyDown(ImGuiKey_E))
        delta += scene.world_up;
      if (glm::length(delta) > 0) {
        delta *= dt * std::max(scene.radius, 0.1f) * (io.KeyShift ? 4 : 1);
        eye += delta;
        pivot += delta;
        camera.w2c = cv_pose(eye, right, down, front);
        changed = true;
      }
    }
  }
  if (!io.WantCaptureKeyboard) {
    if (ImGui::IsKeyPressed(ImGuiKey_Space))
      playing = !playing;
    if (ImGui::IsKeyPressed(ImGuiKey_R)) {
      camera = scene.initial;
      pivot = scene.center;
      changed = true;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_F)) {
      fit();
      changed = true;
    }
    int step = ImGui::IsKeyPressed(ImGuiKey_RightArrow)  ? 1
               : ImGui::IsKeyPressed(ImGuiKey_LeftArrow) ? -1
                                                         : 0;
    if (step) {
      playing = false;
      if (!scene.sample_times.empty()) {
        auto it =
            step > 0 ? std::upper_bound(scene.sample_times.begin(),
                                        scene.sample_times.end(), time + 1e-6f)
                     : std::lower_bound(scene.sample_times.begin(),
                                        scene.sample_times.end(), time - 1e-6f);
        if (step < 0 && it != scene.sample_times.begin())
          --it;
        if (it != scene.sample_times.end())
          time = *it;
      } else
        time = std::clamp(time + step / scene.fps, scene.start, scene.end);
      dirty = true;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_F11)) {
      fullscreen = !fullscreen;
      if (fullscreen) {
        glfwGetWindowPos(window, &old_x, &old_y);
        glfwGetWindowSize(window, &old_w, &old_h);
        auto m = glfwGetPrimaryMonitor();
        auto v = glfwGetVideoMode(m);
        glfwSetWindowMonitor(window, m, 0, 0, v->width, v->height,
                             v->refreshRate);
      } else
        glfwSetWindowMonitor(window, nullptr, old_x, old_y, old_w, old_h, 0);
    }
  }
  if (changed) {
    path_playing = false;
    selected_camera = -1;
    dirty = true;
    last_motion = glfwGetTime();
  }
}
void ViewerApp::draw_ui(ViewerEvents &ev) {
  auto &io = ImGui::GetIO();
  float sw = panel ? 300 : 44;
  sw = std::min(sw, io.DisplaySize.x * 0.45f);
  ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                           ImGuiWindowFlags_NoCollapse |
                           ImGuiWindowFlags_NoSavedSettings |
                           ImGuiWindowFlags_NoTitleBar;
  ImGui::SetNextWindowPos({0, 0});
  ImGui::SetNextWindowSize({sw, std::max(1.f, io.DisplaySize.y - 76)});
  ImGui::Begin("Controls", nullptr, flags);
  if (ImGui::Button(panel ? "Hide panel" : " > "))
    panel = !panel;
  if (panel) {
    ImGui::TextUnformatted("VK-FTGSPP Viewer");
    ImGui::Separator();
    static char source[1024] = "";
    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextWithHint("##source", "Run directory or .npz", source,
                             sizeof(source));
    if (ImGui::Button("Open"))
      ev.open = source;
    ImGui::SameLine();
    if (ImGui::Button("Reload"))
      ev.open = "@reload";
    ImGui::TextWrapped("%s", scene.label.c_str());
    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
      if (ImGui::Button("Fit"))
        fit();
      ImGui::SameLine();
      if (ImGui::Button("Reset")) {
        camera = scene.initial;
        pivot = scene.center;
        dirty = true;
      }
      if (ImGui::Button("Roll -90"))
        roll_camera(glm::radians(-90.0f));
      ImGui::SameLine();
      if (ImGui::Button("Flip 180"))
        roll_camera(glm::radians(180.0f));
      ImGui::SameLine();
      if (ImGui::Button("Roll +90"))
        roll_camera(glm::radians(90.0f));
      ImGui::Checkbox("Fly (WASD / Q E)", &fly);
      ImGui::Checkbox("Axes", &axes);
      ImGui::SameLine();
      ImGui::Checkbox("Bounds", &bounds);
      ImGui::Checkbox("Capture frustums", &frustums);
      int up = scene.world_up.z > 0.5 ? 2 : scene.world_up.y > 0.5 ? 1 : 0;
      if (ImGui::Combo("World up", &up, "-Y\0+Y\0+Z\0")) {
        scene.world_up = up == 2   ? glm::vec3(0, 0, 1)
                         : up == 1 ? glm::vec3(0, 1, 0)
                                   : glm::vec3(0, -1, 0);
      }
      if (ImGui::BeginCombo("Capture camera",
                            selected_camera >= 0
                                ? scene.cameras[selected_camera].name.c_str()
                                : "Free")) {
        for (size_t i = 0; i < scene.cameras.size(); ++i)
          if (ImGui::Selectable(scene.cameras[i].name.c_str(),
                                int(i) == selected_camera)) {
            camera = scene.cameras[i];
            selected_camera = i;
            dirty = true;
          }
        ImGui::EndCombo();
      }
      ImGui::Text("Original aspect: %u:%u (fixed)", camera.width, camera.height);
      if (selected_camera >= 0) {
        if (ImGui::Button("Use capture time")) {
          time = scene.cameras[selected_camera].timestamp;
          playing = false;
          dirty = true;
        }
      }
      float fov = glm::degrees(2 * std::atan(camera.height / (2 * camera.fy)));
      if (ImGui::SliderFloat("Vertical FOV", &fov, 10, 110)) {
        float fy = camera.height / (2 * std::tan(glm::radians(fov) * .5f));
        camera.fx *= fy / camera.fy;
        camera.fy = fy;
        dirty = true;
      }
    }
    if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Combo("Output", &display.mode, "RGB\0Alpha\0Transmittance\0");
      int bg = display.checker                         ? 2
               : glm::length(display.background) > 0.1 ? 1
                                                       : 0;
      if (ImGui::Combo("Background", &bg, "Black\0White\0Checker\0")) {
        display.checker = bg == 2;
        display.background = glm::vec3(bg == 1 ? 1 : 0);
      }
      ImGui::ColorEdit3("Custom color", &display.background.x);
      if (ImGui::IsItemEdited())
        display.checker = false;
      ImGui::SliderFloat("Exposure (EV)", &display.exposure, -4, 4);
      if (ImGui::SliderInt("SH degree", &scene.sh_degree, 0, 3))
        dirty = true;
      if (ImGui::Checkbox("Interactive quality", &interactive)) {
        quality_level = 0;
        render_scale = 1;
        dirty = true;
      }
      if (interactive)
        ImGui::SliderFloat("Target FPS", &target_fps, 15, 90, "%.0f");
      ImGui::Text("Render: %u x %u (%.0f%%)", render->uniforms.image_width,
                  render->uniforms.image_height, render_scale * 100);
    }
    if (ImGui::CollapsingHeader("Capture and paths",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Checkbox("Transparent PNG", &transparent);
      if (ImGui::Button("Screenshot"))
        ev.screenshot = true;
      ImGui::SameLine();
      if (ImGui::Button("Save session"))
        ev.save_session = true;
      if (ImGui::Button("Add keyframe")) {
        auto c = camera;
        c.timestamp = time;
        bookmarks.push_back(c);
        path_times.push_back(path_times.empty() ? 0 : path_times.back() + 1);
        selected_bookmark = bookmarks.size() - 1;
      }
      if (!bookmarks.empty()) {
        if (ImGui::Button(path_playing ? "Stop preview" : "Preview path")) {
          path_playing = !path_playing;
          playing = false;
          if (path_playing)
            path_position = path_times.front();
        }
        if (bookmarks.size() > 1 &&
            ImGui::SliderFloat("Path seconds", &path_position,
                               path_times.front(), path_times.back())) {
          path_playing = false;
          sample_path(path_position);
        }
        for (size_t i = 0; i < bookmarks.size(); ++i) {
          std::string label = "Key " + std::to_string(i + 1) +
                              "  t=" + std::to_string(bookmarks[i].timestamp);
          if (ImGui::Selectable(label.c_str(), selected_bookmark == int(i))) {
            selected_bookmark = i;
            camera = bookmarks[i];
            time = camera.timestamp;
            playing = false;
            dirty = true;
          }
        }
        if (selected_bookmark >= 0) {
          float seconds = path_times[selected_bookmark];
          if (ImGui::InputFloat("Key path seconds", &seconds, .1f, 1.f,
                                "%.3f")) {
            float lo = selected_bookmark
                           ? path_times[selected_bookmark - 1] + .001f
                           : 0.f;
            float hi = size_t(selected_bookmark + 1) < path_times.size()
                           ? path_times[selected_bookmark + 1] - .001f
                           : 1e6f;
            path_times[selected_bookmark] = std::clamp(seconds, lo, hi);
          }
          if (ImGui::Button("Replace")) {
            bookmarks[selected_bookmark] = camera;
            bookmarks[selected_bookmark].timestamp = time;
          }
          ImGui::SameLine();
          if (ImGui::Button("Delete")) {
            bookmarks.erase(bookmarks.begin() + selected_bookmark);
            path_times.erase(path_times.begin() + selected_bookmark);
            selected_bookmark = -1;
          }
        }
        if (ImGui::Button("Save path"))
          ev.save_path = true;
        ImGui::SameLine();
        if (ImGui::Button("Export path"))
          ev.export_path = true;
      }
      if (ImGui::Button("Load path"))
        ev.load_path = true;
    }
    ImGui::Separator();
    ImGui::Text("%.1f FPS", present_fps);
    ImGui::Text("Forward: %.2f ms", render->render_ms);
    ImGui::Text("Display + UI: %.2f ms", display_ms());
    ImGui::Text("%zu Gaussians", render->buffers.num_splats);
    ImGui::Text("Buffers: %.1f MiB",
                render->buffers.getTotalAllocSize() / 1048576.0);
    ImGui::TextWrapped("%s", status.c_str());
    if (ImGui::CollapsingHeader("Controls"))
      ImGui::TextWrapped(
          "Left drag: orbit. Shift+left / middle: pan. Right drag: look. "
          "Ctrl+left: roll. Wheel: dolly. Right+WASD/QE: fly. Space: play. "
          "Arrows: sample. R: reset. F: fit. Ctrl+S: screenshot. F11: "
          "fullscreen.");
  }
  ImGui::End();
  ImGui::SetNextWindowPos({0, std::max(0.f, io.DisplaySize.y - 76)});
  ImGui::SetNextWindowSize({io.DisplaySize.x, 76});
  ImGui::Begin("Timeline", nullptr, flags);
  if (ImGui::Button(playing ? "Pause" : "Play"))
    playing = !playing;
  ImGui::SameLine();
  ImGui::Checkbox("Loop", &loop);
  ImGui::SameLine();
  if (ImGui::Checkbox("Snap", &snap))
    dirty = true;
  ImGui::SameLine();
  ImGui::SetNextItemWidth(90);
  ImGui::SliderFloat("Speed", &speed, 0.1f, 4, "%.2fx");
  ImGui::SetNextItemWidth(std::max(80.f, io.DisplaySize.x - 210));
  if (ImGui::SliderFloat("Seconds", &time, scene.start, scene.end, "%.4f")) {
    playing = false;
    dirty = true;
  }
  ImGui::End();
  if (!io.WantCaptureKeyboard && io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_S))
    ev.screenshot = true;
  float x = sw * io.DisplayFramebufferScale.x,
        w = (io.DisplaySize.x - sw) * io.DisplayFramebufferScale.x,
        h = (io.DisplaySize.y - 76) * io.DisplayFramebufferScale.y;
  Viewport next{x, 0, std::max(1.f, w), std::max(1.f, h)};
  float aspect = float(camera.width) / camera.height;
  if (next.w / next.h > aspect) {
    float target = next.h * aspect;
    next.x += (next.w - target) / 2;
    next.w = target;
  } else {
    float target = next.w / aspect;
    next.y += (next.h - target) / 2;
    next.h = target;
  }
  next.x = std::floor(next.x);
  next.y = std::floor(next.y);
  next.w = std::max(1.f, std::floor(next.w));
  next.h = std::max(1.f, std::floor(next.h));
  if (next.w != viewport.w || next.h != viewport.h)
    dirty = true;
  viewport = next;
}
void ViewerApp::update_quality(double now, bool rendered) {
  if (!interactive) {
    if (quality_level) {
      quality_level = 0;
      render_scale = 1;
      dirty = true;
    }
    return;
  }
  if (!playing && !path_playing && now - last_motion > .15) {
    if (quality_level) {
      quality_level = 0;
      render_scale = 1;
      dirty = true;
    }
    return;
  }
  if (!rendered)
    return;
  float budget = 1000.f / std::max(target_fps, 1.f) * .84f;
  slow_frames = render->render_ms > budget ? slow_frames + 1 : 0;
  fast_frames = render->render_ms < budget * .7 ? fast_frames + 1 : 0;
  if (now - last_quality > .5) {
    int old = quality_level;
    if (slow_frames >= 8)
      quality_level = std::min(2, quality_level + 1);
    else if (fast_frames >= 30)
      quality_level = std::max(0, quality_level - 1);
    if (old != quality_level) {
      render_scale = quality_level == 0 ? 1 : quality_level == 1 ? .85f : .75f;
      dirty = true;
      slow_frames = fast_frames = 0;
      last_quality = now;
    }
  }
}
ViewerEvents ViewerApp::frame() {
  if (closed)
    throw std::runtime_error("Viewer is closed");
  ViewerEvents ev;
  if (!playing && !path_playing && !dirty)
    glfwWaitEventsTimeout(.03);
  else
    glfwPollEvents();
  int w, h;
  glfwGetFramebufferSize(window, &w, &h);
  double now = glfwGetTime();
  float dt = std::min(now - previous, .1);
  double elapsed = now - previous;
  previous = now;
  if (w == 0 || h == 0) {
    glfwWaitEventsTimeout(.1);
    return ev;
  }
  present->wait();
  present->resize(w, h);
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  if (playing && scene.end > scene.start) {
    const float previous_timestamp = effective_time();
    time += elapsed * speed;
    if (time > scene.end) {
      if (loop)
        time = scene.start +
               std::fmod(time - scene.start, scene.end - scene.start);
      else {
        time = scene.end;
        playing = false;
      }
    }
    // Several UI frames can snap to the same captured timestamp.
    dirty = dirty || effective_time() != previous_timestamp;
  }
  if (path_playing && !bookmarks.empty()) {
    path_position += elapsed * speed;
    if (path_position > path_times.back()) {
      if (loop && path_times.back() > path_times.front())
        path_position = path_times.front() +
                        std::fmod(path_position - path_times.front(),
                                  path_times.back() - path_times.front());
      else {
        path_position = path_times.back();
        path_playing = false;
      }
    }
    sample_path(path_position);
  }
  draw_ui(ev);
  input(dt);
  float effective_timestamp = effective_time();
  bool rendered = false;
  if (loaded && dirty) {
    auto c = output_camera();
    const float original_width = c.width, original_height = c.height;
    c.width = std::max(1u, (uint32_t)std::lround(c.width * render_scale));
    c.height = std::max(1u, (uint32_t)std::lround(c.height * render_scale));
    float sx = float(c.width) / original_width, sy = float(c.height) / original_height;
    c.fx *= sx;
    c.cx *= sx;
    c.fy *= sy;
    c.cy *= sy;
    try {
      render->render(c, effective_timestamp, scene.sh_degree, scene.threshold);
      rendered = true;
    } catch (const std::exception &e) {
      status = e.what();
      playing = false;
    }
    dirty = false;
  }
  draw_overlays();
  if (axes) {
    ImVec2 origin((viewport.x + 60) / ImGui::GetIO().DisplayFramebufferScale.x,
                  (viewport.y + viewport.h - 60) /
                      ImGui::GetIO().DisplayFramebufferScale.y);
    auto *list = ImGui::GetForegroundDrawList();
    for (int i = 0; i < 3; ++i) {
      glm::vec3 axis(0);
      axis[i] = 1;
      glm::vec3 view = glm::mat3(camera.w2c) * axis;
      list->AddLine(origin, {origin.x + view.x * 28, origin.y + view.y * 28},
                    i == 0   ? IM_COL32(230, 75, 75, 255)
                    : i == 1 ? IM_COL32(75, 230, 100, 255)
                             : IM_COL32(75, 150, 255, 255),
                    2);
    }
  }
  ImGui::Render();
  present->draw(viewport, display);
  update_quality(now, rendered);
  double current = glfwGetTime();
  if (last_frame > 0) {
    double fps = 1 / std::max(current - last_frame, 1e-6);
    present_fps = present_fps ? present_fps * .9 + fps * .1 : fps;
  }
  last_frame = current;
  ++frame_count;
  return ev;
}
float ViewerApp::effective_time() const {
  if (!snap || scene.sample_times.empty())
    return time;
  auto it = std::lower_bound(scene.sample_times.begin(),
                             scene.sample_times.end(), time);
  if (it == scene.sample_times.end())
    return scene.sample_times.back();
  if (it == scene.sample_times.begin())
    return *it;
  return std::abs(*it - time) < std::abs(*(it - 1) - time) ? *it : *(it - 1);
}
void ViewerApp::sample_path(float t) {
  if (bookmarks.empty())
    return;
  size_t i = 0;
  while (i + 1 < path_times.size() && path_times[i + 1] < t)
    ++i;
  if (i + 1 == bookmarks.size()) {
    camera = bookmarks.back();
    time = camera.timestamp;
  } else {
    float f = std::clamp((t - path_times[i]) /
                             std::max(.001f, path_times[i + 1] - path_times[i]),
                         0.f, 1.f);
    const auto &a = bookmarks[i];
    const auto &b = bookmarks[i + 1];
    camera = a;
    glm::mat4 pa = glm::inverse(a.w2c), pb = glm::inverse(b.w2c);
    glm::quat q = glm::slerp(glm::quat_cast(glm::mat3(pa)),
                             glm::quat_cast(glm::mat3(pb)), f);
    glm::mat4 pose = glm::mat4_cast(q);
    pose[3] = glm::mix(pa[3], pb[3], f);
    camera.w2c = glm::inverse(pose);
    camera.fx = glm::mix(a.fx, b.fx * float(a.width) / b.width, f);
    camera.fy = glm::mix(a.fy, b.fy * float(a.height) / b.height, f);
    camera.cx = glm::mix(a.cx, b.cx * float(a.width) / b.width, f);
    camera.cy = glm::mix(a.cy, b.cy * float(a.height) / b.height, f);
    time = glm::mix(a.timestamp, b.timestamp, f);
    camera.timestamp = time;
  }
  selected_camera = -1;
  dirty = true;
}
void ViewerApp::draw_overlays() {
  if (!bounds && !frustums)
    return;
  auto c = output_camera();
  auto scale = ImGui::GetIO().DisplayFramebufferScale;
  auto *draw = ImGui::GetForegroundDrawList();
  draw->PushClipRect({viewport.x / scale.x, viewport.y / scale.y},
                     {(viewport.x + viewport.w) / scale.x,
                      (viewport.y + viewport.h) / scale.y});
  auto line = [&](glm::vec3 a, glm::vec3 b, ImU32 color) {
    auto project = [&](glm::vec3 p, ImVec2 &out) {
      glm::vec3 v = c.w2c * glm::vec4(p, 1);
      if (v.z < .01)
        return false;
      out = {(viewport.x + c.fx * v.x / v.z + c.cx) / scale.x,
             (viewport.y + c.fy * v.y / v.z + c.cy) / scale.y};
      return true;
    };
    ImVec2 x, y;
    if (project(a, x) && project(b, y))
      draw->AddLine(x, y, color, 1);
  };
  if (bounds) {
    glm::vec3 points[8];
    for (int i = 0; i < 8; ++i)
      points[i] = scene.center +
                  glm::vec3(i & 1 ? 1 : -1, i & 2 ? 1 : -1, i & 4 ? 1 : -1) *
                      scene.radius;
    for (int i = 0; i < 8; ++i)
      for (int bit = 1; bit <= 4; bit *= 2)
        if (!(i & bit))
          line(points[i], points[i | bit], IM_COL32(220, 180, 90, 160));
  }
  if (frustums) {
    for (const auto &cam : scene.cameras) {
      auto pose = glm::inverse(cam.w2c);
      glm::vec3 eye(pose[3]), p[4];
      float z = scene.radius * .15f;
      for (int i = 0; i < 4; ++i) {
        float u = i == 1 || i == 2 ? cam.width : 0, v = i >= 2 ? cam.height : 0;
        p[i] = pose * glm::vec4((u - cam.cx) / cam.fx * z,
                                (v - cam.cy) / cam.fy * z, z, 1);
        line(eye, p[i], IM_COL32(90, 180, 230, 150));
      }
      for (int i = 0; i < 4; ++i)
        line(p[i], p[(i + 1) % 4], IM_COL32(90, 180, 230, 150));
    }
  }
  draw->PopClipRect();
}
std::vector<float> ViewerApp::capture(int w, int h) {
  if (closed || !loaded)
    throw std::runtime_error("No loaded viewer model");
  present->wait();
  auto c = output_camera(w, h);
  render->render(c, effective_time(), scene.sh_degree, scene.threshold);
  auto data = render->readback();
  dirty = true;
  return data;
}
} // namespace ftgspp
