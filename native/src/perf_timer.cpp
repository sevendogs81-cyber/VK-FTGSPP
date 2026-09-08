#include "perf_timer.h"

#include <stack>

namespace PerfTimer {

// names of train stages as strings
#define _(name) #name ,
static std::string _TrainStageNames[TrainStage::END] = {
    PERF_TIMER_TRAIN_STAGES
};
#undef _


struct _TimerObject {
    size_t count = 0;
    double total_time = 0.0;
};

// Synchronous multi-device training submits one independent Vulkan pipeline
// from each Python worker thread.  Timestamp marker stacks must therefore not
// be process-global: interleaved ranks would corrupt one another's nesting and
// make update() observe a different marker count than its query pool.  The
// single-device path keeps identical behavior because all calls stay on one
// thread. Completed measurements, however, belong to the process-wide summary
// returned to the Python coordinator. Protect that accumulator while retaining
// thread-local marker and host-timing state below.
static std::mutex stagesMutex;
static _TimerObject stages[TrainStage::END];

static thread_local std::vector<std::pair<TrainStage, int>> marks;
static thread_local std::vector<TrainStage> pushedMarks;

static thread_local bool hostHold = false;
static thread_local std::chrono::time_point<
    std::chrono::high_resolution_clock> hostStartTime;
static thread_local double hostTimeDelta = -1.0;

// reset timing and counts
void reset() {
    // This starts a new process-wide accumulation interval and must be called
    // at a quiescent boundary. VkSplat initialization does so before worker
    // dispatch. Only the calling thread's in-flight state can be reset because
    // marker ownership is deliberately thread-local.
    {
        std::lock_guard<std::mutex> lock(stagesMutex);
        for (int i = 0; i < int(TrainStage::END); i++) {
            stages[i] = {0, 0.0};
        }
    }
    marks.clear();
    pushedMarks.clear();
    hostHold = false;
    hostTimeDelta = -1.0;
}

void hostTic() {
    if (hostHold)
        _THROW_ERROR("hostTic");
    hostHold = true;
    hostStartTime = std::chrono::high_resolution_clock::now();
}

void hostToc() {
    if (hostTimeDelta < 0.0) {
        hostTimeDelta = 0.0;
        return;
    }
    if (!hostHold)
        _THROW_ERROR("hostToc");
    hostHold = false;
    auto hostEndTime = std::chrono::high_resolution_clock::now();
    hostTimeDelta += std::chrono::duration<double>(hostEndTime-hostStartTime).count();
}

namespace {
constexpr bool isRasterBackwardProbe(TrainStage stage) {
    return stage >= _RasterizeBackwardScheduling_PerPixel &&
        stage < _RasterizeBackwardScheduling_N;
}
}  // namespace

template<TrainStage stage>
Timer<stage>::Timer(VulkanGSPipeline *module) : module(module) {
    then = std::chrono::high_resolution_clock::now();
    const bool write_gpu =
        isRasterBackwardProbe(stage) || module->stageTimestampsEnabled();
    if (write_gpu && module->writeTimestamp(1))
        marks.emplace_back(stage, 1);
}

template<TrainStage stage>
Timer<stage>::~Timer() {
    {
        std::lock_guard<std::mutex> lock(stagesMutex);
        stages[int(stage)].count += 1;
        if (stage >= _RasterizeBackwardScheduling_PerPixel && stage < _RasterizeBackwardScheduling_N)
            stages[RasterizeBackward].count += 1;
    }

    const bool write_gpu =
        isRasterBackwardProbe(stage) || module->stageTimestampsEnabled();
    if (write_gpu && module->writeTimestampNoExcept(-1))
        marks.emplace_back(stage, -1);

    // auto now = std::chrono::high_resolution_clock::now();
    // double dt = std::chrono::duration<double>(now - then).count();

    // PerfTimer::stages[int(stage)].total_time += dt;
}

void pushMarker(VulkanGSPipeline *module) {

    if (!module->writeTimestamp(-1))
        _THROW_ERROR("Failed to write exit timestamp in pushMarker");
    
    int depth = 1;
    for (int i = (int)marks.size()-1; i >= 0; --i) {
        auto [stage, delta] = marks[i];
        depth -= delta;
        if (depth == 0) {
            pushedMarks.push_back(stage);
            marks.emplace_back(stage, -1);
            return;
        }
    }
    _THROW_ERROR("Empty stack in pushMarker");
}

void popMarkers(VulkanGSPipeline *module) {
    {
        std::lock_guard<std::mutex> lock(stagesMutex);
        for (auto stage : pushedMarks)
            stages[int(stage)].total_time += hostTimeDelta;
    }
    while (!pushedMarks.empty()) {
        auto stage = pushedMarks.back();
        pushedMarks.pop_back();
        if (!module->writeTimestamp(1))
            _THROW_ERROR("Failed to write enter timestamp in popMarkers");
        marks.emplace_back(stage, 1);
    }
    hostTimeDelta = 0.0;
}

std::vector<std::pair<int, int>> takeMarks() {
    std::vector<std::pair<int, int>> out;
    out.reserve(marks.size());
    for (const auto& mark : marks)
        out.emplace_back(static_cast<int>(mark.first), mark.second);
    marks.clear();
    return out;
}

std::vector<std::pair<size_t, double>> update(std::vector<double> times) {
    return update(times, takeMarks());
}

std::vector<std::pair<size_t, double>> update(
    const std::vector<double>& times,
    const std::vector<std::pair<int, int>>& markers
) {
    if (times.size() != markers.size())
        _THROW_ERROR(
            "Number of timestamps (" + std::to_string(times.size()) +
            ") and number of marks (" + std::to_string(markers.size()) +
            ") mismatch in batch time update"
        );
    std::vector<std::pair<size_t, double>> results(TrainStage::END, {0, 0.0});
    double accumulated[TrainStage::END] = {};
    std::vector<std::pair<TrainStage, double>> stack;
    for (size_t i = 0; i < times.size(); i++) {
        auto stage = static_cast<TrainStage>(markers[i].first);
        int delta = markers[i].second;
        if (delta == 1) {
            stack.emplace_back(stage, times[i]);
        }
        else {
            if (stack.empty() || stack.back().first != stage)
                _THROW_ERROR("Unbalanced timestamp marker stack in batch time update");
            double dt = times[i] - stack.back().second;
            accumulated[int(stage)] += dt;
            if (stage >= _RasterizeBackwardScheduling_PerPixel && stage < _RasterizeBackwardScheduling_N)
                accumulated[RasterizeBackward] += dt;
            stack.pop_back();
            results[stage].first += 1;
            results[stage].second += dt;
        }
    }
    if (!stack.empty())
        _THROW_ERROR("Unbalanced timestamp marker stack in batch time update");
    {
        std::lock_guard<std::mutex> lock(stagesMutex);
        for (int i = 0; i < int(TrainStage::END); i++)
            stages[i].total_time += accumulated[i];
    }
    return results;
}

// return a summary of timing
std::map<std::string, std::tuple<size_t, double>> get_summary() {
    std::map<std::string, std::tuple<size_t, double>> summary;
    std::lock_guard<std::mutex> lock(stagesMutex);
    for (int i = 0; i < int(TrainStage::END); i++) {
        std::string name = _TrainStageNames[i];
        summary[name] = { stages[i].count, stages[i].total_time };
    }
    return summary;
}

// template instantiation of timers
#define _(name) template struct Timer<name>;
PERF_TIMER_TRAIN_STAGES
#undef _
// template <TrainStage name> Timer<name>::Timer();
// template <TrainStage name> Timer<name>::~Timer();

};  // namespace PerfTimer
