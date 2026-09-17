#include "render/scheduler.h"

#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <iomanip>



ThompsonSamplingScheduler::ThompsonSamplingScheduler(
    size_t num_options, double initial_noise_std,
    size_t max_no_update, double adapt_tau, double warmup_tau,
    uint_fast32_t seed
) : num_options(num_options), stdev_offset(initial_noise_std), stats(num_options) {
    this->adapt_beta = exp(-1.0/adapt_tau);
    this->warmup_beta = exp(-1.0/warmup_tau);
    this->max_no_update = std::max(max_no_update-num_options, 2*num_options);
    rng.seed(seed);
}


void ThompsonSamplingScheduler::update(size_t idx, double val) {

    // decay running stats
    static constexpr double kMinCount = 2.0;
    for (Stat& stat : stats) {
        if (stat.count <= kMinCount)
            continue;
        double factor = fmax(adapt_beta, kMinCount/stat.count);
        stat.count *= factor;
        stat.sum_val *= factor;
        stat.sum_val2 *= factor;
        stat.num_no_update += 1;
    }
    stdev_offset *= warmup_beta;

    // update at idx
    Stat& stat = stats[idx];
    stat.count += 1.0;
    stat.sum_val += val;
    stat.sum_val2 += val*val;
    stat.num_no_update = 0;

    // printf("update %d with %lf - mean=%lf std=%lf stdm=%lf\n",
    //     (int)idx, val, stat.mean(), stat.stdev(), stat.stdev_mean(stdev_offset));
}


size_t ThompsonSamplingScheduler::sample() {

    // make sure all option stats are initialized first
    std::vector<size_t> ustats;
    for (size_t i = 0; i < stats.size(); i++) {
        if (stats[i].count < 2.0) ustats.push_back(i);
        if (stats[i].count < 1.0) ustats.push_back(i);
    }
    if (!ustats.empty()) {
        size_t idx = std::uniform_int_distribution<size_t>(0, ustats.size()-1)(rng);
        return ustats[idx];
    }

    // if sampling is needed, do that one first
    for (size_t i = 0; i < stats.size(); i++) {
        if (stats[i].num_no_update >= max_no_update)
            return i;
    }

    // pick random (Thompson sampling with normal distribution)
    size_t best_idx = (size_t)-1;
    double best_val = std::numeric_limits<double>::max();
    for (size_t i = 0; i < stats.size(); i++) {
        double mean = stats[i].mean();
        double stdev = stats[i].stdev_mean(stdev_offset);
        double val = std::normal_distribution<double>(mean, stdev)(rng);
        if (val < best_val)
            best_idx = i, best_val = val;
    }
    // printf("Pick %d\n", (int)best_idx);
    return best_idx;

}

size_t ThompsonSamplingScheduler::best_mean() const {
    if (stats.empty())
        throw std::runtime_error("empty Thompson scheduler has no best option");
    size_t best = 0;
    double best_value = std::numeric_limits<double>::max();
    for (size_t i = 0; i < stats.size(); ++i) {
        if (stats[i].count <= 0.0)
            continue;
        const double mean = stats[i].sum_val / stats[i].count;
        if (mean < best_value) {
            best = i;
            best_value = mean;
        }
    }
    return best;
}

bool ThompsonSamplingScheduler::fully_initialized(
    size_t minimum_samples) const {
    return !stats.empty() && std::all_of(
        stats.begin(), stats.end(), [minimum_samples](const Stat& stat) {
            return stat.count >= static_cast<double>(minimum_samples);
        });
}

std::string ThompsonSamplingScheduler::checkpoint() const {
    std::ostringstream out;
    out << std::setprecision(17) << num_options << ' ' << stdev_offset << ' '
        << adapt_beta << ' ' << warmup_beta << ' ' << max_no_update << ' ' << rng;
    for (const auto& stat : stats)
        out << ' ' << stat.count << ' ' << stat.sum_val << ' ' << stat.sum_val2
            << ' ' << stat.num_no_update;
    return out.str();
}
void ThompsonSamplingScheduler::restore(const std::string& state) {
    std::istringstream in(state);
    size_t count;
    if (!(in >> count) || count < 1 || count > 5)
        throw std::invalid_argument("invalid raster scheduler option count");
    if (num_options != 0 && count != num_options)
        throw std::invalid_argument("raster scheduler device options changed");
    num_options = count;
    if (!(in >> stdev_offset >> adapt_beta >> warmup_beta >> max_no_update >> std::ws >> rng) ||
        !std::isfinite(stdev_offset) || !std::isfinite(adapt_beta) ||
        !std::isfinite(warmup_beta))
        throw std::invalid_argument("invalid raster scheduler state");
    stats.resize(count);
    for (auto& stat : stats) {
        if (!(in >> stat.count >> stat.sum_val >> stat.sum_val2 >> stat.num_no_update) ||
            !std::isfinite(stat.count) || !std::isfinite(stat.sum_val) ||
            !std::isfinite(stat.sum_val2) || stat.count < 0)
            throw std::invalid_argument("invalid raster scheduler statistics");
    }
    in >> std::ws;
    if (!in.eof()) throw std::invalid_argument("trailing raster scheduler data");
}
