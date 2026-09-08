#include "sync_generation_state.h"
#include "tile_work_control.h"

#include <algorithm>
#include <stdexcept>

SyncGenerationState::SyncGenerationState(uint32_t world_size)
    : world_size_(world_size),
      reported_(world_size, 0),
      reports_(world_size) {
    if (world_size_ < 2 || world_size_ > 8)
        throw std::invalid_argument(
            "sync generation state supports P in [2, 8]");
}

void SyncGenerationState::validate_rank(uint32_t rank) const {
    if (rank >= world_size_)
        throw std::out_of_range("sync generation rank is out of range");
}

bool SyncGenerationState::all_reported_locked() const {
    return std::all_of(
        reported_.begin(), reported_.end(), [](char value) {
            return value != 0;
        });
}

void SyncGenerationState::ensure_generation_locked(uint64_t generation) {
    if (generation == 0)
        throw std::invalid_argument("sync generation zero is reserved");
    if (generation_ == 0 || generation > generation_) {
        generation_ = generation;
        std::fill(reported_.begin(), reported_.end(), 0);
        reports_.assign(world_size_, RankReport{});
        aborted_ = false;
        fatal_ = false;
        abort_code_ = 0;
        reason_.clear();
        return;
    }
    if (generation < generation_)
        throw std::runtime_error(
            "sync generation went backwards from " +
            std::to_string(generation_) + " to " +
            std::to_string(generation));
}

SyncGenerationState::Decision
SyncGenerationState::make_decision_locked(uint32_t rank) const {
    Decision decision;
    if (rank < world_size_ && reported_[rank]) {
        decision.local_overflow = reports_[rank].overflow;
        decision.local_exact_count = reports_[rank].exact_count;
    }
    decision.abort_code = abort_code_;
    decision.aborted = aborted_;
    decision.fatal = fatal_;
    decision.reason = reason_;
    for (uint32_t index = 0; index < world_size_; ++index) {
        if (!reported_[index])
            continue;
        decision.max_exact_count = std::max(
            decision.max_exact_count, reports_[index].exact_count);
        if (reports_[index].overflow)
            decision.aborted = true;
        if (reports_[index].abort_code == ftgspp::kTileAbortPrefix ||
            reports_[index].abort_code == ftgspp::kTileAbortInvariant) {
            decision.fatal = true;
            decision.aborted = true;
            decision.abort_code = reports_[index].abort_code;
        }
    }
    if (decision.fatal && decision.reason.empty())
        decision.reason = "tile invariant/prefix abort";
    else if (decision.aborted && decision.reason.empty())
        decision.reason = "tile_overflow_generation";
    return decision;
}

SyncGenerationState::Decision SyncGenerationState::report_local(
    uint32_t rank, uint64_t generation, RankReport report) {
    validate_rank(rank);
    std::lock_guard<std::mutex> lock(mutex_);
    ensure_generation_locked(generation);
    if (reported_[rank] != 0)
        throw std::runtime_error(
            "sync generation rank reported twice in generation " +
            std::to_string(generation));
    reports_[rank] = report;
    reported_[rank] = 1;
    if (report.abort_code == ftgspp::kTileAbortPrefix ||
        report.abort_code == ftgspp::kTileAbortInvariant) {
        aborted_ = true;
        fatal_ = true;
        abort_code_ = report.abort_code;
        reason_ = "tile invariant/prefix abort";
    } else if (report.overflow) {
        aborted_ = true;
        abort_code_ = report.abort_code;
        if (reason_.empty())
            reason_ = "tile_overflow_generation";
    }
    cv_.notify_all();
    return make_decision_locked(rank);
}

SyncGenerationState::Decision SyncGenerationState::report_and_wait(
    uint32_t rank, uint64_t generation, RankReport report) {
    validate_rank(rank);
    std::unique_lock<std::mutex> lock(mutex_);
    ensure_generation_locked(generation);
    if (reported_[rank] != 0)
        throw std::runtime_error(
            "sync generation rank reported twice in generation " +
            std::to_string(generation));
    reports_[rank] = report;
    reported_[rank] = 1;
    if (report.abort_code == ftgspp::kTileAbortPrefix ||
        report.abort_code == ftgspp::kTileAbortInvariant) {
        aborted_ = true;
        fatal_ = true;
        abort_code_ = report.abort_code;
        reason_ = "tile invariant/prefix abort";
    } else if (report.overflow) {
        aborted_ = true;
        abort_code_ = report.abort_code;
        if (reason_.empty())
            reason_ = "tile_overflow_generation";
    }
    cv_.notify_all();
    cv_.wait(lock, [&]() {
        return all_reported_locked() || fatal_;
    });
    return make_decision_locked(rank);
}

SyncGenerationState::Decision SyncGenerationState::wait_decision(
    uint32_t rank, uint64_t generation) {
    validate_rank(rank);
    std::unique_lock<std::mutex> lock(mutex_);
    if (generation_ != generation)
        throw std::runtime_error(
            "sync generation wait for a different generation");
    cv_.wait(lock, [&]() {
        return all_reported_locked() || fatal_;
    });
    return make_decision_locked(rank);
}

void SyncGenerationState::abort(
    uint64_t generation, const std::string& reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    ensure_generation_locked(generation);
    aborted_ = true;
    reason_ = reason.empty() ? "tile_overflow_generation" : reason;
    cv_.notify_all();
}

void SyncGenerationState::clear_for_replay() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::fill(reported_.begin(), reported_.end(), 0);
    reports_.assign(world_size_, RankReport{});
    aborted_ = false;
    fatal_ = false;
    abort_code_ = 0;
    reason_.clear();
    cv_.notify_all();
}
