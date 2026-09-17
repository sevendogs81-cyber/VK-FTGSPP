#pragma once

#include <condition_variable>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

// Host-side generation contract for replicated synchronous training.
// Each rank reports TileWorkControl after its own compute timeline retires.
// P=2 may start D2H after a local report; a later overflow still aborts the
// mailbox so the peer cannot Adam.
class SyncGenerationState {
public:
    struct RankReport {
        bool overflow = false;
        uint32_t exact_count = 0;
        uint32_t abort_code = 0;
    };

    struct Decision {
        bool aborted = false;
        bool fatal = false;
        bool local_overflow = false;
        uint32_t local_exact_count = 0;
        uint32_t max_exact_count = 0;
        uint32_t abort_code = 0;
        std::string reason;
    };

    explicit SyncGenerationState(uint32_t world_size, double timeout_seconds = 120.0);

    uint32_t world_size() const { return world_size_; }

    // Record this rank and return immediately. Sets aborted if this rank
    // overflowed or a peer already did. Used by P=2 so the fast rank can
    // submit the first gradient bucket without waiting for the peer GPU.
    Decision report_local(
        uint32_t rank, uint64_t generation, RankReport report);

    Decision wait_decision(uint32_t rank, uint64_t generation);

    void abort(uint64_t generation, const std::string& reason);
    void abort_current(const std::string& reason) {
        std::lock_guard<std::mutex> lock(mutex_);
        aborted_ = true;
        // Overflow must still wait for both reports to agree on capacity.
        fatal_ = reason.find("tile_overflow") == std::string::npos;
        reason_ = reason;
        cv_.notify_all();
    }
    void clear_for_replay();

private:
    void validate_rank(uint32_t rank) const;
    void ensure_generation_locked(uint64_t generation);
    Decision make_decision_locked(uint32_t rank) const;
    bool all_reported_locked() const;

    const std::chrono::duration<double> timeout_;
    const uint32_t world_size_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    uint64_t generation_ = 0;
    std::vector<char> reported_;
    std::vector<RankReport> reports_;
    bool aborted_ = false;
    bool fatal_ = false;
    uint32_t abort_code_ = 0;
    std::string reason_;
};
