#pragma once
#include "collective/host_mailbox_owner.h"
#include "collective/sync_generation_state.h"
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>

inline const char* choose_exact_bucket_codec(
    double dense_cost, double sparse_cost, bool sparse_exact,
    double minimum_sparse_gain = 0.05) {
    if (!std::isfinite(dense_cost) || !std::isfinite(sparse_cost) ||
        !std::isfinite(minimum_sparse_gain) || dense_cost <= 0 ||
        sparse_cost <= 0 || minimum_sparse_gain < 0 || minimum_sparse_gain >= 1)
        throw std::invalid_argument("bucket codec costs are invalid");
    return sparse_exact && sparse_cost <= dense_cost * (1.0 - minimum_sparse_gain)
        ? "exact_sorted_sparse" : "dense";
}

// One training run owns one session shared by exactly two engine instances.
// No process-global mailbox: independent runs can coexist in one interpreter.
struct PairSession {
    std::shared_ptr<HostMailboxOwner> dense, sparse, relocation;
    std::shared_ptr<SyncGenerationState> generation;
    std::mutex mutex;
    double timeout_seconds = 120.0;

    void ensure_generation(uint32_t world_size) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!generation)
            generation = std::make_shared<SyncGenerationState>(world_size, timeout_seconds);
        else if (generation->world_size() != world_size)
            throw std::runtime_error("session world size mismatch");
    }
    void release_generation() {
        std::lock_guard<std::mutex> lock(mutex);
        generation.reset();
    }
    void abort(const std::string& reason) {
        if (dense) dense->abort(reason);
        if (sparse) sparse->abort(reason);
        if (relocation) relocation->abort(reason);
        if (generation) generation->abort_current(reason);
    }
    void clear_for_replay() {
        if (dense) dense->clear_abort();
        if (sparse) sparse->clear_abort();
        if (relocation) relocation->clear_abort();
        if (generation) generation->clear_for_replay();
    }
};
