#include "hierarchical_host_collective.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#include <immintrin.h>
#endif

namespace {

HierarchicalHostAlgorithm select_algorithm(
    const HierarchicalHostCollectiveConfig& config) {
    if (config.algorithm != HierarchicalHostAlgorithm::Auto &&
        config.algorithm != HierarchicalHostAlgorithm::Central)
        throw std::invalid_argument(
            "hierarchical host collective only supports central reduce");
    return HierarchicalHostAlgorithm::Central;
}

void check_finite_input(float value) {
    if (!std::isfinite(value))
        throw std::runtime_error(
            "central reducer encountered non-finite input");
}

void check_finite_reduced(float value) {
    if (!std::isfinite(value))
        throw std::runtime_error("central reducer overflowed FP32");
}

void reduce_central_scalar_range(
    const float* const* inputs,
    uint32_t world_size,
    size_t begin,
    size_t end,
    float* output,
    float output_scale) {
    for (size_t index = begin; index < end; ++index) {
        float sum = 0.0f;
        for (uint32_t rank = 0; rank < world_size; ++rank) {
            const float value = inputs[rank][index];
            check_finite_input(value);
            // Exact rank-0..P-1 FP32 reduction order. Do not reorder.
            sum = sum + value;
        }
        const float reduced = sum * output_scale;
        check_finite_reduced(reduced);
        output[index] = reduced;
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
void reduce_central_avx2_range(
    const float* const* inputs,
    uint32_t world_size,
    size_t begin,
    size_t end,
    float* output,
    float output_scale) {
    const size_t aligned_end = begin + ((end - begin) / 8u) * 8u;
    const __m256 scale = _mm256_set1_ps(output_scale);
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 inf = _mm256_set1_ps(
        std::numeric_limits<float>::infinity());
    size_t index = begin;
    for (; index < aligned_end; index += 8u) {
        __m256 acc = _mm256_loadu_ps(inputs[0] + index);
        for (uint32_t rank = 0; rank < world_size; ++rank) {
            const __m256 lane = rank == 0
                ? acc
                : _mm256_loadu_ps(inputs[rank] + index);
            const __m256 abs_lane = _mm256_and_ps(lane, abs_mask);
            const int unordered = _mm256_movemask_ps(
                _mm256_cmp_ps(lane, lane, _CMP_UNORD_Q));
            const int infinite = _mm256_movemask_ps(
                _mm256_cmp_ps(abs_lane, inf, _CMP_EQ_OQ));
            if (unordered | infinite)
                throw std::runtime_error(
                    "central reducer encountered non-finite input");
            if (rank > 0)
                acc = _mm256_add_ps(acc, lane);
        }
        acc = _mm256_mul_ps(acc, scale);
        const __m256 abs_acc = _mm256_and_ps(acc, abs_mask);
        const int unordered = _mm256_movemask_ps(
            _mm256_cmp_ps(acc, acc, _CMP_UNORD_Q));
        const int infinite = _mm256_movemask_ps(
            _mm256_cmp_ps(abs_acc, inf, _CMP_EQ_OQ));
        if (unordered | infinite)
            throw std::runtime_error("central reducer overflowed FP32");
        _mm256_storeu_ps(output + index, acc);
    }
    reduce_central_scalar_range(
        inputs, world_size, index, end, output, output_scale);
}
#endif

void reduce_central_fp32(
    const float* const* inputs,
    uint32_t world_size,
    size_t float_count,
    float* output,
    bool mean_reduction,
    uint32_t shards) {
    if (float_count == 0)
        return;
    const float output_scale = mean_reduction
        ? 1.0f / static_cast<float>(world_size)
        : 1.0f;
    auto reduce_range = [&](size_t begin, size_t end) {
#if defined(__GNUC__) || defined(__clang__)
        if (__builtin_cpu_supports("avx2") && end - begin >= 8u) {
            reduce_central_avx2_range(
                inputs, world_size, begin, end, output, output_scale);
            return;
        }
#endif
        reduce_central_scalar_range(
            inputs, world_size, begin, end, output, output_scale);
    };
    const uint32_t use_shards =
        (float_count >= 32768u && shards > 1u)
            ? std::min<uint32_t>(shards, 8u) : 1u;
    if (use_shards == 1u) {
        reduce_range(0, float_count);
        return;
    }
    std::vector<std::thread> workers;
    std::vector<std::exception_ptr> failures(use_shards);
    const size_t chunk = (float_count + use_shards - 1u) / use_shards;
    workers.reserve(use_shards);
    for (uint32_t shard = 0; shard < use_shards; ++shard) {
        const size_t begin = size_t(shard) * chunk;
        const size_t end = std::min(float_count, begin + chunk);
        if (begin >= end)
            break;
        workers.emplace_back([&, shard, begin, end]() {
            try {
                reduce_range(begin, end);
            } catch (...) {
                failures[shard] = std::current_exception();
            }
        });
    }
    for (auto& worker : workers)
        worker.join();
    for (const auto& failure : failures) {
        if (failure)
            std::rethrow_exception(failure);
    }
}

}  // namespace

void finalize_local_gradient_fp32(
    float* values, size_t float_count, uint32_t local_microsteps) {
    if (local_microsteps == 0)
        throw std::invalid_argument(
            "local gradient finalize requires K >= 1");
    if (float_count != 0 && values == nullptr)
        throw std::invalid_argument(
            "local gradient finalize received a null buffer");

    const float inverse_k = 1.0f / static_cast<float>(local_microsteps);
    for (size_t index = 0; index < float_count; ++index) {
        if (!std::isfinite(values[index]))
            throw std::runtime_error(
                "local gradient finalize encountered non-finite input");
        values[index] = values[index] * inverse_k;
        if (!std::isfinite(values[index]))
            throw std::runtime_error(
                "local gradient finalize overflowed FP32");
    }
}

GradientScalingContract make_gradient_scaling_contract(
    uint32_t local_microsteps, uint32_t world_size) {
    if (local_microsteps == 0 || world_size == 0)
        throw std::invalid_argument(
            "gradient scaling requires positive K and P");
    GradientScalingContract result;
    result.local_microsteps = local_microsteps;
    result.world_size = world_size;
    result.local_finalize_scale =
        1.0f / static_cast<float>(local_microsteps);
    result.collective_scale = 1.0f / static_cast<float>(world_size);
    result.global_scale =
        result.local_finalize_scale * result.collective_scale;
    return result;
}

SortedSparseMeanResult exact_sorted_sparse_mean_fp32(
    const SortedSparseRankInput* ranks,
    uint32_t world_size,
    size_t record_width,
    uint32_t row_count) {
    if (ranks == nullptr || world_size < 2 || world_size > 8)
        throw std::invalid_argument(
            "exact sparse mean requires 2..8 rank inputs");
    if (record_width == 0)
        throw std::invalid_argument(
            "exact sparse mean requires a positive record width");

    std::vector<size_t> positions(world_size, 0);
    for (uint32_t rank = 0; rank < world_size; ++rank) {
        const SortedSparseRankInput& input = ranks[rank];
        if (input.count != 0 &&
            (input.ids == nullptr || input.values == nullptr))
            throw std::invalid_argument(
                "exact sparse mean received a null rank buffer");
        for (size_t index = 0; index < input.count; ++index) {
            const uint32_t id = input.ids[index];
            if (id >= row_count)
                throw std::runtime_error(
                    "exact sparse mean saw an out-of-range id");
            if (index > 0 && input.ids[index] <= input.ids[index - 1])
                throw std::runtime_error(
                    "exact sparse mean requires strictly increasing ids");
            for (size_t lane = 0; lane < record_width; ++lane) {
                if (!std::isfinite(
                        input.values[index * record_width + lane]))
                    throw std::runtime_error(
                        "exact sparse mean encountered non-finite input");
            }
        }
    }

    SortedSparseMeanResult result;
    const float inverse_p = 1.0f / static_cast<float>(world_size);
    std::vector<float> reduced(record_width, 0.0f);
    while (true) {
        bool found = false;
        uint32_t next_id = 0;
        for (uint32_t rank = 0; rank < world_size; ++rank) {
            const size_t position = positions[rank];
            if (position >= ranks[rank].count)
                continue;
            const uint32_t id = ranks[rank].ids[position];
            if (!found || id < next_id) {
                next_id = id;
                found = true;
            }
        }
        if (!found)
            break;
        std::fill(reduced.begin(), reduced.end(), 0.0f);
        for (uint32_t rank = 0; rank < world_size; ++rank) {
            size_t& position = positions[rank];
            if (position >= ranks[rank].count ||
                ranks[rank].ids[position] != next_id)
                continue;
            const float* row =
                ranks[rank].values + position * record_width;
            for (size_t lane = 0; lane < record_width; ++lane)
                reduced[lane] = reduced[lane] + row[lane];
            ++position;
        }
        for (size_t lane = 0; lane < record_width; ++lane) {
            reduced[lane] = reduced[lane] * inverse_p;
            if (!std::isfinite(reduced[lane]))
                throw std::runtime_error(
                    "exact sparse mean overflowed FP32");
        }
        result.ids.push_back(next_id);
        result.values.insert(
            result.values.end(), reduced.begin(), reduced.end());
    }
    return result;
}

size_t count_sorted_sparse_union(
    const uint32_t* const* id_ptrs,
    const size_t* counts,
    uint32_t world_size) {
    if (id_ptrs == nullptr || counts == nullptr ||
        world_size < 2 || world_size > 8)
        throw std::invalid_argument(
            "sorted sparse union count requires 2..8 rank inputs");
    std::vector<size_t> positions(world_size, 0);
    for (uint32_t rank = 0; rank < world_size; ++rank) {
        if (counts[rank] != 0 && id_ptrs[rank] == nullptr)
            throw std::invalid_argument(
                "sorted sparse union count received a null id buffer");
        for (size_t index = 1; index < counts[rank]; ++index) {
            if (id_ptrs[rank][index] <= id_ptrs[rank][index - 1])
                throw std::runtime_error(
                    "sorted sparse union count requires strictly increasing ids");
        }
    }

    size_t union_count = 0;
    while (true) {
        bool found = false;
        uint32_t next_id = 0;
        for (uint32_t rank = 0; rank < world_size; ++rank) {
            const size_t position = positions[rank];
            if (position >= counts[rank])
                continue;
            const uint32_t id = id_ptrs[rank][position];
            if (!found || id < next_id) {
                next_id = id;
                found = true;
            }
        }
        if (!found)
            break;
        ++union_count;
        for (uint32_t rank = 0; rank < world_size; ++rank) {
            size_t& position = positions[rank];
            if (position < counts[rank] && id_ptrs[rank][position] == next_id)
                ++position;
        }
    }
    return union_count;
}

const char* choose_exact_bucket_codec(
    double dense_seconds,
    double sparse_seconds,
    bool sparse_exact,
    double minimum_sparse_gain) {
    if (!std::isfinite(dense_seconds) || !std::isfinite(sparse_seconds) ||
        !std::isfinite(minimum_sparse_gain) ||
        dense_seconds <= 0.0 || sparse_seconds <= 0.0 ||
        minimum_sparse_gain < 0.0 || minimum_sparse_gain >= 1.0)
        throw std::invalid_argument(
            "bucket codec calibration values are invalid");
    if (sparse_exact &&
        sparse_seconds <= dense_seconds * (1.0 - minimum_sparse_gain))
        return "exact_sorted_sparse";
    return "dense";
}

const char* decide_hierarchical_exact_codec(
    const HierarchicalExactCodecVote* votes,
    uint32_t world_size,
    double minimum_sparse_gain,
    uint64_t* union_count) {
    if (votes == nullptr || world_size < 2 || world_size > 8)
        throw std::invalid_argument(
            "hierarchical exact codec requires 2..8 rank votes");
    const uint64_t dense_bytes = votes[0].dense_gaussian_bytes;
    const uint64_t aux_bytes = votes[0].dense_aux_bytes;
    const uint64_t record_bytes = votes[0].record_bytes;
    uint64_t sum_local = 0;
    const uint32_t* id_ptrs[8] = {};
    size_t counts[8] = {};
    for (uint32_t rank = 0; rank < world_size; ++rank) {
        if (votes[rank].dense_gaussian_bytes != dense_bytes ||
            votes[rank].dense_aux_bytes != aux_bytes ||
            votes[rank].record_bytes != record_bytes)
            throw std::runtime_error(
                "hierarchical exact codec saw diverging payload layouts");
        if (votes[rank].local_count > 0 && votes[rank].local_ids == nullptr)
            throw std::invalid_argument(
                "hierarchical exact codec received a null id list");
        if (record_bytes != 0 &&
            votes[rank].local_sparse_bytes !=
                votes[rank].local_count * record_bytes)
            throw std::runtime_error(
                "hierarchical exact codec local bytes do not match record width");
        if (record_bytes == 0 && votes[rank].local_sparse_bytes != 0)
            throw std::invalid_argument(
                "hierarchical exact codec requires record_bytes for a "
                "non-empty sparse vote");
        sum_local += votes[rank].local_sparse_bytes;
        id_ptrs[rank] = votes[rank].local_ids;
        counts[rank] = static_cast<size_t>(votes[rank].local_count);
    }
    const uint64_t observed_union = (sum_local == 0 || record_bytes == 0)
        ? 0u
        : static_cast<uint64_t>(
              count_sorted_sparse_union(id_ptrs, counts, world_size));
    if (union_count != nullptr)
        *union_count = observed_union;
    if (dense_bytes == 0 || sum_local == 0)
        return "dense";
    const uint64_t union_bytes = observed_union * record_bytes;
    const double dense_cost =
        2.0 * double(world_size) * double(dense_bytes + aux_bytes);
    const double sparse_cost =
        double(sum_local) +
        double(world_size) * double(union_bytes) +
        2.0 * double(world_size) * double(aux_bytes);
    return choose_exact_bucket_codec(
        dense_cost, sparse_cost, true, minimum_sparse_gain);
}

HierarchicalHostCollective::Slot::Slot(uint32_t world_size)
    : submitted(world_size, false),
      consumed(world_size, false),
      inputs(world_size, nullptr) {}

HierarchicalHostCollective::SparseSlot::SparseSlot(uint32_t world_size)
    : submitted(world_size, false),
      consumed(world_size, false),
      id_ptrs(world_size, nullptr),
      value_ptrs(world_size, nullptr),
      counts(world_size, 0) {}

HierarchicalHostCollective::HierarchicalHostCollective(
    const HierarchicalHostCollectiveConfig& config)
    : config_(config),
      selected_algorithm_(select_algorithm(config)),
      reducer_shards_(
          config.reducer_shards == 0
              ? std::max<uint32_t>(1, config.world_size)
              : config.reducer_shards),
      attached_(config.world_size, false),
      sparse_slot_(config.world_size),
      active_generation_(config.world_size, 0),
      next_bucket_(config.world_size, 0) {
    if (config_.world_size < 3 || config_.world_size > 8)
        throw std::invalid_argument(
            "hierarchical host collective supports P in [3, 8]");
    if (config_.bucket_count == 0 ||
        config_.bucket_count > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument(
            "hierarchical host collective requires a valid bucket count");
    if (config_.bucket_float_capacity == 0 || config_.ring_depth == 0)
        throw std::invalid_argument(
            "hierarchical host collective requires non-empty ring slots");
    if (config_.local_microsteps == 0)
        throw std::invalid_argument(
            "hierarchical host collective requires K >= 1");

    slots_.reserve(config_.ring_depth);
    for (size_t index = 0; index < config_.ring_depth; ++index)
        slots_.emplace_back(config_.world_size);
}

HierarchicalHostCollective::~HierarchicalHostCollective() {
    abort("hierarchical host collective released");
}

void HierarchicalHostCollective::validate_rank(uint32_t rank) const {
    if (rank >= config_.world_size)
        throw std::out_of_range(
            "hierarchical host collective rank is out of range");
}

bool HierarchicalHostCollective::all_true(
    const std::vector<bool>& values) const {
    return std::all_of(
        values.begin(), values.end(), [](bool value) { return value; });
}

bool HierarchicalHostCollective::all_attached_locked() const {
    return all_true(attached_);
}

void HierarchicalHostCollective::attach(
    uint32_t rank,
    size_t payload_bytes,
    const std::string& layout_signature) {
    validate_rank(rank);
    if (payload_bytes == 0 || layout_signature.empty())
        throw std::invalid_argument(
            "hierarchical host collective requires a non-empty packed ABI");

    std::lock_guard<std::mutex> lock(mutex_);
    if (aborted_) {
        throw std::runtime_error(
            "hierarchical host collective is aborted: " + abort_reason_);
    }
    if (attached_[rank])
        throw std::runtime_error(
            "hierarchical host collective rank was attached twice");
    if (payload_bytes_ == 0) {
        payload_bytes_ = payload_bytes;
        layout_signature_ = layout_signature;
    } else if (
        payload_bytes_ != payload_bytes ||
        layout_signature_ != layout_signature) {
        throw std::runtime_error(
            "hierarchical host collective packed ABI mismatch");
    }
    attached_[rank] = true;
    fully_attached_once_ = all_attached_locked();
    cv_.notify_all();
}

void HierarchicalHostCollective::detach(uint32_t rank) noexcept {
    if (rank >= config_.world_size)
        return;
    std::lock_guard<std::mutex> lock(mutex_);
    attached_[rank] = false;
    if (fully_attached_once_ && !aborted_)
        abort_locked(
            "rank detached before collective completion");
    cv_.notify_all();
}

void HierarchicalHostCollective::validate_rank_sequence_locked(
    uint32_t rank, uint64_t generation, uint32_t bucket) {
    if (generation == 0 || bucket >= config_.bucket_count) {
        abort_locked("invalid generation or bucket");
        throw std::invalid_argument(
            "hierarchical host generation/bucket is invalid");
    }
    if (next_bucket_[rank] != bucket) {
        abort_generation_ = generation;
        abort_locked("rank skipped or reordered a collective bucket");
        throw std::runtime_error(
            "hierarchical host rank bucket sequence mismatch");
    }
    if (active_generation_[rank] != 0 &&
        active_generation_[rank] != generation) {
        abort_generation_ = generation;
        abort_locked("rank skipped or repeated a collective generation");
        throw std::runtime_error(
            "hierarchical host rank generation sequence mismatch");
    }
}

void HierarchicalHostCollective::advance_rank_sequence_locked(
    uint32_t rank, uint64_t generation, uint32_t bucket) {
    if (bucket + 1 == config_.bucket_count) {
        if (generation == std::numeric_limits<uint64_t>::max()) {
            abort_generation_ = generation;
            abort_locked("collective generation overflow");
            throw std::overflow_error(
                "hierarchical host generation overflow");
        }
        active_generation_[rank] = generation + 1;
        next_bucket_[rank] = 0;
    } else {
        active_generation_[rank] = generation;
        next_bucket_[rank] = bucket + 1;
    }
}

void HierarchicalHostCollective::initialize_slot_locked(
    Slot& slot,
    uint64_t generation,
    uint32_t bucket,
    size_t float_count) {
    slot.occupied = true;
    slot.reduced_ready = false;
    slot.generation = generation;
    slot.bucket = bucket;
    slot.float_count = float_count;
    std::fill(slot.submitted.begin(), slot.submitted.end(), false);
    std::fill(slot.consumed.begin(), slot.consumed.end(), false);
    std::fill(slot.inputs.begin(), slot.inputs.end(), nullptr);
    slot.reduced.assign(float_count, 0.0f);
}

void HierarchicalHostCollective::reduce_bucket(
    uint32_t rank,
    uint64_t generation,
    uint32_t bucket,
    const float* local_input,
    size_t float_count,
    float* output) {
    if (output == nullptr)
        throw std::invalid_argument(
            "hierarchical host bucket output is null");
    const float* reduced = acquire_reduced_bucket(
        rank, generation, bucket, local_input, float_count);
    std::copy(reduced, reduced + float_count, output);
    release_bucket(rank, generation, bucket);
}

const float* HierarchicalHostCollective::acquire_reduced_bucket(
    uint32_t rank,
    uint64_t generation,
    uint32_t bucket,
    const float* local_input,
    size_t float_count) {
    validate_rank(rank);
    if (float_count == 0 ||
        float_count > config_.bucket_float_capacity ||
        local_input == nullptr)
        throw std::invalid_argument(
            "hierarchical host bucket memory range is invalid");

    std::unique_lock<std::mutex> lock(mutex_);
    if (!attached_[rank])
        throw std::runtime_error(
            "hierarchical host reduction by detached rank");
    cv_.wait(lock, [&]() {
        return aborted_ || all_attached_locked();
    });
    if (aborted_)
        throw std::runtime_error(
            "hierarchical host collective aborted: " + abort_reason_);

    validate_rank_sequence_locked(rank, generation, bucket);
    Slot& slot = slots_[bucket % slots_.size()];
    while (
        slot.occupied &&
        (slot.generation != generation || slot.bucket != bucket)) {
        if (!slot.consumed[rank]) {
            abort_generation_ = generation;
            abort_locked(
                "ranks entered different generation/bucket contracts");
            throw std::runtime_error(
                "hierarchical host cross-rank generation mismatch");
        }
        cv_.wait(lock, [&]() {
            return aborted_ || !slot.occupied;
        });
        if (aborted_)
            throw std::runtime_error(
                "hierarchical host collective aborted: " + abort_reason_);
    }
    if (!slot.occupied)
        initialize_slot_locked(slot, generation, bucket, float_count);
    if (slot.float_count != float_count) {
        abort_generation_ = generation;
        abort_locked("ranks submitted different bucket sizes");
        throw std::runtime_error(
            "hierarchical host bucket size mismatch");
    }
    if (slot.submitted[rank]) {
        abort_generation_ = generation;
        abort_locked("rank submitted the same bucket twice");
        throw std::runtime_error(
            "hierarchical host duplicate bucket submission");
    }

    slot.inputs[rank] = local_input;
    slot.submitted[rank] = true;
    const bool last_rank = all_true(slot.submitted);
    if (!last_rank) {
        cv_.wait(lock, [&]() {
            return aborted_ || slot.reduced_ready;
        });
        if (aborted_)
            throw std::runtime_error(
                "hierarchical host collective aborted at generation " +
                std::to_string(abort_generation_) + ": " + abort_reason_);
        return slot.reduced.data();
    }

    const float* rank_inputs[8] = {};
    for (uint32_t source = 0; source < config_.world_size; ++source)
        rank_inputs[source] = slot.inputs[source];
    float* reduced = slot.reduced.data();
    const size_t float_count_copy = slot.float_count;
    const uint32_t world_size = config_.world_size;
    const bool mean_reduction = config_.mean_reduction;
    const uint32_t shards = reducer_shards_;
    lock.unlock();
    try {
        reduce_central_fp32(
            rank_inputs, world_size, float_count_copy, reduced,
            mean_reduction, shards);
    } catch (const std::exception& error) {
        lock.lock();
        abort_generation_ = generation;
        abort_locked(error.what());
        throw;
    } catch (...) {
        lock.lock();
        abort_generation_ = generation;
        abort_locked("unknown central reducer failure");
        throw;
    }
    lock.lock();
    if (aborted_)
        throw std::runtime_error(
            "hierarchical host collective aborted at generation " +
            std::to_string(abort_generation_) + ": " + abort_reason_);
    slot.reduced_ready = true;
    cv_.notify_all();
    return slot.reduced.data();
}

const SortedSparseMeanResult&
HierarchicalHostCollective::acquire_reduced_sorted_sparse(
    uint32_t rank,
    uint64_t generation,
    const uint32_t* local_ids,
    const float* local_values,
    size_t local_count,
    size_t record_width,
    uint32_t row_count) {
    validate_rank(rank);
    if (generation == 0 || record_width == 0)
        throw std::invalid_argument(
            "sorted sparse collective generation/record width is invalid");
    if (local_count != 0 &&
        (local_ids == nullptr || local_values == nullptr))
        throw std::invalid_argument(
            "sorted sparse collective received a null local payload");

    std::unique_lock<std::mutex> lock(mutex_);
    if (!attached_[rank])
        throw std::runtime_error(
            "sorted sparse reduction by detached rank");
    cv_.wait(lock, [&]() {
        return aborted_ || all_attached_locked();
    });
    if (aborted_)
        throw std::runtime_error(
            "hierarchical host collective aborted: " + abort_reason_);

    while (sparse_slot_.occupied &&
           sparse_slot_.generation != generation) {
        if (!sparse_slot_.consumed[rank]) {
            abort_generation_ = generation;
            abort_locked(
                "ranks entered different sparse generation contracts");
            throw std::runtime_error(
                "hierarchical host sparse generation mismatch");
        }
        cv_.wait(lock, [&]() {
            return aborted_ || !sparse_slot_.occupied;
        });
        if (aborted_)
            throw std::runtime_error(
                "hierarchical host collective aborted: " + abort_reason_);
    }
    if (!sparse_slot_.occupied) {
        sparse_slot_.occupied = true;
        sparse_slot_.reduced_ready = false;
        sparse_slot_.generation = generation;
        sparse_slot_.record_width = record_width;
        sparse_slot_.row_count = row_count;
        std::fill(
            sparse_slot_.submitted.begin(),
            sparse_slot_.submitted.end(), false);
        std::fill(
            sparse_slot_.consumed.begin(),
            sparse_slot_.consumed.end(), false);
        std::fill(
            sparse_slot_.id_ptrs.begin(),
            sparse_slot_.id_ptrs.end(), nullptr);
        std::fill(
            sparse_slot_.value_ptrs.begin(),
            sparse_slot_.value_ptrs.end(), nullptr);
        std::fill(
            sparse_slot_.counts.begin(), sparse_slot_.counts.end(), 0);
        sparse_slot_.reduced = {};
        sparse_generation_ = generation;
    }
    if (sparse_slot_.record_width != record_width ||
        sparse_slot_.row_count != row_count) {
        abort_generation_ = generation;
        abort_locked("ranks submitted different sparse layouts");
        throw std::runtime_error(
            "hierarchical host sparse layout mismatch");
    }
    if (sparse_slot_.submitted[rank]) {
        abort_generation_ = generation;
        abort_locked("rank submitted sparse records twice");
        throw std::runtime_error(
            "hierarchical host duplicate sparse submission");
    }

    sparse_slot_.id_ptrs[rank] = local_ids;
    sparse_slot_.value_ptrs[rank] = local_values;
    sparse_slot_.counts[rank] = local_count;
    sparse_slot_.submitted[rank] = true;
    const bool last_rank = all_true(sparse_slot_.submitted);
    if (!last_rank) {
        cv_.wait(lock, [&]() {
            return aborted_ || sparse_slot_.reduced_ready;
        });
        if (aborted_)
            throw std::runtime_error(
                "hierarchical host collective aborted at generation " +
                std::to_string(abort_generation_) + ": " + abort_reason_);
        return sparse_slot_.reduced;
    }

    SortedSparseRankInput inputs[8] = {};
    for (uint32_t source = 0; source < config_.world_size; ++source) {
        inputs[source].ids = sparse_slot_.id_ptrs[source];
        inputs[source].values = sparse_slot_.value_ptrs[source];
        inputs[source].count = sparse_slot_.counts[source];
    }
    const uint32_t world_size = config_.world_size;
    lock.unlock();
    SortedSparseMeanResult local_reduced;
    try {
        local_reduced = exact_sorted_sparse_mean_fp32(
            inputs, world_size, record_width, row_count);
    } catch (const std::exception& error) {
        lock.lock();
        abort_generation_ = generation;
        abort_locked(error.what());
        throw;
    } catch (...) {
        lock.lock();
        abort_generation_ = generation;
        abort_locked("unknown sparse reducer failure");
        throw;
    }
    lock.lock();
    if (aborted_)
        throw std::runtime_error(
            "hierarchical host collective aborted at generation " +
            std::to_string(abort_generation_) + ": " + abort_reason_);
    sparse_slot_.reduced = std::move(local_reduced);
    sparse_slot_.reduced_ready = true;
    cv_.notify_all();
    return sparse_slot_.reduced;
}

void HierarchicalHostCollective::release_sorted_sparse(
    uint32_t rank, uint64_t generation) {
    validate_rank(rank);
    std::lock_guard<std::mutex> lock(mutex_);
    if (aborted_)
        throw std::runtime_error(
            "hierarchical host collective aborted at generation " +
            std::to_string(abort_generation_) + ": " + abort_reason_);
    if (!sparse_slot_.occupied || !sparse_slot_.reduced_ready ||
        sparse_slot_.generation != generation ||
        !sparse_slot_.submitted[rank] || sparse_slot_.consumed[rank]) {
        abort_generation_ = generation;
        abort_locked("rank released an unacquired sparse reduction");
        throw std::runtime_error(
            "hierarchical host sparse release mismatch");
    }
    sparse_slot_.consumed[rank] = true;
    if (all_true(sparse_slot_.consumed)) {
        sparse_slot_.occupied = false;
        sparse_slot_.reduced_ready = false;
        std::fill(
            sparse_slot_.id_ptrs.begin(),
            sparse_slot_.id_ptrs.end(), nullptr);
        std::fill(
            sparse_slot_.value_ptrs.begin(),
            sparse_slot_.value_ptrs.end(), nullptr);
        std::fill(
            sparse_slot_.counts.begin(), sparse_slot_.counts.end(), 0);
        sparse_slot_.reduced = {};
    }
    cv_.notify_all();
}

void HierarchicalHostCollective::release_bucket(
    uint32_t rank, uint64_t generation, uint32_t bucket) {
    validate_rank(rank);
    std::lock_guard<std::mutex> lock(mutex_);
    if (aborted_)
        throw std::runtime_error(
            "hierarchical host collective aborted at generation " +
            std::to_string(abort_generation_) + ": " + abort_reason_);
    if (bucket >= config_.bucket_count)
        throw std::out_of_range(
            "hierarchical host released bucket is out of range");
    Slot& slot = slots_[bucket % slots_.size()];
    if (!slot.occupied || !slot.reduced_ready ||
        slot.generation != generation || slot.bucket != bucket ||
        !slot.submitted[rank] || slot.consumed[rank]) {
        abort_generation_ = generation;
        abort_locked("rank released an unacquired reduced bucket");
        throw std::runtime_error(
            "hierarchical host reduced bucket release mismatch");
    }
    slot.consumed[rank] = true;
    advance_rank_sequence_locked(rank, generation, bucket);
    if (all_true(slot.consumed)) {
        slot.occupied = false;
        slot.reduced_ready = false;
        std::fill(slot.inputs.begin(), slot.inputs.end(), nullptr);
        slot.reduced.clear();
    }
    cv_.notify_all();
}

void HierarchicalHostCollective::abort_locked(
    const std::string& reason) noexcept {
    aborted_ = true;
    abort_reason_ = reason.empty()
        ? "unspecified rank failure"
        : reason;
    cv_.notify_all();
}

void HierarchicalHostCollective::abort(
    const std::string& reason) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (abort_generation_ == 0) {
        for (const Slot& slot : slots_) {
            if (slot.occupied)
                abort_generation_ =
                    std::max(abort_generation_, slot.generation);
        }
    }
    abort_locked(reason);
}

void HierarchicalHostCollective::clear_abort() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    aborted_ = false;
    abort_generation_ = 0;
    abort_reason_.clear();
    for (Slot& slot : slots_) {
        slot.occupied = false;
        slot.reduced_ready = false;
        slot.generation = 0;
        slot.bucket = 0;
        slot.float_count = 0;
        std::fill(slot.submitted.begin(), slot.submitted.end(), false);
        std::fill(slot.consumed.begin(), slot.consumed.end(), false);
        std::fill(slot.inputs.begin(), slot.inputs.end(), nullptr);
        slot.reduced.clear();
    }
    sparse_slot_.occupied = false;
    sparse_slot_.reduced_ready = false;
    sparse_slot_.generation = 0;
    std::fill(
        sparse_slot_.submitted.begin(), sparse_slot_.submitted.end(), false);
    std::fill(
        sparse_slot_.consumed.begin(), sparse_slot_.consumed.end(), false);
    std::fill(
        sparse_slot_.id_ptrs.begin(), sparse_slot_.id_ptrs.end(), nullptr);
    std::fill(
        sparse_slot_.value_ptrs.begin(),
        sparse_slot_.value_ptrs.end(), nullptr);
    std::fill(
        sparse_slot_.counts.begin(), sparse_slot_.counts.end(), 0);
    sparse_slot_.reduced = {};
    sparse_generation_ = 0;
    std::fill(active_generation_.begin(), active_generation_.end(), 0);
    std::fill(next_bucket_.begin(), next_bucket_.end(), 0);
    cv_.notify_all();
}

GradientScalingContract
HierarchicalHostCollective::scaling_contract() const {
    return make_gradient_scaling_contract(
        config_.local_microsteps, config_.world_size);
}

const char* HierarchicalHostCollective::algorithm_name(
    HierarchicalHostAlgorithm algorithm) {
    switch (algorithm) {
    case HierarchicalHostAlgorithm::Auto:
        return "auto";
    case HierarchicalHostAlgorithm::Central:
        return "central_hierarchical_reduce_broadcast";
    }
    return "unknown";
}

std::map<std::string, HierarchicalHostCollective::ReportValue>
HierarchicalHostCollective::algorithm_report() const {
    std::lock_guard<std::mutex> lock(mutex_);
    const GradientScalingContract scaling = scaling_contract();
    const size_t attached_count = static_cast<size_t>(std::count(
        attached_.begin(), attached_.end(), true));
    return {
        {"abi", std::string("ftgspp-hierarchical-host-v1")},
        {"requested_algorithm", std::string(algorithm_name(config_.algorithm))},
        {"selected_algorithm", std::string(algorithm_name(selected_algorithm_))},
        {"execution_backend", std::string("cpu_memory_central")},
        {"selection_reason", std::string(
            config_.algorithm == HierarchicalHostAlgorithm::Auto
                ? "P>=3 uses fixed-rank central reduction"
                : "explicit native candidate selection")},
        {"world_size", config_.world_size},
        {"bucket_count", static_cast<uint64_t>(config_.bucket_count)},
        {"bucket_float_capacity",
         static_cast<uint64_t>(config_.bucket_float_capacity)},
        {"ring_depth", static_cast<uint64_t>(config_.ring_depth)},
        {"attached_count", static_cast<uint64_t>(attached_count)},
        {"central_cpu_available", true},
        {"gpu_transport_ready", false},
        {"cpu_memory_implementation", true},
        {"deterministic_fp32_rank_order", true},
        {"local_microsteps", config_.local_microsteps},
        {"local_finalize_scale", scaling.local_finalize_scale},
        {"collective_scale",
         config_.mean_reduction ? scaling.collective_scale : 1.0f},
        {"global_scale",
         config_.mean_reduction ? scaling.global_scale
                                : scaling.local_finalize_scale},
        {"mean_reduction", config_.mean_reduction},
        {"reducer_shards", reducer_shards_},
        {"aborted", aborted_},
        {"abort_generation", abort_generation_},
        {"abort_reason", abort_reason_},
    };
}
