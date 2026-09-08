#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <variant>
#include <vector>

enum class HierarchicalHostAlgorithm {
    Auto,
    Central,
};

struct HierarchicalHostCollectiveConfig {
    uint32_t world_size = 0;
    size_t bucket_count = 0;
    size_t bucket_float_capacity = 0;
    size_t ring_depth = 3;
    uint32_t local_microsteps = 1;
    uint32_t reducer_shards = 0;
    HierarchicalHostAlgorithm algorithm = HierarchicalHostAlgorithm::Auto;
    bool mean_reduction = true;
};

struct GradientScalingContract {
    uint32_t local_microsteps = 1;
    uint32_t world_size = 1;
    float local_finalize_scale = 1.0f;
    float collective_scale = 1.0f;
    float global_scale = 1.0f;
};

// Finalize a rank-local K-view accumulator in place. The collective then
// applies exactly one additional 1/P scale while reducing rank inputs.
void finalize_local_gradient_fp32(
    float* values, size_t float_count, uint32_t local_microsteps);

GradientScalingContract make_gradient_scaling_contract(
    uint32_t local_microsteps, uint32_t world_size);

struct SortedSparseRankInput {
    const uint32_t* ids = nullptr;
    const float* values = nullptr;  // row-major [count * record_width]
    size_t count = 0;
};

struct SortedSparseMeanResult {
    std::vector<uint32_t> ids;
    std::vector<float> values;  // row-major [union * record_width]
};

// Exact P-way merge of strictly ID-sorted sparse records. Missing ranks contribute
// zero and the divisor is always the full world size, never the number of
// ranks that touched an ID.
SortedSparseMeanResult exact_sorted_sparse_mean_fp32(
    const SortedSparseRankInput* ranks,
    uint32_t world_size,
    size_t record_width,
    uint32_t row_count);

// K-way count of the strictly ID-sorted union. Missing ranks contribute
// nothing; the result is the number of distinct IDs, not max(local).
size_t count_sorted_sparse_union(
    const uint32_t* const* id_ptrs,
    const size_t* counts,
    uint32_t world_size);

// Sparse is selected only when exact and at least 5% faster than dense.
// P=2 pair-exchange uses this helper with both ranks' real message bytes.
const char* choose_exact_bucket_codec(
    double dense_seconds,
    double sparse_seconds,
    bool sparse_exact,
    double minimum_sparse_gain = 0.05);

// P>=3 auto_exact vote. Every rank must see the same votes so codec
// decisions cannot diverge (dense vs sparse collectives would deadlock).
struct HierarchicalExactCodecVote {
    uint64_t local_sparse_bytes = 0;
    uint64_t dense_gaussian_bytes = 0;
    uint64_t dense_aux_bytes = 0;
    const uint32_t* local_ids = nullptr;
    uint64_t local_count = 0;
    uint64_t record_bytes = 0;
};

// PCIe model: dense = 2*P*(gaussian+aux);
// sparse = sum(local) + P*union + 2*P*aux. Union is the exact ID union
// of the submitted sorted local id lists, not max(local).
const char* decide_hierarchical_exact_codec(
    const HierarchicalExactCodecVote* votes,
    uint32_t world_size,
    double minimum_sparse_gain = 0.05,
    uint64_t* union_count = nullptr);

// Single-host collective core for exact synchronous training.
//
// This class is the P>=3 central CPU reduce. P=2 uses HostMailboxOwner
// directly from the Python/native pair-exchange path.
class HierarchicalHostCollective {
public:
    using ReportValue = std::variant<
        uint64_t, uint32_t, bool, float, std::string>;

    explicit HierarchicalHostCollective(
        const HierarchicalHostCollectiveConfig& config);
    ~HierarchicalHostCollective();

    HierarchicalHostCollective(const HierarchicalHostCollective&) = delete;
    HierarchicalHostCollective& operator=(
        const HierarchicalHostCollective&) = delete;

    void attach(
        uint32_t rank,
        size_t payload_bytes,
        const std::string& layout_signature);
    void detach(uint32_t rank) noexcept;

    // Blocking CPU-memory collective. Every rank
    // calls with the same generation, bucket and float_count. output may alias
    // local_input. The default result is sum(rank-local-finalized-gradient) / P.
    void reduce_bucket(
        uint32_t rank,
        uint64_t generation,
        uint32_t bucket,
        const float* local_input,
        size_t float_count,
        float* output);

    // Zero-copy variant used by the Vulkan binding. The returned pointer names
    // the one reduced slab shared by every rank for this bucket and remains
    // valid until release_bucket(). Callers copy it into their private Vulkan
    // upload ticket before releasing it.
    const float* acquire_reduced_bucket(
        uint32_t rank,
        uint64_t generation,
        uint32_t bucket,
        const float* local_input,
        size_t float_count);
    void release_bucket(
        uint32_t rank, uint64_t generation, uint32_t bucket);

    // Exact sorted-sparse collective for visible Gaussian records.
    // Every rank submits a strictly ID-sorted local payload; the shared result
    // uses the full-P divisor and remains valid until release_sorted_sparse().
    const SortedSparseMeanResult& acquire_reduced_sorted_sparse(
        uint32_t rank,
        uint64_t generation,
        const uint32_t* local_ids,
        const float* local_values,
        size_t local_count,
        size_t record_width,
        uint32_t row_count);
    void release_sorted_sparse(uint32_t rank, uint64_t generation);

    void abort(const std::string& reason) noexcept;
    void clear_abort() noexcept;

    HierarchicalHostAlgorithm selected_algorithm() const {
        return selected_algorithm_;
    }
    GradientScalingContract scaling_contract() const;
    std::map<std::string, ReportValue> algorithm_report() const;

    static const char* algorithm_name(HierarchicalHostAlgorithm algorithm);

private:
    struct Slot {
        explicit Slot(uint32_t world_size);

        bool occupied = false;
        bool reduced_ready = false;
        uint64_t generation = 0;
        uint32_t bucket = 0;
        size_t float_count = 0;
        std::vector<bool> submitted;
        std::vector<bool> consumed;
        std::vector<const float*> inputs;
        std::vector<float> reduced;
    };

    struct SparseSlot {
        explicit SparseSlot(uint32_t world_size);

        bool occupied = false;
        bool reduced_ready = false;
        uint64_t generation = 0;
        size_t record_width = 0;
        uint32_t row_count = 0;
        std::vector<bool> submitted;
        std::vector<bool> consumed;
        std::vector<const uint32_t*> id_ptrs;
        std::vector<const float*> value_ptrs;
        std::vector<size_t> counts;
        SortedSparseMeanResult reduced;
    };

    void validate_rank(uint32_t rank) const;
    bool all_attached_locked() const;
    bool all_true(const std::vector<bool>& values) const;
    void validate_rank_sequence_locked(
        uint32_t rank, uint64_t generation, uint32_t bucket);
    void advance_rank_sequence_locked(
        uint32_t rank, uint64_t generation, uint32_t bucket);
    void initialize_slot_locked(
        Slot& slot, uint64_t generation, uint32_t bucket, size_t float_count);
    void abort_locked(const std::string& reason) noexcept;

    const HierarchicalHostCollectiveConfig config_;
    const HierarchicalHostAlgorithm selected_algorithm_;
    const uint32_t reducer_shards_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<bool> attached_;
    bool fully_attached_once_ = false;
    size_t payload_bytes_ = 0;
    std::string layout_signature_;
    std::vector<Slot> slots_;
    SparseSlot sparse_slot_;
    uint64_t sparse_generation_ = 0;
    std::vector<uint64_t> active_generation_;
    std::vector<uint32_t> next_bucket_;
    bool aborted_ = false;
    uint64_t abort_generation_ = 0;
    std::string abort_reason_;
};
