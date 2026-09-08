#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <variant>
#include <vector>

// Multi-device collective coordinator for machines where Vulkan peer memory is
// unavailable. Payload storage is borrowed from each trainer's persistently
// mapped Vulkan readback ring, so this object owns synchronization metadata but
// no duplicate byte arena. Peers copy each published mapping into their private
// upload stager; a Vulkan compute pass performs the mean before Adam.
class HostMailboxOwner {
public:
    using ReportValue = std::variant<uint64_t, uint32_t, bool, std::string>;

    HostMailboxOwner(uint32_t world_size, size_t bucket_bytes);

    void attach(
        uint32_t rank,
        size_t payload_bytes,
        const std::string& layout_signature);
    void detach(uint32_t rank) noexcept;

    size_t payload_bytes() const;
    size_t bucket_bytes() const { return bucket_bytes_; }
    size_t bucket_count() const;
    uint32_t world_size() const { return world_size_; }

    // Publish one bucket for a monotonically increasing optimizer generation.
    // The wait releases the mutex, so both rank worker threads may enter it.
    void publish_and_wait(
        uint32_t rank, uint64_t generation, uint32_t bucket_index);
    void publish_borrowed_and_wait(
        uint32_t rank, uint64_t generation, uint32_t bucket_index,
        const void* pointer, size_t bytes);
    const void* peer_borrowed_pointer(
        uint32_t rank, uint32_t peer_rank, uint64_t generation,
        uint32_t bucket_index,
        size_t bytes) const;
    void wait_writable(
        uint32_t rank, uint64_t generation, uint32_t bucket_index);
    void mark_consumed(
        uint32_t rank, uint64_t generation, uint32_t bucket_index);
    void publish_message_and_wait(
        uint32_t rank, uint64_t generation, size_t message_bytes);
    size_t peer_message_bytes(
        uint32_t rank, uint32_t peer_rank, uint64_t generation) const;
    void abort(const std::string& reason) noexcept;
    // Drop a recoverable overflow abort so the same optimizer generation can
    // be replayed after every rank has drained and grown tile capacity.
    void clear_abort() noexcept;

    std::map<std::string, ReportValue> report() const;

private:
    struct BucketState {
        explicit BucketState(uint32_t world_size)
            : published(world_size, 0), consumed(world_size, 0) {}
        std::vector<uint64_t> published;
        std::vector<uint64_t> consumed;
    };
    struct RingSlotState {
        explicit RingSlotState(uint32_t world_size)
            : published_valid(world_size, false),
              published_generation(world_size, 0),
              published_bucket(world_size, 0),
              consumed_valid(world_size, false),
              consumed_generation(world_size, 0),
              consumed_bucket(world_size, 0),
              borrowed_pointer(world_size, nullptr),
              borrowed_bytes(world_size, 0) {}
        std::vector<bool> published_valid;
        std::vector<uint64_t> published_generation;
        std::vector<uint32_t> published_bucket;
        std::vector<bool> consumed_valid;
        std::vector<uint64_t> consumed_generation;
        std::vector<uint32_t> consumed_bucket;
        std::vector<const void*> borrowed_pointer;
        std::vector<size_t> borrowed_bytes;
    };

    void validate_rank(uint32_t rank) const;
    bool all_attached_locked() const;
    bool peer_detached_locked(uint32_t rank) const;

    const uint32_t world_size_;
    const size_t bucket_bytes_;
    static constexpr size_t ring_depth_ = 3;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    size_t payload_bytes_ = 0;
    std::string layout_signature_;
    std::vector<bool> attached_;
    bool fully_attached_once_ = false;
    std::vector<BucketState> buckets_;
    std::vector<RingSlotState> ring_;
    std::vector<uint64_t> message_generation_;
    std::vector<size_t> message_bytes_;
    bool aborted_ = false;
    std::string abort_reason_;
};
