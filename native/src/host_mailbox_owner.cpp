#include "host_mailbox_owner.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace {
size_t checked_bucket_count(size_t payload_bytes, size_t bucket_bytes) {
    if (payload_bytes == 0 || bucket_bytes == 0)
        return 0;
    if (payload_bytes > std::numeric_limits<size_t>::max() - (bucket_bytes - 1))
        throw std::overflow_error("host mailbox bucket-count overflow");
    return (payload_bytes + bucket_bytes - 1) / bucket_bytes;
}
}

HostMailboxOwner::HostMailboxOwner(
    uint32_t world_size, size_t bucket_bytes)
    : world_size_(world_size),
      bucket_bytes_(std::max<size_t>(4, bucket_bytes - bucket_bytes % 4)),
      attached_(world_size, false),
      message_generation_(world_size, 0),
      message_bytes_(world_size, 0) {
    if (world_size_ != 2 && world_size_ != 4 &&
        world_size_ != 6 && world_size_ != 8)
        throw std::invalid_argument(
            "host mailbox v3 supports world_size 2, 4, 6, or 8");
    if (bucket_bytes == 0 || bucket_bytes_ == 0)
        throw std::invalid_argument("host mailbox bucket size must be non-zero");
    ring_.reserve(ring_depth_);
    for (size_t i = 0; i < ring_depth_; ++i)
        ring_.emplace_back(world_size_);
}

void HostMailboxOwner::validate_rank(uint32_t rank) const {
    if (rank >= world_size_)
        throw std::out_of_range("host mailbox rank is out of range");
}

bool HostMailboxOwner::all_attached_locked() const {
    return std::all_of(
        attached_.begin(), attached_.end(), [](bool value) { return value; });
}

bool HostMailboxOwner::peer_detached_locked(uint32_t rank) const {
    for (uint32_t peer = 0; peer < world_size_; ++peer) {
        if (peer != rank && !attached_[peer])
            return true;
    }
    return false;
}

void HostMailboxOwner::attach(
    uint32_t rank,
    size_t payload_bytes,
    const std::string& layout_signature) {
    validate_rank(rank);
    if (payload_bytes == 0 || layout_signature.empty())
        throw std::invalid_argument("host mailbox requires a non-empty packed ABI");
    std::lock_guard<std::mutex> lock(mutex_);
    if (attached_[rank])
        throw std::runtime_error("host mailbox rank was attached twice");
    if (aborted_)
        throw std::runtime_error("host mailbox is aborted: " + abort_reason_);
    if (payload_bytes_ == 0) {
        payload_bytes_ = payload_bytes;
        layout_signature_ = layout_signature;
        const size_t count = checked_bucket_count(payload_bytes_, bucket_bytes_);
        if (count > std::numeric_limits<uint32_t>::max())
            throw std::overflow_error(
                "host mailbox bucket count exceeds the collective ABI");
        buckets_.reserve(count);
        for (size_t i = 0; i < count; ++i)
            buckets_.emplace_back(world_size_);
    } else if (
        payload_bytes_ != payload_bytes || layout_signature_ != layout_signature) {
        throw std::runtime_error(
            "host mailbox replica packed-gradient ABI size/layout mismatch");
    }
    attached_[rank] = true;
    fully_attached_once_ = all_attached_locked();
    cv_.notify_all();
}

void HostMailboxOwner::detach(uint32_t rank) noexcept {
    if (rank >= world_size_)
        return;
    std::lock_guard<std::mutex> lock(mutex_);
    attached_[rank] = false;
    cv_.notify_all();
}

size_t HostMailboxOwner::payload_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return payload_bytes_;
}

size_t HostMailboxOwner::bucket_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buckets_.size();
}

void HostMailboxOwner::wait_writable(
    uint32_t rank, uint64_t generation, uint32_t bucket_index) {
    validate_rank(rank);
    if (generation == 0)
        throw std::invalid_argument("host mailbox generation zero is reserved");
    std::unique_lock<std::mutex> lock(mutex_);
    if (bucket_index >= buckets_.size())
        throw std::out_of_range("host mailbox bucket is out of range");
    RingSlotState& slot = ring_[bucket_index % ring_depth_];
    cv_.wait(lock, [&]() {
        if (aborted_ || !attached_[rank] ||
            (fully_attached_once_ && peer_detached_locked(rank)))
            return true;
        // Exchange threads attach lazily on their first collective. An unused
        // slot is writable only after every rank has attached.
        if (!all_attached_locked())
            return false;
        if (!slot.published_valid[rank])
            return true;
        for (uint32_t peer = 0; peer < world_size_; ++peer) {
            if (peer == rank)
                continue;
            if (!slot.consumed_valid[peer] ||
                slot.consumed_generation[peer] !=
                    slot.published_generation[rank] ||
                slot.consumed_bucket[peer] != slot.published_bucket[rank])
                return false;
        }
        return true;
    });
    if (aborted_)
        throw std::runtime_error("host mailbox exchange aborted: " + abort_reason_);
    if (!attached_[rank])
        throw std::runtime_error("host mailbox rank detached while waiting");
    if (peer_detached_locked(rank))
        throw std::runtime_error("host mailbox peer unavailable before ring reuse");
}

void HostMailboxOwner::publish_and_wait(
    uint32_t rank, uint64_t generation, uint32_t bucket_index) {
    validate_rank(rank);
    if (generation == 0)
        throw std::invalid_argument("host mailbox generation zero is reserved");
    std::unique_lock<std::mutex> lock(mutex_);
    if (bucket_index >= buckets_.size())
        throw std::out_of_range("host mailbox bucket is out of range");
    if (!attached_[rank])
        throw std::runtime_error("host mailbox publish by detached rank");
    BucketState& state = buckets_[bucket_index];
    if (state.published[rank] >= generation)
        throw std::runtime_error("host mailbox bucket generation was published twice");
    state.published[rank] = generation;
    RingSlotState& slot = ring_[bucket_index % ring_depth_];
    slot.published_valid[rank] = true;
    slot.published_generation[rank] = generation;
    slot.published_bucket[rank] = bucket_index;
    cv_.notify_all();
    cv_.wait(lock, [&]() {
        if (aborted_ || (fully_attached_once_ && peer_detached_locked(rank)))
            return true;
        if (!all_attached_locked())
            return false;
        return std::all_of(
            state.published.begin(), state.published.end(),
            [generation](uint64_t value) { return value >= generation; });
    });
    if (aborted_)
        throw std::runtime_error("host mailbox exchange aborted: " + abort_reason_);
    if (peer_detached_locked(rank))
        throw std::runtime_error("host mailbox peer unavailable during exchange");
    for (uint64_t value : state.published) {
        if (value != generation)
            throw std::runtime_error(
                "host mailbox ranks published different optimizer generations");
    }
}

void HostMailboxOwner::publish_borrowed_and_wait(
    uint32_t rank, uint64_t generation, uint32_t bucket_index,
    const void* pointer, size_t bytes) {
    validate_rank(rank);
    if (generation == 0 || pointer == nullptr || bytes == 0 ||
        bytes > bucket_bytes_)
        throw std::invalid_argument("borrowed mailbox publication is invalid");
    std::unique_lock<std::mutex> lock(mutex_);
    if (bucket_index >= buckets_.size())
        throw std::out_of_range("borrowed mailbox bucket is out of range");
    if (!attached_[rank])
        throw std::runtime_error("borrowed mailbox publish by detached rank");
    BucketState& state = buckets_[bucket_index];
    if (state.published[rank] >= generation)
        throw std::runtime_error("borrowed mailbox bucket was published twice");
    RingSlotState& slot = ring_[bucket_index % ring_depth_];
    slot.borrowed_pointer[rank] = pointer;
    slot.borrowed_bytes[rank] = bytes;
    state.published[rank] = generation;
    slot.published_valid[rank] = true;
    slot.published_generation[rank] = generation;
    slot.published_bucket[rank] = bucket_index;
    cv_.notify_all();
    cv_.wait(lock, [&]() {
        if (aborted_ || (fully_attached_once_ && peer_detached_locked(rank)))
            return true;
        if (!all_attached_locked())
            return false;
        return std::all_of(
            state.published.begin(), state.published.end(),
            [generation](uint64_t value) { return value >= generation; });
    });
    if (aborted_)
        throw std::runtime_error(
            "borrowed mailbox exchange aborted: " + abort_reason_);
    if (peer_detached_locked(rank))
        throw std::runtime_error("borrowed mailbox peer unavailable");
    for (uint64_t value : state.published) {
        if (value != generation)
            throw std::runtime_error(
                "borrowed mailbox ranks published different generations");
    }
}

const void* HostMailboxOwner::peer_borrowed_pointer(
    uint32_t rank, uint32_t peer_rank, uint64_t generation,
    uint32_t bucket_index,
    size_t bytes) const {
    validate_rank(rank);
    validate_rank(peer_rank);
    if (peer_rank == rank)
        throw std::invalid_argument("borrowed mailbox peer rank equals local rank");
    std::lock_guard<std::mutex> lock(mutex_);
    if (bucket_index >= buckets_.size())
        throw std::out_of_range("borrowed mailbox bucket is out of range");
    const RingSlotState& slot = ring_[bucket_index % ring_depth_];
    if (!slot.published_valid[peer_rank] ||
        slot.published_generation[peer_rank] != generation ||
        slot.published_bucket[peer_rank] != bucket_index ||
        slot.borrowed_pointer[peer_rank] == nullptr ||
        slot.borrowed_bytes[peer_rank] != bytes)
        throw std::runtime_error(
            "borrowed mailbox peer pointer does not match the bucket ABI");
    return slot.borrowed_pointer[peer_rank];
}

void HostMailboxOwner::mark_consumed(
    uint32_t rank, uint64_t generation, uint32_t bucket_index) {
    validate_rank(rank);
    std::lock_guard<std::mutex> lock(mutex_);
    if (bucket_index >= buckets_.size())
        throw std::out_of_range("host mailbox consumed bucket is out of range");
    BucketState& state = buckets_[bucket_index];
    for (uint64_t value : state.published) {
        if (value != generation)
            throw std::runtime_error(
                "host mailbox consumed an unpublished generation");
    }
    state.consumed[rank] = generation;
    RingSlotState& slot = ring_[bucket_index % ring_depth_];
    slot.consumed_valid[rank] = true;
    slot.consumed_generation[rank] = generation;
    slot.consumed_bucket[rank] = bucket_index;
    cv_.notify_all();
}

void HostMailboxOwner::publish_message_and_wait(
    uint32_t rank, uint64_t generation, size_t message_bytes) {
    validate_rank(rank);
    if (generation == 0)
        throw std::invalid_argument("host mailbox message generation is invalid");
    std::unique_lock<std::mutex> lock(mutex_);
    if (message_bytes > payload_bytes_)
        throw std::invalid_argument("host mailbox message size is invalid");
    if (aborted_)
        throw std::runtime_error("host mailbox is aborted: " + abort_reason_);
    if (!attached_[rank])
        throw std::runtime_error("host mailbox message by detached rank");
    if (message_generation_[rank] >= generation)
        throw std::runtime_error("host mailbox message generation was published twice");
    message_bytes_[rank] = message_bytes;
    message_generation_[rank] = generation;
    cv_.notify_all();
    cv_.wait(lock, [&]() {
        if (aborted_ || (fully_attached_once_ && peer_detached_locked(rank)))
            return true;
        if (!all_attached_locked())
            return false;
        return std::all_of(
            message_generation_.begin(), message_generation_.end(),
            [generation](uint64_t value) { return value >= generation; });
    });
    if (aborted_)
        throw std::runtime_error("host mailbox message aborted: " + abort_reason_);
    if (peer_detached_locked(rank))
        throw std::runtime_error("host mailbox peer detached during message exchange");
    for (uint64_t value : message_generation_) {
        if (value != generation)
            throw std::runtime_error(
                "host mailbox ranks published different message generations");
    }
}

size_t HostMailboxOwner::peer_message_bytes(
    uint32_t rank, uint32_t peer_rank, uint64_t generation) const {
    validate_rank(rank);
    validate_rank(peer_rank);
    if (peer_rank == rank)
        throw std::invalid_argument("host mailbox peer rank equals local rank");
    std::lock_guard<std::mutex> lock(mutex_);
    if (message_generation_[peer_rank] != generation)
        throw std::runtime_error("host mailbox peer message is not published");
    return message_bytes_[peer_rank];
}

void HostMailboxOwner::abort(const std::string& reason) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    aborted_ = true;
    abort_reason_ = reason.empty() ? "unspecified rank failure" : reason;
    cv_.notify_all();
}

void HostMailboxOwner::clear_abort() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    aborted_ = false;
    abort_reason_.clear();
    for (auto& bucket : buckets_) {
        std::fill(bucket.published.begin(), bucket.published.end(), 0);
        std::fill(bucket.consumed.begin(), bucket.consumed.end(), 0);
    }
    for (auto& slot : ring_) {
        std::fill(slot.published_valid.begin(), slot.published_valid.end(), false);
        std::fill(slot.published_generation.begin(), slot.published_generation.end(), 0);
        std::fill(slot.published_bucket.begin(), slot.published_bucket.end(), 0);
        std::fill(slot.consumed_valid.begin(), slot.consumed_valid.end(), false);
        std::fill(slot.consumed_generation.begin(), slot.consumed_generation.end(), 0);
        std::fill(slot.consumed_bucket.begin(), slot.consumed_bucket.end(), 0);
        std::fill(
            slot.borrowed_pointer.begin(), slot.borrowed_pointer.end(), nullptr);
        std::fill(slot.borrowed_bytes.begin(), slot.borrowed_bytes.end(), 0);
    }
    std::fill(message_generation_.begin(), message_generation_.end(), 0);
    std::fill(message_bytes_.begin(), message_bytes_.end(), 0);
    cv_.notify_all();
}

std::map<std::string, HostMailboxOwner::ReportValue>
HostMailboxOwner::report() const {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t attached_count = static_cast<size_t>(std::count(
        attached_.begin(), attached_.end(), true));
    return {
        {"abi", std::string("ftgspp-host-mailbox-v3")},
        {"world_size", world_size_},
        {"payload_bytes", static_cast<uint64_t>(payload_bytes_)},
        {"bucket_bytes", static_cast<uint64_t>(bucket_bytes_)},
        {"bucket_count", static_cast<uint64_t>(buckets_.size())},
        {"ring_depth", static_cast<uint64_t>(ring_depth_)},
        {"host_storage_bytes", static_cast<uint64_t>(0)},
        {"borrowed_storage", true},
        {"attached_count", static_cast<uint64_t>(attached_count)},
        {"cpu_reduction", false},
        {"aborted", aborted_},
    };
}
