#include "hierarchical_host_collective.h"
#include "host_mailbox_owner.h"
#include "sync_generation_state.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

template<typename T>
T report_value(
    const std::map<std::string, HierarchicalHostCollective::ReportValue>& report,
    const std::string& key) {
    const auto found = report.find(key);
    if (found == report.end())
        throw std::runtime_error("missing report key: " + key);
    return std::get<T>(found->second);
}

void attach_all(
    HierarchicalHostCollective& collective, uint32_t world_size) {
    for (uint32_t rank = 0; rank < world_size; ++rank)
        collective.attach(rank, 4096, "dense-fp32-v1");
}

std::vector<std::vector<float>> run_bucket(
    HierarchicalHostCollective& collective,
    uint64_t generation,
    uint32_t bucket,
    const std::vector<std::vector<float>>& inputs) {
    std::vector<std::vector<float>> outputs(
        inputs.size(), std::vector<float>(inputs.front().size(), 0.0f));
    std::vector<std::exception_ptr> failures(inputs.size());
    std::vector<std::thread> ranks;
    for (uint32_t rank = 0; rank < inputs.size(); ++rank) {
        ranks.emplace_back([&, rank]() {
            try {
                collective.reduce_bucket(
                    rank, generation, bucket,
                    inputs[rank].data(), inputs[rank].size(),
                    outputs[rank].data());
            } catch (...) {
                failures[rank] = std::current_exception();
            }
        });
    }
    for (auto& rank : ranks)
        rank.join();
    for (const auto& failure : failures) {
        if (failure)
            std::rethrow_exception(failure);
    }
    return outputs;
}

void test_scaling_contract() {
    std::vector<float> local = {8.0f, -4.0f, 2.0f};
    finalize_local_gradient_fp32(local.data(), local.size(), 4);
    require(
        local == std::vector<float>({2.0f, -1.0f, 0.5f}),
        "local finalize did not apply exactly 1/K");

    const GradientScalingContract scaling =
        make_gradient_scaling_contract(4, 8);
    require(scaling.local_finalize_scale == 0.25f, "wrong 1/K scale");
    require(scaling.collective_scale == 0.125f, "wrong 1/P scale");
    require(scaling.global_scale == 0.03125f, "wrong 1/(PK) scale");
}

void test_pair_path_does_not_construct_central_collective() {
    HierarchicalHostCollectiveConfig config;
    config.world_size = 2;
    config.bucket_count = 2;
    config.bucket_float_capacity = 64;
    bool rejected = false;
    try {
        HierarchicalHostCollective collective(config);
        (void)collective;
    } catch (const std::invalid_argument& error) {
        rejected = std::string(error.what()).find("[3, 8]") != std::string::npos;
    }
    require(
        rejected,
        "P=2 must use HostMailboxOwner directly, not HierarchicalHostCollective");
}

void test_central_fixed_rank_and_slots() {
    HierarchicalHostCollectiveConfig config;
    config.world_size = 4;
    config.bucket_count = 2;
    config.bucket_float_capacity = 8;
    config.ring_depth = 1;
    config.local_microsteps = 2;
    config.algorithm = HierarchicalHostAlgorithm::Central;
    HierarchicalHostCollective collective(config);
    attach_all(collective, config.world_size);

    std::vector<std::vector<float>> first = {
        {1.0e20f, 2.0f, 8.0f},
        {-1.0e20f, 4.0f, 6.0f},
        {3.0f, 6.0f, 4.0f},
        {1.0f, 8.0f, 2.0f},
    };
    const auto output0 = run_bucket(collective, 7, 0, first);
    const std::vector<float> expected0 = {1.0f, 5.0f, 5.0f};
    for (const auto& rank_output : output0)
        require(rank_output == expected0, "central rank outputs differ");

    const std::vector<std::vector<float>> second = {
        {4.0f, 8.0f}, {8.0f, 12.0f}, {12.0f, 16.0f}, {16.0f, 20.0f},
    };
    const auto output1 = run_bucket(collective, 7, 1, second);
    for (const auto& rank_output : output1)
        require(
            rank_output == std::vector<float>({10.0f, 14.0f}),
            "ring slot reuse changed the central result");

    const auto next_generation = run_bucket(collective, 8, 0, first);
    require(
        next_generation.front() == expected0,
        "next generation did not reuse the ring safely");
    const auto report = collective.algorithm_report();
    require(
        report_value<bool>(report, "deterministic_fp32_rank_order"),
        "fixed-rank FP32 contract missing from report");
    require(
        report_value<float>(report, "local_finalize_scale") == 0.5f,
        "algorithm report lost K scaling");
    require(
        report_value<float>(report, "collective_scale") == 0.25f,
        "algorithm report lost P scaling");
}

void test_central_p8_exact_and_single_reduced_slab() {
    HierarchicalHostCollectiveConfig config;
    config.world_size = 8;
    config.bucket_count = 1;
    config.bucket_float_capacity = 8;
    config.ring_depth = 1;
    config.algorithm = HierarchicalHostAlgorithm::Central;
    HierarchicalHostCollective collective(config);
    attach_all(collective, config.world_size);

    std::vector<std::vector<float>> inputs;
    for (uint32_t rank = 0; rank < config.world_size; ++rank) {
        inputs.push_back({
            static_cast<float>(rank + 1),
            static_cast<float>(2 * (rank + 1)),
            static_cast<float>(4 * (rank + 1)),
        });
    }
    std::vector<const float*> reduced_pointers(config.world_size, nullptr);
    std::vector<std::exception_ptr> failures(config.world_size);
    std::vector<std::thread> ranks;
    for (uint32_t rank = 0; rank < config.world_size; ++rank) {
        ranks.emplace_back([&, rank]() {
            try {
                reduced_pointers[rank] = collective.acquire_reduced_bucket(
                    rank, 1, 0, inputs[rank].data(), inputs[rank].size());
                require(
                    std::vector<float>(
                        reduced_pointers[rank],
                        reduced_pointers[rank] + inputs[rank].size()) ==
                        std::vector<float>({4.5f, 9.0f, 18.0f}),
                    "P=8 central reduction is not exact");
                collective.release_bucket(rank, 1, 0);
            } catch (...) {
                failures[rank] = std::current_exception();
            }
        });
    }
    for (auto& rank : ranks)
        rank.join();
    for (const auto& failure : failures) {
        if (failure)
            std::rethrow_exception(failure);
    }
    for (const float* pointer : reduced_pointers)
        require(
            pointer == reduced_pointers.front(),
            "P=8 ranks did not borrow one shared reduced slab");
}

void test_generation_mismatch_aborts_waiters() {
    HierarchicalHostCollectiveConfig config;
    config.world_size = 3;
    config.bucket_count = 1;
    config.bucket_float_capacity = 4;
    config.algorithm = HierarchicalHostAlgorithm::Central;
    HierarchicalHostCollective collective(config);
    attach_all(collective, config.world_size);

    const float value = 1.0f;
    std::vector<uint8_t> failed(3, 0);
    std::vector<std::thread> ranks;
    for (uint32_t rank = 0; rank < 3; ++rank) {
        ranks.emplace_back([&, rank]() {
            float output = 0.0f;
            try {
                collective.reduce_bucket(
                    rank, rank == 1 ? 2 : 1, 0,
                    &value, 1, &output);
            } catch (const std::exception&) {
                failed[rank] = 1;
            }
        });
    }
    for (auto& rank : ranks)
        rank.join();
    require(
        failed[0] && failed[1] && failed[2],
        "generation mismatch did not wake and fail every rank");
    const auto report = collective.algorithm_report();
    require(
        report_value<bool>(report, "aborted"),
        "generation mismatch did not persist abort state");
    require(
        report_value<uint64_t>(report, "abort_generation") != 0,
        "abort report omitted its generation");
}

void test_exact_sorted_sparse_mean_uses_full_world_size() {
    // Rank contributions for IDs 0/1/2 with record_width=2. Rank 2 is empty.
    const uint32_t ids0[] = {0u, 2u};
    const float values0[] = {2.0f, 4.0f, 6.0f, 8.0f};
    const uint32_t ids1[] = {1u, 2u};
    const float values1[] = {10.0f, 12.0f, 14.0f, 16.0f};
    const uint32_t ids3[] = {0u};
    const float values3[] = {18.0f, 20.0f};
    SortedSparseRankInput ranks[4] = {
        {ids0, values0, 2},
        {ids1, values1, 2},
        {nullptr, nullptr, 0},
        {ids3, values3, 1},
    };
    const auto reduced = exact_sorted_sparse_mean_fp32(ranks, 4, 2, 3);
    require(reduced.ids.size() == 3, "sparse union missed an id");
    require(reduced.ids[0] == 0u && reduced.ids[1] == 1u && reduced.ids[2] == 2u,
            "sparse union is not ID sorted");
    require(
        reduced.values == std::vector<float>(
            {5.0f, 6.0f, 2.5f, 3.0f, 5.0f, 6.0f}),
        "sparse mean did not divide by the full world size");
    require(
        std::string(choose_exact_bucket_codec(1.0, 0.9, true)) ==
            "exact_sorted_sparse",
        "codec should prefer a 10% faster exact sparse path");
    require(
        std::string(choose_exact_bucket_codec(1.0, 0.96, true)) == "dense",
        "codec must fall back when sparse gain is under 5%");
}

void test_hierarchical_exact_codec_is_identical_across_ranks() {
    const uint32_t ids0[] = {0u, 1u};
    const uint32_t ids1[] = {0u, 2u, 3u, 4u};
    const uint32_t ids2[] = {1u, 2u};
    const uint32_t ids3[] = {4u};
    HierarchicalExactCodecVote votes[4] = {
        {2000u, 10000u, 0u, ids0, 2u, 1000u},
        {4000u, 10000u, 0u, ids1, 4u, 1000u},
        {2000u, 10000u, 0u, ids2, 2u, 1000u},
        {1000u, 10000u, 0u, ids3, 1u, 1000u},
    };
    uint64_t union_count = 0;
    const char* first = decide_hierarchical_exact_codec(
        votes, 4, 0.05, &union_count);
    require(union_count == 5u, "overlapping P>=3 union should be exact ID count");
    for (uint32_t rank = 0; rank < 4; ++rank) {
        HierarchicalExactCodecVote rotated[4];
        for (uint32_t index = 0; index < 4; ++index)
            rotated[index] = votes[(index + rank) % 4];
        require(
            std::string(decide_hierarchical_exact_codec(rotated, 4)) ==
                std::string(first),
            "P>=3 codec decision must not depend on vote presentation order "
            "of identical payloads");
    }
    const uint32_t same0[] = {0u, 1u};
    const uint32_t same1[] = {0u, 1u};
    const uint32_t same2[] = {0u, 1u};
    const uint32_t same3[] = {0u, 1u};
    HierarchicalExactCodecVote tiny[4] = {
        {100u, 100000u, 0u, same0, 2u, 50u},
        {100u, 100000u, 0u, same1, 2u, 50u},
        {100u, 100000u, 0u, same2, 2u, 50u},
        {100u, 100000u, 0u, same3, 2u, 50u},
    };
    require(
        std::string(decide_hierarchical_exact_codec(tiny, 4)) ==
            "exact_sorted_sparse",
        "cheap exact sparse should beat dense on the P>=3 PCIe model");
    HierarchicalExactCodecVote empty[3] = {
        {0u, 8000u}, {0u, 8000u}, {0u, 8000u},
    };
    require(
        std::string(decide_hierarchical_exact_codec(empty, 3)) == "dense",
        "empty sparse payloads must fall back to dense");
}

void test_hierarchical_exact_codec_uses_real_union_not_max_local() {
    constexpr uint32_t kCount = 500;
    constexpr uint64_t kRecordBytes = 100;
    constexpr uint64_t kLocalBytes = kCount * kRecordBytes;
    constexpr uint64_t kDenseBytes = 100000;
    std::vector<std::vector<uint32_t>> disjoint(4, std::vector<uint32_t>(kCount));
    std::vector<std::vector<uint32_t>> overlap(4, std::vector<uint32_t>(kCount));
    for (uint32_t rank = 0; rank < 4; ++rank) {
        for (uint32_t index = 0; index < kCount; ++index) {
            disjoint[rank][index] = rank * kCount + index;
            overlap[rank][index] = index;
        }
    }
    HierarchicalExactCodecVote disjoint_votes[4];
    HierarchicalExactCodecVote overlap_votes[4];
    for (uint32_t rank = 0; rank < 4; ++rank) {
        disjoint_votes[rank] = {
            kLocalBytes, kDenseBytes, 0u,
            disjoint[rank].data(), kCount, kRecordBytes,
        };
        overlap_votes[rank] = {
            kLocalBytes, kDenseBytes, 0u,
            overlap[rank].data(), kCount, kRecordBytes,
        };
    }
    uint64_t disjoint_union = 0;
    uint64_t overlap_union = 0;
    require(
        std::string(decide_hierarchical_exact_codec(
            disjoint_votes, 4, 0.05, &disjoint_union)) == "dense",
        "disjoint visible sets must cost P*union, not P*max(local)");
    require(
        disjoint_union == 4u * kCount,
        "disjoint union should be the sum of local counts");
    require(
        std::string(decide_hierarchical_exact_codec(
            overlap_votes, 4, 0.05, &overlap_union)) ==
            "exact_sorted_sparse",
        "fully overlapping visible sets should keep sparse cheaper than dense");
    require(
        overlap_union == kCount,
        "overlapping union should equal one rank's local count");

    const uint32_t* none[3] = {nullptr, nullptr, nullptr};
    const size_t zero[3] = {0, 0, 0};
    require(
        count_sorted_sparse_union(none, zero, 3) == 0,
        "empty union count should be zero");
}

void test_central_vector_payload_rank_order_mean() {
    HierarchicalHostCollectiveConfig config;
    config.world_size = 4;
    config.bucket_count = 1;
    config.bucket_float_capacity = 64;
    config.algorithm = HierarchicalHostAlgorithm::Central;
    HierarchicalHostCollective collective(config);
    attach_all(collective, config.world_size);

    std::vector<std::vector<float>> inputs(4, std::vector<float>(32, 0.0f));
    for (uint32_t rank = 0; rank < 4; ++rank) {
        for (size_t index = 0; index < 32; ++index)
            inputs[rank][index] = static_cast<float>(rank + 1);
    }
    const auto output = run_bucket(collective, 1, 0, inputs);
    const std::vector<float> expected(32, 2.5f);
    for (const auto& rank_output : output)
        require(rank_output == expected, "AVX2/scalar central mean diverged");
}

void test_central_sorted_sparse_collective_p4() {
    HierarchicalHostCollectiveConfig config;
    config.world_size = 4;
    config.bucket_count = 1;
    config.bucket_float_capacity = 8;
    config.algorithm = HierarchicalHostAlgorithm::Central;
    HierarchicalHostCollective collective(config);
    attach_all(collective, config.world_size);

    const uint32_t ids0[] = {0u, 2u};
    const float values0[] = {4.0f, 8.0f, 12.0f, 16.0f};
    const uint32_t ids1[] = {1u};
    const float values1[] = {20.0f, 24.0f};
    const uint32_t ids2[] = {0u, 1u};
    const float values2[] = {28.0f, 32.0f, 36.0f, 40.0f};
    const uint32_t* ids3 = nullptr;
    const float* values3 = nullptr;
    const uint32_t* all_ids[] = {ids0, ids1, ids2, ids3};
    const float* all_values[] = {values0, values1, values2, values3};
    const size_t counts[] = {2, 1, 2, 0};
    std::vector<SortedSparseMeanResult> outputs(4);
    std::vector<std::exception_ptr> failures(4);
    std::vector<std::thread> ranks;
    for (uint32_t rank = 0; rank < 4; ++rank) {
        ranks.emplace_back([&, rank]() {
            try {
                const auto& reduced = collective.acquire_reduced_sorted_sparse(
                    rank, 11, all_ids[rank], all_values[rank], counts[rank],
                    2, 3);
                outputs[rank] = reduced;
                collective.release_sorted_sparse(rank, 11);
            } catch (...) {
                failures[rank] = std::current_exception();
            }
        });
    }
    for (auto& rank : ranks)
        rank.join();
    for (const auto& failure : failures) {
        if (failure)
            std::rethrow_exception(failure);
    }
    const std::vector<uint32_t> expected_ids = {0u, 1u, 2u};
    const std::vector<float> expected_values = {
        8.0f, 10.0f, 14.0f, 16.0f, 3.0f, 4.0f};
    for (const auto& output : outputs) {
        require(output.ids == expected_ids, "P4 sparse collective ids differ");
        require(
            output.values == expected_values,
            "P4 sparse collective values differ");
    }
}

void test_non_finite_input_aborts_every_rank() {
    HierarchicalHostCollectiveConfig config;
    config.world_size = 4;
    config.bucket_count = 1;
    config.bucket_float_capacity = 4;
    config.algorithm = HierarchicalHostAlgorithm::Central;
    HierarchicalHostCollective collective(config);
    attach_all(collective, config.world_size);

    std::vector<std::vector<float>> inputs(
        config.world_size, std::vector<float>({1.0f, 2.0f}));
    inputs[2][1] = std::numeric_limits<float>::quiet_NaN();
    std::vector<uint8_t> failed(config.world_size, 0);
    std::vector<std::thread> ranks;
    for (uint32_t rank = 0; rank < config.world_size; ++rank) {
        ranks.emplace_back([&, rank]() {
            float output[2] = {};
            try {
                collective.reduce_bucket(
                    rank, 1, 0, inputs[rank].data(), 2, output);
            } catch (const std::exception&) {
                failed[rank] = 1;
            }
        });
    }
    for (auto& rank : ranks)
        rank.join();
    require(
        std::all_of(
            failed.begin(), failed.end(),
            [](uint8_t value) { return value != 0; }),
        "non-finite input did not abort every rank");
    const auto report = collective.algorithm_report();
    require(
        report_value<bool>(report, "aborted"),
        "non-finite reduction did not persist abort state");
    require(
        report_value<uint64_t>(report, "abort_generation") == 1,
        "non-finite reduction reported the wrong abort generation");
}

void test_mailbox_clear_abort_allows_replay() {
    HostMailboxOwner owner(2, 64);
    owner.attach(0, 64, "dense-v1");
    owner.attach(1, 64, "dense-v1");
    owner.abort("tile_overflow_generation");
    bool threw = false;
    try {
        owner.wait_writable(0, 1, 0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "aborted mailbox did not wake wait_writable");
    owner.clear_abort();
    owner.wait_writable(0, 1, 0);
}

void test_sync_generation_p2_local_abort_and_p3_barrier() {
    {
        SyncGenerationState state(2);
        SyncGenerationState::RankReport overflow;
        overflow.overflow = true;
        overflow.exact_count = 1000;
        const auto first = state.report_local(0, 7, overflow);
        require(first.aborted, "P=2 local overflow did not abort");
        require(first.max_exact_count == 1000, "P=2 lost exact_count");
        SyncGenerationState::RankReport ok;
        ok.exact_count = 400;
        const auto second = state.report_local(1, 7, ok);
        require(second.aborted, "P=2 peer did not see abort");
        require(second.max_exact_count == 1000, "P=2 max exact used the overflow rank");
        state.clear_for_replay();
        const auto replay = state.report_local(0, 7, ok);
        require(!replay.aborted, "clear_for_replay left abort latched");
    }
    {
        SyncGenerationState state(3);
        std::vector<SyncGenerationState::Decision> decisions(3);
        std::vector<std::thread> ranks;
        for (uint32_t rank = 0; rank < 3; ++rank) {
            ranks.emplace_back([&, rank]() {
                SyncGenerationState::RankReport report;
                report.overflow = rank == 1;
                report.exact_count = rank == 1 ? 2048u : 128u;
                decisions[rank] = state.report_and_wait(rank, 3, report);
            });
        }
        for (auto& rank : ranks)
            rank.join();
        for (const auto& decision : decisions) {
            require(decision.aborted, "P=3 overflow was not global");
            require(
                decision.max_exact_count == 2048,
                "P=3 did not broadcast max exact_count");
        }
    }
}

}  // namespace

int main() {
    try {
        test_scaling_contract();
        test_pair_path_does_not_construct_central_collective();
        test_central_fixed_rank_and_slots();
        test_central_p8_exact_and_single_reduced_slab();
        test_exact_sorted_sparse_mean_uses_full_world_size();
        test_hierarchical_exact_codec_is_identical_across_ranks();
        test_hierarchical_exact_codec_uses_real_union_not_max_local();
        test_central_vector_payload_rank_order_mean();
        test_central_sorted_sparse_collective_p4();
        test_generation_mismatch_aborts_waiters();
        test_non_finite_input_aborts_every_rank();
        test_mailbox_clear_abort_allows_replay();
        test_sync_generation_p2_local_abort_and_p3_barrier();
    } catch (const std::exception& error) {
        std::cerr << "hierarchical host collective test failed: "
                  << error.what() << '\n';
        return 1;
    }
    std::cout << "hierarchical host collective contracts passed\n";
    return 0;
}
