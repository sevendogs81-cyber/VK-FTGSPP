#include "collective/host_mailbox_owner.h"
#include "collective/sync_generation_state.h"
#include "collective/pair_session.h"
#include "render/scheduler.h"
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class Function> void rejects(Function fn) {
    bool threw = false;
    try { fn(); } catch (const std::exception&) { threw = true; }
    require(threw, "operation should reject invalid state");
}
int main() {
    try {
        ThompsonSamplingScheduler scheduler(2);
        scheduler.update(0, 0.001);
        scheduler.update(1, 0.002);
        auto encoded = scheduler.checkpoint();
        ThompsonSamplingScheduler restored(2);
        restored.restore(encoded);
        require(encoded == restored.checkpoint(), "scheduler checkpoint did not round trip");
        rejects([] { HostMailboxOwner owner(4, 16); });
        rejects([] { SyncGenerationState state(3); });
        rejects([] { HostMailboxOwner owner(2, 16, 0); });
        require(std::string(choose_exact_bucket_codec(100, 96, true)) == "dense",
                "codec must save at least five percent");
        require(std::string(choose_exact_bucket_codec(100, 95, true)) == "exact_sorted_sparse",
                "exact five percent threshold");
        HostMailboxOwner owner(2, 16, 0.2);
        owner.attach(0, 16, "pair-test");
        owner.attach(1, 16, "pair-test");
        auto rank = std::async(std::launch::async, [&] { owner.publish_and_wait(0, 1, 0); });
        owner.publish_and_wait(1, 1, 0);
        rank.get();
        owner.abort("injected failure");
        rejects([&] { owner.publish_and_wait(0, 2, 0); });
        owner.clear_abort();
        rank = std::async(std::launch::async, [&] { owner.publish_and_wait(0, 2, 0); });
        owner.publish_and_wait(1, 2, 0);
        rank.get();
        HostMailboxOwner missing_peer(2, 16, 0.02);
        missing_peer.attach(0, 16, "timeout");
        const auto begin = std::chrono::steady_clock::now();
        rejects([&] { missing_peer.publish_and_wait(0, 1, 0); });
        require(std::chrono::steady_clock::now() - begin < std::chrono::seconds(1),
                "missing peer did not time out");
        SyncGenerationState state(2, 0.02);
        SyncGenerationState::RankReport ok{}, overflow{};
        overflow.overflow = true;
        overflow.exact_count = 4096;
        require(!state.report_local(0, 7, ok).aborted, "fresh generation aborted");
        require(state.report_local(1, 7, overflow).aborted, "overflow not global");
        require(state.wait_decision(0, 7).max_exact_count == 4096, "lost capacity");
        state.clear_for_replay();
        state.report_local(0, 7, ok);
        state.report_local(1, 7, ok);
        require(!state.wait_decision(0, 7).aborted, "replay kept abort state");
        rejects([&] { state.report_local(0, 7, ok); });
        SyncGenerationState missing_report(2, 0.02);
        missing_report.report_local(0, 1, ok);
        rejects([&] { missing_report.wait_decision(0, 1); });
        SyncGenerationState mismatch(2, 0.02);
        mismatch.report_local(0, 1, ok);
        rejects([&] { mismatch.report_local(1, 2, ok); });
        require(mismatch.wait_decision(0, 1).fatal, "generation mismatch did not abort peer");
        PairSession first, second;
        first.ensure_generation(2);
        second.ensure_generation(2);
        first.abort("failure");
        second.generation->report_local(0, 1, ok);
        second.generation->report_local(1, 1, ok);
        require(!second.generation->wait_decision(0, 1).aborted, "sessions share state");
        std::cout << "pair collective contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
