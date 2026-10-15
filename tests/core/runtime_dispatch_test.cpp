#include "../check.hpp"
#include "../../src/runtime/model_impl.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <unistd.h>

class Source final : public guild::core::ExpertSource {
public:
    bool missing = false;
    std::vector<uint8_t> weights;
    const uint8_t* blob(int64_t, int64_t) override { return missing ? nullptr : weights.data(); }
};

int main() {
    using namespace guild;
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / ("guild-dispatch-test-" + std::to_string(getpid()));
    fs::create_directories(root);
    kernels::cpu::NativeFmt fmt;
    std::string err;
    CHECK(kernels::cpu::native_fmt(8, 8, 2560, 640, fmt, err));
    {
        std::ofstream layout(root / "native_experts.txt");
        layout << "# guild native experts v3: (n_expert 2)\n0 8 8 0 " << fmt.bytes << "\n";
    }
    CHECK(kernels::cpu::expert_layout_load(root.string(), 1, 2, err));
    std::vector<std::pair<int32_t, int32_t>> ranked = {{0, 0}, {0, 1}}, parsed;
    const std::string profile = (root / "profile.bin").string();
    CHECK(core::write_expert_profile(profile, 1, 2, ranked, err));
    int64_t slots = 0;
    CHECK(core::read_expert_profile(profile, 1, 2, parsed, slots, err));
    CHECK(parsed == ranked && slots == 2);
    {
        std::ofstream bad(profile, std::ios::binary | std::ios::trunc);
        const uint32_t header[5] = {1, 1, 2, UINT32_MAX, UINT32_MAX};
        bad.write("STRP", 4);
        bad.write(reinterpret_cast<const char*>(header), sizeof(header));
    }
    CHECK(!core::read_expert_profile(profile, 1, 2, parsed, slots, err));
    CHECK(err.find("counts") != std::string::npos);
    Source source;
    source.weights.assign(fmt.bytes, 0);
    kernels::cpu::ExpertPool pool(1, false, true);
    runtime::Drive drive;
    drive.d.src = &source;
    drive.d.pool = &pool;
    drive.d.n_expert = 2;
    const int32_t residency[2] = {-1, -1};
    drive.d.host_res = residency;
    std::vector<float> x(2560, 0), out(2560, 1);
    int32_t id = 0;
    runtime::drive_pool_multi(&drive, x.data(), &id, 1, 1, out.data(), 0);
    CHECK(!drive.d.failed);
    CHECK(std::all_of(out.begin(), out.end(), [](float v) { return v == 0; }));

    // One expert is GPU-owned and leaves its mapped host row untouched. A stale
    // NaN in that row must not reject the healthy CPU result in the other row.
    {
        const int32_t mixed_residency[2] = {0, -1};
        const int32_t mixed_ids[2] = {0, 1};
        int32_t counts[4] = {}, start[3] = {}, dst[2] = {}, tokens[2] = {}, start2[3] = {};
        unsigned long long ptr[2] = {}, ptr2[2] = {};
        core::GpuPlanSink plan;
        plan.counts = counts; plan.start = start; plan.dst = dst; plan.tok = tokens;
        plan.start2 = start2; plan.ptr = ptr; plan.ptr2 = ptr2; plan.cap = 2;
        runtime::Drive mixed;
        mixed.d.src = &source; mixed.d.pool = &pool; mixed.d.n_expert = 2;
        mixed.d.host_res = mixed_residency; mixed.d.plan = &plan;
        mixed.d.cache_base = source.weights.data(); mixed.d.cache_blob = int64_t(fmt.bytes);
        std::vector<float> mixed_out(5120, std::numeric_limits<float>::quiet_NaN());
        runtime::drive_pool_multi(&mixed, x.data(), mixed_ids, 1, 2, mixed_out.data(), 0);
        CHECK(!mixed.d.failed && counts[1] == 1 && dst[0] == 0);
        CHECK(std::isnan(mixed_out[0]));
        CHECK(std::all_of(mixed_out.begin() + 2560, mixed_out.end(), [](float v) { return v == 0; }));
    }

    auto rejects = [&](const std::string& message) {
        bool rejected = false;
        try { runtime::drive_pool_multi(&drive, x.data(), &id, 1, 1, out.data(), 0); }
        catch (const std::exception& e) {
            rejected = true;
            CHECK(std::string(e.what()).find(message) != std::string::npos);
        }
        CHECK(rejected && drive.d.failed);
    };
    source.missing = true;
    rejects("could not produce a blob");
    source.missing = false;
    rejects("could not produce a blob"); // failure remains latched

    // A new dispatch can be tested independently; the live runtime never clears
    // a failed dispatch and instead requires a model reload.
    drive.d.failed = false;
    x[3] = std::numeric_limits<float>::quiet_NaN();
    rejects("expert input: nonfinite");
    x[3] = 0;
    drive.d.failed = false;
    drive.d.job_of.clear();
    const uint16_t nan = 0x7e00; // corrupt a down-projection Q8_0 scale
    std::memcpy(source.weights.data() + fmt.down_off, &nan, sizeof(nan));
    rejects("expert output: nonfinite");
    fs::remove_all(root);
    std::puts("runtime_dispatch_test: source errors, failure latch and nonfinite expert outputs rejected");
}
