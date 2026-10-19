// Optional development-only CUPTI activity collector. No runtime dependency.
// c++ -std=c++17 -O2 -fPIC -shared tools/cuda_activity_trace.cpp -lcupti -o build-cuda12/libguild_activity.so
// ornith_prefill_bench --activity trace.tsv --activity-library ./build-cuda12/libguild_activity.so ...
// Concurrent kernel activities preserve overlap (never CUPTI_ACTIVITY_KIND_KERNEL).
#include <cupti.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <chrono>

namespace {
std::FILE* output = nullptr;
std::mutex mutex;
void check(CUptiResult r) {
    if (r != CUPTI_SUCCESS) {
        const char* message = nullptr;
        cuptiGetResultString(r, &message);
        throw std::runtime_error(std::string("CUPTI: ") + (message ? message : "unknown error"));
    }
}
void CUPTIAPI request(uint8_t** buffer, size_t* size, size_t* max_records) {
    *size = 8 << 20;
    *buffer = static_cast<uint8_t*>(std::malloc(*size));
    if (!*buffer)
        std::abort();
    *max_records = 0;
}
void CUPTIAPI complete(CUcontext context, uint32_t stream, uint8_t* buffer, size_t, size_t valid) {
    std::lock_guard<std::mutex> lock(mutex);
    CUpti_Activity* record = nullptr;
    CUptiResult result;
    while ((result = cuptiActivityGetNextRecord(buffer, valid, &record)) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            const auto& k = *reinterpret_cast<const CUpti_ActivityKernel9*>(record);
            std::fprintf(output, "kernel\t%llu\t%llu\t%u\t0\t%u\t%s\t%llu\t%u\t%u\n", (unsigned long long) k.start,
                         (unsigned long long) k.end, k.streamId, k.correlationId, k.name,
                         (unsigned long long) k.graphNodeId, k.registersPerThread, k.localMemoryPerThread);
        } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            const auto& m = *reinterpret_cast<const CUpti_ActivityMemcpy5*>(record);
            std::fprintf(output, "copy\t%llu\t%llu\t%u\t%llu\t%u\tcopy_%u_%u_%u\t%llu\t0\t0\n",
                         (unsigned long long) m.start, (unsigned long long) m.end, m.streamId,
                         (unsigned long long) m.bytes, m.correlationId, m.copyKind, m.srcKind, m.dstKind,
                         (unsigned long long) m.graphNodeId);
        } else if (record->kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
            const auto& a = *reinterpret_cast<const CUpti_ActivityAPI*>(record);
            const char* name = nullptr;
            cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, a.cbid, &name);
            std::fprintf(output, "api\t%llu\t%llu\t%u\t0\t%u\t%s\t0\t0\t0\n", (unsigned long long) a.start,
                         (unsigned long long) a.end, a.threadId, a.correlationId, name ? name : "unknown");
        }
    }
    if (result != CUPTI_ERROR_MAX_LIMIT_REACHED)
        std::fprintf(output, "error\t0\t0\t0\t0\t0\trecord_status_%u\t0\t0\t0\n", unsigned(result));
    size_t dropped = 0;
    const auto status = cuptiActivityGetNumDroppedRecords(context, stream, &dropped);
    if (dropped || status != CUPTI_SUCCESS)
        std::fprintf(output, "dropped\t0\t0\t0\t%zu\t0\tstatus_%u\t0\t0\t0\n", dropped, unsigned(status));
    std::free(buffer);
}
} // namespace

extern "C" void guild_activity_start(const char* path) {
    output = std::fopen(path, "w");
    if (!output)
        throw std::runtime_error("cannot open activity output");
    std::fputs(
        "kind\tstart_ns\tend_ns\tstream_or_thread\tbytes\tcorrelation\tname\tgraph_node\tregisters\tlocal_bytes\n",
        output);
    check(cuptiActivityRegisterCallbacks(request, complete));
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
}
extern "C" void guild_activity_mark(const char* name) {
    uint64_t timestamp = 0;
    check(cuptiGetTimestamp(&timestamp));
    std::lock_guard<std::mutex> lock(mutex);
    const auto host =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    std::fprintf(output, "mark\t%llu\t%llu\t0\t%llu\t0\t%s\t0\t0\t0\n", (unsigned long long) timestamp,
                 (unsigned long long) timestamp, (unsigned long long) host, name);
}
extern "C" void guild_activity_stop() {
    check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
    check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
    check(cuptiActivityFlushAll(0));
    std::lock_guard<std::mutex> lock(mutex);
    std::fclose(output);
    output = nullptr;
}
