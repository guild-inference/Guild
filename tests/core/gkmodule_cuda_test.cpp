#include "../check.hpp"
#include "../signing_fixture.hpp"
#include "guild/kernels/registry.hpp"
#include "guild/model/runner.hpp"
#include "guild/models/sha256.hpp"
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace guild;
namespace fs = std::filesystem;
int main() {
    const auto root = fs::current_path() / "gkmodule-cuda-work";
    CHECK(!fs::exists(root)); fs::create_directories(root / "source");
    fs::copy_file(GUILD_SCAN_CUDA_SO, root / "source/module.so");
    std::string err;
    abi::GKernelMetadata m;
    m.package_kind = "gkmodule"; m.module_abi = 1; m.scope = "module";
    m.package_id = "example.scan.cuda"; m.version = "1.0.0"; m.publisher_id = abi::kOfficialGuildPublisherId;
    m.library_path = "module.so"; m.supported_backends = GUILD_BACKEND_CUDA;
    m.provided_ops = {"example.scan.cuda.prefix_sum", "example.scan.cuda.exclusive_sum"};
    m.artifacts.push_back({"module.so", models::Sha256::hash_file((root / "source/module.so").string()), fs::file_size(root / "source/module.so"), "shared_library"});
    CHECK(m.sign_with_seed_hex(abi::kOfficialGuildReleaseKeyId, abi::kOfficialGuildDevSeedHex, err));
    { std::ofstream f(root / "source/.gkmetadata"); f << m.to_json_string(); }
    kernels::KernelRegistry reg; model::ModelRunnerRegistry runners;
    abi::GKernelPackageManager pm((root / "store").string(), &reg, &runners);
    CHECK(pm.init(err)); abi::trust_test_signing_key(pm.security_policy());
    CHECK(pm.install_package((root / "source").string(), true, err) == GUILD_STATUS_OK);
    float *input = nullptr, *output = nullptr;
    CHECK(cudaMalloc(&input, 16) == cudaSuccess); CHECK(cudaMalloc(&output, 16) == cudaSuccess);
    float host_in[]{1,2,3,4}, host_out[4]{};
    CHECK(cudaMemcpy(input, host_in, 16, cudaMemcpyHostToDevice) == cudaSuccess);
    auto tensor = [](float* ptr) {
        guild_tensor_desc_t t{}; t.struct_size = sizeof(t); t.dtype = GUILD_DTYPE_F32; t.n_dims = 1;
        t.dims[0] = 4; t.strides_bytes[0] = 4; t.buffer.struct_size = sizeof(t.buffer);
        t.buffer.data = ptr; t.buffer.byte_size = 16; t.buffer.alignment = 4;
        t.buffer.memory_space = GUILD_MEM_DEVICE_VRAM; t.buffer.lifetime = GUILD_LIFETIME_SESSION;
        return t;
    };
    const auto in = tensor(input), out = tensor(output);
    guild_operation_query_v1_t query{}; query.struct_size = sizeof(query); query.version = 1;
    query.operation_id = "example.sequence.prefix_sum"; query.min_version = query.max_version = GUILD_MAKE_ABI_VERSION(1,0,0);
    query.allowed_backends = GUILD_BACKEND_CUDA; query.available_ram_bytes = query.available_vram_bytes = UINT64_MAX;
    query.inputs = &in; query.n_inputs = 1; query.outputs = &out; query.n_outputs = 1;
    auto api = reg.operation_api(); const auto caps = kernels::query_host_device_capabilities();
    cudaStream_t stream{}; CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    guild_exec_context_v1_t ctx{}; ctx.struct_size = sizeof(ctx); ctx.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
    ctx.backend = GUILD_BACKEND_CUDA; ctx.device_stream = stream;
    for (auto mode : {GUILD_SYNC_STREAM_ORDERED, GUILD_SYNC_GRAPH_CAPTURE}) {
        query.sync_mode = ctx.sync_mode = mode;
        guild_operation_binding_t binding = nullptr; guild_error_info_t e{};
        CHECK(api.resolve(api.user, &query, &caps, &binding, &e) == GUILD_STATUS_OK);
        cudaGraph_t graph{}; cudaGraphExec_t executable{};
        if (mode == GUILD_SYNC_GRAPH_CAPTURE) CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess);
        CHECK(api.execute(binding, &ctx, nullptr, 0, &e) == GUILD_STATUS_OK);
        if (mode == GUILD_SYNC_GRAPH_CAPTURE) {
            CHECK(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
            CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0) == cudaSuccess);
            CHECK(cudaGraphLaunch(executable, stream) == cudaSuccess);
        }
        CHECK(pm.disable_package(m.package_id, err) == GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY);
        cudaEvent_t done{}; CHECK(cudaEventCreate(&done) == cudaSuccess); CHECK(cudaEventRecord(done, stream) == cudaSuccess);
        if (executable) {
            CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
            CHECK(cudaGraphExecDestroy(executable) == cudaSuccess); CHECK(cudaGraphDestroy(graph) == cudaSuccess);
        }
        api.release(binding); // waits before dropping the last provider binding
        CHECK(cudaEventQuery(done) == cudaSuccess); CHECK(cudaEventDestroy(done) == cudaSuccess);
        CHECK(cudaMemcpy(host_out, output, 16, cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(host_out[0] == 1 && host_out[1] == 3 && host_out[2] == 6 && host_out[3] == 10);
    }
    CHECK(pm.disable_package(m.package_id, err) == GUILD_STATUS_OK);
    CHECK(cudaStreamDestroy(stream) == cudaSuccess); CHECK(cudaFree(input) == cudaSuccess); CHECK(cudaFree(output) == cudaSuccess);
    fs::remove_all(root);
    std::cout << "gkmodule CUDA launch, graph capture and stream-safe release passed\n";
}
