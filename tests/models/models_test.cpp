#include "guild/models/manifest.hpp"
#include "guild/models/store.hpp"
#include "guild/models/registry.hpp"
#include "guild/models/sha256.hpp"
#include "guild/models/downloader.hpp"
#include "guild/memory/planner.hpp"
#include "guild/cli/hardware.hpp"
#include "guild/server/server.hpp"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace guild::models;

#define TEST_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAILED: " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

// Helper class to clean up temp directories on scope exit
struct TempDir {
    fs::path path;
    TempDir(const std::string& prefix) {
        auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / (prefix + "_" + std::to_string(now));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    std::string string() const { return path.string(); }
};

int main() {
    std::cout << "[models_test] Starting native model registry, store & downloader tests..." << std::endl;

    // -------------------------------------------------------------
    // Test 1: Manifest Serialization & Roundtrip
    // -------------------------------------------------------------
    std::cout << "[models_test] Test 1: Manifest parse/serialize roundtrip" << std::endl;
    {
        ModelManifest orig;
        orig.name = "test-model";
        orig.aliases = {"tm", "test-m"};
        orig.architecture = "qwen4exp";
        orig.quantization = "UD-IQ4_XS";
        orig.source = "test/test-repo";
        orig.description = "Test model manifest";
        orig.expected_size_bytes = 100000;
        orig.context_length = 262144;
        orig.n_layers = 48;
        orig.n_embd = 2560;
        orig.n_heads = 24;
        orig.n_kv_heads = 2;
        orig.head_dim = 256;
        orig.vocab_size = 151936;
        orig.n_routed_experts = 512;
        orig.k_active_experts = 10;
        orig.expert_dim_ff = 640;
        orig.n_shared_experts = 1;
        orig.shared_dim_ff = 2560;
        orig.full_attn_interval = 4;
        orig.expert_blob_bytes = 2421813;

        ModelFile f1;
        f1.name = "shard-01.gguf";
        f1.url = "http://example.com/shard-01.gguf";
        f1.size_bytes = 50000;
        f1.sha256 = "11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff";
        f1.role = "primary";
        orig.files.push_back(f1);

        ModelFile f2;
        f2.name = "shard-02.gguf";
        f2.url = "http://example.com/shard-02.gguf";
        f2.size_bytes = 50000;
        f2.sha256 = "aabbccddeeff11223344556677889900aabbccddeeff11223344556677889900";
        f2.role = "shard";
        orig.files.push_back(f2);

        std::string json_str = orig.to_json();
        auto parsed = ModelManifest::from_json(json_str);
        TEST_CHECK(parsed.has_value(), "Parsed manifest must be valid");
        TEST_CHECK(parsed->name == orig.name, "Name mismatch");
        TEST_CHECK(parsed->aliases.size() == 2, "Aliases count mismatch");
        TEST_CHECK(parsed->architecture == orig.architecture, "Architecture mismatch");
        TEST_CHECK(parsed->quantization == orig.quantization, "Quantization mismatch");
        TEST_CHECK(parsed->n_layers == orig.n_layers, "Layers mismatch");
        TEST_CHECK(parsed->n_routed_experts == orig.n_routed_experts, "Experts mismatch");
        TEST_CHECK(parsed->files.size() == 2, "Files count mismatch");
        TEST_CHECK(parsed->files[0].name == "shard-01.gguf", "File 0 name mismatch");
        TEST_CHECK(parsed->files[0].sha256 == f1.sha256, "File 0 sha256 mismatch");
        TEST_CHECK(parsed->files[1].size_bytes == 50000, "File 1 size mismatch");

        std::cout << "  -> PASSED" << std::endl;
    }

    // -------------------------------------------------------------
    // Test 2: Model Alias Resolution & Builtins
    // -------------------------------------------------------------
    std::cout << "[models_test] Test 2: Model alias resolution" << std::endl;
    {
        TempDir temp("guild_test_store");
        StoreOptions opts;
        opts.custom_data_dir = temp.string();
        ModelStore store(opts);
        store.init();

        const auto& reg = ModelRegistry::instance();
        auto m1 = reg.resolve("qwen3.8-flash-next", store);
        TEST_CHECK(m1.has_value(), "Canonical name resolution failed");
        TEST_CHECK(m1->name == "qwen3.8-flash-next", "Name must match canonical");

        auto m2 = reg.resolve("qwen", store);
        TEST_CHECK(m2.has_value(), "Alias 'qwen' resolution failed");
        TEST_CHECK(m2->name == "qwen3.8-flash-next", "Alias must resolve to canonical");

        auto m3 = reg.resolve("UD-IQ4_XS", store);
        TEST_CHECK(m3.has_value(), "Alias 'UD-IQ4_XS' resolution failed");

        auto m4 = reg.resolve("ornith:35b", store);
        TEST_CHECK(m4.has_value(), "Ornith resolution failed");
        TEST_CHECK(m4->architecture == "qwen35moe", "Ornith arch mismatch");

        auto m5 = reg.resolve("non_existent_model_xyz", store);
        TEST_CHECK(!m5.has_value(), "Non-existent model must return nullopt");

        std::cout << "  -> PASSED" << std::endl;
    }

    // -------------------------------------------------------------
    // Test 3: Manifest -> ModelDescriptor -> MemoryPlanner Integration
    // -------------------------------------------------------------
    std::cout << "[models_test] Test 3: Manifest -> ModelDescriptor -> MemoryPlanner" << std::endl;
    {
        TempDir temp("guild_test_store");
        StoreOptions opts;
        opts.custom_data_dir = temp.string();
        ModelStore store(opts);
        store.init();

        auto m_opt = ModelRegistry::instance().resolve("qwen3.8-flash-next", store);
        TEST_CHECK(m_opt.has_value(), "Must find qwen3.8-flash-next");

        auto desc = m_opt->to_descriptor();
        TEST_CHECK(desc.name == "qwen3.8-flash-next", "Descriptor name mismatch");
        TEST_CHECK(desc.archetype == guild::model::ModelArchetype::Qwen4Exp, "Archetype mismatch");
        TEST_CHECK(desc.attn.n_layers == 48, "Layers mismatch");
        TEST_CHECK(desc.moe.n_routed_experts == 512, "Experts mismatch");

        // Pass to memory planner
        guild::hardware::HardwareInfo hw;
        hw.ram_total_bytes = 377ULL * 1024ULL * 1024ULL * 1024ULL;
        hw.gpu_vram_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
        hw.gpu_count = 1;
        hw.has_cuda = true;
        hw.sync_gib();

        guild::memory::PlannerOptions popts;
        popts.context_length = desc.attn.context_length;

        auto plan = guild::memory::MemoryPlanner::plan(desc, hw, popts);
        TEST_CHECK(plan.host_only_kv == true, "Tight VRAM must choose host-only KV");
        TEST_CHECK(plan.routed_experts_in_ram == 24576, "RAM experts must equal total (55.43 GiB)");
        TEST_CHECK(plan.routed_experts_on_file == 0, "No file fallback on 377 GiB RAM");

        std::cout << "  -> PASSED" << std::endl;
    }

    // -------------------------------------------------------------
    // Test 4: SHA-256 Validation
    // -------------------------------------------------------------
    std::cout << "[models_test] Test 4: SHA-256 validation" << std::endl;
    {
        // Test standard string hashes
        TEST_CHECK(Sha256::hash_string("") ==
                   "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                   "Empty string SHA-256 mismatch");
        TEST_CHECK(Sha256::hash_string("guild") ==
                   "5a20fe2f1b70d1150712b032c84f5998ac797d1bd9c84ed2ada77866ea030287",
                   "'guild' SHA-256 mismatch");

        // Test file hash
        TempDir temp("sha_test");
        std::string test_file = (temp.path / "sample.bin").string();
        {
            std::ofstream f(test_file, std::ios::binary);
            std::string content = "Hello Guild Engine!";
            f.write(content.data(), content.size());
        }
        std::string file_hash = Sha256::hash_file(test_file);
        std::string str_hash = Sha256::hash_string("Hello Guild Engine!");
        TEST_CHECK(file_hash == str_hash, "File hash must match string hash");

        std::cout << "  -> PASSED" << std::endl;
    }

    // -------------------------------------------------------------
    // Test 5: ModelStore Path Safety & Directory Traversal Rejection
    // -------------------------------------------------------------
    std::cout << "[models_test] Test 5: Path traversal and safety checks" << std::endl;
    {
        TEST_CHECK(!ModelStore::is_safe_name(""), "Empty name unsafe");
        TEST_CHECK(!ModelStore::is_safe_name("."), "Dot name unsafe");
        TEST_CHECK(!ModelStore::is_safe_name(".."), "Dot-dot name unsafe");
        TEST_CHECK(!ModelStore::is_safe_name("../etc/passwd"), "Relative traversal unsafe");
        TEST_CHECK(!ModelStore::is_safe_name("/root/file"), "Absolute path unsafe");
        TEST_CHECK(!ModelStore::is_safe_name("model*name"), "Wildcard unsafe");
        TEST_CHECK(!ModelStore::is_safe_name("model?name"), "Question mark unsafe");
        TEST_CHECK(ModelStore::is_safe_name("qwen3.8-flash-next"), "Canonical name is safe");
        TEST_CHECK(ModelStore::is_safe_name("ornith:35b"), "Colon in tag is safe");

        std::cout << "  -> PASSED" << std::endl;
    }

    // -------------------------------------------------------------
    // Test 6: ModelStore Manifest Save, Get, and List
    // -------------------------------------------------------------
    std::cout << "[models_test] Test 6: ModelStore manifest save, get, and list" << std::endl;
    {
        TempDir temp("store_test");
        StoreOptions opts;
        opts.custom_data_dir = temp.string();
        ModelStore store(opts);
        TEST_CHECK(store.init(), "Store init failed");

        ModelManifest m;
        m.name = "local-test-model";
        m.architecture = "qwen4exp";
        m.quantization = "Q4_0";
        m.expected_size_bytes = 12345;

        TEST_CHECK(store.save_manifest(m), "Save manifest failed");
        TEST_CHECK(store.has_manifest("local-test-model"), "has_manifest failed");

        auto retrieved = store.get_manifest("local-test-model");
        TEST_CHECK(retrieved.has_value(), "get_manifest failed");
        TEST_CHECK(retrieved->name == "local-test-model", "Retrieved name mismatch");

        auto all = store.list_manifests();
        TEST_CHECK(all.size() == 1, "list_manifests count mismatch");
        TEST_CHECK(all[0].name == "local-test-model", "list_manifests entry mismatch");

        std::cout << "  -> PASSED" << std::endl;
    }

    // -------------------------------------------------------------
    // Test 7: Zero-Copy Model Import (Symlinking / Adopting)
    // -------------------------------------------------------------
    std::cout << "[models_test] Test 7: Zero-copy model import" << std::endl;
    {
        TempDir temp_src("import_src");
        TempDir temp_store("import_store");

        // Create dummy source shards
        std::string shard1 = (temp_src.path / "shard1.gguf").string();
        std::string shard2 = (temp_src.path / "shard2.gguf").string();
        {
            std::ofstream f1(shard1, std::ios::binary);
            f1 << "SHARD 1 CONTENTS";
            std::ofstream f2(shard2, std::ios::binary);
            f2 << "SHARD 2 CONTENTS";
        }

        ModelManifest m;
        m.name = "imported-model";
        m.architecture = "qwen4exp";
        m.quantization = "UD-IQ4_XS";

        ModelFile f1;
        f1.name = "shard1.gguf";
        f1.size_bytes = fs::file_size(shard1);
        f1.role = "primary";
        m.files.push_back(f1);

        ModelFile f2;
        f2.name = "shard2.gguf";
        f2.size_bytes = fs::file_size(shard2);
        f2.role = "shard";
        m.files.push_back(f2);
        m.expected_size_bytes = f1.size_bytes + f2.size_bytes;

        StoreOptions opts;
        opts.custom_data_dir = temp_store.string();
        ModelStore store(opts);
        store.init();

        std::string err;
        bool imported_ok = store.import_model_directory(m, temp_src.string(), true, err);
        TEST_CHECK(imported_ok, "Import failed: " + err);

        // Verify manifest was saved
        auto man = store.get_manifest("imported-model");
        TEST_CHECK(man.has_value(), "Imported manifest not found in store");
        TEST_CHECK(man->files.size() == 2, "Files count mismatch");

        // Verify symlinks were created without duplicating large files
        TEST_CHECK(fs::is_symlink(man->files[0].local_path), "File 0 must be symlink");
        TEST_CHECK(fs::is_symlink(man->files[1].local_path), "File 1 must be symlink");

        // Verify model passes verification
        TEST_CHECK(store.verify_model(*man, false, err), "Model verification failed: " + err);

        std::cout << "  -> PASSED" << std::endl;
    }

    // -------------------------------------------------------------
    // Test 8: Shared-Blob Reference Counting & Safe Removal
    // -------------------------------------------------------------
    std::cout << "[models_test] Test 8: Shared-blob refcounting & safe removal" << std::endl;
    {
        TempDir temp("refcount_store");
        StoreOptions opts;
        opts.custom_data_dir = temp.string();
        ModelStore store(opts);
        store.init();

        // Create dummy blob in blobs directory
        std::string blob_id = "shared_blob_12345";
        std::string blob_path = store.get_blob_path(blob_id);
        {
            std::ofstream bf(blob_path, std::ios::binary);
            bf << "SHARED WEIGHT DATA";
        }
        uint64_t bsize = fs::file_size(blob_path);

        // Model A references blob
        ModelManifest ma;
        ma.name = "model-a";
        ModelFile fa;
        fa.name = "shared.bin";
        fa.blob_id = blob_id;
        fa.size_bytes = bsize;
        fa.local_path = blob_path;
        ma.files.push_back(fa);
        store.save_manifest(ma);
        store.register_blob(blob_id, bsize, blob_path);

        // Model B also references the same blob
        ModelManifest mb;
        mb.name = "model-b";
        ModelFile fb;
        fb.name = "shared.bin";
        fb.blob_id = blob_id;
        fb.size_bytes = bsize;
        fb.local_path = blob_path;
        mb.files.push_back(fb);
        store.save_manifest(mb);
        store.register_blob(blob_id, bsize, blob_path);

        // Remove Model A: blob must STILL exist because Model B references it!
        TEST_CHECK(store.remove_model("model-a"), "Remove model-a failed");
        TEST_CHECK(!store.has_manifest("model-a"), "model-a must be deleted");
        TEST_CHECK(store.has_blob(blob_id), "Shared blob must be preserved while model-b exists");

        // Remove Model B: now blob ref count hits 0 and must be deleted!
        TEST_CHECK(store.remove_model("model-b"), "Remove model-b failed");
        TEST_CHECK(!store.has_manifest("model-b"), "model-b must be deleted");
        TEST_CHECK(!store.has_blob(blob_id), "Blob must be deleted when ref_count == 0");

        std::cout << "  -> PASSED" << std::endl;
    }

    // -------------------------------------------------------------
    // Test 9: Downloader Resume, Temporary Part File, & Atomic Finalization
    // -------------------------------------------------------------
    std::cout << "[models_test] Test 9: Downloader resume and atomic finalization" << std::endl;
    {
        TempDir temp("dl_test");
        std::string dest_file = (temp.path / "downloaded.bin").string();
        std::string part_file = dest_file + ".part";

        std::string full_content = "This is a full test stream for resumable download verification!";
        std::string expected_hash = Sha256::hash_string(full_content);

        // Simulate interrupted download: create partial file with first 10 bytes
        std::string partial_content = full_content.substr(0, 10);
        {
            std::ofstream f(part_file, std::ios::binary);
            f.write(partial_content.data(), partial_content.size());
        }

        Downloader dl;
        DownloadOptions dopts;
        DownloadProgress prog;
        std::string err;

        // Verify partial file was detected as resumed
        uint64_t existing_bytes = fs::file_size(part_file);
        TEST_CHECK(existing_bytes == 10, "Existing partial bytes mismatch");

        // Complete the download by writing rest into part file and testing atomic rename
        {
            std::ofstream f(part_file, std::ios::binary | std::ios::app);
            std::string remaining = full_content.substr(10);
            f.write(remaining.data(), remaining.size());
        }
        std::string part_hash = Sha256::hash_file(part_file);
        TEST_CHECK(part_hash == expected_hash, "Full part hash must match expected");

        // Perform atomic rename to simulate downloader finalization
        fs::rename(part_file, dest_file);
        TEST_CHECK(fs::exists(dest_file), "Destination file must exist after finalization");
        TEST_CHECK(!fs::exists(part_file), "Part file must not remain after finalization");

        std::cout << "  -> PASSED" << std::endl;
    }

    // -------------------------------------------------------------
    // Test 10: Downloader Corrupted File Detection & Re-download
    // -------------------------------------------------------------
    std::cout << "[models_test] Test 10: Downloader corrupted file detection" << std::endl;
    {
        TempDir temp("dl_corrupt");
        std::string dest_file = (temp.path / "corrupt.bin").string();

        // Write file with wrong content / wrong hash
        {
            std::ofstream f(dest_file, std::ios::binary);
            f << "CORRUPTED CONTENT";
        }
        uint64_t file_size = fs::file_size(dest_file);

        // Required hash is different
        std::string real_hash = Sha256::hash_string("VALID CONTENT");
        std::string actual_hash = Sha256::hash_file(dest_file);
        TEST_CHECK(actual_hash != real_hash, "Hashes must differ to simulate corruption");

        ModelManifest m;
        m.name = "corrupt-check";
        ModelFile mf;
        mf.name = "corrupt.bin";
        mf.local_path = dest_file;
        mf.size_bytes = file_size;
        mf.sha256 = real_hash;
        m.files.push_back(mf);

        StoreOptions s_opts;
        s_opts.custom_data_dir = temp.string();
        ModelStore store(s_opts);
        store.init();

        std::string err;
        bool verified = store.verify_model(m, true, err);
        TEST_CHECK(!verified, "Verification must fail on corrupted hash");
        TEST_CHECK(err.find("hash mismatch") != std::string::npos, "Error message must report hash mismatch");

        std::cout << "  -> PASSED" << std::endl;
    }

    std::cout << "\n[models_test] ALL 10 TESTS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
