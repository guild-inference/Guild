// tests/core/gpack_test.cpp - Phase 4 GPACK v1 container, GBT1 binary tokenizer,
// safetensors import, corruption/truncation/crash recovery, and concurrency tests.
#include "guild/artifact/gpack.hpp"
#include "guild/models/sha256.hpp"
#include "guild/runtime/tokenizer.hpp"
#include "guild/server/json.hpp"
#include "gguf_fixture.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace guild::artifact;
using namespace guild::runtime;

namespace {

int g_passed = 0;
int g_failed = 0;

void check(bool cond, const char* msg, int line, const std::string& extra = "") {
    if (cond) {
        ++g_passed;
    } else {
        ++g_failed;
        std::cerr << "[FAIL] line " << line << ": " << msg;
        if (!extra.empty()) std::cerr << " (" << extra << ")";
        std::cerr << "\n";
    }
}
#define CHECK(c) check((c), #c, __LINE__)
#define CHECK_ERR(c, e) check((c), #c, __LINE__, (e))

// Write a minimal valid safetensors model directory (single or sharded)
void create_safetensors_fixture(const fs::path& dir, bool sharded) {
    fs::create_directories(dir);
    {
        std::ofstream cfg(dir / "config.json");
        cfg << R"({
  "architecture": "qwen3_5_moe",
  "hidden_size": 16,
  "num_hidden_layers": 2,
  "num_attention_heads": 2,
  "num_key_value_heads": 2,
  "head_dim": 8,
  "vocab_size": 32,
  "max_position_embeddings": 512,
  "num_experts": 4,
  "num_experts_per_tok": 2,
  "moe_intermediate_size": 8,
  "full_attention_interval": 2
})";
    }

    auto write_st_file = [](const fs::path& file_path,
                            const std::vector<std::tuple<std::string, std::string, std::vector<int64_t>, std::vector<uint8_t>>>& tensors) {
        guild::server::json::JsonValue hroot;
        uint64_t off = 0;
        for (const auto& [name, dtype, shape, data] : tensors) {
            guild::server::json::JsonValue tdesc;
            tdesc["dtype"] = dtype;
            guild::server::json::JsonValue sh(std::vector<guild::server::json::JsonValue>{});
            for (int64_t d : shape) sh.arr_val.push_back(guild::server::json::JsonValue(d));
            tdesc["shape"] = std::move(sh);
            guild::server::json::JsonValue offs(std::vector<guild::server::json::JsonValue>{
                guild::server::json::JsonValue(static_cast<int64_t>(off)),
                guild::server::json::JsonValue(static_cast<int64_t>(off + data.size()))
            });
            tdesc["data_offsets"] = std::move(offs);
            hroot[name] = std::move(tdesc);
            off += data.size();
        }
        const std::string hstr = hroot.dump();
        const uint64_t hlen = hstr.size();
        std::ofstream out(file_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&hlen), 8);
        out.write(hstr.data(), static_cast<std::streamsize>(hstr.size()));
        for (const auto& [name, dtype, shape, data] : tensors) {
            if (!data.empty()) {
                out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
            }
        }
    };

    auto make_bytes = [](size_t n, uint8_t seed) {
        std::vector<uint8_t> b(n);
        for (size_t i = 0; i < n; ++i) b[i] = static_cast<uint8_t>((seed + i * 13) & 0xFF);
        return b;
    };

    std::vector<std::tuple<std::string, std::string, std::vector<int64_t>, std::vector<uint8_t>>> shard1;
    std::vector<std::tuple<std::string, std::string, std::vector<int64_t>, std::vector<uint8_t>>> shard2;

    // token_embd.weight: [32, 16] BF16 = 1024 B
    shard1.push_back({"token_embd.weight", "BF16", {32, 16}, make_bytes(32 * 16 * 2, 1)});
    // output_norm.weight: [16] F32 = 64 B
    shard1.push_back({"output_norm.weight", "F32", {16}, make_bytes(16 * 4, 2)});
    // output.weight: [32, 16] BF16 = 1024 B
    (sharded ? shard2 : shard1).push_back({"output.weight", "BF16", {32, 16}, make_bytes(32 * 16 * 2, 3)});

    for (int l = 0; l < 2; ++l) {
        const std::string pfx = "blk." + std::to_string(l) + ".";
        auto& target = (sharded && l == 1) ? shard2 : shard1;
        target.push_back({pfx + "attn_norm.weight", "F32", {16}, make_bytes(16 * 4, static_cast<uint8_t>(10 + l))});
        target.push_back({pfx + "ffn_gate_inp.weight", "BF16", {4, 16}, make_bytes(4 * 16 * 2, static_cast<uint8_t>(20 + l))});
        // Experts: [4, 8, 16] BF16 -> 4 * 8 * 16 * 2 = 1024 B
        target.push_back({pfx + "ffn_gate_exps.weight", "BF16", {4, 8, 16}, make_bytes(4 * 8 * 16 * 2, static_cast<uint8_t>(30 + l))});
        target.push_back({pfx + "ffn_up_exps.weight", "BF16", {4, 8, 16}, make_bytes(4 * 8 * 16 * 2, static_cast<uint8_t>(40 + l))});
        target.push_back({pfx + "ffn_down_exps.weight", "BF16", {4, 16, 8}, make_bytes(4 * 16 * 8 * 2, static_cast<uint8_t>(50 + l))});
    }

    if (!sharded) {
        write_st_file(dir / "model.safetensors", shard1);
    } else {
        write_st_file(dir / "model-00001-of-00002.safetensors", shard1);
        write_st_file(dir / "model-00002-of-00002.safetensors", shard2);
        guild::server::json::JsonValue idx;
        guild::server::json::JsonValue wmap;
        for (const auto& t : shard1) wmap[std::get<0>(t)] = "model-00001-of-00002.safetensors";
        for (const auto& t : shard2) wmap[std::get<0>(t)] = "model-00002-of-00002.safetensors";
        idx["weight_map"] = std::move(wmap);
        std::ofstream out(dir / "model.safetensors.index.json");
        out << idx.dump_pretty();
    }
}

void create_tokenizer_fixture(const fs::path& tok_dir) {
    fs::create_directories(tok_dir);
    {
        std::ofstream vf(tok_dir / "vocab.json");
        vf << R"({
  "h": 0,
  "e": 1,
  "l": 2,
  "o": 3,
  "Ġ": 4,
  "w": 5,
  "r": 6,
  "d": 7,
  "he": 8,
  "ll": 9,
  "hell": 10,
  "hello": 11,
  "Ġw": 12,
  "Ġwo": 13,
  "Ġwor": 14,
  "Ġworl": 15,
  "Ġworld": 16,
  "<|endoftext|>": 17,
  "<|im_start|>": 18,
  "<|im_end|>": 19
})";
    }
    {
        std::ofstream mf(tok_dir / "merges.txt");
        mf << "h e\n"
           << "l l\n"
           << "he ll\n"
           << "hell o\n"
           << "Ġ w\n"
           << "Ġw o\n"
           << "Ġwo r\n"
           << "Ġwor l\n"
           << "Ġworl d\n";
    }
    {
        std::ofstream tf(tok_dir / "tokenizer.json");
        tf << R"({
  "model": "gpt2",
  "pre": "qwen35",
  "vocab_size": 20,
  "n_merges": 9,
  "special_ids": {
    "tokenizer.ggml.eos_token_id": 19
  }
})";
    }
    {
        std::ofstream ttf(tok_dir / "token_type.json");
        ttf << "[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3]\n";
    }
    {
        std::ofstream ctf(tok_dir / "chat_template.jinja");
        ctf << "{% macro render_content(msg) %}{{ msg.content }}{% endmacro %}"
            << "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ render_content(message) }}<|im_end|>\n{% endfor %}";
    }
}

void test_gbt1_binary_tokenizer(const fs::path& work_dir) {
    const fs::path tok_dir = work_dir / "tok_fixture";
    create_tokenizer_fixture(tok_dir);

    Tokenizer tok1;
    std::string err;
    CHECK(tok1.load(tok_dir.string(), err));

    std::vector<uint8_t> bin;
    CHECK(tok1.serialize_binary(bin, err));
    CHECK(bin.size() > 32);

    Tokenizer tok2;
    CHECK(tok2.load_binary(bin.data(), bin.size(), err));
    CHECK(tok2.vocab_size() == tok1.vocab_size());

    const std::vector<std::string> prompts = {
        "hello world",
        "<|im_start|>hello world<|im_end|>",
        "hell hello world hello"
    };
    for (const auto& p : prompts) {
        std::vector<int32_t> ids1, ids2;
        CHECK(tok1.encode(p, ids1, true, err));
        CHECK(tok2.encode(p, ids2, true, err));
        CHECK(ids1 == ids2);
        std::string dec1 = tok1.decode(ids1, false);
        std::string dec2 = tok2.decode(ids2, false);
        CHECK(!dec1.empty());
        CHECK(dec1 == dec2);
    }

    // Corrupt GBT1 magic or truncate -> must fail closed
    if (!bin.empty()) {
        std::vector<uint8_t> bad_bin = bin;
        bad_bin[0] ^= 0xFF;
        Tokenizer tok_bad;
        CHECK(!tok_bad.load_binary(bad_bin.data(), bad_bin.size(), err));
        CHECK(!tok_bad.load_binary(bin.data(), 12, err));
    }

    // If real Ornith/Qwen tokenizer exists on disk, benchmark & verify 100% parity
    const fs::path real_tok = "/mnt/models-ssd/Strata-data/packs/ornith-1.5-35b/tokenizer";
    if (fs::exists(real_tok / "tokenizer.json")) {
        const auto t0 = std::chrono::steady_clock::now();
        Tokenizer rtok_json;
        CHECK(rtok_json.load(real_tok.string(), err));
        const double json_load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        std::vector<uint8_t> rbin;
        CHECK(rtok_json.serialize_binary(rbin, err));

        const auto t1 = std::chrono::steady_clock::now();
        Tokenizer rtok_bin;
        CHECK(rtok_bin.load_binary(rbin.data(), rbin.size(), err));
        const double bin_load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();

        std::cout << "[GBT1] Real 248k-vocab tokenizer load: JSON=" << json_load_ms
                  << " ms vs GBT1=" << bin_load_ms << " ms (" << (rbin.size() >> 20) << " MiB)\n";
        CHECK(bin_load_ms < json_load_ms);

        const std::vector<std::string> real_cases = {
            "<|im_start|>user\nWrite a fast C++20 SIMD matrix multiplication kernel with AVX-512.<|im_end|>\n<|im_start|>assistant\n",
            "你好，世界！这是一个测试：1234567890 !== === >= <= -> :: std::vector<uint64_t>",
            "   \n\t  multiple   spaces   and\nnewlines\n\n"
        };
        for (const auto& rc : real_cases) {
            std::vector<int32_t> a, b;
            CHECK(rtok_json.encode(rc, a, true, err));
            CHECK(rtok_bin.encode(rc, b, true, err));
            CHECK(a == b);
        }
    }
}

void test_safetensors_and_gpack_integrity(const fs::path& work_dir) {
    // 1. Single-shard safetensors -> GPACK
    const fs::path st_single_dir = work_dir / "st_single";
    create_safetensors_fixture(st_single_dir, false);
    const fs::path gpack_single = work_dir / "single.gpack";

    GPackBuildOptions bopts;
    bopts.primary_input_path = st_single_dir.string();
    bopts.output_path = gpack_single.string();
    bopts.model_name = "test-st-single";
    GPackBuildReport breport;
    std::string err;
    CHECK(GPackBuilder::build(bopts, breport, err));
    CHECK(fs::exists(gpack_single));
    CHECK(breport.tensor_count == 13);
    CHECK(breport.manifest_sha256.size() == 64);

    // 2. Sharded safetensors -> GPACK
    const fs::path st_sharded_dir = work_dir / "st_sharded";
    create_safetensors_fixture(st_sharded_dir, true);
    const fs::path gpack_sharded = work_dir / "sharded.gpack";
    bopts.primary_input_path = (st_sharded_dir / "model.safetensors.index.json").string();
    bopts.output_path = gpack_sharded.string();
    GPackBuildReport sreport;
    CHECK(GPackBuilder::build(bopts, sreport, err));
    CHECK(fs::exists(gpack_sharded));
    CHECK(sreport.tensor_count == 13);
    // Single-shard and multi-shard safetensors of identical tensors must produce identical tensor index!
    GPackReader r_single, r_sharded;
    CHECK(r_single.open(gpack_single.string(), err, true));
    CHECK(r_sharded.open(gpack_sharded.string(), err, true));
    CHECK(r_single.tensors().size() == r_sharded.tensors().size());
    CHECK(r_single.total_expert_bytes() == r_sharded.total_expert_bytes());
    CHECK(r_single.dense_pool_bytes() == r_sharded.dense_pool_bytes());

    // 3. Test ABI model reader over GPackReader
    auto abi_reader = r_single.make_abi_model_reader();
    int64_t n_embd = 0;
    CHECK(abi_reader.get_i64(abi_reader.user, "qwen3_5_moe.embedding_length", &n_embd) == GUILD_STATUS_OK);
    CHECK(n_embd == 16);
    guild_tensor_desc_t tdesc{};
    CHECK(abi_reader.get_tensor(abi_reader.user, "token_embd.weight", &tdesc) == GUILD_STATUS_OK);
    CHECK(tdesc.n_dims == 2);
    CHECK(tdesc.dims[0] == 16 && tdesc.dims[1] == 32);

    // 4. Test concurrent GPackReader access across 8 threads
    std::atomic<bool> thread_ok{true};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&]() {
            for (int iter = 0; iter < 10; ++iter) {
                GPackReader tr;
                std::string terr;
                if (!tr.open(gpack_single.string(), terr, true)) {
                    thread_ok.store(false);
                    return;
                }
                if (tr.tensors().size() != 13 || tr.expert_layers().size() != 2) {
                    thread_ok.store(false);
                    return;
                }
            }
        });
    }
    for (auto& th : threads) th.join();
    CHECK(thread_ok.load());

    // 5. Corruption, truncation, and version compatibility tests
    std::ifstream orig_f(gpack_single, std::ios::binary);
    std::vector<uint8_t> raw((std::istreambuf_iterator<char>(orig_f)), std::istreambuf_iterator<char>());
    CHECK(raw.size() > 256);

    auto write_corrupt = [&](const std::string& name, const std::vector<uint8_t>& bytes) -> std::string {
        fs::path p = work_dir / name;
        std::ofstream out(p, std::ios::binary);
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        return p.string();
    };

    // 5a. Truncated file
    {
        std::vector<uint8_t> trunc(raw.begin(), raw.begin() + raw.size() / 2);
        GPackReader r;
        CHECK(!r.open(write_corrupt("trunc.gpack", trunc), err, false));
    }
    // 5b. Bad header magic
    {
        std::vector<uint8_t> bad_hdr = raw;
        bad_hdr[0] = 'X';
        GPackReader r;
        CHECK(!r.open(write_corrupt("bad_hdr.gpack", bad_hdr), err, false));
    }
    // 5c. Unsupported min_reader_version
    {
        std::vector<uint8_t> future_ver = raw;
        GPackHeaderV1 hdr{};
        std::memcpy(&hdr, future_ver.data(), sizeof(hdr));
        hdr.min_reader_version = 99;
        std::memcpy(future_ver.data(), &hdr, sizeof(hdr));
        GPackReader r;
        CHECK(!r.open(write_corrupt("future_ver.gpack", future_ver), err, false));
    }
    // 5d. Bad footer magic
    {
        std::vector<uint8_t> bad_ftr = raw;
        bad_ftr[bad_ftr.size() - 1] ^= 0xFF;
        GPackReader r;
        CHECK(!r.open(write_corrupt("bad_ftr.gpack", bad_ftr), err, false));
    }
    // 5e. Corrupted manifest payload byte (detected even in fast open mode)
    {
        std::vector<uint8_t> bad_manifest = raw;
        GPackHeaderV1 hdr{};
        std::memcpy(&hdr, bad_manifest.data(), sizeof(hdr));
        bad_manifest[static_cast<size_t>(hdr.manifest_offset + 4)] ^= 0x01;
        GPackReader r;
        CHECK(!r.open(write_corrupt("bad_manifest.gpack", bad_manifest), err, false));
    }
    // 5f. Corrupted tensor data byte (detected in full verification mode)
    {
        std::vector<uint8_t> bad_data = raw;
        const auto* dense_sec = r_single.find_section(GPackSectionKind::DensePackBlob);
        CHECK(dense_sec != nullptr);
        bad_data[static_cast<size_t>(dense_sec->offset + 8)] ^= 0xFF;
        GPackReader r;
        CHECK(!r.open(write_corrupt("bad_data.gpack", bad_data), err, true));
    }
    // 5g. Failed conversion leaves no partial output file at output_path
    {
        const fs::path bad_out = work_dir / "should_not_exist.gpack";
        GPackBuildOptions fail_opts;
        fail_opts.primary_input_path = (work_dir / "nonexistent_input.gguf").string();
        fail_opts.output_path = bad_out.string();
        GPackBuildReport frep;
        CHECK(!GPackBuilder::build(fail_opts, frep, err));
        CHECK(!fs::exists(bad_out));
    }
}

void test_gguf_single_and_split_conversion(const fs::path& work_dir) {
    const fs::path gguf_dir = work_dir / "gguf_conv";
    fs::create_directories(gguf_dir);

    std::vector<fixture::Kv> base_kv = {
        fixture::str("general.architecture", "qwen35moe"),
        fixture::str("general.name", "tiny-ornith"),
        fixture::u32("qwen35moe.block_count", 1),
        fixture::u32("qwen35moe.full_attention_interval", 1),
        fixture::u32("qwen35moe.embedding_length", 256),
        fixture::u32("qwen35moe.attention.head_count", 4),
        fixture::u32("qwen35moe.attention.head_count_kv", 2),
        fixture::u32("qwen35moe.attention.key_length", 256),
        fixture::u32("qwen35moe.expert_count", 4),
        fixture::u32("qwen35moe.expert_used_count", 2),
        fixture::u32("qwen35moe.expert_feed_forward_length", 256),
        fixture::u32("qwen35moe.context_length", 512)
    };

    std::vector<fixture::Tensor> t_shard1 = {
        {"token_embd.weight", {256, 32}, 18, 1}, // IQ3_XXS
        {"output_norm.weight", {256}, 0, 2},     // F32
        {"blk.0.attn_norm.weight", {256}, 0, 3},
        {"blk.0.ffn_gate_inp.weight", {256, 4}, 0, 4}
    };
    std::vector<fixture::Tensor> t_shard2 = {
        {"output.weight", {256, 32}, 13, 5}, // Q5_K
        {"blk.0.ffn_gate_exps.weight", {256, 256, 4}, 18, 6},
        {"blk.0.ffn_up_exps.weight", {256, 256, 4}, 18, 7},
        {"blk.0.ffn_down_exps.weight", {256, 256, 4}, 10, 8} // Q2_K
    };

    std::vector<fixture::Tensor> all_t = t_shard1;
    all_t.insert(all_t.end(), t_shard2.begin(), t_shard2.end());

    // 1. Write single-file GGUF
    const fs::path single_gguf = gguf_dir / "tiny.gguf";
    fixture::write(single_gguf, base_kv, all_t);

    // 2. Write 2-shard split GGUF
    const fs::path split1 = gguf_dir / "tiny-00001-of-00002.gguf";
    const fs::path split2 = gguf_dir / "tiny-00002-of-00002.gguf";
    auto kv1 = base_kv;
    auto sk1 = fixture::split_keys(0, 2, all_t.size());
    kv1.insert(kv1.end(), sk1.begin(), sk1.end());
    auto sk2 = fixture::split_keys(1, 2, all_t.size());
    fixture::write(split1, kv1, t_shard1);
    fixture::write(split2, sk2, t_shard2);

    // 3. Convert both to embedded-expert GPACK
    const fs::path out_single = gguf_dir / "from_single.gpack";
    const fs::path out_split  = gguf_dir / "from_split.gpack";
    const fs::path out_ext    = gguf_dir / "from_split_ext.gpack";
    std::string err;

    GPackBuildOptions opts1;
    opts1.primary_input_path = single_gguf.string();
    opts1.output_path = out_single.string();
    opts1.model_name = "tiny-ornith";
    GPackBuildReport rep1;
    CHECK_ERR(GPackBuilder::build(opts1, rep1, err), err);

    GPackBuildOptions opts2;
    opts2.primary_input_path = split1.string();
    opts2.output_path = out_split.string();
    opts2.model_name = "tiny-ornith";
    GPackBuildReport rep2;
    CHECK_ERR(GPackBuilder::build(opts2, rep2, err), err);

    CHECK(rep1.tensor_count == rep2.tensor_count);
    CHECK(rep1.expert_blob_bytes == rep2.expert_blob_bytes);
    CHECK(rep1.dense_pack_bytes == rep2.dense_pack_bytes);

    GPackReader gr1, gr2;
    CHECK_ERR(gr1.open(out_single.string(), err, true), err);
    CHECK_ERR(gr2.open(out_split.string(), err, true), err);
    // Verify ExpertBlobs SHA-256 is identical between single-file GGUF and 2-shard split GGUF!
    const auto* es1 = gr1.find_section(GPackSectionKind::ExpertBlobs);
    const auto* es2 = gr2.find_section(GPackSectionKind::ExpertBlobs);
    CHECK(es1 != nullptr && es2 != nullptr);
    if (es1 && es2) {
        CHECK(std::memcmp(es1->sha256, es2->sha256, 32) == 0);
    }

    // 4. Convert split GGUF with external experts (--external-experts)
    opts2.output_path = out_ext.string();
    opts2.embed_experts = false;
    GPackBuildReport rep_ext;
    CHECK(GPackBuilder::build(opts2, rep_ext, err));
    GPackReader gr_ext;
    CHECK(gr_ext.open(out_ext.string(), err, true));
    CHECK(!gr_ext.has_embedded_experts());
    CHECK(gr_ext.has_external_experts());
    CHECK(!gr_ext.external_expert_primary_gguf().empty());
    CHECK(gr_ext.expert_layers().size() == 1);
    if (!gr_ext.expert_layers().empty()) {
        CHECK(gr_ext.expert_layers()[0].gate_shard == "tiny-00002-of-00002.gguf");
    }

    // 5. Truncated GGUF shard must be rejected
    const fs::path trunc_gguf = gguf_dir / "trunc.gguf";
    fixture::write(trunc_gguf, base_kv, all_t, 512);
    GPackBuildOptions bad_opts;
    bad_opts.primary_input_path = trunc_gguf.string();
    bad_opts.output_path = (gguf_dir / "bad_trunc.gpack").string();
    GPackBuildReport bad_rep;
    CHECK(!GPackBuilder::build(bad_opts, bad_rep, err));
    CHECK(!fs::exists(bad_opts.output_path));
}

} // namespace

int main() {
    const fs::path work_dir = fs::current_path() / "gpack_test_tmp";
    std::error_code ec;
    fs::remove_all(work_dir, ec);
    fs::create_directories(work_dir, ec);

    test_gbt1_binary_tokenizer(work_dir);
    test_safetensors_and_gpack_integrity(work_dir);
    test_gguf_single_and_split_conversion(work_dir);

    fs::remove_all(work_dir, ec);
    std::cout << "gpack_test: " << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed == 0 ? 0 : 1;
}
