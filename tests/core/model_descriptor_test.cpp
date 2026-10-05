#include "guild/model/model_descriptor.hpp"
#include "guild/model/archetype.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>

int main() {
    using namespace guild::model;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "assertion failed at line %d: %s\n", __LINE__, #cond); \
        return 1; \
    } \
} while (0)

    // 1. String conversion checks
    CHECK(std::strcmp(archetype_to_string(ModelArchetype::Qwen4Exp), "qwen4exp") == 0);
    CHECK(std::strcmp(archetype_to_string(ModelArchetype::Qwen35MoE), "qwen35moe") == 0);
    CHECK(std::strcmp(archetype_to_string(ModelArchetype::GLM), "glm") == 0);
    CHECK(std::strcmp(archetype_to_string(ModelArchetype::DeepSeek), "deepseek") == 0);
    CHECK(std::strcmp(archetype_to_string(ModelArchetype::Mixtral), "mixtral") == 0);

    CHECK(archetype_from_string("Qwen4Exp") == ModelArchetype::Qwen4Exp);
    CHECK(archetype_from_string("qwen35moe") == ModelArchetype::Qwen35MoE);
    CHECK(archetype_from_string("GLM") == ModelArchetype::GLM);
    CHECK(archetype_from_string("deepseek") == ModelArchetype::DeepSeek);
    CHECK(archetype_from_string("mixtral") == ModelArchetype::Mixtral);

    // 2. Registry verification
    auto& reg = ArchetypeRegistry::instance();
    CHECK(reg.find(ModelArchetype::Qwen4Exp) != nullptr);
    CHECK(reg.find(ModelArchetype::Qwen35MoE) != nullptr);
    CHECK(reg.find(ModelArchetype::GLM) != nullptr);
    CHECK(reg.find(ModelArchetype::DeepSeek) != nullptr);
    CHECK(reg.find(ModelArchetype::Mixtral) != nullptr);

    CHECK(reg.find("qwen4exp") != nullptr);
    CHECK(reg.find("glm") != nullptr);
    CHECK(reg.find("deepseek") != nullptr);
    CHECK(reg.find("mixtral") != nullptr);

    // 3. Geometry and layer classification
    ModelDescriptor desc;
    desc.name = "Qwen3.8-Flash-Next";
    desc.archetype = ModelArchetype::Qwen4Exp;
    desc.attn.n_embd = 2560;
    desc.attn.n_layers = 48;
    desc.attn.n_heads = 24;
    desc.attn.n_kv_heads = 2;
    desc.attn.head_dim = 256;
    desc.attn.context_length = 262144;
    desc.attn.pattern = AttentionPattern::HybridGDN;
    desc.attn.full_attn_interval = 4;

    desc.moe.n_routed_experts = 512;
    desc.moe.k_active_experts = 10;
    desc.moe.expert_dim_ff = 640;

    CHECK(desc.moe.total_routed_experts(desc.attn.n_layers) == 48 * 512); // 24576
    CHECK(desc.attn.n_full_attn_layers() == 12);
    CHECK(desc.attn.n_recurrent_layers() == 36);

    CHECK(!desc.attn.is_full_attn_layer(0));
    CHECK(!desc.attn.is_full_attn_layer(1));
    CHECK(!desc.attn.is_full_attn_layer(2));
    CHECK(desc.attn.is_full_attn_layer(3));
    CHECK(desc.attn.is_full_attn_layer(7));
    CHECK(desc.attn.is_full_attn_layer(47));

    // 4. Memory estimation checks
    // KV FP16 (2 bytes): 12 full-attention layers * 2 (K+V) * 2 kv_heads * 256 head_dim * 2 bytes = 24,576 bytes/tok
    // 262,144 tokens * 24,576 = 6,442,450,944 bytes (exactly 6.0 GiB)
    const uint64_t kv_fp16 = desc.estimate_kv_bytes(262144, 2);
    CHECK(kv_fp16 == 6442450944ULL);

    // Routed weights FP16: 24,576 experts * 3 * 640 * 2560 * 2 bytes = 241,591,910,400 bytes (~225 GiB)
    const uint64_t routed_fp16 = desc.estimate_routed_weights_bytes(2);
    CHECK(routed_fp16 == 24576ULL * 3ULL * 640ULL * 2560ULL * 2ULL);

    std::printf("model_descriptor_test: all assertions PASSED\n");
    return 0;
}
