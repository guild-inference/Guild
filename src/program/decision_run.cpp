// src/program/decision_run.cpp - typed decisions (the /v1/systemone format) from a decision checkpoint such as
// StartLux-Decision-v1-35B-A3B: each request's questions are rendered as prompts and read from the option letters'
// next-token logits (include/guild/decision/decision.hpp).  No text is generated.
//
//   decision_run --native M.gguf --pack P --requests in.jsonl [--out out.jsonl] [--detail] [--fresh] [--repeat N]
//
// in.jsonl: one request a line, {"id", "state", "questions": {name: {"type", "instructions", "criteria"}}}.
// out.jsonl: per request {"id", "answers", "usage", "timing"[, "questions"]}, or {"id", "error"}.
#include "guild/decision/decision.hpp"
#include "guild/program/model_options.hpp"
#include "guild/runtime/chat_template.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using guild::decision::JsonValue;

void usage() {
    std::cout << "Usage: decision_run [options]\n"
              << guild::program::model_options_usage() << "Decisions:\n"
              << "  --requests <file>          JSON lines, one /v1/systemone request each ({\"id\", \"state\", "
                 "\"questions\"})\n"
              << "  --request <json>           One request\n"
              << "  --out <file>               Where the answers go (default: standard output)\n"
              << "  --decision-config <file>   The checkpoint's decision_config.json (default: in the tokenizer "
                 "directory)\n"
              << "  --chat-template <file>     The chat template (default: the tokenizer directory's)\n"
              << "  --detail                   Each question's prompt tokens, letter logits and probabilities too\n"
              << "  --fresh                    Forget the previous request's prompts before each request\n"
              << "  --no-share                 Read every question's prompt whole (no shared-beginning checkpoint)\n"
              << "  --repeat <n>               Answer each request n times (timing in the last answer's line)\n";
}

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

JsonValue obj() {
    return JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
}

} // namespace

int main(int argc, char** argv) try {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr)
        setenv("CUDA_MODULE_LOADING", "EAGER", 0);

    guild::program::ModelOptions mo;
    mo.max_context = 65536; // the KV cells this holds fit the GPU with every attention layer's K/V there
    std::string requests_path, request_json, out_path, config_path, template_path;
    bool detail = false, fresh = false, share = true;
    int repeat = 1;
    for (int i = 1; i < argc; ++i) {
        const int r = guild::program::parse_model_option(argc, argv, i, mo, "decision_run");
        if (r == 2)
            return 2;
        if (r == 1)
            continue;
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires an argument\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else if (a == "--requests") {
            requests_path = next("--requests");
        } else if (a == "--request") {
            request_json = next("--request");
        } else if (a == "--out") {
            out_path = next("--out");
        } else if (a == "--decision-config") {
            config_path = next("--decision-config");
        } else if (a == "--chat-template") {
            template_path = next("--chat-template");
        } else if (a == "--detail") {
            detail = true;
        } else if (a == "--fresh") {
            fresh = true;
        } else if (a == "--no-share") {
            share = false;
        } else if (a == "--repeat") {
            repeat = std::max(1, std::atoi(next("--repeat").c_str()));
        } else {
            std::fprintf(stderr, "decision_run: unknown option %s (see --help)\n", a.c_str());
            return 2;
        }
    }
    if (requests_path.empty() == request_json.empty()) {
        std::fprintf(stderr, "decision_run: give --requests <file> or --request <json>\n");
        return 2;
    }
    if (mo.tokenizer_dir.empty() && !mo.pack_dir.empty())
        mo.tokenizer_dir = mo.pack_dir + "/tokenizer";
    if (config_path.empty())
        config_path = mo.tokenizer_dir + "/decision_config.json";
    guild::decision::Config cfg;
    std::string err;
    if (!guild::decision::load_config(config_path, cfg, err)) {
        std::fprintf(stderr, "decision_run: %s\n", err.c_str());
        return 2;
    }
    guild::runtime::ChatTemplate tmpl;
    if (!tmpl.load(template_path.empty() ? mo.tokenizer_dir : template_path, err)) {
        std::fprintf(stderr, "decision_run: chat template: %s\n", err.c_str());
        return 2;
    }

    std::vector<std::string> lines;
    if (!request_json.empty()) {
        lines.push_back(request_json);
    } else {
        std::ifstream in(requests_path);
        if (!in) {
            std::fprintf(stderr, "decision_run: cannot read %s\n", requests_path.c_str());
            return 2;
        }
        for (std::string line; std::getline(in, line);)
            if (line.find_first_not_of(" \t\r") != std::string::npos)
                lines.push_back(line);
    }

    const auto t_load = std::chrono::steady_clock::now();
    guild::program::LoadedModel loaded;
    if (const int rc = guild::program::load_model(mo, "decision_run", loaded); rc != 0)
        return rc;
    std::fprintf(stderr, "decision_run: loaded in %.1f s, context %lld, %zu requests\n", ms_since(t_load) / 1000,
                 (long long) loaded.context, lines.size());
    guild::decision::Decider decider(*loaded.session, loaded.model->tokenizer(), tmpl, cfg);
    decider.share_prefix = share;
    decider.max_length = loaded.context;
    if (!decider.check(err)) {
        std::fprintf(stderr, "decision_run: %s\n", err.c_str());
        return 1;
    }

    std::ofstream out_file;
    if (!out_path.empty()) {
        out_file.open(out_path);
        if (!out_file) {
            std::fprintf(stderr, "decision_run: cannot write %s\n", out_path.c_str());
            return 2;
        }
    }
    std::ostream& out = out_path.empty() ? std::cout : out_file;
    int failed = 0;
    for (const auto& line : lines) {
        std::string perr;
        auto req = JsonValue::parse(line, perr);
        JsonValue rec = obj();
        rec.obj_val.emplace_back("id", req && req->is_object() && req->contains("id") ? (*req)["id"] : JsonValue());
        if (!req) {
            rec.obj_val.emplace_back("error", "invalid JSON: " + perr);
            out << guild::decision::dump(rec) << "\n" << std::flush;
            ++failed;
            continue;
        }
        JsonValue answers;
        guild::decision::Usage use;
        std::map<std::string, guild::decision::QuestionDetail> det;
        std::vector<double> ms;
        bool ok = true;
        for (int r = 0; r < repeat && ok; ++r) {
            if (fresh)
                loaded.session->reset();
            const auto t0 = std::chrono::steady_clock::now();
            ok = decider.decide(*req, answers, use, detail ? &det : nullptr, err);
            ms.push_back(ms_since(t0));
        }
        if (!ok) {
            rec.obj_val.emplace_back("error", err);
            out << guild::decision::dump(rec) << "\n" << std::flush;
            std::fprintf(stderr, "decision_run: %s: %s\n", guild::decision::dump(rec["id"]).c_str(), err.c_str());
            ++failed;
            if (!loaded.model->is_ready()) // a runtime failure: the model refuses further work
                return 1;
            continue;
        }
        rec.obj_val.emplace_back("answers", answers);
        JsonValue u = obj();
        u.obj_val.emplace_back("input_tokens", JsonValue((int64_t) use.input_tokens));
        u.obj_val.emplace_back("output_tokens", JsonValue((int64_t) 0));
        rec.obj_val.emplace_back("usage", u);
        double mean = 0, var = 0;
        for (double v : ms)
            mean += v / (double) ms.size();
        for (double v : ms)
            var += (v - mean) * (v - mean) / (double) std::max<size_t>(1, ms.size() - 1);
        JsonValue t = obj();
        JsonValue runs(std::vector<JsonValue>{});
        for (double v : ms)
            runs.arr_val.emplace_back(v);
        t.obj_val.emplace_back("ms", runs);
        t.obj_val.emplace_back("mean_ms", mean);
        t.obj_val.emplace_back("stdev_ms", ms.size() > 1 ? std::sqrt(var) : 0.0);
        t.obj_val.emplace_back("prefill_ms", use.prefill_ms);
        t.obj_val.emplace_back("window_ms", use.window_ms);
        t.obj_val.emplace_back("cached_tokens", JsonValue((int64_t) use.cached_tokens));
        t.obj_val.emplace_back("passes", JsonValue((int64_t) use.passes));
        rec.obj_val.emplace_back("timing", t);
        if (detail) {
            JsonValue qs = obj();
            for (const auto& [k, d] : det) {
                JsonValue q = obj();
                q.obj_val.emplace_back("type", d.type);
                JsonValue opts(std::vector<JsonValue>{}), toks(std::vector<JsonValue>{}), z(std::vector<JsonValue>{}),
                    p(std::vector<JsonValue>{});
                for (const auto& o : d.options)
                    opts.arr_val.emplace_back(o);
                for (int32_t id : d.tokens)
                    toks.arr_val.emplace_back((int64_t) id);
                for (float v : d.letter_logits)
                    z.arr_val.emplace_back((double) v);
                for (double v : d.probs)
                    p.arr_val.emplace_back(v);
                q.obj_val.emplace_back("options", opts);
                q.obj_val.emplace_back("tokens", toks);
                q.obj_val.emplace_back("letter_logits", z);
                q.obj_val.emplace_back("temperature", d.temperature);
                q.obj_val.emplace_back("probs", p);
                qs.obj_val.emplace_back(k, q);
            }
            rec.obj_val.emplace_back("questions", qs);
        }
        out << guild::decision::dump(rec) << "\n" << std::flush;
        std::fprintf(stderr, "decision_run: %s: %lld tokens (%lld from the shared beginning), %.1f ms\n",
                     guild::decision::dump(rec["id"]).c_str(), (long long) use.input_tokens,
                     (long long) use.cached_tokens, ms.back());
    }
    return failed ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "decision_run: %s\n", e.what());
    return 1;
}
