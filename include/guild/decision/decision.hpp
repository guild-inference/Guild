// include/guild/decision/decision.hpp - typed decisions (the TypeSafe /v1/systemone format) read from a causal LM's
// next-token logits, as the StartLux-Decision checkpoints define them (their startlux_decision/ package: jevfmt.py
// renders each question as one chat prompt, model.py reads the option letters' logits at its last position).
//
// A request is a state (text or JSON) and named questions of three types: choice (one of up to 26 options; wider
// lists run in rounds), noul (yes / no) and score (2 to 10 ordered levels).  Each question becomes one prompt:
// a fixed system line, "Evidence:" with the state, the question and its lettered options, then the checkpoint's
// thinking-off assistant prefix.  The answer is softmax(z / T) over the letters' logits z, T the question type's
// temperature from the checkpoint's decision_config.json.  Nothing is generated.
//
// The questions of a request repeat its state, so their prompts begin with the same tokens: the shared beginning is
// read once (GuildSession::score keeps a checkpoint there) and each question reads only its own tail.
#pragma once

#include "guild/runtime/chat_template.hpp"
#include "guild/runtime/session.hpp"
#include "guild/runtime/tokenizer.hpp"
#include "guild/server/json.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace guild::decision {

using JsonValue = guild::server::json::JsonValue;

/// decision_config.json
struct Config {
    std::string format;                        ///< "StartLux-Decision-v1"
    std::vector<int32_t> letter_ids;           ///< the token ids of A..Z
    std::map<std::string, double> temperature; ///< per question type
    int max_options = 26;                      ///< options a prompt can list
    int wide_group = 25, wide_keep = 3;        ///< a wider choice: groups of about this size, the top `keep` of each
    double wide_residual = 1e-3;               ///< the probability mass the options left out of the final round share
};
bool load_config(const std::string& path, Config& cfg, std::string& err);

struct Option {
    std::string id;
    std::optional<std::string> criterion;
};

/// One question as it is rendered (jevfmt.from_systemone's record)
struct Row {
    std::string id;
    std::string type; ///< choice, noul or score
    std::string state;
    std::string instructions;
    std::vector<Option> options;
};

// ---- the rendering (jevfmt.py)
/// A /v1/systemone question spec on `state` -> its record; false (with err) for a spec the format refuses
bool from_systemone(const JsonValue& state, const JsonValue& spec, const std::string& qid, Row& row, std::string& err);
/// jevfmt.validate
bool validate(const Row& row, std::string& err);
/// The user message: Evidence / Question / Options
std::string user_message(const Row& row);
/// The system line every prompt starts with
extern const char* const kSystem;
/// A record's prompt tokens: the chat template with the thinking-off prefix, tokenized with its special tokens
bool render_ids(const Row& row, const runtime::ChatTemplate& tmpl, const runtime::Tokenizer& tok,
                std::vector<int32_t>& ids, std::string& err);

// ---- Python's text conversions, as the reference applies them
/// json.dumps(v, ensure_ascii=False)
std::string py_json_dumps(const JsonValue& v);
/// The state as the prompt shows it (jevfmt.state_text: long arrays carry their element index)
std::string state_text(const JsonValue& state);
/// str.strip() (Unicode whitespace)
std::string py_strip(const std::string& s);

struct QuestionDetail {
    std::string type;
    std::vector<std::string> options;
    std::vector<int32_t> tokens;
    std::vector<float> letter_logits;
    double temperature = 1.0;
    std::vector<double> probs;
};

struct Usage {
    int64_t input_tokens = 0;
    int64_t cached_tokens = 0; ///< of those, taken from a checkpoint (the shared beginning) instead of read
    double prefill_ms = 0, window_ms = 0;
    int64_t passes = 0; ///< prompts scored
};

class Decider {
public:
    Decider(runtime::GuildSession& session, const runtime::Tokenizer& tok, const runtime::ChatTemplate& tmpl,
            const Config& cfg)
        : session_(session),
          tok_(tok),
          tmpl_(tmpl),
          cfg_(cfg) {
    }
    /// Checks that the chat template ends with the thinking-off prefix and that every letter is its own token after
    /// it, with the ids decision_config.json names (jevfmt.check_tokenizer).
    bool check(std::string& err) const;
    /// One request: answers (an object, the /v1/systemone format), usage; `detail` gets each question the model read
    /// in one pass (wide choices' rounds and single options are only in the answers).  Images are refused.
    bool decide(const JsonValue& request, JsonValue& answers, Usage& usage,
                std::map<std::string, QuestionDetail>* detail, std::string& err);
    /// Prompts sharing their first tokens read them once (a checkpoint); off, every prompt is read whole.
    bool share_prefix = true;
    int64_t max_length = 262144;

private:
    /// The letter logits of rows (each <= max_options options), in their option order; tokens counted in usage
    bool logits(const std::vector<Row>& rows, std::vector<std::vector<float>>& out,
                std::vector<std::vector<int32_t>>* tokens, Usage& usage, std::string& err);
    bool probs(const std::vector<Row>& rows, std::vector<std::vector<double>>& out, Usage& usage,
               std::vector<std::vector<float>>* raw, std::vector<std::vector<int32_t>>* tokens, std::string& err);
    bool wide(const JsonValue& state, const JsonValue& spec, std::vector<std::pair<std::string, double>>& out,
              Usage& usage, std::string& err);
    double temperature(const std::string& type) const;

    runtime::GuildSession& session_;
    const runtime::Tokenizer& tok_;
    const runtime::ChatTemplate& tmpl_;
    const Config& cfg_;
};

/// TypeSafe's confidence of a choice answer: (p_max - 1/n) / (1 - 1/n), clamped to [0, 1]
double choice_confidence(const std::vector<double>& p);
/// ... of a score answer: 1 - the probability-weighted distance in levels from the most likely level, relative to
/// that of an even spread from the middle level, floored at 0
double score_confidence(const std::vector<double>& p);

/// JSON text of a value, numbers with 9 significant digits (answers, details)
std::string dump(const JsonValue& v);

} // namespace guild::decision
