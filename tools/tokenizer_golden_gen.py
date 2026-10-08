#!/usr/bin/env python3
import json
import jinja2
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, Regex, AddedToken

PACKS = {
    "qwen": "/mnt/models-ssd/Strata-data/packs/unsloth-ud-iq4_xs/tokenizer",
    "ornith": "/mnt/models-ssd/Strata-data/packs/ornith-1.5-35b/tokenizer",
}

def create_tokenizer(p):
    with open(f"{p}/tokenizer.json") as f:
        info = json.load(f)
    bpe = models.BPE.from_file(f"{p}/vocab.json", f"{p}/merges.txt")
    tok = Tokenizer(bpe)
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(info["pre_pattern"]), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    ])
    tok.decoder = decoders.ByteLevel()

    with open(f"{p}/token_type.json") as f:
        token_types = json.load(f)
    with open(f"{p}/vocab.json") as f:
        vocab = json.load(f)

    inv_vocab = {v: k for k, v in vocab.items()}
    special_tokens = []
    for idx, ttype in enumerate(token_types):
        if ttype in (3, 4):
            special_tokens.append(inv_vocab[idx])

    tok.add_special_tokens([AddedToken(t, special=True) for t in special_tokens])
    return tok

def create_jinja_env(p):
    with open(f"{p}/chat_template.jinja") as f:
        template_str = f.read()
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    return env.from_string(template_str)

def main():
    qwen_tok = create_tokenizer(PACKS["qwen"])
    qwen_tpl = create_jinja_env(PACKS["qwen"])
    ornith_tpl = create_jinja_env(PACKS["ornith"])

    test_strings = [
        ("empty", ""),
        ("english_simple", "Hello, world!"),
        ("english_sentences", "The quick brown fox jumps over the lazy dog. 1234567890!"),
        ("contractions", "I'm you're he's she'll we'd they've it's won't can't didn't"),
        ("punctuation", "!@#$%^&*()_+-=[]{}|;':\",./<>?`~"),
        ("multiple_spaces", "This    has    multiple   spaces    between words."),
        ("tabs_and_newlines", "Line 1\nLine 2\r\n\tIndented line\n\n\nTriple newline\t\tDouble tab"),
        ("leading_trailing_ws", "   \t\n  Leading and trailing whitespace.  \n\t  "),
        ("code_and_json", "{\"key\": \"value\", \"numbers\": [1, 2, 3], \"nested\": {\"valid\": true}}"),
        ("code_python", "def calculate_sum(a: int, b: int) -> int:\n    # Return the sum\n    return a + b\n\nprint(calculate_sum(10, 20))"),
        ("hindi", "नमस्ते दुनिया! यह एक परीक्षण संदेश है। 12345"),
        ("chinese", "你好，世界！这是一个测试句子。包含中文标点符号：“”‘’；：，。？！"),
        ("mixed_multilingual", "Hello 你好 नमस्ते Привет World 🌍 123 !"),
        ("emoji", "👋 🌍 🚀 💻 🦊 🐱 🐶 🍕 🍣 🍦"),
        ("combining_unicode", "e\u0301 a\u0300 u\u0308 c\u0327 n\u0303 o\u0302"),
        ("special_ids_text", "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\n<think>\n"),
        ("long_input", "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 50),
        ("consecutive_newlines_spaces", "\n\n   \n\n\t\t\t\n   \n"),
        ("math_and_symbols", "∀x ∈ ℝ, ∃y: y > x; α + β = γ; 2 + 2 = 4 ≠ 5; √4 = 2; π ≈ 3.14159"),
    ]

    cases = []
    for name, s in test_strings:
        enc_normal = qwen_tok.encode(s)
        dec_normal = qwen_tok.decode(enc_normal.ids, skip_special_tokens=False)
        cases.append({
            "name": name,
            "text": s,
            "tokens": enc_normal.ids,
            "decoded": dec_normal,
            "allow_special": True
        })

    # Chat template test cases
    chat_cases = []

    dialog_1 = [
        {"role": "user", "content": "What is 2 + 2?"}
    ]
    dialog_2 = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ]
    dialog_3 = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"},
        {"role": "user", "content": "What's the weather in Tokyo?"}
    ]
    dialog_4 = [
        {"role": "user", "content": "First line\nSecond line with indentation\n    Indented"}
    ]

    for d_name, dialog in [("single_user", dialog_1),
                           ("system_and_user", dialog_2),
                           ("multi_turn", dialog_3),
                           ("multiline_user", dialog_4)]:
        q_rendered = qwen_tpl.render(messages=dialog, add_generation_prompt=True)
        q_tokens = qwen_tok.encode(q_rendered).ids
        chat_cases.append({
            "name": f"qwen_{d_name}",
            "model": "qwen",
            "messages": dialog,
            "rendered": q_rendered,
            "tokens": q_tokens
        })

        o_rendered = ornith_tpl.render(messages=dialog, add_generation_prompt=True)
        o_tokens = qwen_tok.encode(o_rendered).ids
        chat_cases.append({
            "name": f"ornith_{d_name}",
            "model": "ornith",
            "messages": dialog,
            "rendered": o_rendered,
            "tokens": o_tokens
        })

    out_data = {
        "text_cases": cases,
        "chat_cases": chat_cases
    }

    with open("tests/data/tokenizer_golden.json", "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(cases)} text cases and {len(chat_cases)} chat cases.")

if __name__ == "__main__":
    main()
