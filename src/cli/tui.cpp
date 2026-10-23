#include "guild/cli/tui.hpp"

#include "guild/artifact/gpack.hpp"
#include "guild/cli/ansi.hpp"
#include "guild/cli/hardware.hpp"
#include "guild/gexec/gexec_contract.hpp"
#include "guild/gexec/gexec_optimizer.hpp"
#include "guild/memory/planner.hpp"
#include "guild/model/archetype.hpp"
#include "guild/model/model_descriptor.hpp"
#include "guild/models/downloader.hpp"
#include "guild/models/manifest.hpp"
#include "guild/models/registry.hpp"
#include "guild/models/store.hpp"
#include "guild/runtime/chat_template.hpp"
#include "guild/runtime/version.hpp"
#include "guild/server/engine.hpp"
#include "guild/server/native_engine.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <conio.h>
#include <io.h>
#include <windows.h>
#else
#include <poll.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace guild::cli {

namespace {

namespace fs = std::filesystem;

std::atomic<bool> g_tui_sigint{false};
std::atomic<bool> g_tui_sigwinch{false};

void handle_tui_sigint(int) {
    g_tui_sigint.store(true, std::memory_order_relaxed);
}

#if !defined(_WIN32)
void handle_tui_sigwinch(int) {
    g_tui_sigwinch.store(true, std::memory_order_relaxed);
}
#endif

// RAII Terminal Mode Guard: enters alternate screen + raw/cbreak input and restores on exit
class TerminalGuard {
public:
    explicit TerminalGuard(bool interactive) : interactive_(interactive) {
        if (!interactive_)
            return;
#if defined(_WIN32)
        hOut_ = GetStdHandle(STD_OUTPUT_HANDLE);
        hIn_ = GetStdHandle(STD_INPUT_HANDLE);
        if (hOut_ != INVALID_HANDLE_VALUE && GetConsoleMode(hOut_, &orig_out_mode_)) {
            has_out_mode_ = true;
            SetConsoleMode(hOut_, orig_out_mode_ | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
        if (hIn_ != INVALID_HANDLE_VALUE && GetConsoleMode(hIn_, &orig_in_mode_)) {
            has_in_mode_ = true;
            DWORD mode = orig_in_mode_;
            mode &= ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT | ENABLE_PROCESSED_INPUT);
            mode |= ENABLE_WINDOW_INPUT;
            SetConsoleMode(hIn_, mode);
        }
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
#else
        if (isatty(STDIN_FILENO) && tcgetattr(STDIN_FILENO, &orig_termios_) == 0) {
            has_termios_ = true;
            struct termios raw = orig_termios_;
            raw.c_lflag &= static_cast<tcflag_t>(~(ECHO | ICANON | IEXTEN | ISIG));
            raw.c_iflag &= static_cast<tcflag_t>(~(IXON | ICRNL));
            raw.c_cc[VMIN] = 0;
            raw.c_cc[VTIME] = 0;
            tcsetattr(STDIN_FILENO, TCSANOW, &raw);
        }
#endif
        std::cout << "\033[?1049h\033[?25l\033[2J\033[H" << std::flush;
    }

    ~TerminalGuard() {
        restore();
    }

    void restore() {
        if (restored_)
            return;
        restored_ = true;
        if (!interactive_)
            return;
        std::cout << "\033[0m\033[?25h\033[?1049l" << std::flush;
#if defined(_WIN32)
        if (has_out_mode_ && hOut_ != INVALID_HANDLE_VALUE) {
            SetConsoleMode(hOut_, orig_out_mode_);
        }
        if (has_in_mode_ && hIn_ != INVALID_HANDLE_VALUE) {
            SetConsoleMode(hIn_, orig_in_mode_);
        }
#else
        if (has_termios_) {
            tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios_);
        }
#endif
    }

private:
    bool interactive_ = false;
    bool restored_ = false;
#if defined(_WIN32)
    HANDLE hOut_ = INVALID_HANDLE_VALUE;
    HANDLE hIn_ = INVALID_HANDLE_VALUE;
    DWORD orig_out_mode_ = 0;
    DWORD orig_in_mode_ = 0;
    bool has_out_mode_ = false;
    bool has_in_mode_ = false;
#else
    struct termios orig_termios_ {};
    bool has_termios_ = false;
#endif
};

enum class KeyKind {
    None,
    Char,
    Enter,
    Newline, // Alt+Enter or Ctrl+J
    Backspace,
    DeleteKey,
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    PageUp,
    PageDown,
    Tab,
    Escape,
    CtrlC,
    CtrlQ,
    CtrlL,
    CtrlR,
    CtrlU,
    CtrlD,
    F1,
    F2,
    F3,
    F4,
    F5,
};

struct KeyEvent {
    KeyKind kind = KeyKind::None;
    std::string text;
};

void query_terminal_size(int& cols, int& rows, int override_cols, int override_rows) {
    if (override_cols > 0 && override_rows > 0) {
        cols = override_cols;
        rows = override_rows;
        return;
    }
    cols = 120;
    rows = 36;
#if defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO csbi{};
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE && GetConsoleScreenBufferInfo(hOut, &csbi)) {
        int c = csbi.srWindow.Right - csbi.srWindow.Left + 1;
        int r = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
        if (c >= 40)
            cols = c;
        if (r >= 16)
            rows = r;
    }
#else
    struct winsize ws {};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
        if (ws.ws_col >= 40)
            cols = ws.ws_col;
        if (ws.ws_row >= 16)
            rows = ws.ws_row;
    }
#endif
    if (const char* env_c = std::getenv("COLUMNS")) {
        int c = std::atoi(env_c);
        if (c >= 40)
            cols = c;
    }
    if (const char* env_r = std::getenv("LINES")) {
        int r = std::atoi(env_r);
        if (r >= 16)
            rows = r;
    }
}

KeyEvent read_key_event(int timeout_ms) {
#if defined(_WIN32)
    HANDLE hIn = GetStdHandle(STD_INPUT_HANDLE);
    if (hIn == INVALID_HANDLE_VALUE)
        return {};
    DWORD wait_res = WaitForSingleObject(hIn, static_cast<DWORD>(timeout_ms));
    if (wait_res != WAIT_OBJECT_0)
        return {};
    INPUT_RECORD ir{};
    DWORD read_cnt = 0;
    while (PeekConsoleInputW(hIn, &ir, 1, &read_cnt) && read_cnt > 0) {
        ReadConsoleInputW(hIn, &ir, 1, &read_cnt);
        if (ir.EventType == WINDOW_BUFFER_SIZE_EVENT) {
            return {};
        }
        if (ir.EventType != KEY_EVENT || !ir.Event.KeyEvent.bKeyDown)
            continue;
        const WORD vk = ir.Event.KeyEvent.wVirtualKeyCode;
        const WCHAR wc = ir.Event.KeyEvent.uChar.UnicodeChar;
        const DWORD ctrl = ir.Event.KeyEvent.dwControlKeyState;
        const bool alt_down = (ctrl & (LEFT_ALT_PRESSED | RIGHT_ALT_PRESSED)) != 0;

        if (vk == VK_F1)
            return {KeyKind::F1, ""};
        if (vk == VK_F2)
            return {KeyKind::F2, ""};
        if (vk == VK_F3)
            return {KeyKind::F3, ""};
        if (vk == VK_F4)
            return {KeyKind::F4, ""};
        if (vk == VK_F5)
            return {KeyKind::F5, ""};
        if (vk == VK_UP)
            return {KeyKind::Up, ""};
        if (vk == VK_DOWN)
            return {KeyKind::Down, ""};
        if (vk == VK_LEFT)
            return {KeyKind::Left, ""};
        if (vk == VK_RIGHT)
            return {KeyKind::Right, ""};
        if (vk == VK_HOME)
            return {KeyKind::Home, ""};
        if (vk == VK_END)
            return {KeyKind::End, ""};
        if (vk == VK_PRIOR)
            return {KeyKind::PageUp, ""};
        if (vk == VK_NEXT)
            return {KeyKind::PageDown, ""};
        if (vk == VK_DELETE)
            return {KeyKind::DeleteKey, ""};
        if (vk == VK_ESCAPE)
            return {KeyKind::Escape, ""};
        if (vk == VK_TAB)
            return {KeyKind::Tab, ""};
        if (vk == VK_RETURN) {
            return {alt_down ? KeyKind::Newline : KeyKind::Enter, ""};
        }
        if (vk == VK_BACK)
            return {KeyKind::Backspace, ""};

        if (wc == 3)
            return {KeyKind::CtrlC, ""};
        if (wc == 17)
            return {KeyKind::CtrlQ, ""};
        if (wc == 12)
            return {KeyKind::CtrlL, ""};
        if (wc == 18)
            return {KeyKind::CtrlR, ""};
        if (wc == 21)
            return {KeyKind::CtrlU, ""};
        if (wc == 4)
            return {KeyKind::CtrlD, ""};
        if (wc == 10)
            return {KeyKind::Newline, ""};
        if (wc >= 32) {
            char utf8[8] = {0};
            int n = WideCharToMultiByte(CP_UTF8, 0, &wc, 1, utf8, sizeof(utf8) - 1, nullptr, nullptr);
            if (n > 0)
                return {KeyKind::Char, std::string(utf8, static_cast<size_t>(n))};
        }
    }
    return {};
#else
    struct pollfd pfd {};
    pfd.fd = STDIN_FILENO;
    pfd.events = POLLIN;
    int ret = poll(&pfd, 1, timeout_ms);
    if (ret <= 0 || !(pfd.revents & POLLIN))
        return {};

    unsigned char ch = 0;
    if (read(STDIN_FILENO, &ch, 1) != 1)
        return {};

    if (ch == 3)
        return {KeyKind::CtrlC, ""};
    if (ch == 17)
        return {KeyKind::CtrlQ, ""};
    if (ch == 12)
        return {KeyKind::CtrlL, ""};
    if (ch == 18)
        return {KeyKind::CtrlR, ""};
    if (ch == 21)
        return {KeyKind::CtrlU, ""};
    if (ch == 4)
        return {KeyKind::CtrlD, ""};
    if (ch == 9)
        return {KeyKind::Tab, ""};
    if (ch == 10)
        return {KeyKind::Newline, ""};
    if (ch == 13)
        return {KeyKind::Enter, ""};
    if (ch == 127 || ch == 8)
        return {KeyKind::Backspace, ""};

    if (ch == 27) {
        struct pollfd epfd {};
        epfd.fd = STDIN_FILENO;
        epfd.events = POLLIN;
        if (poll(&epfd, 1, 15) <= 0)
            return {KeyKind::Escape, ""};

        char seq[8] = {0};
        ssize_t n = read(STDIN_FILENO, seq, 1);
        if (n <= 0)
            return {KeyKind::Escape, ""};

        // Alt+Enter
        if (seq[0] == '\r' || seq[0] == '\n')
            return {KeyKind::Newline, ""};

        if (seq[0] == '[' || seq[0] == 'O') {
            if (poll(&epfd, 1, 15) <= 0)
                return {KeyKind::Escape, ""};
            if (read(STDIN_FILENO, &seq[1], 1) <= 0)
                return {KeyKind::Escape, ""};

            if (seq[0] == 'O') {
                if (seq[1] == 'P')
                    return {KeyKind::F1, ""};
                if (seq[1] == 'Q')
                    return {KeyKind::F2, ""};
                if (seq[1] == 'R')
                    return {KeyKind::F3, ""};
                if (seq[1] == 'S')
                    return {KeyKind::F4, ""};
                if (seq[1] == 'H')
                    return {KeyKind::Home, ""};
                if (seq[1] == 'F')
                    return {KeyKind::End, ""};
            }

            if (seq[1] == 'A')
                return {KeyKind::Up, ""};
            if (seq[1] == 'B')
                return {KeyKind::Down, ""};
            if (seq[1] == 'C')
                return {KeyKind::Right, ""};
            if (seq[1] == 'D')
                return {KeyKind::Left, ""};
            if (seq[1] == 'H')
                return {KeyKind::Home, ""};
            if (seq[1] == 'F')
                return {KeyKind::End, ""};

            if (seq[1] >= '0' && seq[1] <= '9') {
                if (poll(&epfd, 1, 15) > 0 && read(STDIN_FILENO, &seq[2], 1) == 1) {
                    if (seq[2] == '~') {
                        if (seq[1] == '1' || seq[1] == '7')
                            return {KeyKind::Home, ""};
                        if (seq[1] == '4' || seq[1] == '8')
                            return {KeyKind::End, ""};
                        if (seq[1] == '3')
                            return {KeyKind::DeleteKey, ""};
                        if (seq[1] == '5')
                            return {KeyKind::PageUp, ""};
                        if (seq[1] == '6')
                            return {KeyKind::PageDown, ""};
                    } else if (seq[1] == '1' && poll(&epfd, 1, 15) > 0 && read(STDIN_FILENO, &seq[3], 1) == 1 &&
                               seq[3] == '~') {
                        if (seq[2] == '1')
                            return {KeyKind::F1, ""};
                        if (seq[2] == '2')
                            return {KeyKind::F2, ""};
                        if (seq[2] == '3')
                            return {KeyKind::F3, ""};
                        if (seq[2] == '4')
                            return {KeyKind::F4, ""};
                        if (seq[2] == '5')
                            return {KeyKind::F5, ""};
                    }
                }
            }
        }
        return {KeyKind::Escape, ""};
    }

    if (ch >= 32) {
        std::string utf8(1, static_cast<char>(ch));
        int extra = 0;
        if ((ch & 0xE0) == 0xC0)
            extra = 1;
        else if ((ch & 0xF0) == 0xE0)
            extra = 2;
        else if ((ch & 0xF8) == 0xF0)
            extra = 3;
        for (int i = 0; i < extra; ++i) {
            unsigned char next = 0;
            if (read(STDIN_FILENO, &next, 1) == 1)
                utf8.push_back(static_cast<char>(next));
        }
        return {KeyKind::Char, utf8};
    }
    return {};
#endif
}

// UTF-8 safe character count and truncation for terminal columns
size_t utf8_display_len(const std::string& s) {
    size_t count = 0;
    for (unsigned char c : s) {
        if ((c & 0xC0) != 0x80)
            ++count;
    }
    return count;
}

std::string utf8_truncate_or_pad(const std::string& s, size_t target_cols, bool pad_right = true) {
    std::string out;
    size_t cols = 0;
    size_t i = 0;
    while (i < s.size() && cols < target_cols) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t char_bytes = 1;
        if ((c & 0xE0) == 0xC0)
            char_bytes = 2;
        else if ((c & 0xF0) == 0xE0)
            char_bytes = 3;
        else if ((c & 0xF8) == 0xF0)
            char_bytes = 4;
        if (i + char_bytes > s.size())
            char_bytes = 1;
        out.append(s, i, char_bytes);
        i += char_bytes;
        ++cols;
    }
    if (pad_right && cols < target_cols) {
        out.append(target_cols - cols, ' ');
    }
    return out;
}

std::vector<std::string> wrap_text_lines(const std::string& text, size_t width) {
    if (width < 8)
        width = 8;
    std::vector<std::string> out;
    std::stringstream ss(text);
    std::string paragraph;
    while (std::getline(ss, paragraph, '\n')) {
        if (paragraph.empty()) {
            out.emplace_back("");
            continue;
        }
        std::string current;
        size_t cur_cols = 0;
        size_t i = 0;
        while (i < paragraph.size()) {
            unsigned char c = static_cast<unsigned char>(paragraph[i]);
            size_t char_bytes = 1;
            if ((c & 0xE0) == 0xC0)
                char_bytes = 2;
            else if ((c & 0xF0) == 0xE0)
                char_bytes = 3;
            else if ((c & 0xF8) == 0xF0)
                char_bytes = 4;
            if (i + char_bytes > paragraph.size())
                char_bytes = 1;
            if (cur_cols + 1 > width) {
                out.push_back(current);
                current.clear();
                cur_cols = 0;
            }
            current.append(paragraph, i, char_bytes);
            i += char_bytes;
            ++cur_cols;
        }
        if (!current.empty())
            out.push_back(current);
    }
    if (out.empty())
        out.emplace_back("");
    return out;
}

void split_reasoning_and_answer(const std::string& raw, std::string& reasoning, std::string& answer,
                                bool& in_thinking) {
    reasoning.clear();
    answer.clear();
    in_thinking = false;
    std::string work = raw;
    const std::string open_tag = "<think>";
    const std::string close_tag = "</think>";

    auto open_pos = work.find(open_tag);
    if (open_pos != std::string::npos) {
        std::string before = work.substr(0, open_pos);
        auto close_pos = work.find(close_tag, open_pos + open_tag.size());
        if (close_pos != std::string::npos) {
            reasoning = work.substr(open_pos + open_tag.size(), close_pos - (open_pos + open_tag.size()));
            answer = before + work.substr(close_pos + close_tag.size());
        } else {
            in_thinking = true;
            reasoning = work.substr(open_pos + open_tag.size());
            answer = before;
        }
    } else {
        auto close_pos = work.find(close_tag);
        if (close_pos != std::string::npos) {
            reasoning = work.substr(0, close_pos);
            answer = work.substr(close_pos + close_tag.size());
        } else {
            answer = work;
        }
    }
    // Trim leading newlines on answer
    while (!answer.empty() && (answer.front() == '\n' || answer.front() == '\r'))
        answer.erase(answer.begin());
    while (!reasoning.empty() && (reasoning.front() == '\n' || reasoning.front() == '\r'))
        reasoning.erase(reasoning.begin());
}

enum class TuiTab {
    Chat = 0,
    Models = 1,
    Config = 2,
    Telemetry = 3,
    Help = 4,
};

struct ChatEntry {
    std::string role; // "user", "assistant", "system"
    std::string raw_content;
    double prompt_tok_s = 0.0;
    double decode_tok_s = 0.0;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    std::string strategy_summary;
    std::string finish_reason;

    ChatEntry() = default;
    ChatEntry(std::string r, std::string c) : role(std::move(r)), raw_content(std::move(c)) {
    }
};

struct ModelCatalogItem {
    std::string name;
    std::string architecture;
    std::string quantization;
    std::string local_path;
    std::string pack_dir;
    std::string status; // "READY", "INSTALLED", "AVAILABLE"
    int routed_experts = 0;
    int active_experts = 0;
    uint64_t size_bytes = 0;
};

std::string render_bar(double ratio, size_t width) {
    ratio = std::max(0.0, std::min(1.0, ratio));
    size_t filled = static_cast<size_t>(ratio * static_cast<double>(width) + 0.5);
    std::string bar = "[";
    for (size_t i = 0; i < width; ++i) {
        bar += (i < filled) ? "#" : "-";
    }
    bar += "]";
    return bar;
}

} // namespace

int run_tui(const TuiOptions& opts) {
    g_tui_sigint.store(false);
    g_tui_sigwinch.store(false);
    std::signal(SIGINT, handle_tui_sigint);
#if !defined(_WIN32)
    std::signal(SIGWINCH, handle_tui_sigwinch);
#endif

    const bool interactive_tty = opts.script_commands.empty() && is_tty();
    TerminalGuard term_guard(interactive_tty);

    models::StoreOptions st_opts;
    st_opts.custom_data_dir = opts.data_dir;
    models::ModelStore store(st_opts);
    store.init();

    HardwareProfile hw = detect_hardware();

    // Build model catalog from built-ins + installed manifests + known local GPACKs
    auto refresh_models = [&]() {
        std::vector<ModelCatalogItem> items;
        auto builtins = models::ModelRegistry::instance().list_available();
        auto installed = store.list_manifests();

        for (const auto& b : builtins) {
            ModelCatalogItem item;
            item.name = b.name;
            item.architecture = b.architecture;
            item.quantization = b.quantization;
            item.routed_experts = static_cast<int>(b.n_routed_experts);
            item.active_experts = static_cast<int>(b.k_active_experts);
            item.size_bytes = b.expected_size_bytes;
            item.status = store.has_manifest(b.name) ? "INSTALLED" : "AVAILABLE";

            for (const auto& f : b.files) {
                if (!f.local_path.empty() && fs::exists(f.local_path)) {
                    item.local_path = f.local_path;
                    item.status = "READY";
                    break;
                }
            }
            if (b.name == "qwen3.8-flash-next") {
                if (fs::exists("/mnt/models-ssd/Strata-data/gpacks/qwen3.8-flash-next-ext.gpack")) {
                    item.local_path = "/mnt/models-ssd/Strata-data/gpacks/qwen3.8-flash-next-ext.gpack";
                    item.status = "READY (GPACK)";
                } else if (fs::exists("/mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/"
                                      "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf")) {
                    item.local_path = "/mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/"
                                      "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf";
                    item.pack_dir = "/mnt/models-ssd/Strata-data/packs/unsloth-ud-iq4_xs";
                    item.status = "READY";
                }
            } else if (b.name == "ornith:35b") {
                if (fs::exists("/mnt/models-ssd/Strata-data/gpacks/ornith-1.5-35b-ext.gpack")) {
                    item.local_path = "/mnt/models-ssd/Strata-data/gpacks/ornith-1.5-35b-ext.gpack";
                    item.status = "READY (GPACK)";
                } else if (fs::exists("/home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf")) {
                    item.local_path = "/home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf";
                    item.pack_dir = "/mnt/models-ssd/Strata-data/packs/ornith-1.5-35b";
                    item.status = "READY";
                }
            }
            items.push_back(std::move(item));
        }

        for (const auto& m : installed) {
            bool already = false;
            for (const auto& existing : items) {
                if (existing.name == m.name) {
                    already = true;
                    break;
                }
            }
            if (!already) {
                ModelCatalogItem item;
                item.name = m.name;
                item.architecture = m.architecture;
                item.quantization = m.quantization;
                item.routed_experts = static_cast<int>(m.n_routed_experts);
                item.active_experts = static_cast<int>(m.k_active_experts);
                item.size_bytes = m.total_size_bytes();
                item.status = "INSTALLED";
                const auto* prim = m.find_file_by_role("primary");
                if (prim && !prim->local_path.empty()) {
                    item.local_path = prim->local_path;
                    if (fs::exists(item.local_path))
                        item.status = "READY";
                }
                items.push_back(std::move(item));
            }
        }
        return items;
    };

    std::vector<ModelCatalogItem> catalog = refresh_models();

    // Runtime configuration state
    std::string active_model_name = opts.initial_model;
    if (active_model_name.empty()) {
        // Default to ornith:35b if ready on disk, otherwise qwen3.8-flash-next
        active_model_name = "qwen3.8-flash-next";
        for (const auto& item : catalog) {
            if (item.name == "ornith:35b" && item.status.rfind("READY", 0) == 0) {
                active_model_name = item.name;
                break;
            }
        }
    }

    std::string decode_strategy = opts.decode_strategy.empty() ? "adaptive" : opts.decode_strategy;
    int spec_max_draft = opts.spec_max_draft > 0 ? opts.spec_max_draft : 4;
    int max_tokens = opts.max_tokens > 0 ? opts.max_tokens : 256;
    int64_t prefill_batch = opts.prefill_batch;
    float temperature = opts.temperature;
    std::string gexec_path = opts.gexec_path;
    bool show_reasoning = true;
    bool force_mock = opts.force_mock;

    // Engine & active model state
    std::mutex state_mutex;
    std::shared_ptr<server::IInferenceEngine> engine;
    std::shared_ptr<server::NativeInferenceEngine> native_engine_ptr;
    model::ModelDescriptor active_desc;
    memory::ExecutionPlan active_plan;
    runtime::StartupProfile active_startup;
    runtime::RuntimeTelemetry latest_telemetry;
    bool is_mock_engine = false;
    std::string status_banner = "Ready";

    std::vector<ChatEntry> messages;
    int chat_scroll_offset = 0;
    int selected_model_idx = 0;
    int selected_config_idx = 0;
    TuiTab current_tab = TuiTab::Chat;

    // Multiline prompt buffer
    std::string prompt_buf;
    size_t cursor_pos = 0;

    // Async generation state
    std::atomic<bool> is_generating{false};
    std::thread gen_thread;

    auto join_gen_thread = [&]() {
        if (gen_thread.joinable()) {
            gen_thread.join();
        }
    };

    auto load_model_engine = [&](const std::string& target_model) {
        if (is_generating.load()) {
            if (engine)
                engine->stop();
            join_gen_thread();
        }

        std::lock_guard<std::mutex> lock(state_mutex);
        active_model_name = target_model;
        status_banner = "Loading model '" + target_model + "'...";

        auto m_opt = models::ModelRegistry::instance().resolve(target_model, store);
        if (!m_opt.has_value()) {
            m_opt = models::ModelRegistry::instance().find_builtin(target_model);
        }

        if (m_opt.has_value()) {
            active_desc = m_opt->to_descriptor();
        } else if (artifact::GPackReader::is_gpack_file(target_model)) {
            artifact::GPackReader r;
            std::string gerr;
            if (r.open(target_model, gerr, false)) {
                active_desc = r.descriptor();
            }
        } else {
            active_desc.name = target_model;
            active_desc.arch_name = "qwen4exp";
            active_desc.archetype = model::ModelArchetype::Qwen4Exp;
            active_desc.attn.n_layers = 48;
            active_desc.attn.n_embd = 2560;
            active_desc.attn.n_heads = 24;
            active_desc.attn.n_kv_heads = 2;
            active_desc.attn.head_dim = 256;
            active_desc.attn.context_length = 262144;
            active_desc.attn.vocab_size = 151936;
            active_desc.attn.full_attn_interval = 4;
            active_desc.moe.n_routed_experts = 512;
            active_desc.moe.k_active_experts = 10;
            active_desc.moe.expert_dim_ff = 640;
            active_desc.moe.n_shared_experts = 1;
            active_desc.moe.shared_dim_ff = 2560;
            active_desc.moe.expert_blob_bytes = 2421813;
        }

        memory::PlannerOptions popts;
        popts.context_length = std::min<int64_t>(active_desc.attn.context_length, 32768);
        active_plan = memory::MemoryPlanner::plan(active_desc, hw, popts);
        active_plan.prefill_batch = prefill_batch;
        active_plan.decode_strategy = decode_strategy;
        active_plan.spec_max_draft = spec_max_draft;

        std::string native_path;
        std::string pack_dir;
        std::string profile_bin = "/home/ubuntu/Guild/data/expert-profile.bin";
        if (!fs::exists(profile_bin))
            profile_bin = "/home/ubuntu/Strata/data/expert-profile.bin";
        std::string mtp_dir;

        if (artifact::GPackReader::is_gpack_file(target_model)) {
            native_path = target_model;
        } else if (target_model == "ornith:35b" || target_model == "ornith") {
            if (fs::exists("/mnt/models-ssd/Strata-data/gpacks/ornith-1.5-35b-ext.gpack")) {
                native_path = "/mnt/models-ssd/Strata-data/gpacks/ornith-1.5-35b-ext.gpack";
            } else if (fs::exists("/home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf")) {
                native_path = "/home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf";
                pack_dir = "/mnt/models-ssd/Strata-data/packs/ornith-1.5-35b";
            }
            profile_bin.clear();
            if (gexec_path.empty() && fs::exists("/home/ubuntu/Guild/build-cuda12/phase66/gexec/ornith-65k-s0.gexec")) {
                gexec_path = "/home/ubuntu/Guild/build-cuda12/phase66/gexec/ornith-65k-s0.gexec";
            }
        } else if (target_model == "qwen3.8-flash-next" || target_model == "qwen") {
            if (fs::exists("/mnt/models-ssd/Strata-data/gpacks/qwen3.8-flash-next-ext.gpack")) {
                native_path = "/mnt/models-ssd/Strata-data/gpacks/qwen3.8-flash-next-ext.gpack";
                mtp_dir = "/mnt/models-ssd/Strata-data/mtp/rt";
            } else if (fs::exists("/mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/"
                                  "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf")) {
                native_path = "/mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/"
                              "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf";
                pack_dir = "/mnt/models-ssd/Strata-data/packs/unsloth-ud-iq4_xs";
                mtp_dir = "/mnt/models-ssd/Strata-data/mtp/rt";
            }
            if (gexec_path.empty() && fs::exists("/home/ubuntu/Guild/build-cuda12/phase66/gexec/qwen-65k-s0.gexec")) {
                gexec_path = "/home/ubuntu/Guild/build-cuda12/phase66/gexec/qwen-65k-s0.gexec";
            }
        } else if (m_opt.has_value()) {
            const auto* prim = m_opt->find_file_by_role("primary");
            if (prim && !prim->local_path.empty() && fs::exists(prim->local_path)) {
                native_path = prim->local_path;
            }
            if (m_opt->metadata.count("pack_dir"))
                pack_dir = m_opt->metadata.at("pack_dir");
        }

        const bool is_gpack = artifact::GPackReader::is_gpack_file(native_path);
        const bool can_load_native = !force_mock && hw.has_cuda &&
                                     (is_gpack || (!native_path.empty() && !pack_dir.empty() && fs::exists(pack_dir)));

        native_engine_ptr.reset();
        engine.reset();

        if (can_load_native) {
            server::NativeInferenceEngineOptions n_opts;
            n_opts.model_name = active_desc.name;
            n_opts.desc = active_desc;
            n_opts.plan = active_plan;
            n_opts.paths.primary_model_path = native_path;
            n_opts.paths.pack_dir = pack_dir;
            n_opts.paths.expert_profile_path = profile_bin;
            n_opts.paths.mtp_dir = mtp_dir;
            if (!pack_dir.empty())
                n_opts.paths.tokenizer_dir = pack_dir + "/tokenizer";
            n_opts.paths.gexec_path = gexec_path;

            auto candidate = std::make_shared<server::NativeInferenceEngine>(std::move(n_opts));
            std::string init_err;
            if (candidate->init(init_err)) {
                native_engine_ptr = candidate;
                engine = candidate;
                is_mock_engine = false;
                if (candidate->model()) {
                    active_startup = candidate->model()->startup_profile();
                }
                hw = detect_hardware();
                status_banner = "Loaded native model: " + active_desc.name +
                                (active_startup.gexec_active ? " [GEXEC ACTIVE]" : " [PORTABLE]");
                return;
            } else {
                status_banner = "Native load fallback (" + init_err + ") -> Mock preview ready";
            }
        }

        auto mock = std::make_shared<server::MockInferenceEngine>(
            active_desc.name, active_plan.context_length,
            "<think>\nAnalyzing MoE routing across " + std::to_string(active_desc.moe.n_routed_experts) +
                " routed experts (" + std::to_string(active_desc.moe.k_active_experts) +
                " active per token) with strategy=" + decode_strategy +
                ".\n</think>\n"
                "Guild native TUI session active on " +
                hw.summary_cpu() + " (" + hw.summary_gpu() + "). How can I help you today?");
        if (opts.mock_delay_ms > 0) {
            mock->set_delay_ms(opts.mock_delay_ms);
        }
        engine = mock;
        is_mock_engine = true;
        active_startup = runtime::StartupProfile{};
        active_startup.container_format = is_gpack ? "gpack.v1" : "mock-preview";
        if (!gexec_path.empty()) {
            gexec::GExecReader greader;
            std::string gerr;
            if (greader.open(gexec_path, gerr)) {
                active_startup.gexec_active = true;
                active_startup.gexec_path = gexec_path;
                active_startup.gexec_cache_key = greader.identity().compute_cache_key();
                if (greader.has_section(gexec::GExecSectionKind::AutotuneTable) &&
                    !greader.autotune_table().records.empty()) {
                    active_startup.gexec_cpu_expert_impl = greader.autotune_table().records[0].selected_impl_id;
                }
            }
        }
        if (force_mock) {
            status_banner = "Ready (" + active_desc.name + " · Mock Engine)";
        } else if (!hw.has_cuda) {
            status_banner = "Ready (" + active_desc.name + " · No CUDA GPU detected, running interactive preview)";
        } else if (native_path.empty()) {
            status_banner = "Ready (" + active_desc.name + " · Run /pull or F2 Models to download weights)";
        }
    };

    load_model_engine(active_model_name);

    messages.push_back(
        {"system",
         "Welcome to Guild Native TUI (" + std::string(GUILD_VERSION) +
             "). Use F1-F5 to switch views, Enter to send, Alt+Enter/Ctrl+J for newline, Esc to cancel generation, "
             "Ctrl+R to toggle reasoning, Ctrl+Q or /quit to exit."});

    auto start_generation = [&](const std::string& user_prompt) {
        if (is_generating.load())
            return;

        {
            std::lock_guard<std::mutex> lock(state_mutex);
            messages.push_back({"user", user_prompt});
            messages.push_back({"assistant", ""});
            chat_scroll_offset = 0;
            status_banner = "Generating (" + decode_strategy + ", max_draft=" + std::to_string(spec_max_draft) +
                            ")... Press Esc to cancel";
        }

        is_generating.store(true);
        join_gen_thread();

        gen_thread = std::thread([&, user_prompt]() {
            server::InferenceRequest req;
            std::shared_ptr<server::IInferenceEngine> cur_engine;
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                cur_engine = engine;
                req.model = active_desc.name;
                req.max_tokens = max_tokens;
                req.sampling.temperature = temperature;
            }

            if (!cur_engine) {
                is_generating.store(false);
                return;
            }

            if (const auto* tpl = cur_engine->chat_template(); tpl && tpl->is_loaded()) {
                std::vector<runtime::ChatMessage> chat_msgs;
                {
                    std::lock_guard<std::mutex> lock(state_mutex);
                    for (const auto& m : messages) {
                        if ((m.role == "user" || m.role == "assistant") && !m.raw_content.empty()) {
                            chat_msgs.push_back({m.role, m.raw_content});
                        }
                    }
                }
                std::string tpl_err;
                if (!tpl->render(chat_msgs, true, req.prompt, tpl_err)) {
                    req.prompt = "<|im_start|>user\n" + user_prompt + "<|im_end|>\n<|im_start|>assistant\n";
                }
            } else {
                req.prompt = "<|im_start|>user\n" + user_prompt + "<|im_end|>\n<|im_start|>assistant\n";
            }

            std::string enc_err;
            cur_engine->encode(req.prompt, req.prompt_tokens, enc_err);

            server::GenerationResult gen_res;
            auto on_tok = [&](const server::TokenOutput& tok) -> bool {
                if (g_tui_sigint.exchange(false)) {
                    cur_engine->stop();
                    return false;
                }
                if (tok.is_special)
                    return true;
                std::lock_guard<std::mutex> lock(state_mutex);
                if (!messages.empty() && messages.back().role == "assistant") {
                    messages.back().raw_content += tok.text;
                }
                return true;
            };

            // If NativeInferenceEngine is active, update its plan strategy before generating
            if (native_engine_ptr && native_engine_ptr->session()) {
                runtime::GenerationRequest g_req;
                g_req.prompt = req.prompt;
                g_req.prompt_tokens = req.prompt_tokens;
                g_req.max_new_tokens = max_tokens;
                g_req.temperature = temperature;
                g_req.decode_strategy = decode_strategy;
                g_req.spec_max_draft = spec_max_draft;

                runtime::RuntimeTelemetry telem;
                std::string g_err;
                auto n_tok_cb = [&](const runtime::TokenEvent& ev) -> bool {
                    if (g_tui_sigint.exchange(false)) {
                        native_engine_ptr->stop();
                        return false;
                    }
                    if (ev.is_special)
                        return true;
                    std::lock_guard<std::mutex> lock(state_mutex);
                    if (!messages.empty() && messages.back().role == "assistant") {
                        messages.back().raw_content += ev.text;
                    }
                    return true;
                };
                auto n_prefill_cb = [&](const runtime::PrefillProgressEvent& pe) {
                    std::lock_guard<std::mutex> lock(state_mutex);
                    latest_telemetry.prompt_tok_s = pe.tok_s;
                    latest_telemetry.prompt_ms = pe.ms;
                };
                native_engine_ptr->session()->generate(g_req, n_tok_cb, telem, g_err, n_prefill_cb);
                std::lock_guard<std::mutex> lock(state_mutex);
                latest_telemetry = telem;
                if (!messages.empty() && messages.back().role == "assistant") {
                    messages.back().prompt_tok_s = telem.prompt_tok_s;
                    messages.back().decode_tok_s = telem.decode_tok_s;
                    messages.back().prompt_tokens = telem.prompt_tokens;
                    messages.back().completion_tokens = telem.completion_tokens;
                    messages.back().finish_reason = telem.finish_reason;
                    messages.back().strategy_summary = telem.decode_strategy_final + " (" +
                                                       std::to_string(telem.drafts_accepted) + "/" +
                                                       std::to_string(telem.drafts_offered) + " drafts)";
                }
                status_banner = telem.finish_reason == "cancel"
                                    ? "Generation cancelled by user"
                                    : ("Completed: " + std::to_string(telem.completion_tokens) + " tokens @ " +
                                       std::to_string(static_cast<int>(telem.decode_tok_s)) + " tok/s");
            } else {
                cur_engine->generate_stream(req, on_tok, gen_res);
                std::lock_guard<std::mutex> lock(state_mutex);
                latest_telemetry = cur_engine->last_telemetry();
                latest_telemetry.decode_strategy_mode = decode_strategy;
                if (!messages.empty() && messages.back().role == "assistant") {
                    messages.back().prompt_tok_s = gen_res.prompt_tok_s;
                    messages.back().decode_tok_s = gen_res.decode_tok_s;
                    messages.back().prompt_tokens = gen_res.prompt_tokens;
                    messages.back().completion_tokens = gen_res.completion_tokens;
                    messages.back().finish_reason = gen_res.finish_reason;
                    messages.back().strategy_summary = decode_strategy + " (" +
                                                       std::to_string(gen_res.drafts_accepted) + "/" +
                                                       std::to_string(gen_res.drafts_offered) + " drafts)";
                }
                status_banner = gen_res.finish_reason == "cancel"
                                    ? "Generation cancelled by user"
                                    : ("Completed: " + std::to_string(gen_res.completion_tokens) + " tokens @ " +
                                       std::to_string(static_cast<int>(gen_res.decode_tok_s)) + " tok/s");
            }

            is_generating.store(false);
        });
    };

    auto cycle_strategy = [&](int dir) {
        static const std::vector<std::string> kStrategies = {"adaptive", "auto",      "mtp",
                                                             "lookup",   "retrieval", "normal"};
        auto it = std::find(kStrategies.begin(), kStrategies.end(), decode_strategy);
        int idx = (it == kStrategies.end()) ? 0 : static_cast<int>(std::distance(kStrategies.begin(), it));
        idx = (idx + dir + static_cast<int>(kStrategies.size())) % static_cast<int>(kStrategies.size());
        decode_strategy = kStrategies[static_cast<size_t>(idx)];
        active_plan.decode_strategy = decode_strategy;
        status_banner = "Decode strategy set to '" + decode_strategy + "'";
    };

    auto execute_slash_command = [&](const std::string& cmd_line) -> bool {
        std::istringstream iss(cmd_line);
        std::string cmd;
        iss >> cmd;
        if (cmd == "/quit" || cmd == "/exit") {
            return false;
        }
        if (cmd == "/clear" || cmd == "/reset") {
            std::lock_guard<std::mutex> lock(state_mutex);
            messages.clear();
            if (native_engine_ptr && native_engine_ptr->session()) {
                native_engine_ptr->session()->reset();
                native_engine_ptr->session()->clear_retrieval_corpus();
            }
            status_banner = "Conversation and session KV/retrieval state cleared";
            return true;
        }
        if (cmd == "/cancel" || cmd == "/stop") {
            if (is_generating.load() && engine) {
                engine->stop();
                status_banner = "Cancelling generation...";
            }
            return true;
        }
        if (cmd == "/models") {
            current_tab = TuiTab::Models;
            return true;
        }
        if (cmd == "/config") {
            current_tab = TuiTab::Config;
            return true;
        }
        if (cmd == "/telemetry" || cmd == "/gexec") {
            std::string arg;
            if (iss >> arg) {
                gexec_path = arg;
                load_model_engine(active_model_name);
            } else {
                current_tab = TuiTab::Telemetry;
            }
            return true;
        }
        if (cmd == "/help") {
            current_tab = TuiTab::Help;
            return true;
        }
        if (cmd == "/chat") {
            current_tab = TuiTab::Chat;
            return true;
        }
        if (cmd == "/reasoning") {
            show_reasoning = !show_reasoning;
            status_banner = std::string("Reasoning display: ") + (show_reasoning ? "expanded" : "collapsed");
            return true;
        }
        if (cmd == "/model") {
            std::string target;
            if (iss >> target) {
                load_model_engine(target);
            } else {
                current_tab = TuiTab::Models;
            }
            return true;
        }
        if (cmd == "/strategy") {
            std::string strat;
            if (iss >> strat) {
                decode_strategy = strat;
                active_plan.decode_strategy = strat;
                status_banner = "Decode strategy set to '" + strat + "'";
            } else {
                cycle_strategy(1);
            }
            return true;
        }
        if (cmd == "/draft") {
            int d = 4;
            if (iss >> d) {
                spec_max_draft = std::max(0, std::min(7, d));
                active_plan.spec_max_draft = spec_max_draft;
                status_banner = "Max speculative draft set to " + std::to_string(spec_max_draft);
            }
            return true;
        }
        if (cmd == "/temp") {
            float t = 0.0f;
            if (iss >> t) {
                temperature = std::max(0.0f, std::min(2.0f, t));
                status_banner = "Sampling temperature set to " + std::to_string(temperature);
            }
            return true;
        }
        if (cmd == "/max") {
            int m = 256;
            if (iss >> m && m > 0) {
                max_tokens = std::min(32768, m);
                status_banner = "Max new tokens set to " + std::to_string(max_tokens);
            }
            return true;
        }
        if (cmd == "/resize") {
            int c = 120, r = 36;
            if (iss >> c >> r) {
                status_banner = "Resized viewport to " + std::to_string(c) + "x" + std::to_string(r);
            }
            return true;
        }
        status_banner = "Unknown command '" + cmd + "' (try /models, /strategy, /draft, /clear, /help, /quit)";
        return true;
    };

    int cols = 120, rows = 36;
    query_terminal_size(cols, rows, opts.override_cols, opts.override_rows);

    std::string last_rendered_frame;
    auto render_screen = [&]() -> std::string {
        std::lock_guard<std::mutex> lock(state_mutex);
        std::ostringstream out;
        if (interactive_tty) {
            out << "\033[H";
        }

        const bool use_color = interactive_tty;
        const char* c_reset = use_color ? "\033[0m" : "";
        const char* c_bold = use_color ? "\033[1m" : "";
        const char* c_dim = use_color ? "\033[2m" : "";
        const char* c_cyan = use_color ? "\033[36m" : "";
        const char* c_green = use_color ? "\033[32m" : "";
        const char* c_yellow = use_color ? "\033[33m" : "";
        const char* c_magenta = use_color ? "\033[35m" : "";
        const char* c_bg_bar = use_color ? "\033[48;5;236m\033[97m" : "";

        const size_t total_w = static_cast<size_t>(std::max(60, cols));
        const size_t total_h = static_cast<size_t>(std::max(18, rows));

        // 1. Top Header Bar (Row 1)
        std::string hdr_left = " GUILD TUI " + std::string(GUILD_VERSION) + " | Model: " + active_desc.name +
                               (is_mock_engine ? " [Preview]" : " [Native]") + " | Arch: " + active_desc.arch_name +
                               " (" + std::to_string(active_desc.moe.n_routed_experts) + "x" +
                               std::to_string(active_desc.moe.k_active_experts) + " MoE)";
        out << c_bg_bar << c_bold << utf8_truncate_or_pad(hdr_left, total_w) << c_reset << "\n";

        // 2. Navigation Tab Bar (Row 2)
        auto tab_label = [&](TuiTab t, const char* name) {
            if (current_tab == t)
                return std::string(c_bold) + c_green + "[" + name + "]" + c_reset;
            return std::string(c_dim) + " " + name + " " + c_reset;
        };
        std::cout.flush();
        std::string nav_plain =
            " [F1] Chat  [F2] Models  [F3] Config  [F4] Telemetry & GEXEC  [F5] Help  |  Ctrl+Q Exit";
        out << tab_label(TuiTab::Chat, "F1 Chat") << " " << tab_label(TuiTab::Models, "F2 Models") << " "
            << tab_label(TuiTab::Config, "F3 Config") << " " << tab_label(TuiTab::Telemetry, "F4 Telemetry & GEXEC")
            << " " << tab_label(TuiTab::Help, "F5 Help") << c_dim << "  |  Esc: Cancel  Ctrl+R: Reasoning  Ctrl+Q: Quit"
            << "\033[K" << c_reset << "\n";
        (void) nav_plain;

        out << std::string(total_w, '-') << "\n";

        // Body height = total_h - 7 (3 header rows + 4 footer/prompt rows)
        const size_t body_h = total_h > 8 ? (total_h - 7) : 10;
        const bool split_sidebar = (total_w >= 96);
        const size_t side_w = split_sidebar ? 38 : 0;
        const size_t main_w = split_sidebar ? (total_w - side_w - 3) : total_w;

        // Build Main Pane lines
        std::vector<std::string> main_lines;
        if (current_tab == TuiTab::Chat) {
            for (const auto& m : messages) {
                if (m.role == "system") {
                    for (const auto& l : wrap_text_lines("[System] " + m.raw_content, main_w)) {
                        main_lines.push_back(std::string(c_dim) + utf8_truncate_or_pad(l, main_w) + c_reset);
                    }
                } else if (m.role == "user") {
                    for (const auto& l : wrap_text_lines("You > " + m.raw_content, main_w)) {
                        main_lines.push_back(std::string(c_bold) + c_cyan + utf8_truncate_or_pad(l, main_w) + c_reset);
                    }
                } else {
                    std::string reasoning, answer;
                    bool in_think = false;
                    split_reasoning_and_answer(m.raw_content, reasoning, answer, in_think);
                    if (!reasoning.empty()) {
                        if (show_reasoning) {
                            std::string r_hdr =
                                in_think ? "  [Reasoning... (Ctrl+R to fold)]" : "  [Reasoning (Ctrl+R to fold)]";
                            main_lines.push_back(std::string(c_magenta) + utf8_truncate_or_pad(r_hdr, main_w) +
                                                 c_reset);
                            for (const auto& rl : wrap_text_lines("  | " + reasoning, main_w)) {
                                main_lines.push_back(std::string(c_dim) + utf8_truncate_or_pad(rl, main_w) + c_reset);
                            }
                        } else {
                            std::string folded = "  [Reasoning collapsed (" +
                                                 std::to_string(utf8_display_len(reasoning)) +
                                                 " chars) — press Ctrl+R to expand]";
                            main_lines.push_back(std::string(c_dim) + utf8_truncate_or_pad(folded, main_w) + c_reset);
                        }
                    }
                    if (!answer.empty() || reasoning.empty()) {
                        for (const auto& al : wrap_text_lines("Guild > " + answer, main_w)) {
                            main_lines.push_back(std::string(c_green) + utf8_truncate_or_pad(al, main_w) + c_reset);
                        }
                    }
                    if (m.completion_tokens > 0) {
                        std::ostringstream meta;
                        meta << "   └─ " << std::fixed << std::setprecision(1) << m.decode_tok_s << " tok/s decode, "
                             << m.prompt_tok_s << " tok/s prefill (" << m.completion_tokens << " tok, "
                             << m.strategy_summary << ", " << m.finish_reason << ")";
                        main_lines.push_back(std::string(c_dim) + utf8_truncate_or_pad(meta.str(), main_w) + c_reset);
                    }
                }
                main_lines.push_back(utf8_truncate_or_pad("", main_w));
            }
        } else if (current_tab == TuiTab::Models) {
            main_lines.push_back(std::string(c_bold) +
                                 utf8_truncate_or_pad("Interactive Model Selector & Store Manager", main_w) + c_reset);
            main_lines.push_back(
                std::string(c_dim) +
                utf8_truncate_or_pad("Use Up/Down to select, Enter to load model, /pull <name> to download:", main_w) +
                c_reset);
            main_lines.push_back(utf8_truncate_or_pad("", main_w));
            for (size_t i = 0; i < catalog.size(); ++i) {
                const auto& item = catalog[i];
                const bool sel = (static_cast<int>(i) == selected_model_idx);
                const bool act = (item.name == active_model_name);
                std::ostringstream row;
                row << (sel ? " > " : "   ") << std::left << std::setw(22) << item.name << " " << std::setw(10)
                    << item.architecture << " " << std::setw(10) << item.quantization << " " << std::setw(14)
                    << ("[" + item.status + "]") << (act ? " *ACTIVE*" : "");
                main_lines.push_back((sel ? std::string(c_bold) + c_cyan : std::string("")) +
                                     utf8_truncate_or_pad(row.str(), main_w) + c_reset);
                if (!item.local_path.empty()) {
                    main_lines.push_back(std::string(c_dim) +
                                         utf8_truncate_or_pad("      path: " + item.local_path, main_w) + c_reset);
                }
            }
        } else if (current_tab == TuiTab::Config) {
            main_lines.push_back(std::string(c_bold) +
                                 utf8_truncate_or_pad("Runtime & Speculative Decoding Configuration", main_w) +
                                 c_reset);
            main_lines.push_back(std::string(c_dim) +
                                 utf8_truncate_or_pad("Use Up/Down to select, Left/Right or Enter to adjust:", main_w) +
                                 c_reset);
            main_lines.push_back(utf8_truncate_or_pad("", main_w));

            std::vector<std::pair<std::string, std::string>> cfg_rows = {
                {"Decode Strategy", decode_strategy + " (adaptive | auto | mtp | lookup | retrieval | normal)"},
                {"Spec Max Draft (k)",
                 std::to_string(spec_max_draft) + " tokens (window T = " + std::to_string(1 + spec_max_draft) + ")"},
                {"Sampling Temperature", std::to_string(temperature) + (temperature == 0.0f ? " (Greedy)" : "")},
                {"Max New Tokens", std::to_string(max_tokens)},
                {"Prefill Batch Size", prefill_batch == 0 ? "0 (Auto)" : std::to_string(prefill_batch)},
                {"Reasoning Display", show_reasoning ? "Expanded (<think> visible)" : "Collapsed (Ctrl+R)"},
                {"GEXEC Artifact", gexec_path.empty() ? "(auto / portable fallback)" : gexec_path},
            };
            for (size_t i = 0; i < cfg_rows.size(); ++i) {
                const bool sel = (static_cast<int>(i) == selected_config_idx);
                std::string line = (sel ? " > " : "   ") + cfg_rows[i].first + ": " + cfg_rows[i].second;
                main_lines.push_back((sel ? std::string(c_bold) + c_yellow : std::string("")) +
                                     utf8_truncate_or_pad(line, main_w) + c_reset);
            }
        } else if (current_tab == TuiTab::Telemetry) {
            main_lines.push_back(std::string(c_bold) +
                                 utf8_truncate_or_pad("Deep-Dive Runtime, Speculative & GEXEC Telemetry", main_w) +
                                 c_reset);
            main_lines.push_back(utf8_truncate_or_pad("", main_w));

            std::ostringstream t1, t2, t3, t4, t5, t6;
            t1 << "Model / Container : " << active_desc.name << " (" << active_startup.container_format << ")";
            t2 << "Prefill Throughput: " << std::fixed << std::setprecision(2) << latest_telemetry.prompt_tok_s
               << " tok/s (" << latest_telemetry.prompt_ms << " ms, " << latest_telemetry.prompt_tokens << " tokens)";
            t3 << "Decode Throughput : " << std::fixed << std::setprecision(2) << latest_telemetry.decode_tok_s
               << " tok/s (" << latest_telemetry.decode_ms << " ms, " << latest_telemetry.completion_tokens
               << " tokens)";
            t4 << "Spec Arbitration  : mode=" << latest_telemetry.decode_strategy_mode
               << " initial=" << latest_telemetry.decode_strategy_initial
               << " final=" << latest_telemetry.decode_strategy_final;
            t5 << "Spec Windows      : Normal=" << latest_telemetry.spec_normal_windows
               << " MTP=" << latest_telemetry.spec_mtp_windows << " Lookup=" << latest_telemetry.spec_lookup_windows
               << " Retr=" << latest_telemetry.spec_retrieval_windows
               << " Switches=" << latest_telemetry.spec_strategy_switches;
            t6 << "GEXEC Lockfile    : " << (active_startup.gexec_active ? "ACTIVE" : "INACTIVE (Portable)")
               << (active_startup.gexec_cpu_expert_impl.empty() ? ""
                                                                : (" [" + active_startup.gexec_cpu_expert_impl + "]"));

            for (const auto& s : {t1.str(), t2.str(), t3.str(), t4.str(), t5.str(), t6.str()}) {
                main_lines.push_back(utf8_truncate_or_pad(s, main_w));
            }
            if (!gexec_path.empty() && fs::exists(gexec_path)) {
                gexec::GExecReader greader;
                std::string gerr;
                if (greader.open(gexec_path, gerr)) {
                    main_lines.push_back(utf8_truncate_or_pad(
                        "GEXEC Cache Key   : " + greader.identity().compute_cache_key().substr(0, 32) + "...", main_w));
                    if (greader.has_section(gexec::GExecSectionKind::HeterogeneousPlan)) {
                        const auto& hp = greader.heterogeneous_plan();
                        if (hp.spec_calibrated && hp.verify_window_ms.size() >= 4) {
                            std::ostringstream vc;
                            vc << "Calibrated V(1..4): " << std::fixed << std::setprecision(2) << hp.verify_window_ms[0]
                               << " / " << hp.verify_window_ms[1] << " / " << hp.verify_window_ms[2] << " / "
                               << hp.verify_window_ms[3] << " ms (CV=" << std::setprecision(3) << hp.verify_noise_cv
                               << ")";
                            main_lines.push_back(utf8_truncate_or_pad(vc.str(), main_w));
                        }
                    }
                }
            }
        } else {
            main_lines.push_back(std::string(c_bold) +
                                 utf8_truncate_or_pad("Guild Native TUI Keyboard Shortcuts & Slash Commands", main_w) +
                                 c_reset);
            main_lines.push_back(utf8_truncate_or_pad("", main_w));
            const char* help_items[] = {
                "  F1..F5               Switch between Chat, Models, Config, Telemetry, and Help",
                "  Enter                Submit prompt (or load selected model in F2 Models)",
                "  Alt+Enter / Ctrl+J   Insert newline in multiline prompt (or end line with '\\')",
                "  Esc / Ctrl+C         Cancel in-flight generation immediately",
                "  PgUp / PgDn          Scroll conversation history up / down",
                "  Ctrl+R               Expand or collapse <think>...</think> reasoning blocks",
                "  Ctrl+L / /clear      Clear conversation & reset session KV/retrieval datastore",
                "  Ctrl+Q / /quit       Restore terminal and exit Guild TUI",
                "  /model <name>        Switch active model (e.g. /model ornith:35b)",
                "  /strategy <mode>     Set decode strategy (adaptive|mtp|lookup|retrieval|normal)",
                "  /draft <0..7>        Set maximum speculative draft depth k",
                "  /temp <0.0..2.0>     Set sampling temperature (0.0 = greedy)",
                "  /gexec <path.gexec>  Bind hardware-calibrated GEXEC artifact",
            };
            for (const char* h : help_items) {
                main_lines.push_back(utf8_truncate_or_pad(h, main_w));
            }
        }

        // Build Right Sidebar lines (Hardware, RAM/VRAM, KV, Experts, Speculative, GEXEC)
        std::vector<std::string> side_lines;
        if (split_sidebar) {
            const double ram_used = std::max(0.0, hw.ram_total_gib - hw.ram_free_gib);
            const double ram_ratio = hw.ram_total_gib > 0.0 ? (ram_used / hw.ram_total_gib) : 0.0;
            const double vram_used =
                hw.gpu_vram_free_gib > 0.0
                    ? std::max(0.0, hw.gpu_vram_gib - hw.gpu_vram_free_gib)
                    : ((active_plan.dense_gpu_bytes + active_plan.kv_staging_bytes) / (1024.0 * 1024.0 * 1024.0));
            const double vram_ratio = hw.gpu_vram_gib > 0.0 ? (vram_used / hw.gpu_vram_gib) : 0.0;

            auto push_side = [&](const std::string& s, const char* col = "") {
                side_lines.push_back(std::string(col) + utf8_truncate_or_pad(s, side_w) + (col[0] ? c_reset : ""));
            };

            push_side("HARDWARE MONITOR", c_bold);
            push_side("CPU : " + hw.summary_cpu());
            push_side("Cores: " + std::to_string(hw.cpu_physical_cores) + "C/" +
                      std::to_string(hw.cpu_logical_threads) + "T " +
                      (hw.isa_avx512 ? "[AVX-512]" : (hw.isa_avx2 ? "[AVX2]" : "[Scalar]")));
            {
                std::ostringstream rs;
                rs << "RAM : " << render_bar(ram_ratio, 10) << " " << std::fixed << std::setprecision(1) << ram_used
                   << "/" << hw.ram_total_gib << "G";
                push_side(rs.str());
            }
            push_side("GPU : " + hw.summary_gpu());
            {
                std::ostringstream vs;
                vs << "VRAM: " << render_bar(vram_ratio, 10) << " " << std::fixed << std::setprecision(1) << vram_used
                   << "/" << hw.gpu_vram_gib << "G";
                push_side(vs.str());
            }
            push_side("");
            push_side("KV & EXPERT RESIDENCY", c_bold);
            push_side("KV  : " + std::string(memory::kv_precision_to_string(active_plan.kv_format)) + " · " +
                      memory::kv_mode_to_string(active_plan.kv_mode));
            {
                std::ostringstream kvs;
                kvs << "Host KV: " << std::fixed << std::setprecision(2)
                    << (active_plan.full_host_kv_bytes / (1024.0 * 1024.0 * 1024.0))
                    << "G | Stg: " << std::setprecision(0) << (active_plan.kv_staging_bytes / (1024.0 * 1024.0)) << "M";
                push_side(kvs.str());
            }
            {
                std::ostringstream exs;
                exs << "Experts: RAM " << active_plan.routed_experts_in_ram << " | GPU "
                    << active_plan.routed_experts_in_gpu << " | File " << active_plan.routed_experts_on_file;
                push_side(exs.str());
            }
            push_side("");
            push_side("THROUGHPUT & SPECULATION", c_bold);
            {
                std::ostringstream pfs;
                pfs << "Prefill: " << std::fixed << std::setprecision(1) << latest_telemetry.prompt_tok_s << " tok/s ("
                    << latest_telemetry.prompt_tokens << " tok)";
                push_side(pfs.str(), c_cyan);
            }
            {
                std::ostringstream dcs;
                dcs << "Decode : " << std::fixed << std::setprecision(2) << latest_telemetry.decode_tok_s << " tok/s ("
                    << latest_telemetry.completion_tokens << " tok)";
                push_side(dcs.str(), c_green);
            }
            push_side("Strategy: " + decode_strategy + " (k=" + std::to_string(spec_max_draft) + ")");
            {
                const int off = latest_telemetry.drafts_offered;
                const int acc = latest_telemetry.drafts_accepted;
                const double pct = off > 0 ? (100.0 * acc / off) : 0.0;
                std::ostringstream drs;
                drs << "Drafts : " << acc << "/" << off << " (" << std::fixed << std::setprecision(1) << pct << "%)";
                push_side(drs.str());
            }
            push_side("");
            push_side("GEXEC OPTIMIZATION", c_bold);
            push_side(std::string("State  : ") + (active_startup.gexec_active ? "LOCKED (.gexec)" : "PORTABLE"));
            if (!active_startup.gexec_cpu_expert_impl.empty()) {
                push_side("Kernel : " + active_startup.gexec_cpu_expert_impl);
            }
        }

        // Scroll window calculation for main pane
        size_t start_line = 0;
        if (current_tab == TuiTab::Chat && main_lines.size() > body_h) {
            const size_t max_scroll = main_lines.size() - body_h;
            const size_t clamped_scroll =
                static_cast<size_t>(std::max(0, std::min(chat_scroll_offset, static_cast<int>(max_scroll))));
            start_line = max_scroll - clamped_scroll;
        }

        for (size_t r = 0; r < body_h; ++r) {
            std::string left_cell =
                (start_line + r < main_lines.size()) ? main_lines[start_line + r] : utf8_truncate_or_pad("", main_w);
            if (split_sidebar) {
                std::string right_cell = (r < side_lines.size()) ? side_lines[r] : utf8_truncate_or_pad("", side_w);
                out << left_cell << " | " << right_cell << "\033[K\n";
            } else {
                out << left_cell << "\033[K\n";
            }
        }

        // Footer + Multiline Prompt Bar (4 rows)
        out << std::string(total_w, '-') << "\n";
        out << c_yellow << utf8_truncate_or_pad(" Status: " + status_banner, total_w) << c_reset << "\033[K\n";

        std::string display_prompt = prompt_buf;
        for (size_t i = 0; i < display_prompt.size(); ++i) {
            if (display_prompt[i] == '\n') {
                display_prompt.replace(i, 1, " ↵ ");
            }
        }
        std::string prompt_line = " Prompt > " + display_prompt + (interactive_tty ? "█" : "");
        out << c_bold << utf8_truncate_or_pad(prompt_line, total_w) << c_reset << "\033[K\n";
        out << c_dim
            << utf8_truncate_or_pad(
                   " [Enter] Send  [Alt+Enter/Ctrl+J] Newline  [/models] Models  [/strategy] Strategy  [/clear] Reset",
                   total_w)
            << c_reset << "\033[K";

        return out.str();
    };

    // Automated script execution mode for deterministic testing
    if (!opts.script_commands.empty()) {
        for (const auto& scmd : opts.script_commands) {
            if (scmd.empty())
                continue;
            if (scmd[0] == '/') {
                if (!execute_slash_command(scmd))
                    break;
            } else if (scmd == "KEY:F1") {
                current_tab = TuiTab::Chat;
            } else if (scmd == "KEY:F2") {
                current_tab = TuiTab::Models;
            } else if (scmd == "KEY:F3") {
                current_tab = TuiTab::Config;
            } else if (scmd == "KEY:F4") {
                current_tab = TuiTab::Telemetry;
            } else if (scmd == "KEY:F5") {
                current_tab = TuiTab::Help;
            } else if (scmd == "KEY:CTRL_R") {
                show_reasoning = !show_reasoning;
            } else if (scmd == "KEY:ESC") {
                if (is_generating.load() && engine)
                    engine->stop();
            } else if (scmd.rfind("WAIT:", 0) == 0) {
                int ms = std::atoi(scmd.substr(5).c_str());
                std::this_thread::sleep_for(std::chrono::milliseconds(ms));
            } else {
                start_generation(scmd);
                // Wait up to 30s unless next command is KEY:ESC or WAIT:
                if (&scmd == &opts.script_commands.back()) {
                    join_gen_thread();
                }
            }
        }
        join_gen_thread();
        last_rendered_frame = render_screen();
        if (opts.dump_screen_on_exit) {
            std::cout << last_rendered_frame << "\n";
        }
        return 0;
    }

    // Main interactive event loop
    bool running = true;
    while (running) {
        if (g_tui_sigwinch.exchange(false)) {
            query_terminal_size(cols, rows, opts.override_cols, opts.override_rows);
            std::cout << "\033[2J\033[H" << std::flush;
        }
        if (g_tui_sigint.exchange(false)) {
            if (is_generating.load() && engine) {
                engine->stop();
                status_banner = "Generation cancelled (Ctrl+C). Press Ctrl+Q to exit.";
            } else {
                running = false;
                break;
            }
        }

        hw.sync_gib();
        std::string frame = render_screen();
        if (frame != last_rendered_frame) {
            std::cout << frame << std::flush;
            last_rendered_frame = frame;
        }

        KeyEvent ev = read_key_event(35);
        if (ev.kind == KeyKind::None) {
            if (!interactive_tty) {
                // Non-interactive EOF
                std::string piped_line;
                if (!std::getline(std::cin, piped_line)) {
                    break;
                }
                if (!piped_line.empty()) {
                    if (piped_line[0] == '/') {
                        if (!execute_slash_command(piped_line))
                            break;
                    } else {
                        start_generation(piped_line);
                        join_gen_thread();
                    }
                }
            }
            continue;
        }

        switch (ev.kind) {
        case KeyKind::CtrlQ: running = false; break;
        case KeyKind::CtrlC:
        case KeyKind::Escape:
            if (is_generating.load() && engine) {
                engine->stop();
                status_banner = "Cancelling generation...";
            } else if (ev.kind == KeyKind::Escape && current_tab != TuiTab::Chat) {
                current_tab = TuiTab::Chat;
            }
            break;
        case KeyKind::CtrlL: execute_slash_command("/clear"); break;
        case KeyKind::CtrlR:
            show_reasoning = !show_reasoning;
            status_banner = std::string("Reasoning display: ") + (show_reasoning ? "expanded" : "collapsed");
            break;
        case KeyKind::F1: current_tab = TuiTab::Chat; break;
        case KeyKind::F2:
            catalog = refresh_models();
            current_tab = TuiTab::Models;
            break;
        case KeyKind::F3: current_tab = TuiTab::Config; break;
        case KeyKind::F4: current_tab = TuiTab::Telemetry; break;
        case KeyKind::F5: current_tab = TuiTab::Help; break;
        case KeyKind::Tab: current_tab = static_cast<TuiTab>((static_cast<int>(current_tab) + 1) % 5); break;
        case KeyKind::PageUp:
        case KeyKind::CtrlU: chat_scroll_offset += 5; break;
        case KeyKind::PageDown:
        case KeyKind::CtrlD: chat_scroll_offset = std::max(0, chat_scroll_offset - 5); break;
        case KeyKind::Up:
            if (current_tab == TuiTab::Models && !catalog.empty()) {
                selected_model_idx =
                    (selected_model_idx - 1 + static_cast<int>(catalog.size())) % static_cast<int>(catalog.size());
            } else if (current_tab == TuiTab::Config) {
                selected_config_idx = (selected_config_idx + 6) % 7;
            } else {
                chat_scroll_offset += 1;
            }
            break;
        case KeyKind::Down:
            if (current_tab == TuiTab::Models && !catalog.empty()) {
                selected_model_idx = (selected_model_idx + 1) % static_cast<int>(catalog.size());
            } else if (current_tab == TuiTab::Config) {
                selected_config_idx = (selected_config_idx + 1) % 7;
            } else {
                chat_scroll_offset = std::max(0, chat_scroll_offset - 1);
            }
            break;
        case KeyKind::Left:
            if (current_tab == TuiTab::Config) {
                if (selected_config_idx == 0)
                    cycle_strategy(-1);
                else if (selected_config_idx == 1)
                    spec_max_draft = std::max(0, spec_max_draft - 1);
                else if (selected_config_idx == 2)
                    temperature = std::max(0.0f, temperature - 0.1f);
                else if (selected_config_idx == 3)
                    max_tokens = std::max(32, max_tokens / 2);
                else if (selected_config_idx == 5)
                    show_reasoning = !show_reasoning;
            } else if (cursor_pos > 0) {
                --cursor_pos;
            }
            break;
        case KeyKind::Right:
            if (current_tab == TuiTab::Config) {
                if (selected_config_idx == 0)
                    cycle_strategy(1);
                else if (selected_config_idx == 1)
                    spec_max_draft = std::min(7, spec_max_draft + 1);
                else if (selected_config_idx == 2)
                    temperature = std::min(2.0f, temperature + 0.1f);
                else if (selected_config_idx == 3)
                    max_tokens = std::min(8192, max_tokens * 2);
                else if (selected_config_idx == 5)
                    show_reasoning = !show_reasoning;
            } else if (cursor_pos < prompt_buf.size()) {
                ++cursor_pos;
            }
            break;
        case KeyKind::Home: cursor_pos = 0; break;
        case KeyKind::End: cursor_pos = prompt_buf.size(); break;
        case KeyKind::Backspace:
            if (cursor_pos > 0 && !prompt_buf.empty()) {
                prompt_buf.erase(cursor_pos - 1, 1);
                --cursor_pos;
            }
            break;
        case KeyKind::DeleteKey:
            if (cursor_pos < prompt_buf.size()) {
                prompt_buf.erase(cursor_pos, 1);
            }
            break;
        case KeyKind::Newline:
            prompt_buf.insert(cursor_pos, "\n");
            ++cursor_pos;
            break;
        case KeyKind::Enter:
            if (current_tab == TuiTab::Models && prompt_buf.empty() && !catalog.empty()) {
                const auto& picked = catalog[static_cast<size_t>(selected_model_idx)];
                load_model_engine(picked.name);
                current_tab = TuiTab::Chat;
                break;
            }
            if (!prompt_buf.empty() && prompt_buf.back() == '\\') {
                prompt_buf.back() = '\n';
                cursor_pos = prompt_buf.size();
                break;
            }
            if (!prompt_buf.empty()) {
                std::string submitted = prompt_buf;
                prompt_buf.clear();
                cursor_pos = 0;
                if (submitted[0] == '/') {
                    if (!execute_slash_command(submitted)) {
                        running = false;
                    }
                } else {
                    current_tab = TuiTab::Chat;
                    start_generation(submitted);
                }
            }
            break;
        case KeyKind::Char:
            prompt_buf.insert(cursor_pos, ev.text);
            cursor_pos += ev.text.size();
            break;
        default: break;
        }
    }

    if (is_generating.load() && engine) {
        engine->stop();
    }
    join_gen_thread();
    term_guard.restore();
    return 0;
}

int cmd_tui(int argc, char** argv) {
    TuiOptions opts;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: guild tui [model] [options]\n"
                      << "  --model <name>               Initial model to load (e.g. ornith:35b, qwen3.8-flash-next)\n"
                      << "  --data-dir <path>            Custom data directory for Guild model store\n"
                      << "  --gexec <path.gexec>         Hardware-calibrated GEXEC artifact to bind\n"
                      << "  --decode-strategy <strategy> Decode strategy: auto|normal|mtp|lookup|retrieval|adaptive\n"
                      << "  --spec-max-draft <1..7>      Maximum speculative draft depth (default: 4)\n"
                      << "  --max-tokens <N>             Default max generation tokens (default: 256)\n"
                      << "  --temperature <float>        Default sampling temperature (default: 0.0 greedy)\n"
                      << "  --prefill-batch <0..512>     Prompt prefill batch size (0 = auto)\n"
                      << "  --mock                       Run with mock inference engine for testing\n"
                      << "  --mock-delay-ms <ms>         Per-token delay in mock mode (for cancellation tests)\n"
                      << "  --cols <N> --rows <N>        Override terminal dimensions\n"
                      << "  --script <cmd>               Execute scripted command/prompt (repeatable)\n"
                      << "  --dump-screen                Print rendered TUI frame on exit\n";
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            opts.initial_model = argv[++i];
        } else if (arg == "--data-dir" && i + 1 < argc) {
            opts.data_dir = argv[++i];
        } else if (arg == "--gexec" && i + 1 < argc) {
            opts.gexec_path = argv[++i];
        } else if (arg == "--decode-strategy" && i + 1 < argc) {
            opts.decode_strategy = argv[++i];
        } else if (arg == "--spec-max-draft" && i + 1 < argc) {
            opts.spec_max_draft = std::atoi(argv[++i]);
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            opts.max_tokens = std::atoi(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            opts.temperature = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--prefill-batch" && i + 1 < argc) {
            opts.prefill_batch = std::atoll(argv[++i]);
        } else if (arg == "--mock") {
            opts.force_mock = true;
        } else if (arg == "--mock-delay-ms" && i + 1 < argc) {
            opts.mock_delay_ms = std::atoi(argv[++i]);
        } else if (arg == "--cols" && i + 1 < argc) {
            opts.override_cols = std::atoi(argv[++i]);
        } else if (arg == "--rows" && i + 1 < argc) {
            opts.override_rows = std::atoi(argv[++i]);
        } else if (arg == "--script" && i + 1 < argc) {
            opts.script_commands.push_back(argv[++i]);
        } else if (arg == "--dump-screen") {
            opts.dump_screen_on_exit = true;
        } else if (!arg.empty() && arg[0] != '-' && opts.initial_model.empty()) {
            opts.initial_model = arg;
        }
    }
    return run_tui(opts);
}

} // namespace guild::cli
