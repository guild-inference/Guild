// The HTTP front end: a hand-rolled HTTP/1.1 server (one poll-driven listener thread, one worker thread per
// connection) that speaks the OpenAI and Anthropic wire formats on top of `IInferenceEngine`.
//
// It deliberately keeps no model logic: routing parses the request into `InferenceRequest`, calls the engine, and
// serializes the result or the SSE stream back.  The engine is chosen by the CLI (`--native` vs a subprocess
// engine), and both are used through the same interface, so the streaming path is identical for either.
//
// Concurrency contract: the listener thread and the per-connection workers only share the engine, and the engine's
// generation is serialized internally (the runtime's model-wide generation mutex), so multiple in-flight HTTP
// requests queue rather than run in parallel.  A client that disconnects mid-stream stops the callback, which
// cancels the generation; that is what keeps a killed `curl` from holding the model busy.
//
// Security default: the listener binds to `127.0.0.1` unless the caller passes another `--host`, and there is no
// authentication here at all.  The Python server in `serve/` is the one that refuses a non-loopback bind without an
// API key and checks that key on every request; exposing this binary directly is a deliberate act.

#include "guild/server/server.hpp"
#include "guild/server/openai.hpp"
#include "guild/runtime/chat_template.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <exception>
#include <iostream>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace guild::server {

namespace {

#if defined(_WIN32)
void ensure_winsock() {
    static bool initialized = false;
    if (!initialized) {
        WSADATA wsa{};
        WSAStartup(MAKEWORD(2, 2), &wsa);
        initialized = true;
    }
}
inline void close_socket_fd(int fd) {
    if (fd >= 0)
        closesocket(static_cast<SOCKET>(fd));
}
inline void shutdown_socket_rw(int fd) {
    if (fd >= 0)
        shutdown(static_cast<SOCKET>(fd), SD_BOTH);
}
constexpr int kSendNoSig = 0;
#else
inline void ensure_winsock() {
}
inline void close_socket_fd(int fd) {
    if (fd >= 0)
        close(fd);
}
inline void shutdown_socket_rw(int fd) {
    if (fd >= 0)
        shutdown(fd, SHUT_RDWR);
}
constexpr int kSendNoSig = MSG_NOSIGNAL;
#endif

bool send_all(int fd, const char* data, size_t len) {
    size_t total_sent = 0;
    while (total_sent < len) {
        const auto sent = send(fd, data + total_sent, static_cast<int>(len - total_sent), kSendNoSig);
        if (sent <= 0) {
            return false;
        }
        total_sent += static_cast<size_t>(sent);
    }
    return true;
}

bool send_response(int fd, const HttpResponse& res) {
    std::string raw = res.serialize();
    return send_all(fd, raw.data(), raw.size());
}

std::string sse_data(const std::string& payload) {
    std::string event = "data: ";
    for (char c : payload) {
        event += c;
        if (c == '\n')
            event += "data: ";
    }
    return event + "\n\n";
}

bool generate_checked(IInferenceEngine& engine, const InferenceRequest& req, StreamCallback callback,
                      GenerationResult& result) {
    try {
        const bool ok =
            callback ? engine.generate_stream(req, std::move(callback), result) : engine.generate(req, result);
        if (!ok || result.finish_reason == "error")
            return result.fail(result.error_message);
        return true;
    } catch (const std::exception& e) {
        return result.fail(std::string("Inference failed: ") + e.what());
    } catch (...) {
        return result.fail("Inference failed with an unknown exception");
    }
}

} // namespace

Server::Server(std::shared_ptr<IInferenceEngine> engine, ServerOptions options)
    : engine_(std::move(engine)),
      options_(std::move(options)) {
    ensure_winsock();
#if defined(_WIN32)
    wake_pipe_[0] = 0;
    wake_pipe_[1] = 0;
#else
    if (pipe(wake_pipe_) == 0) {
        for (int i = 0; i < 2; ++i) {
            const int flags = fcntl(wake_pipe_[i], F_GETFL, 0);
            if (flags >= 0)
                fcntl(wake_pipe_[i], F_SETFL, flags | O_NONBLOCK);
            fcntl(wake_pipe_[i], F_SETFD, FD_CLOEXEC);
        }
    } else {
        wake_pipe_[0] = -1;
        wake_pipe_[1] = -1;
    }
#endif
}

Server::~Server() {
    stop();
#if !defined(_WIN32)
    for (int i = 0; i < 2; ++i) {
        if (wake_pipe_[i] >= 0) {
            close(wake_pipe_[i]);
            wake_pipe_[i] = -1;
        }
    }
#endif
}

void Server::request_stop() {
    running_.store(false, std::memory_order_release);
#if !defined(_WIN32)
    if (wake_pipe_[1] >= 0) {
        const char byte = 1;
        const ssize_t ignored = ::write(wake_pipe_[1], &byte, 1);
        (void) ignored;
    }
#endif
}

bool Server::start() {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    if (wake_pipe_[0] < 0 || wake_pipe_[1] < 0) {
        if (options_.verbose) {
            std::cerr << "[server] Wake pipe unavailable\n";
        }
        return false;
    }
    if (started_) {
        if (running_.load(std::memory_order_acquire))
            return true;
        // A prior run exited via request_stop() without stop(); join and close before restarting.
        if (listen_fd_ >= 0)
            shutdown_socket_rw(listen_fd_);
        queue_cv_.notify_all();
        if (listener_thread_.joinable())
            listener_thread_.join();
        if (listen_fd_ >= 0) {
            close_socket_fd(listen_fd_);
            listen_fd_ = -1;
        }
        for (auto& w : workers_) {
            if (w.joinable())
                w.join();
        }
        workers_.clear();
        started_ = false;
    }

#if !defined(_WIN32)
    // Drain any residual wakeup bytes from a previous stop()/request_stop().
    char drain[64];
    while (::read(wake_pipe_[0], drain, sizeof(drain)) > 0) {}

    signal(SIGPIPE, SIG_IGN);
#endif

    int sock_fd = static_cast<int>(socket(AF_INET, SOCK_STREAM, 0));
    if (sock_fd < 0) {
        if (options_.verbose) {
            std::cerr << "[server] Failed to create socket: " << std::strerror(errno) << "\n";
        }
        return false;
    }
#if !defined(_WIN32)
    fcntl(sock_fd, F_SETFD, FD_CLOEXEC);
#endif

    int opt = 1;
    setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(options_.port));
    if (inet_pton(AF_INET, options_.host.c_str(), &addr.sin_addr) <= 0) {
        if (options_.verbose) {
            std::cerr << "[server] Invalid host address: " << options_.host << "\n";
        }
        close_socket_fd(sock_fd);
        return false;
    }

    if (bind(sock_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        if (options_.verbose) {
            std::cerr << "[server] Bind failed on " << options_.host << ":" << options_.port << ": "
                      << std::strerror(errno) << "\n";
        }
        close_socket_fd(sock_fd);
        return false;
    }

    if (options_.port == 0) {
        sockaddr_in actual_addr{};
        socklen_t len = sizeof(actual_addr);
        if (getsockname(sock_fd, reinterpret_cast<sockaddr*>(&actual_addr), &len) == 0) {
            bound_port_.store(ntohs(actual_addr.sin_port), std::memory_order_release);
        } else {
            bound_port_.store(options_.port, std::memory_order_release);
        }
    } else {
        bound_port_.store(options_.port, std::memory_order_release);
    }

    if (listen(sock_fd, 128) != 0) {
        if (options_.verbose) {
            std::cerr << "[server] Listen failed: " << std::strerror(errno) << "\n";
        }
        close_socket_fd(sock_fd);
        return false;
    }

    listen_fd_ = sock_fd;
    running_.store(true, std::memory_order_release);
    started_ = true;

    int num_workers = options_.worker_threads > 0 ? options_.worker_threads : 4;
    for (int i = 0; i < num_workers; ++i) {
        workers_.emplace_back(&Server::worker_loop, this);
    }

    // Pass descriptors by value so listener_loop never reads mutable member fields.
    // Descriptor ownership stays with the Server until after listener_thread_.join().
    listener_thread_ = std::thread(&Server::listener_loop, this, listen_fd_, wake_pipe_[0]);

    return true;
}

void Server::run() {
    while (running_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    stop();
}

void Server::stop() {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    request_stop();
    if (!started_)
        return;

    if (listen_fd_ >= 0) {
        shutdown_socket_rw(listen_fd_);
    }

    // Cancel any in-flight engine generation before waiting for worker threads.
    if (engine_) {
        engine_->stop();
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        for (int fd : client_queue_) {
            close_socket_fd(fd);
        }
        client_queue_.clear();
        // Shutdown (but do not close) active client sockets so workers blocked in recv()/send()
        // wake immediately while retaining exclusive ownership of close(client_fd).
        for (int fd : active_clients_) {
            shutdown_socket_rw(fd);
        }
    }

    queue_cv_.notify_all();

    if (listener_thread_.joinable()) {
        listener_thread_.join();
    }

    // Only close listen_fd_ after listener_thread_ has exited, preventing use-after-close
    // and file-descriptor number reuse while poll()/accept() can run.
    if (listen_fd_ >= 0) {
        close_socket_fd(listen_fd_);
        listen_fd_ = -1;
    }
#if !defined(_WIN32)
    char drain[64];
    while (::read(wake_pipe_[0], drain, sizeof(drain)) > 0) {}
#endif

    for (auto& w : workers_) {
        if (w.joinable()) {
            w.join();
        }
    }
    workers_.clear();
    started_ = false;
}

void Server::listener_loop(int listen_fd, int wake_fd) {
    while (running_.load(std::memory_order_acquire)) {
#if defined(_WIN32)
        (void) wake_fd;
        WSAPOLLFD pfd{};
        pfd.fd = static_cast<SOCKET>(listen_fd);
        pfd.events = POLLRDNORM;
        int ret = WSAPoll(&pfd, 1, 50);
        if (ret < 0)
            break;
        if (!running_.load(std::memory_order_acquire))
            break;
        if (ret == 0)
            continue;
        if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))
            break;
        const bool ready_to_accept = (pfd.revents & POLLRDNORM) != 0;
#else
        struct pollfd pfds[2]{};
        pfds[0].fd = listen_fd;
        pfds[0].events = POLLIN;
        pfds[1].fd = wake_fd;
        pfds[1].events = POLLIN;

        int ret = poll(pfds, 2, 100);
        if (ret < 0) {
            if (errno == EINTR)
                continue;
            break;
        }
        if (!running_.load(std::memory_order_acquire) || (pfds[1].revents & (POLLIN | POLLERR | POLLHUP))) {
            break;
        }
        if (ret == 0) {
            continue;
        }

        if (pfds[0].revents & (POLLERR | POLLHUP | POLLNVAL)) {
            break;
        }
        const bool ready_to_accept = (pfds[0].revents & POLLIN) != 0;
#endif

        if (ready_to_accept) {
            sockaddr_in client_addr{};
            socklen_t client_len = sizeof(client_addr);
            int client_fd = static_cast<int>(accept(listen_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len));
            if (client_fd >= 0) {
#if defined(_WIN32)
                DWORD timeout_ms = 30000;
                setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout_ms),
                           sizeof(timeout_ms));
                setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&timeout_ms),
                           sizeof(timeout_ms));
#else
                fcntl(client_fd, F_SETFD, FD_CLOEXEC);
                // Set send/recv timeouts
                struct timeval tv {};
                tv.tv_sec = 30;
                tv.tv_usec = 0;
                setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
                setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif

                {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    if (!running_.load(std::memory_order_acquire)) {
                        close_socket_fd(client_fd);
                        break;
                    }
                    if (client_queue_.size() < 64) {
                        client_queue_.push_back(client_fd);
                        queue_cv_.notify_one();
                    } else {
                        // Queue full: reject immediately
                        const char busy_msg[] =
                            "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 1\r\nContent-Length: 0\r\n\r\n";
                        send(client_fd, busy_msg, static_cast<int>(sizeof(busy_msg) - 1), kSendNoSig);
                        close_socket_fd(client_fd);
                    }
                }
            } else if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
                if (!running_.load(std::memory_order_acquire))
                    break;
            }
        }
    }
}

void Server::worker_loop() {
    for (;;) {
        int client_fd = -1;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock,
                           [this] { return !client_queue_.empty() || !running_.load(std::memory_order_acquire); });

            if (!running_.load(std::memory_order_acquire)) {
                while (!client_queue_.empty()) {
                    close_socket_fd(client_queue_.back());
                    client_queue_.pop_back();
                }
                return;
            }

            client_fd = client_queue_.front();
            client_queue_.erase(client_queue_.begin());
            active_clients_.push_back(client_fd);
        }

        if (client_fd >= 0) {
            handle_client(client_fd);
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                auto it = std::find(active_clients_.begin(), active_clients_.end(), client_fd);
                if (it != active_clients_.end()) {
                    active_clients_.erase(it);
                }
            }
            close_socket_fd(client_fd);
        }
    }
}

void Server::handle_client(int client_fd) {
    std::string raw_buffer;
    char buf[4096];
    size_t header_end = std::string::npos;

    while (header_end == std::string::npos) {
        const auto n = recv(client_fd, buf, static_cast<int>(sizeof(buf)), 0);
        if (n <= 0)
            return;
        raw_buffer.append(buf, static_cast<size_t>(n));
        header_end = raw_buffer.find("\r\n\r\n");
        if (raw_buffer.size() > 65536) {
            send_response(client_fd,
                          HttpResponse::json(400, OpenAiFormatter::format_error("Request headers too large")));
            return;
        }
    }

    HttpRequest req;
    std::string header_part = raw_buffer.substr(0, header_end + 2);
    if (!HttpRequest::parse_headers(header_part, req)) {
        send_response(client_fd, HttpResponse::json(400, OpenAiFormatter::format_error("Malformed HTTP request")));
        return;
    }

    size_t content_len = req.content_length();
    if (content_len > MAX_BODY_SIZE) {
        send_response(client_fd, HttpResponse::json(413, OpenAiFormatter::format_error("Payload too large",
                                                                                       "invalid_request_error", "",
                                                                                       "payload_too_large")));
        return;
    }
    req.body = raw_buffer.substr(header_end + 4);

    while (req.body.size() < content_len) {
        const auto n = recv(client_fd, buf, static_cast<int>(sizeof(buf)), 0);
        if (n <= 0)
            break;
        req.body.append(buf, static_cast<size_t>(n));
    }

    // CORS preflight
    if (req.method == "OPTIONS") {
        send_response(client_fd, HttpResponse::cors_preflight());
        return;
    }

    // Routing
    if (req.method == "GET" && (req.path == "/health" || req.path == "/api/health")) {
        auto start = std::chrono::steady_clock::now();
        telemetry_.record_request_start();

        std::string json_body =
            OpenAiFormatter::format_health(engine_ ? engine_->model_name() : "None",
                                           engine_ ? engine_->max_context() : 0, engine_ ? engine_->is_ready() : false);
        send_response(client_fd, HttpResponse::json(200, json_body));

        auto end = std::chrono::steady_clock::now();
        RequestMetrics m;
        m.method = "GET";
        m.path = req.path;
        m.status_code = 200;
        m.duration_s = std::chrono::duration<double>(end - start).count();
        telemetry_.record_request_finish(m);
        log_request(m);
        return;
    }

    if (req.method == "GET" && (req.path == "/v1/models" || req.path == "/models")) {
        auto start = std::chrono::steady_clock::now();
        telemetry_.record_request_start();

        std::vector<std::string> names;
        if (engine_)
            names.push_back(engine_->model_name());
        else
            names.push_back("Qwen3.8-Flash-Next");

        std::string json_body = OpenAiFormatter::format_models(names);
        send_response(client_fd, HttpResponse::json(200, json_body));

        auto end = std::chrono::steady_clock::now();
        RequestMetrics m;
        m.method = "GET";
        m.path = req.path;
        m.status_code = 200;
        m.duration_s = std::chrono::duration<double>(end - start).count();
        telemetry_.record_request_finish(m);
        log_request(m);
        return;
    }

    if (req.method == "GET" && req.path.rfind("/v1/models/", 0) == 0) {
        std::string requested_id = req.path.substr(11);
        std::string active_id = engine_ ? engine_->model_name() : "Qwen3.8-Flash-Next";
        if (requested_id == active_id) {
            send_response(client_fd, HttpResponse::json(200, OpenAiFormatter::format_model(active_id)));
        } else {
            send_response(client_fd, HttpResponse::json(404, OpenAiFormatter::format_error(
                                                                 "The model '" + requested_id + "' does not exist",
                                                                 "invalid_request_error", "model", "model_not_found")));
        }
        return;
    }

    if (req.method == "POST" && (req.path == "/v1/chat/completions" || req.path == "/chat/completions")) {
        process_chat_completions(client_fd, req);
        return;
    }

    if (req.method == "POST" && (req.path == "/v1/completions" || req.path == "/completions")) {
        process_completions(client_fd, req);
        return;
    }

    if (req.method == "POST" && (req.path == "/v1/responses" || req.path == "/responses")) {
        process_responses(client_fd, req);
        return;
    }

    // 404
    RequestMetrics m;
    m.method = req.method;
    m.path = req.path;
    m.status_code = 404;
    m.outcome = "not_found";
    send_response(client_fd, HttpResponse::json(404, OpenAiFormatter::format_error("Not found", "invalid_request_error",
                                                                                   "", "not_found")));
    log_request(m);
}

void Server::process_chat_completions(int client_fd, const HttpRequest& req) {
    auto start_time = std::chrono::steady_clock::now();
    telemetry_.record_request_start();

    ChatCompletionRequest chat_req;
    std::string err_type, err_param, err_code, err_msg;

    if (!OpenAiParser::parse_chat_request(req.body, chat_req, err_type, err_param, err_code, err_msg)) {
        RequestMetrics m;
        m.method = req.method;
        m.path = req.path;
        m.status_code = 400;
        m.outcome = "error";
        m.duration_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        send_response(client_fd,
                      HttpResponse::json(400, OpenAiFormatter::format_error(err_msg, err_type, err_param, err_code)));
        telemetry_.record_request_finish(m);
        log_request(m);
        return;
    }

    if (!engine_ || !engine_->is_ready()) {
        RequestMetrics m;
        m.method = req.method;
        m.path = req.path;
        m.status_code = 503;
        m.outcome = "error";
        m.duration_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        send_response(client_fd,
                      HttpResponse::json(503, OpenAiFormatter::format_error("Engine not ready", "server_error", "",
                                                                            "engine_not_ready")));
        telemetry_.record_request_finish(m);
        log_request(m);
        return;
    }

    InferenceRequest inf_req;
    inf_req.request_id = OpenAiFormatter::generate_id();
    inf_req.model = chat_req.model.empty() ? engine_->model_name() : chat_req.model;

    runtime::ChatTemplateOptions tpl_opts;
    tpl_opts.add_generation_prompt = true;
    tpl_opts.enable_thinking = chat_req.enable_thinking;
    tpl_opts.reasoning_effort = chat_req.reasoning_effort;
    tpl_opts.tools = chat_req.tools;
    tpl_opts.tool_choice = chat_req.tool_choice;

    const auto* tpl = engine_->chat_template();
    const bool has_loaded_tpl = (tpl != nullptr && tpl->is_loaded());
    if (has_loaded_tpl) {
        std::string tpl_err;
        if (!tpl->render(chat_req.messages, tpl_opts, inf_req.prompt, tpl_err)) {
            RequestMetrics m;
            m.method = req.method;
            m.path = req.path;
            m.status_code = 400;
            m.outcome = "error";
            m.duration_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
            send_response(client_fd,
                          HttpResponse::json(400, OpenAiFormatter::format_error(tpl_err, "invalid_request_error",
                                                                                "messages", "template_error")));
            telemetry_.record_request_finish(m);
            log_request(m);
            return;
        }
    } else {
        inf_req.prompt = OpenAiFormatter::render_chatml(chat_req.messages, tpl_opts);
    }

    const bool prompt_opened_think =
        has_loaded_tpl && chat_req.enable_thinking && (chat_req.reasoning_effort != "none");

    std::string enc_err;
    if (!engine_->encode(inf_req.prompt, inf_req.prompt_tokens, enc_err)) {
        RequestMetrics m;
        m.method = req.method;
        m.path = req.path;
        m.status_code = 400;
        m.outcome = "error";
        m.duration_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        send_response(client_fd,
                      HttpResponse::json(400, OpenAiFormatter::format_error(enc_err, "invalid_request_error", "prompt",
                                                                            "tokenization_failed")));
        telemetry_.record_request_finish(m);
        log_request(m);
        return;
    }

    if (chat_req.max_tokens.has_value()) {
        inf_req.max_tokens = chat_req.max_tokens.value();
    }
    if (chat_req.temperature.has_value()) {
        inf_req.sampling.temperature = chat_req.temperature.value();
    }
    if (chat_req.top_p.has_value()) {
        inf_req.sampling.top_p = chat_req.top_p.value();
    }
    if (chat_req.top_k.has_value()) {
        inf_req.sampling.top_k = chat_req.top_k.value();
    }
    if (chat_req.seed.has_value()) {
        inf_req.sampling.seed = chat_req.seed.value();
    }
    inf_req.sampling.stop = chat_req.stop;

    int64_t created =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    if (chat_req.stream) {
        // SSE Headers
        const char sse_header[] = "HTTP/1.1 200 OK\r\n"
                                  "Content-Type: text/event-stream; charset=utf-8\r\n"
                                  "Cache-Control: no-cache\r\n"
                                  "Connection: keep-alive\r\n"
                                  "Access-Control-Allow-Origin: *\r\n\r\n";

        // Delay success headers until inference has produced a checked token. An
        // initialization/prefill failure must still be an HTTP error response.
        bool stream_started = false;
        bool client_alive = true;
        OpenAiOutputParser output_parser(prompt_opened_think, chat_req.tools, /*stream_tools=*/true);

        std::string first_chunk =
            "data: " +
            OpenAiFormatter::format_chat_chunk(inf_req.request_id, inf_req.model, created, "assistant", "", "") +
            "\n\n";
        auto start_stream = [&]() {
            if (stream_started)
                return client_alive;
            stream_started = true;
            client_alive = send_all(client_fd, sse_header, sizeof(sse_header) - 1) &&
                           send_all(client_fd, first_chunk.data(), first_chunk.size());
            return client_alive;
        };

        auto emit_parsed_events = [&](const std::vector<ParsedOutputEvent>& events) -> bool {
            for (const auto& ev : events) {
                std::string chunk_json;
                if (ev.kind == ParsedEventKind::Reasoning && !ev.text.empty()) {
                    chunk_json = OpenAiFormatter::format_chat_chunk(inf_req.request_id, inf_req.model, created, "", "",
                                                                    "", ev.text);
                } else if (ev.kind == ParsedEventKind::Content && !ev.text.empty()) {
                    chunk_json =
                        OpenAiFormatter::format_chat_chunk(inf_req.request_id, inf_req.model, created, "", ev.text, "");
                } else if (ev.kind == ParsedEventKind::ToolCallStart) {
                    chunk_json = OpenAiFormatter::format_chat_tool_call_chunk(inf_req.request_id, inf_req.model,
                                                                              created, ev.tool_index, ev.call.id,
                                                                              ev.call.function.name, "");
                } else if (ev.kind == ParsedEventKind::ToolCallArgs && !ev.text.empty()) {
                    chunk_json = OpenAiFormatter::format_chat_tool_call_chunk(inf_req.request_id, inf_req.model,
                                                                              created, ev.tool_index, "", "", ev.text);
                } else {
                    continue;
                }
                std::string chunk_data = "data: " + chunk_json + "\n\n";
                if (!send_all(client_fd, chunk_data.data(), chunk_data.size())) {
                    client_alive = false;
                    return false;
                }
            }
            return true;
        };

        auto stream_cb = [&](const TokenOutput& tok) -> bool {
            if (!start_stream())
                return false;
            if (tok.is_special)
                return true;
            auto evs = output_parser.feed(tok.text, true);
            return emit_parsed_events(evs);
        };

        GenerationResult gen_res;
        bool ok = generate_checked(*engine_, inf_req, stream_cb, gen_res);

        if (client_alive && ok) {
            start_stream();
            auto tail_evs = output_parser.finish();
            emit_parsed_events(tail_evs);

            const std::string effective_finish = (output_parser.has_tool_calls() && gen_res.finish_reason == "stop")
                                                     ? "tool_calls"
                                                     : gen_res.finish_reason;
            std::string final_chunk =
                "data: " +
                (chat_req.stream_include_usage
                     ? OpenAiFormatter::format_chat_usage_chunk(
                           inf_req.request_id, inf_req.model, created, effective_finish, gen_res.prompt_tokens,
                           gen_res.completion_tokens, output_parser.reasoning_tokens())
                     : OpenAiFormatter::format_chat_chunk(inf_req.request_id, inf_req.model, created, "", "",
                                                          effective_finish)) +
                "\n\n";
            send_all(client_fd, final_chunk.data(), final_chunk.size());

            const char done_msg[] = "data: [DONE]\n\n";
            send_all(client_fd, done_msg, sizeof(done_msg) - 1);
        } else if (client_alive) {
            const std::string error =
                OpenAiFormatter::format_error(gen_res.error_message, "server_error", "", "inference_failed");
            if (!stream_started)
                send_response(client_fd, HttpResponse::json(500, error));
            else {
                const std::string event = sse_data(error);
                send_all(client_fd, event.data(), event.size());
            }
        }

        auto end_time = std::chrono::steady_clock::now();
        RequestMetrics m;
        m.method = req.method;
        m.path = req.path;
        m.status_code = !ok && !stream_started ? 500 : 200;
        m.duration_s = std::chrono::duration<double>(end_time - start_time).count();
        if (m.duration_s < 0.01 && gen_res.decode_ms > 0.0) {
            m.duration_s = (gen_res.prompt_ms + gen_res.decode_ms) / 1000.0;
        }
        m.prompt_tokens = gen_res.prompt_tokens;
        m.completion_tokens = gen_res.completion_tokens;
        m.prompt_ms = gen_res.prompt_ms;
        m.decode_ms = gen_res.decode_ms;
        m.prompt_tok_s = gen_res.prompt_tok_s;
        m.decode_tok_s = gen_res.decode_tok_s;
        m.ram_expert_hits = gen_res.ram_blobs;
        m.file_expert_reads = gen_res.file_blobs;
        m.gpu_cache_hits = gen_res.gpu_cache_hits;
        m.drafts_accepted = gen_res.drafts_accepted;
        m.drafts_offered = gen_res.drafts_offered;
        m.context_tokens = gen_res.prompt_tokens + gen_res.completion_tokens;
        m.streamed = true;
        m.outcome = !client_alive ? "disconnected" : ok ? "ok" : "error";

        telemetry_.record_request_finish(m);
        log_request(m);
    } else {
        GenerationResult gen_res;
        const bool ok = generate_checked(*engine_, inf_req, nullptr, gen_res);

        OpenAiOutputParser output_parser(prompt_opened_think, chat_req.tools, /*stream_tools=*/false);
        if (ok) {
            if (!gen_res.tokens.empty()) {
                for (const auto& tok : gen_res.tokens) {
                    if (!tok.is_special) {
                        output_parser.feed(tok.text, true);
                    }
                }
            } else {
                output_parser.feed(gen_res.text, true);
            }
            output_parser.finish();
        }

        std::string json_res =
            ok ? OpenAiFormatter::format_chat_completion(
                     inf_req.request_id, inf_req.model, created, output_parser.content(), gen_res.finish_reason,
                     gen_res.prompt_tokens, gen_res.completion_tokens, output_parser.reasoning_content(),
                     output_parser.tool_calls(), output_parser.reasoning_tokens())
               : OpenAiFormatter::format_error(gen_res.error_message, "server_error", "", "inference_failed");

        send_response(client_fd, HttpResponse::json(ok ? 200 : 500, json_res));

        auto end_time = std::chrono::steady_clock::now();
        RequestMetrics m;
        m.method = req.method;
        m.path = req.path;
        m.status_code = ok ? 200 : 500;
        m.duration_s = std::chrono::duration<double>(end_time - start_time).count();
        if (m.duration_s < 0.01 && gen_res.decode_ms > 0.0) {
            m.duration_s = (gen_res.prompt_ms + gen_res.decode_ms) / 1000.0;
        }
        m.prompt_tokens = gen_res.prompt_tokens;
        m.completion_tokens = gen_res.completion_tokens;
        m.prompt_ms = gen_res.prompt_ms;
        m.decode_ms = gen_res.decode_ms;
        m.prompt_tok_s = gen_res.prompt_tok_s;
        m.decode_tok_s = gen_res.decode_tok_s;
        m.ram_expert_hits = gen_res.ram_blobs;
        m.file_expert_reads = gen_res.file_blobs;
        m.gpu_cache_hits = gen_res.gpu_cache_hits;
        m.drafts_accepted = gen_res.drafts_accepted;
        m.drafts_offered = gen_res.drafts_offered;
        m.context_tokens = gen_res.prompt_tokens + gen_res.completion_tokens;
        m.streamed = false;
        m.outcome = ok ? "ok" : "error";

        telemetry_.record_request_finish(m);
        log_request(m);
    }
}

void Server::process_responses(int client_fd, const HttpRequest& req) {
    auto start_time = std::chrono::steady_clock::now();
    telemetry_.record_request_start();

    ResponsesRequest resp_req;
    std::string err_type, err_param, err_code, err_msg;
    if (!OpenAiParser::parse_responses_request(req.body, resp_req, err_type, err_param, err_code, err_msg)) {
        RequestMetrics m;
        m.method = req.method;
        m.path = req.path;
        m.status_code = 400;
        m.outcome = "error";
        m.duration_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        send_response(client_fd,
                      HttpResponse::json(400, OpenAiFormatter::format_error(err_msg, err_type, err_param, err_code)));
        telemetry_.record_request_finish(m);
        log_request(m);
        return;
    }

    if (!engine_ || !engine_->is_ready()) {
        send_response(client_fd,
                      HttpResponse::json(503, OpenAiFormatter::format_error("Engine not ready", "server_error", "",
                                                                            "engine_not_ready")));
        return;
    }

    const auto& chat_req = resp_req.chat;
    InferenceRequest inf_req;
    inf_req.request_id = OpenAiFormatter::generate_id("resp_");
    inf_req.model = chat_req.model.empty() ? engine_->model_name() : chat_req.model;

    runtime::ChatTemplateOptions tpl_opts;
    tpl_opts.add_generation_prompt = true;
    tpl_opts.enable_thinking = chat_req.enable_thinking;
    tpl_opts.reasoning_effort = chat_req.reasoning_effort;
    tpl_opts.tools = chat_req.tools;
    tpl_opts.tool_choice = chat_req.tool_choice;

    const auto* tpl = engine_->chat_template();
    const bool has_loaded_tpl = (tpl != nullptr && tpl->is_loaded());
    if (has_loaded_tpl) {
        std::string tpl_err;
        if (!tpl->render(chat_req.messages, tpl_opts, inf_req.prompt, tpl_err)) {
            send_response(client_fd,
                          HttpResponse::json(400, OpenAiFormatter::format_error(tpl_err, "invalid_request_error",
                                                                                "input", "template_error")));
            return;
        }
    } else {
        inf_req.prompt = OpenAiFormatter::render_chatml(chat_req.messages, tpl_opts);
    }

    const bool prompt_opened_think =
        has_loaded_tpl && chat_req.enable_thinking && (chat_req.reasoning_effort != "none");

    std::string enc_err;
    if (!engine_->encode(inf_req.prompt, inf_req.prompt_tokens, enc_err)) {
        send_response(client_fd,
                      HttpResponse::json(400, OpenAiFormatter::format_error(enc_err, "invalid_request_error", "input",
                                                                            "tokenization_failed")));
        return;
    }

    if (chat_req.max_tokens.has_value())
        inf_req.max_tokens = chat_req.max_tokens.value();
    if (chat_req.temperature.has_value())
        inf_req.sampling.temperature = chat_req.temperature.value();
    if (chat_req.top_p.has_value())
        inf_req.sampling.top_p = chat_req.top_p.value();

    int64_t created =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    if (chat_req.stream) {
        const char sse_header[] = "HTTP/1.1 200 OK\r\n"
                                  "Content-Type: text/event-stream; charset=utf-8\r\n"
                                  "Cache-Control: no-cache\r\n"
                                  "Connection: keep-alive\r\n"
                                  "Access-Control-Allow-Origin: *\r\n\r\n";

        bool stream_started = false;
        bool client_alive = true;
        OpenAiOutputParser output_parser(prompt_opened_think, chat_req.tools, /*stream_tools=*/true);

        auto send_sse_event = [&](const std::string& event_name, const std::string& data_json) -> bool {
            std::string pkt = "event: " + event_name + "\ndata: " + data_json + "\n\n";
            if (!send_all(client_fd, pkt.data(), pkt.size())) {
                client_alive = false;
                return false;
            }
            return true;
        };

        auto start_stream = [&]() {
            if (stream_started)
                return client_alive;
            stream_started = true;
            if (!send_all(client_fd, sse_header, sizeof(sse_header) - 1)) {
                client_alive = false;
                return false;
            }
            json::JsonValue created_ev(std::vector<std::pair<std::string, json::JsonValue>>{});
            created_ev["type"] = "response.created";
            json::JsonValue r_obj(std::vector<std::pair<std::string, json::JsonValue>>{});
            r_obj["id"] = inf_req.request_id;
            r_obj["object"] = "response";
            r_obj["status"] = "in_progress";
            r_obj["model"] = inf_req.model;
            created_ev["response"] = std::move(r_obj);
            return send_sse_event("response.created", created_ev.dump());
        };

        auto emit_events = [&](const std::vector<ParsedOutputEvent>& evs) -> bool {
            for (const auto& ev : evs) {
                if (ev.kind == ParsedEventKind::Reasoning && !ev.text.empty()) {
                    json::JsonValue d(std::vector<std::pair<std::string, json::JsonValue>>{});
                    d["type"] = "response.reasoning_text.delta";
                    d["delta"] = ev.text;
                    if (!send_sse_event("response.reasoning_text.delta", d.dump()))
                        return false;
                } else if (ev.kind == ParsedEventKind::Content && !ev.text.empty()) {
                    json::JsonValue d(std::vector<std::pair<std::string, json::JsonValue>>{});
                    d["type"] = "response.output_text.delta";
                    d["delta"] = ev.text;
                    if (!send_sse_event("response.output_text.delta", d.dump()))
                        return false;
                } else if (ev.kind == ParsedEventKind::ToolCallArgs && !ev.text.empty()) {
                    json::JsonValue d(std::vector<std::pair<std::string, json::JsonValue>>{});
                    d["type"] = "response.function_call_arguments.delta";
                    d["call_id"] = ev.call.id;
                    d["delta"] = ev.text;
                    if (!send_sse_event("response.function_call_arguments.delta", d.dump()))
                        return false;
                }
            }
            return true;
        };

        auto stream_cb = [&](const TokenOutput& tok) -> bool {
            if (!start_stream())
                return false;
            if (tok.is_special)
                return true;
            return emit_events(output_parser.feed(tok.text, true));
        };

        GenerationResult gen_res;
        bool ok = generate_checked(*engine_, inf_req, stream_cb, gen_res);
        if (client_alive && ok) {
            start_stream();
            emit_events(output_parser.finish());
            std::string final_resp = OpenAiFormatter::format_responses_object(
                inf_req.request_id, inf_req.model, created, output_parser.content(), output_parser.reasoning_content(),
                output_parser.tool_calls(), gen_res.finish_reason, gen_res.prompt_tokens, gen_res.completion_tokens,
                output_parser.reasoning_tokens());
            json::JsonValue parsed_resp;
            std::string perr;
            json::JsonValue::parse(final_resp, parsed_resp, perr);
            json::JsonValue comp_ev(std::vector<std::pair<std::string, json::JsonValue>>{});
            comp_ev["type"] = "response.completed";
            comp_ev["response"] = std::move(parsed_resp);
            send_sse_event("response.completed", comp_ev.dump());
        } else if (client_alive && !stream_started) {
            send_response(client_fd,
                          HttpResponse::json(500, OpenAiFormatter::format_error(gen_res.error_message, "server_error",
                                                                                "", "inference_failed")));
        }
    } else {
        GenerationResult gen_res;
        const bool ok = generate_checked(*engine_, inf_req, nullptr, gen_res);
        OpenAiOutputParser output_parser(prompt_opened_think, chat_req.tools, /*stream_tools=*/false);
        if (ok) {
            for (const auto& tok : gen_res.tokens) {
                if (!tok.is_special)
                    output_parser.feed(tok.text, true);
            }
            output_parser.finish();
        }
        std::string json_res =
            ok ? OpenAiFormatter::format_responses_object(
                     inf_req.request_id, inf_req.model, created, output_parser.content(),
                     output_parser.reasoning_content(), output_parser.tool_calls(), gen_res.finish_reason,
                     gen_res.prompt_tokens, gen_res.completion_tokens, output_parser.reasoning_tokens())
               : OpenAiFormatter::format_error(gen_res.error_message, "server_error", "", "inference_failed");
        send_response(client_fd, HttpResponse::json(ok ? 200 : 500, json_res));
    }
}

void Server::process_completions(int client_fd, const HttpRequest& req) {
    auto start_time = std::chrono::steady_clock::now();
    telemetry_.record_request_start();

    CompletionRequest comp_req;
    std::string err_type, err_param, err_code, err_msg;

    if (!OpenAiParser::parse_completion_request(req.body, comp_req, err_type, err_param, err_code, err_msg)) {
        RequestMetrics m;
        m.method = req.method;
        m.path = req.path;
        m.status_code = 400;
        m.outcome = "error";
        m.duration_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        send_response(client_fd,
                      HttpResponse::json(400, OpenAiFormatter::format_error(err_msg, err_type, err_param, err_code)));
        telemetry_.record_request_finish(m);
        log_request(m);
        return;
    }

    if (!engine_ || !engine_->is_ready()) {
        RequestMetrics m;
        m.method = req.method;
        m.path = req.path;
        m.status_code = 503;
        m.outcome = "error";
        m.duration_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        send_response(client_fd,
                      HttpResponse::json(503, OpenAiFormatter::format_error("Engine not ready", "server_error", "",
                                                                            "engine_not_ready")));
        telemetry_.record_request_finish(m);
        log_request(m);
        return;
    }

    InferenceRequest inf_req;
    inf_req.request_id = OpenAiFormatter::generate_id("cmpl-");
    inf_req.model = comp_req.model.empty() ? engine_->model_name() : comp_req.model;
    inf_req.prompt = comp_req.prompt;

    std::string enc_err;
    if (!engine_->encode(inf_req.prompt, inf_req.prompt_tokens, enc_err)) {
        RequestMetrics m;
        m.method = req.method;
        m.path = req.path;
        m.status_code = 400;
        m.outcome = "error";
        m.duration_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        send_response(client_fd,
                      HttpResponse::json(400, OpenAiFormatter::format_error(enc_err, "invalid_request_error", "prompt",
                                                                            "tokenization_failed")));
        telemetry_.record_request_finish(m);
        log_request(m);
        return;
    }

    if (comp_req.max_tokens.has_value()) {
        inf_req.max_tokens = comp_req.max_tokens.value();
    }
    if (comp_req.temperature.has_value()) {
        inf_req.sampling.temperature = comp_req.temperature.value();
    }
    if (comp_req.top_p.has_value()) {
        inf_req.sampling.top_p = comp_req.top_p.value();
    }
    if (comp_req.seed.has_value()) {
        inf_req.sampling.seed = comp_req.seed.value();
    }
    inf_req.sampling.stop = comp_req.stop;

    int64_t created =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    if (comp_req.stream) {
        const char sse_header[] = "HTTP/1.1 200 OK\r\n"
                                  "Content-Type: text/event-stream; charset=utf-8\r\n"
                                  "Cache-Control: no-cache\r\n"
                                  "Connection: keep-alive\r\n"
                                  "Access-Control-Allow-Origin: *\r\n\r\n";

        bool stream_started = false;
        bool client_alive = true;
        auto start_stream = [&]() {
            if (stream_started)
                return client_alive;
            stream_started = true;
            client_alive = send_all(client_fd, sse_header, sizeof(sse_header) - 1);
            return client_alive;
        };
        auto stream_cb = [&](const TokenOutput& tok) -> bool {
            if (!start_stream())
                return false;
            std::string chunk_data =
                "data: " +
                OpenAiFormatter::format_completion_chunk(inf_req.request_id, inf_req.model, created, tok.text, "") +
                "\n\n";
            if (!send_all(client_fd, chunk_data.data(), chunk_data.size())) {
                client_alive = false;
                return false;
            }
            return true;
        };

        GenerationResult gen_res;
        bool ok = generate_checked(*engine_, inf_req, stream_cb, gen_res);

        if (client_alive && ok) {
            start_stream();
            std::string final_chunk = "data: " +
                                      OpenAiFormatter::format_completion_chunk(inf_req.request_id, inf_req.model,
                                                                               created, "", gen_res.finish_reason) +
                                      "\n\n";
            send_all(client_fd, final_chunk.data(), final_chunk.size());

            const char done_msg[] = "data: [DONE]\n\n";
            send_all(client_fd, done_msg, sizeof(done_msg) - 1);
        } else if (client_alive) {
            const std::string error =
                OpenAiFormatter::format_error(gen_res.error_message, "server_error", "", "inference_failed");
            if (!stream_started)
                send_response(client_fd, HttpResponse::json(500, error));
            else {
                const std::string event = sse_data(error);
                send_all(client_fd, event.data(), event.size());
            }
        }

        auto end_time = std::chrono::steady_clock::now();
        RequestMetrics m;
        m.method = req.method;
        m.path = req.path;
        m.status_code = !ok && !stream_started ? 500 : 200;
        m.duration_s = std::chrono::duration<double>(end_time - start_time).count();
        if (m.duration_s < 0.01 && gen_res.decode_ms > 0.0) {
            m.duration_s = (gen_res.prompt_ms + gen_res.decode_ms) / 1000.0;
        }
        m.prompt_tokens = gen_res.prompt_tokens;
        m.completion_tokens = gen_res.completion_tokens;
        m.prompt_ms = gen_res.prompt_ms;
        m.decode_ms = gen_res.decode_ms;
        m.prompt_tok_s = gen_res.prompt_tok_s;
        m.decode_tok_s = gen_res.decode_tok_s;
        m.ram_expert_hits = gen_res.ram_blobs;
        m.file_expert_reads = gen_res.file_blobs;
        m.gpu_cache_hits = gen_res.gpu_cache_hits;
        m.drafts_accepted = gen_res.drafts_accepted;
        m.drafts_offered = gen_res.drafts_offered;
        m.context_tokens = gen_res.prompt_tokens + gen_res.completion_tokens;
        m.streamed = true;
        m.outcome = !client_alive ? "disconnected" : ok ? "ok" : "error";

        telemetry_.record_request_finish(m);
        log_request(m);
    } else {
        GenerationResult gen_res;
        const bool ok = generate_checked(*engine_, inf_req, nullptr, gen_res);

        std::string json_res =
            ok ? OpenAiFormatter::format_completion(inf_req.request_id, inf_req.model, created, gen_res.text,
                                                    gen_res.finish_reason, gen_res.prompt_tokens,
                                                    gen_res.completion_tokens)
               : OpenAiFormatter::format_error(gen_res.error_message, "server_error", "", "inference_failed");

        send_response(client_fd, HttpResponse::json(ok ? 200 : 500, json_res));

        auto end_time = std::chrono::steady_clock::now();
        RequestMetrics m;
        m.method = req.method;
        m.path = req.path;
        m.status_code = ok ? 200 : 500;
        m.duration_s = std::chrono::duration<double>(end_time - start_time).count();
        if (m.duration_s < 0.01 && gen_res.decode_ms > 0.0) {
            m.duration_s = (gen_res.prompt_ms + gen_res.decode_ms) / 1000.0;
        }
        m.prompt_tokens = gen_res.prompt_tokens;
        m.completion_tokens = gen_res.completion_tokens;
        m.prompt_ms = gen_res.prompt_ms;
        m.decode_ms = gen_res.decode_ms;
        m.prompt_tok_s = gen_res.prompt_tok_s;
        m.decode_tok_s = gen_res.decode_tok_s;
        m.ram_expert_hits = gen_res.ram_blobs;
        m.file_expert_reads = gen_res.file_blobs;
        m.gpu_cache_hits = gen_res.gpu_cache_hits;
        m.drafts_accepted = gen_res.drafts_accepted;
        m.drafts_offered = gen_res.drafts_offered;
        m.context_tokens = gen_res.prompt_tokens + gen_res.completion_tokens;
        m.streamed = false;
        m.outcome = ok ? "ok" : "error";

        telemetry_.record_request_finish(m);
        log_request(m);
    }
}

void Server::log_request(const RequestMetrics& m) {
    if (options_.quiet)
        return;

    if (options_.json_telemetry) {
        std::cout << Telemetry::format_jsonl(m) << std::endl;
    } else if (options_.no_tui) {
        std::cout << Telemetry::format_log_line(m) << std::endl;
    } else {
        std::cout << Telemetry::format_live_line(m) << std::endl;
    }
}

} // namespace guild::server
