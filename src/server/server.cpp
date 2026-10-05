#include "guild/server/server.hpp"
#include "guild/server/openai.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

namespace guild::server {

namespace {

bool send_all(int fd, const char* data, size_t len) {
    size_t total_sent = 0;
    while (total_sent < len) {
        ssize_t sent = send(fd, data + total_sent, len - total_sent, MSG_NOSIGNAL);
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

} // namespace

Server::Server(std::shared_ptr<IInferenceEngine> engine, ServerOptions options)
    : engine_(std::move(engine)), options_(std::move(options)) {}

Server::~Server() {
    stop();
}

bool Server::start() {
    if (running_.load()) return true;

    signal(SIGPIPE, SIG_IGN);

    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        if (options_.verbose) {
            std::cerr << "[server] Failed to create socket: " << std::strerror(errno) << "\n";
        }
        return false;
    }

    int opt = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(options_.port));
    if (inet_pton(AF_INET, options_.host.c_str(), &addr.sin_addr) <= 0) {
        if (options_.verbose) {
            std::cerr << "[server] Invalid host address: " << options_.host << "\n";
        }
        close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }

    if (bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        if (options_.verbose) {
            std::cerr << "[server] Bind failed on " << options_.host << ":" << options_.port
                      << ": " << std::strerror(errno) << "\n";
        }
        close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }

    if (options_.port == 0) {
        sockaddr_in actual_addr{};
        socklen_t len = sizeof(actual_addr);
        if (getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&actual_addr), &len) == 0) {
            bound_port_ = ntohs(actual_addr.sin_port);
        } else {
            bound_port_ = options_.port;
        }
    } else {
        bound_port_ = options_.port;
    }

    if (listen(listen_fd_, 128) != 0) {
        if (options_.verbose) {
            std::cerr << "[server] Listen failed: " << std::strerror(errno) << "\n";
        }
        close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }

    stopped_.store(false);
    running_.store(true);

    int num_workers = options_.worker_threads > 0 ? options_.worker_threads : 4;
    for (int i = 0; i < num_workers; ++i) {
        workers_.emplace_back(&Server::worker_loop, this);
    }

    listener_thread_ = std::thread(&Server::listener_loop, this);

    return true;
}

void Server::run() {
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    stop();
}

void Server::stop() {
    running_.store(false);
    if (stopped_.exchange(true)) return;

    if (listen_fd_ >= 0) {
        shutdown(listen_fd_, SHUT_RDWR);
        close(listen_fd_);
        listen_fd_ = -1;
    }

    queue_cv_.notify_all();

    if (listener_thread_.joinable()) {
        listener_thread_.join();
    }

    for (auto& w : workers_) {
        if (w.joinable()) {
            w.join();
        }
    }
    workers_.clear();

    if (engine_) {
        engine_->stop();
    }
}

void Server::listener_loop() {
    while (running_.load()) {
        struct pollfd pfd{};
        pfd.fd = listen_fd_;
        pfd.events = POLLIN;

        int ret = poll(&pfd, 1, 100);
        if (ret < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (ret == 0) {
            continue; // timeout, check running_
        }

        if (pfd.revents & POLLIN) {
            sockaddr_in client_addr{};
            socklen_t client_len = sizeof(client_addr);
            int client_fd = accept(listen_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
            if (client_fd >= 0) {
                // Set send/recv timeouts
                struct timeval tv{};
                tv.tv_sec = 30;
                tv.tv_usec = 0;
                setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
                setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

                {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    client_queue_.push_back(client_fd);
                }
                queue_cv_.notify_one();
            }
        }
    }
}

void Server::worker_loop() {
    while (running_.load()) {
        int client_fd = -1;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return !client_queue_.empty() || !running_.load();
            });

            if (!running_.load() && client_queue_.empty()) {
                return;
            }

            if (!client_queue_.empty()) {
                client_fd = client_queue_.front();
                client_queue_.erase(client_queue_.begin());
            }
        }

        if (client_fd >= 0) {
            handle_client(client_fd);
            close(client_fd);
        }
    }
}

void Server::handle_client(int client_fd) {
    std::string raw_buffer;
    char buf[4096];
    size_t header_end = std::string::npos;

    while (header_end == std::string::npos) {
        ssize_t n = recv(client_fd, buf, sizeof(buf), 0);
        if (n <= 0) return;
        raw_buffer.append(buf, static_cast<size_t>(n));
        header_end = raw_buffer.find("\r\n\r\n");
        if (raw_buffer.size() > 65536) {
            send_response(client_fd, HttpResponse::json(400, OpenAiFormatter::format_error("Request headers too large")));
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
    req.body = raw_buffer.substr(header_end + 4);

    while (req.body.size() < content_len) {
        ssize_t n = recv(client_fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
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

        std::string json_body = OpenAiFormatter::format_health(
            engine_ ? engine_->model_name() : "None",
            engine_ ? engine_->max_context() : 0,
            engine_ ? engine_->is_ready() : false);
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
        if (engine_) names.push_back(engine_->model_name());
        else names.push_back("Qwen3.8-Flash-Next");

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

    if (req.method == "POST" && req.path == "/v1/chat/completions") {
        process_chat_completions(client_fd, req);
        return;
    }

    if (req.method == "POST" && req.path == "/v1/completions") {
        process_completions(client_fd, req);
        return;
    }

    // 404
    RequestMetrics m;
    m.method = req.method;
    m.path = req.path;
    m.status_code = 404;
    m.outcome = "not_found";
    send_response(client_fd, HttpResponse::json(404, OpenAiFormatter::format_error("Not found", "invalid_request_error", "", "not_found")));
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
        send_response(client_fd, HttpResponse::json(400, OpenAiFormatter::format_error(err_msg, err_type, err_param, err_code)));
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
        send_response(client_fd, HttpResponse::json(503, OpenAiFormatter::format_error("Engine not ready", "server_error", "", "engine_not_ready")));
        telemetry_.record_request_finish(m);
        log_request(m);
        return;
    }

    InferenceRequest inf_req;
    inf_req.request_id = OpenAiFormatter::generate_id();
    inf_req.model = chat_req.model.empty() ? engine_->model_name() : chat_req.model;
    inf_req.prompt = OpenAiFormatter::render_chatml(chat_req.messages);
    if (chat_req.max_tokens.has_value()) {
        inf_req.max_tokens = chat_req.max_tokens.value();
    }
    if (chat_req.temperature.has_value()) {
        inf_req.sampling.temperature = chat_req.temperature.value();
    }
    if (chat_req.top_p.has_value()) {
        inf_req.sampling.top_p = chat_req.top_p.value();
    }
    if (chat_req.seed.has_value()) {
        inf_req.sampling.seed = chat_req.seed.value();
    }
    inf_req.sampling.stop = chat_req.stop;

    int64_t created = std::chrono::duration_cast<std::chrono::seconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();

    if (chat_req.stream) {
        // SSE Headers
        const char sse_header[] =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream; charset=utf-8\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n\r\n";

        if (!send_all(client_fd, sse_header, sizeof(sse_header) - 1)) {
            RequestMetrics m;
            m.method = req.method;
            m.path = req.path;
            m.status_code = 200;
            m.streamed = true;
            m.outcome = "disconnected";
            telemetry_.record_request_finish(m);
            log_request(m);
            return;
        }

        // Send initial chunk with role
        std::string first_chunk = "data: " + OpenAiFormatter::format_chat_chunk(
            inf_req.request_id, inf_req.model, created, "assistant", "", "") + "\n\n";

        if (!send_all(client_fd, first_chunk.data(), first_chunk.size())) {
            RequestMetrics m;
            m.method = req.method;
            m.path = req.path;
            m.status_code = 200;
            m.streamed = true;
            m.outcome = "disconnected";
            telemetry_.record_request_finish(m);
            log_request(m);
            return;
        }

        bool client_alive = true;
        auto stream_cb = [&](const TokenOutput& tok) -> bool {
            std::string chunk_data = "data: " + OpenAiFormatter::format_chat_chunk(
                inf_req.request_id, inf_req.model, created, "", tok.text, "") + "\n\n";
            if (!send_all(client_fd, chunk_data.data(), chunk_data.size())) {
                client_alive = false;
                return false;
            }
            return true;
        };

        GenerationResult gen_res;
        bool ok = engine_->generate_stream(inf_req, stream_cb, gen_res);

        if (client_alive && ok) {
            std::string final_chunk = "data: " + OpenAiFormatter::format_chat_chunk(
                inf_req.request_id, inf_req.model, created, "", "", gen_res.finish_reason) + "\n\n";
            send_all(client_fd, final_chunk.data(), final_chunk.size());

            const char done_msg[] = "data: [DONE]\n\n";
            send_all(client_fd, done_msg, sizeof(done_msg) - 1);
        }

        auto end_time = std::chrono::steady_clock::now();
        RequestMetrics m;
        m.method = req.method;
        m.path = req.path;
        m.status_code = 200;
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
        m.context_tokens = gen_res.prompt_tokens + gen_res.completion_tokens;
        m.streamed = true;
        m.outcome = client_alive ? "ok" : "disconnected";

        telemetry_.record_request_finish(m);
        log_request(m);
    } else {
        GenerationResult gen_res;
        engine_->generate(inf_req, gen_res);

        std::string json_res = OpenAiFormatter::format_chat_completion(
            inf_req.request_id, inf_req.model, created, gen_res.text,
            gen_res.finish_reason, gen_res.prompt_tokens, gen_res.completion_tokens);

        send_response(client_fd, HttpResponse::json(200, json_res));

        auto end_time = std::chrono::steady_clock::now();
        RequestMetrics m;
        m.method = req.method;
        m.path = req.path;
        m.status_code = 200;
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
        m.context_tokens = gen_res.prompt_tokens + gen_res.completion_tokens;
        m.streamed = false;
        m.outcome = "ok";

        telemetry_.record_request_finish(m);
        log_request(m);
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
        send_response(client_fd, HttpResponse::json(400, OpenAiFormatter::format_error(err_msg, err_type, err_param, err_code)));
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
        send_response(client_fd, HttpResponse::json(503, OpenAiFormatter::format_error("Engine not ready", "server_error", "", "engine_not_ready")));
        telemetry_.record_request_finish(m);
        log_request(m);
        return;
    }

    InferenceRequest inf_req;
    inf_req.request_id = OpenAiFormatter::generate_id("cmpl-");
    inf_req.model = comp_req.model.empty() ? engine_->model_name() : comp_req.model;
    inf_req.prompt = comp_req.prompt;
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

    int64_t created = std::chrono::duration_cast<std::chrono::seconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();

    if (comp_req.stream) {
        const char sse_header[] =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream; charset=utf-8\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n\r\n";

        if (!send_all(client_fd, sse_header, sizeof(sse_header) - 1)) {
            RequestMetrics m;
            m.method = req.method;
            m.path = req.path;
            m.status_code = 200;
            m.streamed = true;
            m.outcome = "disconnected";
            telemetry_.record_request_finish(m);
            log_request(m);
            return;
        }

        bool client_alive = true;
        auto stream_cb = [&](const TokenOutput& tok) -> bool {
            std::string chunk_data = "data: " + OpenAiFormatter::format_completion_chunk(
                inf_req.request_id, inf_req.model, created, tok.text, "") + "\n\n";
            if (!send_all(client_fd, chunk_data.data(), chunk_data.size())) {
                client_alive = false;
                return false;
            }
            return true;
        };

        GenerationResult gen_res;
        bool ok = engine_->generate_stream(inf_req, stream_cb, gen_res);

        if (client_alive && ok) {
            std::string final_chunk = "data: " + OpenAiFormatter::format_completion_chunk(
                inf_req.request_id, inf_req.model, created, "", gen_res.finish_reason) + "\n\n";
            send_all(client_fd, final_chunk.data(), final_chunk.size());

            const char done_msg[] = "data: [DONE]\n\n";
            send_all(client_fd, done_msg, sizeof(done_msg) - 1);
        }

        auto end_time = std::chrono::steady_clock::now();
        RequestMetrics m;
        m.method = req.method;
        m.path = req.path;
        m.status_code = 200;
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
        m.context_tokens = gen_res.prompt_tokens + gen_res.completion_tokens;
        m.streamed = true;
        m.outcome = client_alive ? "ok" : "disconnected";

        telemetry_.record_request_finish(m);
        log_request(m);
    } else {
        GenerationResult gen_res;
        engine_->generate(inf_req, gen_res);

        std::string json_res = OpenAiFormatter::format_completion(
            inf_req.request_id, inf_req.model, created, gen_res.text,
            gen_res.finish_reason, gen_res.prompt_tokens, gen_res.completion_tokens);

        send_response(client_fd, HttpResponse::json(200, json_res));

        auto end_time = std::chrono::steady_clock::now();
        RequestMetrics m;
        m.method = req.method;
        m.path = req.path;
        m.status_code = 200;
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
        m.context_tokens = gen_res.prompt_tokens + gen_res.completion_tokens;
        m.streamed = false;
        m.outcome = "ok";

        telemetry_.record_request_finish(m);
        log_request(m);
    }
}

void Server::log_request(const RequestMetrics& m) {
    if (options_.quiet) return;

    if (options_.json_telemetry) {
        std::cout << Telemetry::format_jsonl(m) << std::endl;
    } else if (options_.no_tui) {
        std::cout << Telemetry::format_log_line(m) << std::endl;
    } else {
        std::cout << Telemetry::format_live_line(m) << std::endl;
    }
}

} // namespace guild::server
