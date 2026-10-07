#include "guild/server/engine.hpp"
#include "guild/server/json.hpp"
#include "guild/server/openai.hpp"
#include "guild/server/server.hpp"

#include <arpa/inet.h>
#include "../check.hpp"
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

class FailingEngine final : public guild::server::MockInferenceEngine {
public:
    enum Mode { ReturnFalse, Throw, FalseSuccess };
    FailingEngine(Mode mode, bool partial) : mode_(mode), partial_(partial) {}
    bool generate(const guild::server::InferenceRequest&, guild::server::GenerationResult& r) override {
        return fail(r);
    }
    bool generate_stream(const guild::server::InferenceRequest&, guild::server::StreamCallback cb,
                         guild::server::GenerationResult& r) override {
        if (partial_ && cb) cb({123, "checked prefix", false});
        return fail(r);
    }
private:
    bool fail(guild::server::GenerationResult& r) {
        if (mode_ == Throw) throw std::runtime_error("injected inference failure");
        r.text = "must not become a completion";
        r.completion_tokens = 1;
        r.error_message = "injected inference failure";
        if (mode_ == FalseSuccess) {
            r.finish_reason = "error";
            return true;
        }
        return false;
    }
    Mode mode_;
    bool partial_;
};

struct HttpResponseData {
    int status_code = 0;
    std::string status_message;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
};

HttpResponseData send_http_request(int port,
                                   const std::string& method,
                                   const std::string& path,
                                   const std::string& body = "",
                                   const std::vector<std::pair<std::string, std::string>>& extra_headers = {}) {
    HttpResponseData res;
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        return res;
    }

    struct timeval tv{};
    tv.tv_sec = 10;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    if (connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close(sock);
        return res;
    }

    std::string raw_req = method + " " + path + " HTTP/1.1\r\n";
    raw_req += "Host: 127.0.0.1:" + std::to_string(port) + "\r\n";
    raw_req += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    raw_req += "Connection: close\r\n";
    for (const auto& h : extra_headers) {
        raw_req += h.first + ": " + h.second + "\r\n";
    }
    raw_req += "\r\n";
    raw_req += body;

    send(sock, raw_req.data(), raw_req.size(), 0);

    std::string raw_resp;
    char buf[4096];
    while (true) {
        ssize_t n = recv(sock, buf, sizeof(buf), 0);
        if (n <= 0) break;
        raw_resp.append(buf, static_cast<size_t>(n));
    }
    close(sock);

    size_t header_end = raw_resp.find("\r\n\r\n");
    if (header_end == std::string::npos) return res;

    std::string header_part = raw_resp.substr(0, header_end);
    res.body = raw_resp.substr(header_end + 4);

    size_t line_end = header_part.find("\r\n");
    if (line_end != std::string::npos) {
        std::string status_line = header_part.substr(0, line_end);
        size_t s1 = status_line.find(' ');
        if (s1 != std::string::npos) {
            size_t s2 = status_line.find(' ', s1 + 1);
            std::string code_str = status_line.substr(s1 + 1, s2 == std::string::npos ? std::string::npos : s2 - s1 - 1);
            res.status_code = std::atoi(code_str.c_str());
            if (s2 != std::string::npos) {
                res.status_message = status_line.substr(s2 + 1);
            }
        }
    }

    return res;
}

} // namespace

int main() {
    signal(SIGPIPE, SIG_IGN);
    std::cout << "[server_test] Starting native HTTP server test suite...\n";

    auto mock_engine = std::make_shared<guild::server::MockInferenceEngine>(
        "Qwen3.8-Flash-Next", 262144, "Hello from the native Guild engine!");

    guild::server::ServerOptions opts;
    opts.host = "127.0.0.1";
    opts.port = 0; // Ephemeral port
    opts.quiet = true;

    auto server = std::make_unique<guild::server::Server>(mock_engine, opts);
    bool started = server->start();
    CHECK(started && "Server failed to start on ephemeral port");
    (void) started;

    int port = server->bound_port();
    CHECK(port > 0 && "Bound port must be > 0");
    std::cout << "[server_test] Server listening on ephemeral port " << port << "\n";

    // ------------------------------------------------------------------------
    // Test 1: GET /health returns 200
    // ------------------------------------------------------------------------
    {
        std::cout << "[server_test] Test 1: GET /health\n";
        auto resp = send_http_request(port, "GET", "/health");
        CHECK(resp.status_code == 200);

        guild::server::json::JsonValue root;
        std::string err;
        CHECK(guild::server::json::JsonValue::parse(resp.body, root, err));
        CHECK(root.is_object());
        CHECK(root["status"].as_string() == "ok");
        CHECK(root["model"].as_string() == "Qwen3.8-Flash-Next");
        CHECK(root["max_context"].as_int() == 262144);
        std::cout << "  -> PASSED (status: 200, status=ok)\n";
    }

    // ------------------------------------------------------------------------
    // Test 2: GET /v1/models returns current model
    // ------------------------------------------------------------------------
    {
        std::cout << "[server_test] Test 2: GET /v1/models\n";
        auto resp = send_http_request(port, "GET", "/v1/models");
        CHECK(resp.status_code == 200);

        guild::server::json::JsonValue root;
        std::string err;
        CHECK(guild::server::json::JsonValue::parse(resp.body, root, err));
        CHECK(root.is_object());
        CHECK(root["object"].as_string() == "list");
        CHECK(root["data"].is_array());
        CHECK(root["data"].size() >= 1);
        CHECK(root["data"][0]["id"].as_string() == "Qwen3.8-Flash-Next");
        std::cout << "  -> PASSED (status: 200, model found)\n";
    }

    // ------------------------------------------------------------------------
    // Test 3: POST /v1/chat/completions non-stream
    // ------------------------------------------------------------------------
    {
        std::cout << "[server_test] Test 3: POST /v1/chat/completions non-stream\n";
        std::string req_json = "{\"model\":\"Qwen3.8-Flash-Next\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"temperature\":0.7,\"stream\":false}";
        auto resp = send_http_request(port, "POST", "/v1/chat/completions", req_json, {{"Content-Type", "application/json"}});
        CHECK(resp.status_code == 200);

        guild::server::json::JsonValue root;
        std::string err;
        CHECK(guild::server::json::JsonValue::parse(resp.body, root, err));
        CHECK(root.is_object());
        CHECK(root["object"].as_string() == "chat.completion");
        CHECK(root["choices"].is_array());
        CHECK(root["choices"].size() >= 1);
        CHECK(root["choices"][0]["message"]["role"].as_string() == "assistant");
        CHECK(!root["choices"][0]["message"]["content"].as_string().empty());
        CHECK(root["choices"][0]["finish_reason"].as_string() == "stop");
        CHECK(root["usage"]["prompt_tokens"].as_int() > 0);
        CHECK(root["usage"]["completion_tokens"].as_int() > 0);
        std::cout << "  -> PASSED (status: 200, valid choices, usage reported)\n";
    }

    // ------------------------------------------------------------------------
    // Test 4: POST /v1/chat/completions streaming SSE
    // ------------------------------------------------------------------------
    {
        std::cout << "[server_test] Test 4: POST /v1/chat/completions streaming SSE\n";
        std::string req_json = "{\"model\":\"Qwen3.8-Flash-Next\",\"messages\":[{\"role\":\"user\",\"content\":\"Tell me a joke\"}],\"stream\":true}";
        auto resp = send_http_request(port, "POST", "/v1/chat/completions", req_json, {{"Content-Type", "application/json"}});
        CHECK(resp.status_code == 200);
        CHECK(resp.body.find("data: ") != std::string::npos);
        CHECK(resp.body.find("chat.completion.chunk") != std::string::npos);
        CHECK(resp.body.find("data: [DONE]") != std::string::npos);
        std::cout << "  -> PASSED (status: 200, received SSE chunks and [DONE])\n";
    }

    // ------------------------------------------------------------------------
    // Test 5: Malformed JSON returns 400
    // ------------------------------------------------------------------------
    {
        std::cout << "[server_test] Test 5: Malformed JSON returns 400\n";
        std::string req_json = "{\"model\": \"invalid json ... missing brace";
        auto resp = send_http_request(port, "POST", "/v1/chat/completions", req_json, {{"Content-Type", "application/json"}});
        CHECK(resp.status_code == 400);

        guild::server::json::JsonValue root;
        std::string err;
        CHECK(guild::server::json::JsonValue::parse(resp.body, root, err));
        CHECK(root.is_object());
        CHECK(root["error"].is_object());
        CHECK(root["error"]["type"].as_string() == "invalid_request_error");
        CHECK(root["error"]["code"].as_string() == "parse_error");
        std::cout << "  -> PASSED (status: 400, structured parse_error)\n";
    }

    // ------------------------------------------------------------------------
    // Test 6: Unsupported parameters return structured 400
    // ------------------------------------------------------------------------
    {
        std::cout << "[server_test] Test 6: Unsupported parameters return structured 400\n";
        std::string req_json = "{\"model\":\"Qwen3.8-Flash-Next\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"unknown_experimental_feature\":true}";
        auto resp = send_http_request(port, "POST", "/v1/chat/completions", req_json, {{"Content-Type", "application/json"}});
        CHECK(resp.status_code == 400);

        guild::server::json::JsonValue root;
        std::string err;
        CHECK(guild::server::json::JsonValue::parse(resp.body, root, err));
        CHECK(root.is_object());
        CHECK(root["error"].is_object());
        CHECK(root["error"]["type"].as_string() == "invalid_request_error");
        CHECK(root["error"]["code"].as_string() == "unsupported_parameter");
        CHECK(root["error"]["param"].as_string() == "unknown_experimental_feature");
        std::cout << "  -> PASSED (status: 400, unsupported_parameter reported)\n";
    }

    // ------------------------------------------------------------------------
    // Test 7: Concurrent requests behave correctly
    // ------------------------------------------------------------------------
    {
        std::cout << "[server_test] Test 7: Concurrent requests\n";
        const int num_clients = 6;
        std::vector<std::thread> threads;
        std::vector<int> status_codes(num_clients, 0);

        for (int i = 0; i < num_clients; ++i) {
            threads.emplace_back([port, i, &status_codes]() {
                std::string req_json = "{\"model\":\"Qwen3.8-Flash-Next\",\"messages\":[{\"role\":\"user\",\"content\":\"Concurrent test\"}]}";
                auto resp = send_http_request(port, "POST", "/v1/chat/completions", req_json, {{"Content-Type", "application/json"}});
                status_codes[i] = resp.status_code;
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        for (int i = 0; i < num_clients; ++i) {
            CHECK(status_codes[i] == 200);
        }
        std::cout << "  -> PASSED (6 concurrent requests succeeded with 200)\n";
    }

    // ------------------------------------------------------------------------
    // Test 8: Client disconnect during stream does not crash server
    // ------------------------------------------------------------------------
    {
        std::cout << "[server_test] Test 8: Client disconnect during stream\n";
        // Configure artificial delay to ensure request is streaming when client drops
        mock_engine->set_delay_ms(10);

        int sock = socket(AF_INET, SOCK_STREAM, 0);
        CHECK(sock >= 0);

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(port));
        inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

        CHECK(connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);

        std::string req = "POST /v1/chat/completions HTTP/1.1\r\n"
                          "Host: 127.0.0.1:" + std::to_string(port) + "\r\n"
                          "Content-Type: application/json\r\n"
                          "Content-Length: 95\r\n"
                          "Connection: close\r\n\r\n"
                          "{\"model\":\"Qwen3.8-Flash-Next\",\"messages\":[{\"role\":\"user\",\"content\":\"Long stream\"}],\"stream\":true}";

        send(sock, req.data(), req.size(), 0);

        // Read only first chunk
        char buf[128];
        recv(sock, buf, sizeof(buf), 0);

        // Disconnect abruptly
        close(sock);

        // Reset delay
        mock_engine->set_delay_ms(0);

        // Wait a moment for server worker to handle EPIPE
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // Verify server is alive and functioning normally
        auto health_resp = send_http_request(port, "GET", "/health");
        CHECK(health_resp.status_code == 200);
        std::cout << "  -> PASSED (client disconnect handled cleanly, server healthy)\n";
    }

    // ------------------------------------------------------------------------
    // Test 9: Clean SIGINT / stop shutdown
    // ------------------------------------------------------------------------
    {
        std::cout << "[server_test] Test 9: Clean stop shutdown\n";
        CHECK(server->is_running());
        server->stop();
        CHECK(!server->is_running());

        // Connect attempt should fail
        auto resp = send_http_request(port, "GET", "/health");
        CHECK(resp.status_code == 0 && "Connection should fail after server.stop()");
        std::cout << "  -> PASSED (server stopped cleanly, socket closed)\n";
    }

    // ------------------------------------------------------------------------
    // Test 10: Port-already-in-use fails cleanly
    // ------------------------------------------------------------------------
    {
        std::cout << "[server_test] Test 10: Port-already-in-use fails cleanly\n";

        // Bind a dummy socket
        int blocker_fd = socket(AF_INET, SOCK_STREAM, 0);
        CHECK(blocker_fd >= 0);
        int opt = 1;
        setsockopt(blocker_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = 0; // ephemeral
        inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        CHECK(bind(blocker_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
        CHECK(listen(blocker_fd, 1) == 0);

        socklen_t len = sizeof(addr);
        getsockname(blocker_fd, reinterpret_cast<sockaddr*>(&addr), &len);
        int blocked_port = ntohs(addr.sin_port);

        guild::server::ServerOptions bad_opts;
        bad_opts.host = "127.0.0.1";
        bad_opts.port = blocked_port;
        bad_opts.quiet = true;

        guild::server::Server conflict_server(mock_engine, bad_opts);
        bool started_conflict = conflict_server.start();
        CHECK(!started_conflict && "Server.start() must return false when port is occupied");
        (void) started_conflict;
        CHECK(!conflict_server.is_running());

        close(blocker_fd);
        std::cout << "  -> PASSED (port conflict detected and refused without crashing)\n";
    }

    // Both endpoint families must handle false returns, exceptions, and an
    // inconsistent engine reporting true with finish_reason=error.
    for (auto mode : {FailingEngine::ReturnFalse, FailingEngine::Throw, FailingEngine::FalseSuccess}) {
        for (bool partial : {false, true}) {
            auto failing = std::make_shared<FailingEngine>(mode, partial);
            guild::server::Server failing_server(failing, opts);
            CHECK(failing_server.start());
            for (bool chat : {false, true}) {
                for (bool stream : {false, true}) {
                    const std::string path = chat ? "/v1/chat/completions" : "/v1/completions";
                    const std::string body = std::string(chat ? "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],"
                                                             : "{\"prompt\":\"Hi\",") +
                                             "\"stream\":" + (stream ? "true}" : "false}");
                    auto resp = send_http_request(failing_server.bound_port(), "POST", path, body);
                    CHECK(resp.status_code == (stream && partial ? 200 : 500));
                    CHECK(resp.body.find("injected inference failure") != std::string::npos);
                    CHECK(resp.body.find("inference_failed") != std::string::npos);
                    CHECK(resp.body.find("data: [DONE]") == std::string::npos);
                    CHECK(resp.body.find("\"finish_reason\":\"stop\"") == std::string::npos);
                    CHECK(resp.body.find("must not become a completion") == std::string::npos);
                    if (!stream || !partial) {
                        guild::server::json::JsonValue root;
                        std::string err;
                        CHECK(guild::server::json::JsonValue::parse(resp.body, root, err));
                        CHECK(root["error"]["code"].as_string() == "inference_failed");
                        CHECK(!root.contains("choices"));
                    }
                }
            }
            CHECK(failing_server.telemetry().snapshot().active_requests == 0);
            CHECK(failing_server.telemetry().snapshot().completed_requests == 4);
            failing_server.stop();
        }
    }

    std::cout << "\n[server_test] lifecycle, success and failure checks PASSED (Release-active)\n";
    return 0;
}
