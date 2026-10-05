#pragma once

#include "guild/server/engine.hpp"
#include "guild/server/request.hpp"
#include "guild/server/response.hpp"
#include "guild/server/telemetry.hpp"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace guild::server {

struct ServerOptions {
    std::string host = "127.0.0.1";
    int port = 11434;
    bool verbose = false;
    bool quiet = false;
    bool json_telemetry = false;
    bool no_tui = false;
    int worker_threads = 4;
};

class Server {
public:
    Server(std::shared_ptr<IInferenceEngine> engine, ServerOptions options);
    ~Server();

    // Starts listening. Returns false on failure (e.g. port already in use).
    bool start();

    // Runs until stop() is called or signal received
    void run();

    // Stops server and cleans up resources
    void stop();

    // Async-signal-safe request to stop
    void request_stop() { running_.store(false); }

    bool is_running() const { return running_.load(); }
    int bound_port() const { return bound_port_; }
    const ServerOptions& options() const { return options_; }
    Telemetry& telemetry() { return telemetry_; }

private:
    std::shared_ptr<IInferenceEngine> engine_;
    ServerOptions options_;
    Telemetry telemetry_;

    std::atomic<bool> running_{false};
    std::atomic<bool> stopped_{false};
    int listen_fd_{-1};
    int bound_port_{0};

    std::thread listener_thread_;
    std::vector<std::thread> workers_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::vector<int> client_queue_;

    void listener_loop();
    void worker_loop();
    void handle_client(int client_fd);

    void process_chat_completions(int client_fd, const HttpRequest& req);
    void process_completions(int client_fd, const HttpRequest& req);
    void log_request(const RequestMetrics& m);
};

} // namespace guild::server
