#include "../check.hpp"
#include "guild/models/store.hpp"
#include "guild/models/registry.hpp"

#include <chrono>
#include <csignal>
#include <filesystem>
#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

struct ChildResult { int status; std::string output; };

ChildResult run(const std::string& exe, std::vector<std::string> args, bool stop_when_listening = false) {
    int pipefd[2];
    CHECK(pipe(pipefd) == 0);
    pid_t pid = fork();
    CHECK(pid >= 0);
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        int input = open("/dev/null", O_RDONLY);
        dup2(input, STDIN_FILENO);
        close(input);
        args.insert(args.begin(), exe);
        std::vector<char*> argv;
        for (auto& arg : args) argv.push_back(arg.data());
        argv.push_back(nullptr);
        execv(exe.c_str(), argv.data());
        _exit(127);
    }
    close(pipefd[1]);
    std::string output;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    bool stopped = false;
    for (;;) {
        if (std::chrono::steady_clock::now() >= deadline) {
            kill(pid, SIGKILL);
            int status; waitpid(pid, &status, 0);
            std::fprintf(stderr, "CLI child timed out: %s\n", output.c_str());
            CHECK(false);
        }
        pollfd pfd{pipefd[0], POLLIN, 0};
        if (poll(&pfd, 1, 100) <= 0) continue;
        char buf[4096];
        ssize_t n = read(pipefd[0], buf, sizeof(buf));
        if (n <= 0) break;
        output.append(buf, size_t(n));
        if (stop_when_listening && !stopped && output.find("Guild listening") != std::string::npos) {
            kill(pid, SIGTERM);
            stopped = true;
        }
    }
    close(pipefd[0]);
    int status;
    CHECK(waitpid(pid, &status, 0) == pid);
    CHECK(WIFEXITED(status));
    return {WEXITSTATUS(status), output};
}

int main(int argc, char** argv) {
    CHECK(argc == 2);
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / ("guild-cli-failure-" + std::to_string(getpid()));
    guild::models::ModelStore store({root.string()});
    CHECK(store.init());
    auto manifest = guild::models::ModelRegistry::instance().find_builtin("qwen");
    CHECK(manifest.has_value());
    manifest->name = "missing-assets";
    manifest->aliases.clear();
    for (auto& file : manifest->files) file.local_path = (root / file.name).string();
    manifest->metadata["pack_dir"] = (root / "missing-pack").string();
    CHECK(store.save_manifest(*manifest));
    for (const std::string command : {"run", "serve"}) {
        auto failed = run(argv[1], {command, "missing-assets", "--data-dir", root.string(), "--quiet", "--port", "0"});
        CHECK(failed.status != 0);
        CHECK(failed.output.find("Guild listening") == std::string::npos);
        CHECK(failed.output.find(">>>") == std::string::npos);
        CHECK(failed.output.find("Hello! I am Guild") == std::string::npos);
        auto explicit_mock = run(argv[1], {command, "missing-assets", "--mock", "--data-dir", root.string(),
                                          "--quiet", "--port", "0"}, command == "serve");
        CHECK(explicit_mock.status == 0);
        CHECK(explicit_mock.output.find("explicit --mock") != std::string::npos);
    }
    fs::remove_all(root);
    std::puts("cli_failure_test: missing assets cannot start mock inference implicitly");
}
