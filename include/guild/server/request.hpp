#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <unordered_map>
#include <vector>

namespace guild::server {

constexpr size_t MAX_HEADER_SIZE = 65536;       // 64 KiB
constexpr size_t MAX_HEADERS_COUNT = 100;       // Max headers
constexpr size_t MAX_BODY_SIZE = 33554432;      // 32 MiB

struct HttpRequest {
    std::string method;
    std::string path;
    std::string raw_query;
    std::unordered_map<std::string, std::string> query_params;
    std::unordered_map<std::string, std::string> headers;
    std::string body;

    std::string get_header(const std::string& name) const {
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        auto it = headers.find(lower);
        if (it != headers.end()) return it->second;
        return "";
    }

    size_t content_length() const {
        std::string cl = get_header("content-length");
        if (cl.empty()) return 0;
        for (char c : cl) {
            if (!std::isdigit(static_cast<unsigned char>(c))) return 0;
        }
        try {
            unsigned long len = std::stoul(cl);
            if (len > MAX_BODY_SIZE) return 0;
            return static_cast<size_t>(len);
        } catch (...) {
            return 0;
        }
    }

    static bool parse_headers(const std::string& raw_header_text, HttpRequest& req) {
        if (raw_header_text.size() > MAX_HEADER_SIZE) return false;
        size_t pos = 0;
        size_t line_end = raw_header_text.find("\r\n", pos);
        if (line_end == std::string::npos) return false;

        std::string request_line = raw_header_text.substr(pos, line_end - pos);
        pos = line_end + 2;

        // Parse method, target, version
        size_t s1 = request_line.find(' ');
        if (s1 == std::string::npos) return false;
        size_t s2 = request_line.find(' ', s1 + 1);
        if (s2 == std::string::npos) return false;

        req.method = request_line.substr(0, s1);
        std::string target = request_line.substr(s1 + 1, s2 - s1 - 1);

        size_t qpos = target.find('?');
        if (qpos != std::string::npos) {
            req.path = target.substr(0, qpos);
            req.raw_query = target.substr(qpos + 1);
            // Parse query params
            size_t qp = 0;
            while (qp < req.raw_query.size()) {
                size_t amp = req.raw_query.find('&', qp);
                std::string pair = req.raw_query.substr(qp, (amp == std::string::npos) ? std::string::npos : amp - qp);
                size_t eq = pair.find('=');
                if (eq != std::string::npos) {
                    req.query_params[pair.substr(0, eq)] = pair.substr(eq + 1);
                } else if (!pair.empty()) {
                    req.query_params[pair] = "";
                }
                if (amp == std::string::npos) break;
                qp = amp + 1;
            }
        } else {
            req.path = target;
        }

        // Headers
        while (pos < raw_header_text.size()) {
            line_end = raw_header_text.find("\r\n", pos);
            if (line_end == std::string::npos) line_end = raw_header_text.size();
            std::string header_line = raw_header_text.substr(pos, line_end - pos);
            pos = line_end + 2;

            if (header_line.empty()) break;
            size_t colon = header_line.find(':');
            if (colon != std::string::npos) {
                std::string key = header_line.substr(0, colon);
                std::string val = header_line.substr(colon + 1);
                // trim
                while (!val.empty() && (val.front() == ' ' || val.front() == '\t')) val.erase(val.begin());
                while (!val.empty() && (val.back() == ' ' || val.back() == '\t')) val.pop_back();

                std::transform(key.begin(), key.end(), key.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                req.headers[key] = val;
                if (req.headers.size() > MAX_HEADERS_COUNT) return false;
            }
        }

        return true;
    }
};

} // namespace guild::server
