#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace guild::server {

struct HttpResponse {
    int status_code = 200;
    std::string status_message = "OK";
    std::unordered_map<std::string, std::string> headers;
    std::string body;

    static std::string status_to_message(int code) {
        switch (code) {
            case 200: return "OK";
            case 204: return "No Content";
            case 400: return "Bad Request";
            case 404: return "Not Found";
            case 405: return "Method Not Allowed";
            case 409: return "Conflict";
            case 500: return "Internal Server Error";
            case 502: return "Bad Gateway";
            case 503: return "Service Unavailable";
            default: return "Unknown";
        }
    }

    static HttpResponse json(int code, const std::string& json_str) {
        HttpResponse res;
        res.status_code = code;
        res.status_message = status_to_message(code);
        res.headers["Content-Type"] = "application/json; charset=utf-8";
        res.headers["Content-Length"] = std::to_string(json_str.size());
        res.headers["Access-Control-Allow-Origin"] = "*";
        res.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS";
        res.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization";
        res.headers["Connection"] = "close";
        res.body = json_str;
        return res;
    }

    static HttpResponse text(int code, const std::string& text_str) {
        HttpResponse res;
        res.status_code = code;
        res.status_message = status_to_message(code);
        res.headers["Content-Type"] = "text/plain; charset=utf-8";
        res.headers["Content-Length"] = std::to_string(text_str.size());
        res.headers["Access-Control-Allow-Origin"] = "*";
        res.headers["Connection"] = "close";
        res.body = text_str;
        return res;
    }

    static HttpResponse cors_preflight() {
        HttpResponse res;
        res.status_code = 204;
        res.status_message = "No Content";
        res.headers["Access-Control-Allow-Origin"] = "*";
        res.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS";
        res.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization";
        res.headers["Access-Control-Max-Age"] = "86400";
        res.headers["Content-Length"] = "0";
        res.headers["Connection"] = "close";
        return res;
    }

    std::string serialize() const {
        std::string out = "HTTP/1.1 " + std::to_string(status_code) + " " + status_message + "\r\n";
        for (const auto& kv : headers) {
            out += kv.first + ": " + kv.second + "\r\n";
        }
        out += "\r\n";
        out += body;
        return out;
    }
};

} // namespace guild::server
