#include "guild/models/downloader.hpp"
#include "guild/models/sha256.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <thread>

#if defined(GUILD_USE_WINHTTP)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winhttp.h>
#else
#include <curl/curl.h>
#endif

namespace guild::models {

namespace fs = std::filesystem;

namespace {

#if !defined(GUILD_USE_WINHTTP)
struct CurlProgressContext {
    Downloader* downloader{nullptr};
    const DownloadOptions* options{nullptr};
    DownloadProgress* progress{nullptr};
    uint64_t resume_offset{0};
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point last_report_time;
    curl_off_t last_dlnow{0};
};

int curl_xfer_cb(void* clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t, curl_off_t) {
    auto* ctx = static_cast<CurlProgressContext*>(clientp);
    if (!ctx || !ctx->downloader)
        return 0;

    if (ctx->downloader->is_cancelled()) {
        return 1; // Abort transfer
    }

    auto now = std::chrono::steady_clock::now();
    double elapsed_total = std::chrono::duration<double>(now - ctx->start_time).count();

    ctx->progress->file_downloaded_bytes = ctx->resume_offset + static_cast<uint64_t>(dlnow);
    if (dltotal > 0) {
        ctx->progress->file_total_bytes = ctx->resume_offset + static_cast<uint64_t>(dltotal);
        if (ctx->progress->file_total_bytes > 0) {
            ctx->progress->percentage = (static_cast<double>(ctx->progress->file_downloaded_bytes) /
                                         static_cast<double>(ctx->progress->file_total_bytes)) *
                                        100.0;
        }
    }

    if (elapsed_total > 0.05) {
        ctx->progress->speed_bytes_sec = static_cast<double>(dlnow) / elapsed_total;
    }

    // Call user callback periodically (at least every 100ms or on completion)
    double report_delta = std::chrono::duration<double>(now - ctx->last_report_time).count();
    if (report_delta >= 0.1 || (dltotal > 0 && dlnow >= dltotal)) {
        ctx->last_report_time = now;
        if (ctx->options && ctx->options->progress_cb) {
            if (!ctx->options->progress_cb(*ctx->progress)) {
                return 1; // Aborted by user callback
            }
        }
    }

    return 0;
}
#endif

} // namespace

Downloader::Downloader() {
#if !defined(GUILD_USE_WINHTTP)
    curl_global_init(CURL_GLOBAL_DEFAULT);
#endif
}

Downloader::~Downloader() {
#if !defined(GUILD_USE_WINHTTP)
    curl_global_cleanup();
#endif
}

void Downloader::cancel() {
    cancel_requested_.store(true);
}

std::string Downloader::format_bytes(uint64_t bytes) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1);
    if (bytes < 1024) {
        ss << bytes << " B";
    } else if (bytes < 1024 * 1024) {
        ss << (bytes / 1024.0) << " KiB";
    } else if (bytes < 1024ULL * 1024 * 1024) {
        ss << (bytes / (1024.0 * 1024.0)) << " MiB";
    } else {
        ss << (bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB";
    }
    return ss.str();
}

std::string Downloader::format_speed(double bytes_per_sec) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1);
    if (bytes_per_sec < 1024 * 1024) {
        ss << (bytes_per_sec / 1024.0) << " KiB/s";
    } else if (bytes_per_sec < 1024.0 * 1024 * 1024) {
        ss << (bytes_per_sec / (1024.0 * 1024.0)) << " MiB/s";
    } else {
        ss << (bytes_per_sec / (1024.0 * 1024.0 * 1024.0)) << " GiB/s";
    }
    return ss.str();
}

uint64_t Downloader::get_available_disk_space(const std::string& path) {
    std::error_code ec;
    fs::path p(path);
    if (!fs::exists(p)) {
        p = p.parent_path();
    }
    auto space = fs::space(p, ec);
    if (ec)
        return 0;
    return space.available;
}

bool Downloader::download_file(const std::string& url, const std::string& destination_path, uint64_t expected_size,
                               const std::string& expected_sha256, const DownloadOptions& options,
                               DownloadProgress& progress, std::string& err_msg) {
    fs::path dest(destination_path);
    std::error_code ec;
    fs::create_directories(dest.parent_path(), ec);

    // 1. Check if finalized file already exists
    if (!options.force && fs::exists(dest, ec)) {
        uint64_t current_size = fs::file_size(dest, ec);
        if (expected_size == 0 || current_size == expected_size) {
            if (options.verify_hash && !expected_sha256.empty()) {
                std::string hash = Sha256::hash_file(dest.string());
                if (hash == expected_sha256) {
                    progress.file_downloaded_bytes = current_size;
                    progress.file_total_bytes = current_size;
                    progress.percentage = 100.0;
                    return true;
                }
                // Corrupted: remove and redownload
                fs::remove(dest, ec);
            } else {
                progress.file_downloaded_bytes = current_size;
                progress.file_total_bytes = current_size;
                progress.percentage = 100.0;
                return true;
            }
        } else {
            // Wrong size: remove and redownload
            fs::remove(dest, ec);
        }
    }

    std::string part_path = destination_path + ".part";
    fs::path part(part_path);

    int retries = 0;
    while (retries <= options.max_retries) {
        if (cancel_requested_.load()) {
            err_msg = "Download cancelled by user";
            return false;
        }

        uint64_t resume_offset = 0;
        if (!options.force && fs::exists(part, ec)) {
            resume_offset = fs::file_size(part, ec);
            if (expected_size > 0 && resume_offset > expected_size) {
                // Part file is corrupted / oversize, start fresh
                fs::remove(part, ec);
                resume_offset = 0;
            }
        } else if (options.force && fs::exists(part, ec)) {
            fs::remove(part, ec);
        }

        progress.is_resumed = (resume_offset > 0);
        progress.file_downloaded_bytes = resume_offset;
        if (expected_size > 0)
            progress.file_total_bytes = expected_size;

        FILE* fp = std::fopen(part_path.c_str(), resume_offset > 0 ? "ab" : "wb");
        if (!fp) {
            err_msg = "Cannot open destination file: " + part_path + " (" + std::strerror(errno) + ")";
            return false;
        }

#if defined(GUILD_USE_WINHTTP)
        bool transfer_ok = false;
        if (url.rfind("file://", 0) == 0) {
            std::string src_file = url.substr(7);
            if (src_file.size() >= 3 && src_file[0] == '/' && src_file[2] == ':')
                src_file.erase(0, 1);
            FILE* in_fp = std::fopen(src_file.c_str(), "rb");
            if (in_fp) {
                if (resume_offset > 0)
                    std::fseek(in_fp, static_cast<long>(resume_offset), SEEK_SET);
                char buf[65536];
                size_t n = 0;
                uint64_t dlnow = 0;
                transfer_ok = true;
                while ((n = std::fread(buf, 1, sizeof(buf), in_fp)) > 0) {
                    if (cancel_requested_.load()) {
                        transfer_ok = false;
                        break;
                    }
                    std::fwrite(buf, 1, n, fp);
                    dlnow += n;
                    progress.file_downloaded_bytes = resume_offset + dlnow;
                    if (options.progress_cb && !options.progress_cb(progress)) {
                        transfer_ok = false;
                        break;
                    }
                }
                std::fclose(in_fp);
            } else {
                err_msg = "Cannot open file URL: " + src_file;
            }
        } else {
            int wlen = MultiByteToWideChar(CP_UTF8, 0, url.c_str(), -1, nullptr, 0);
            std::wstring wurl(static_cast<size_t>(std::max(1, wlen)), L'\0');
            MultiByteToWideChar(CP_UTF8, 0, url.c_str(), -1, &wurl[0], wlen);
            URL_COMPONENTS uc{};
            uc.dwStructSize = sizeof(uc);
            wchar_t host[256] = {0};
            wchar_t path[2048] = {0};
            uc.lpszHostName = host;
            uc.dwHostNameLength = 255;
            uc.lpszUrlPath = path;
            uc.dwUrlPathLength = 2047;
            if (WinHttpCrackUrl(wurl.c_str(), 0, 0, &uc)) {
                HINTERNET hSession = WinHttpOpen(L"guild/0.2.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                                 WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
                if (hSession) {
                    HINTERNET hConnect = WinHttpConnect(hSession, host, uc.nPort, 0);
                    if (hConnect) {
                        DWORD flags = (uc.nScheme == INTERNET_SCHEME_HTTPS) ? WINHTTP_FLAG_SECURE : 0;
                        HINTERNET hReq = WinHttpOpenRequest(hConnect, L"GET", path, nullptr, WINHTTP_NO_REFERER,
                                                            WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
                        if (hReq) {
                            if (resume_offset > 0) {
                                std::wstring range_hdr = L"Range: bytes=" + std::to_wstring(resume_offset) + L"-";
                                WinHttpAddRequestHeaders(hReq, range_hdr.c_str(), static_cast<DWORD>(-1L),
                                                         WINHTTP_ADDREQ_FLAG_ADD);
                            }
                            if (WinHttpSendRequest(hReq, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0,
                                                   0, 0) &&
                                WinHttpReceiveResponse(hReq, nullptr)) {
                                DWORD status_code = 0;
                                DWORD sc_size = sizeof(status_code);
                                WinHttpQueryHeaders(hReq, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                                                    WINHTTP_HEADER_NAME_BY_INDEX, &status_code, &sc_size,
                                                    WINHTTP_NO_HEADER_INDEX);
                                if (status_code >= 200 && status_code < 300) {
                                    transfer_ok = true;
                                    DWORD avail = 0;
                                    uint64_t dlnow = 0;
                                    char buf[65536];
                                    while (WinHttpQueryDataAvailable(hReq, &avail) && avail > 0) {
                                        if (cancel_requested_.load()) {
                                            transfer_ok = false;
                                            break;
                                        }
                                        DWORD to_read = std::min<DWORD>(avail, sizeof(buf));
                                        DWORD read_bytes = 0;
                                        if (!WinHttpReadData(hReq, buf, to_read, &read_bytes) || read_bytes == 0)
                                            break;
                                        std::fwrite(buf, 1, read_bytes, fp);
                                        dlnow += read_bytes;
                                        progress.file_downloaded_bytes = resume_offset + dlnow;
                                        if (options.progress_cb && !options.progress_cb(progress)) {
                                            transfer_ok = false;
                                            break;
                                        }
                                    }
                                } else {
                                    err_msg = "HTTP status " + std::to_string(status_code);
                                }
                            } else {
                                err_msg = "WinHTTP request failed";
                            }
                            WinHttpCloseHandle(hReq);
                        }
                        WinHttpCloseHandle(hConnect);
                    }
                    WinHttpCloseHandle(hSession);
                }
            } else {
                err_msg = "Invalid URL: " + url;
            }
        }
        std::fclose(fp);
        const bool ok_res = transfer_ok;
#else
        CURL* curl = curl_easy_init();
        if (!curl) {
            std::fclose(fp);
            err_msg = "Failed to initialize CURL";
            return false;
        }

        CurlProgressContext ctx;
        ctx.downloader = this;
        ctx.options = &options;
        ctx.progress = &progress;
        ctx.resume_offset = resume_offset;
        ctx.start_time = std::chrono::steady_clock::now();
        ctx.last_report_time = ctx.start_time;

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "guild/" GUILD_VERSION);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, fwrite);
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, curl_xfer_cb);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &ctx);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

        if (resume_offset > 0) {
            curl_easy_setopt(curl, CURLOPT_RESUME_FROM_LARGE, static_cast<curl_off_t>(resume_offset));
        }

        CURLcode res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        std::fclose(fp);
        if (res != CURLE_OK) {
            err_msg = curl_easy_strerror(res);
        }
        const bool ok_res = (res == CURLE_OK);
#endif

        if (cancel_requested_.load()) {
            err_msg = "Download cancelled by user";
            return false;
        }

        if (ok_res) {
            // Validate downloaded size
            uint64_t actual_size = fs::file_size(part, ec);
            if (expected_size > 0 && actual_size != expected_size) {
                err_msg = "Download incomplete: expected " + std::to_string(expected_size) + " bytes, got " +
                          std::to_string(actual_size) + " bytes";
                retries++;
                std::this_thread::sleep_for(std::chrono::seconds(1 << retries));
                continue;
            }

            // Validate SHA-256
            if (options.verify_hash && !expected_sha256.empty()) {
                std::string hash = Sha256::hash_file(part_path);
                if (hash != expected_sha256) {
                    err_msg = "SHA-256 hash mismatch for " + dest.filename().string() + ": expected " +
                              expected_sha256 + ", got " + hash;
                    // Hash corrupted: remove part file and retry
                    fs::remove(part, ec);
                    retries++;
                    std::this_thread::sleep_for(std::chrono::seconds(1 << retries));
                    continue;
                }
            }

            // Atomic rename from .part to destination
            fs::rename(part, dest, ec);
            if (ec) {
                err_msg = "Failed to finalize downloaded file: " + ec.message();
                return false;
            }

            progress.file_downloaded_bytes = actual_size;
            progress.file_total_bytes = actual_size;
            progress.percentage = 100.0;
            return true;
        }

        // Handle error & retry
        retries++;
        if (retries <= options.max_retries) {
            std::this_thread::sleep_for(std::chrono::seconds(1 << retries));
        }
    }

    return false;
}

bool Downloader::download_model(const ModelManifest& manifest, ModelStore& store, const DownloadOptions& options,
                                std::string& err_msg) {
    store.init();
    cancel_requested_.store(false);

    // Check disk space
    uint64_t needed = manifest.expected_size_bytes;
    uint64_t free_space = get_available_disk_space(store.models_dir());
    if (free_space > 0 && needed > 0 && free_space < needed) {
        err_msg = "Insufficient disk space: need " + format_bytes(needed) + ", but only " + format_bytes(free_space) +
                  " available";
        return false;
    }

    DownloadProgress progress;
    progress.total_files = static_cast<int>(manifest.files.size());
    progress.total_expected_bytes = manifest.expected_size_bytes;
    progress.total_downloaded_bytes = 0;

    ModelManifest updated = manifest;
    fs::path m_ref_dir = fs::path(store.refs_dir()) / manifest.name;
    std::error_code ec;
    fs::create_directories(m_ref_dir, ec);

    for (size_t i = 0; i < updated.files.size(); ++i) {
        auto& f = updated.files[i];
        progress.current_file_index = static_cast<int>(i + 1);
        progress.current_file_name = f.name;

        // Destination in blobs directory or refs directory
        std::string dest_path = (m_ref_dir / f.name).string();
        if (f.url.empty()) {
            err_msg = "File " + f.name + " has no download URL configured";
            return false;
        }

        bool ok = download_file(f.url, dest_path, f.size_bytes, f.sha256, options, progress, err_msg);
        if (!ok) {
            return false;
        }

        progress.total_downloaded_bytes += f.size_bytes;
        f.local_path = dest_path;
        f.blob_id = f.name;
        store.register_blob(f.blob_id, f.size_bytes, dest_path);
    }

    // Save finalized manifest into store
    if (!store.save_manifest(updated)) {
        err_msg = "Failed to save model manifest into store";
        return false;
    }

    return true;
}

} // namespace guild::models
