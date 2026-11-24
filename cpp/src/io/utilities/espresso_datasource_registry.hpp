/*
 * Thread-safe `datasource* -> filepath` registry and parsed `FileMetaData` cache for unmodified
 * `Velox CudfSplitReader` (`make_datasources` -> `datasource::create(datasource*)` ->
 * `read_parquet_footers` -> `hybrid_scan_multifile`) SmartSSD offload.
 *
 * Inputs / Outputs:
 *   - Input : Live `cudf::io::datasource const*` pointers and local POSIX file paths.
 *   - Output: Canonical local file path attached to `FileMetaData::key_value_metadata` under
 *             `_espresso_filepath`, plus cached `FileMetaData` entries when `$ESPRESSO_CSD_ENABLE=1`.
 */

#pragma once

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet_schema.hpp>

#include <cstddef>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace cudf::io::espresso_detail {

inline constexpr char const* ESPRESSO_FILEPATH_META_KEY = "_espresso_filepath";
inline constexpr std::size_t MAX_CACHED_FOOTERS         = 16384;

/// Strip `file://` or `file:` URI prefixes so downstream `realpath` sees a plain POSIX path.
[[nodiscard]] inline std::string normalize_local_filepath(std::string_view raw)
{
  constexpr std::string_view file_uri_3 = "file:///";
  constexpr std::string_view file_uri_1 = "file:/";
  if (raw.size() >= file_uri_3.size() && raw.substr(0, file_uri_3.size()) == file_uri_3) {
    return std::string(raw.substr(file_uri_3.size() - 1));
  }
  if (raw.size() >= file_uri_1.size() && raw.substr(0, file_uri_1.size()) == file_uri_1) {
    return std::string(raw.substr(file_uri_1.size() - 1));
  }
  return std::string(raw);
}

/// Thread-safe map tracking live `datasource*` (and non-owning `user_datasource_wrapper*`) to file paths.
/// Entries are removed in the `datasource` subclass destructors so pointer reuse can never alias stale paths.
class EspressoDatasourceRegistry {
 public:
  static EspressoDatasourceRegistry& instance()
  {
    static EspressoDatasourceRegistry reg;
    return reg;
  }

  void register_path(cudf::io::datasource const* src, std::string_view raw_path)
  {
    if (src == nullptr || raw_path.empty()) { return; }
    std::string norm = normalize_local_filepath(raw_path);
    std::lock_guard const lock(_mtx);
    _paths[src] = std::move(norm);
  }

  void unregister_path(cudf::io::datasource const* src) noexcept
  {
    if (src == nullptr) { return; }
    std::lock_guard const lock(_mtx);
    _paths.erase(src);
  }

  [[nodiscard]] std::optional<std::string> lookup_path(cudf::io::datasource const* src) const
  {
    if (src == nullptr) { return std::nullopt; }
    std::lock_guard const lock(_mtx);
    auto const it = _paths.find(src);
    if (it == _paths.end()) { return std::nullopt; }
    return it->second;
  }

 private:
  mutable std::mutex _mtx;
  std::unordered_map<cudf::io::datasource const*, std::string> _paths;
};

/// Thread-safe cache of `parquet::FileMetaData` keyed by `(filepath, byte_size)` for repeated splits/queries.
class EspressoFooterCache {
 public:
  static EspressoFooterCache& instance()
  {
    static EspressoFooterCache cache;
    return cache;
  }

  [[nodiscard]] std::optional<parquet::FileMetaData> lookup(std::string const& filepath,
                                                            std::size_t file_size) const
  {
    std::string const key = make_key(filepath, file_size);
    std::lock_guard const lock(_mtx);
    auto const it = _entries.find(key);
    if (it == _entries.end()) { return std::nullopt; }
    return it->second;
  }

  void store(std::string const& filepath,
             std::size_t file_size,
             parquet::FileMetaData const& meta)
  {
    std::string const key = make_key(filepath, file_size);
    std::lock_guard const lock(_mtx);
    if (_entries.size() >= MAX_CACHED_FOOTERS) { _entries.clear(); }
    _entries[key] = meta;
  }

 private:
  [[nodiscard]] static std::string make_key(std::string const& filepath, std::size_t file_size)
  {
    return filepath + "|" + std::to_string(file_size);
  }

  mutable std::mutex _mtx;
  std::unordered_map<std::string, parquet::FileMetaData> _entries;
};

}  // namespace cudf::io::espresso_detail
