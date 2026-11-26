/*
 * Thread-safe `datasource* -> filepath` registry, shared GPU-backed dataset footer-hash index, and footer cache
 * for unmodified `Velox CudfSplitReader` and `Sirius` (`hybrid_scan_reader` / `hybrid_scan_multifile` /
 * `detail::reader`) SmartSSD and View-Free self-join late-materialization offload.
 *
 * Inputs / Outputs:
 *   - Input : Live `cudf::io::datasource const*` pointers, local POSIX file paths, raw Parquet footer
 *             byte spans, and `parquet::FileMetaData`.
 *   - Output: Canonical `.parquet` file path resolved via `_espresso_filepath` (`EspressoDatasourceRegistry`
 *             or `EspressoDatasetFooterIndex` backed by Espresso's process-wide GPU NVMe fingerprint cache over
 *             `$ESPRESSO_CSD_VIEW_ROOT` / `$ESPRESSO_DATASET_ROOT`), plus cached `FileMetaData` entries.
 */

#pragma once

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <espresso/csd_gpu_footer.cuh>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

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

/// Compute 64-bit FNV-1a hash mixed with `len` (delegates to Espresso's CPU/GPU bit-identical helper).
[[nodiscard]] inline std::uint64_t hash_parquet_footer_bytes(std::uint8_t const* data, std::size_t len) noexcept
{
  return espresso::csd::hash_parquet_footer_bytes(data, len);
}

/// Normalize `parquet::KeyValue::value` across cuDF 26.08 (`std::string`) and cuDF 26.12 (`std::optional<std::string>`).
[[nodiscard]] inline std::string_view kv_value_view(std::string const& value_str) noexcept
{
  return value_str;
}

template <typename OptionalString>
[[nodiscard]] inline std::string_view kv_value_view(OptionalString const& opt_value) noexcept
{
  return opt_value.has_value() ? std::string_view{*opt_value} : std::string_view{};
}

/// Extract `_espresso_filepath` from `FileMetaData::key_value_metadata`.
[[nodiscard]] inline std::string extract_espresso_metadata_filepath(parquet::FileMetaData const& fmd)
{
  for (auto const& kv : fmd.key_value_metadata) {
    if (kv.key == ESPRESSO_FILEPATH_META_KEY) {
      std::string_view const val = kv_value_view(kv.value);
      if (!val.empty()) { return normalize_local_filepath(val); }
    }
  }
  return {};
}

/// Reconstruct `parquet_reader_options` with `source_info{paths}` when `options.get_source().filepaths()` is empty.
[[nodiscard]] inline std::optional<parquet_reader_options> try_options_with_metadata_filepaths(
  parquet_reader_options const& options,
  std::span<parquet::FileMetaData const> metadatas)
{
  if (!options.get_source().filepaths().empty() || metadatas.empty()) { return std::nullopt; }
  std::vector<std::string> paths;
  paths.reserve(metadatas.size());
  for (auto const& fmd : metadatas) {
    std::string p = extract_espresso_metadata_filepath(fmd);
    if (p.empty()) { return std::nullopt; }
    paths.push_back(std::move(p));
  }
  parquet_reader_options out = options;
  out.set_source(source_info{paths});
  return out;
}

/// Thread-safe map tracking live `datasource*` to file paths.
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

/// Shared `(footer_len, fnv1a_64(footer_bytes)) -> filepath` index over `$ESPRESSO_CSD_VIEW_ROOT` /
/// `$ESPRESSO_DATASET_ROOT`, backed by Espresso's process-wide fingerprint cache (`espresso::csd`).
/// Pre-warmed via 8-SmartSSD GPU NVMe inside `espresso_csd_eager_open_from_env()` (`0 ms` in Query 1, `0` host
/// `pread`s), with 16-thread parallel POSIX fallback when `GpuRuntime` is not yet open.
class EspressoDatasetFooterIndex {
 public:
  static EspressoDatasetFooterIndex& instance()
  {
    static EspressoDatasetFooterIndex idx;
    return idx;
  }

  [[nodiscard]] std::optional<std::string> lookup_by_footer_bytes(std::uint8_t const* data,
                                                                  std::size_t len)
  {
    return espresso::csd::lookup_footer_fingerprint_path(data, len);
  }

  [[nodiscard]] std::optional<std::string> lookup_datasource_tail(cudf::io::datasource* src)
  {
    if (src == nullptr) { return std::nullopt; }
    if (espresso::csd::warm_footer_fingerprints_from_env() == 0u) { return std::nullopt; }
    std::size_t const file_sz = src->size();
    if (file_sz < 12) { return std::nullopt; }
    auto tail8 = src->host_read(file_sz - 8, 8);
    if (!tail8 || tail8->size() < 8) { return std::nullopt; }
    if (std::memcmp(tail8->data() + 4, "PAR1", 4) != 0) { return std::nullopt; }
    std::uint32_t flen = 0;
    std::memcpy(&flen, tail8->data(), sizeof(std::uint32_t));
    if (flen == 0 || std::size_t(flen) + 12 > file_sz) { return std::nullopt; }
    auto fbuf = src->host_read(file_sz - 8 - flen, flen);
    if (!fbuf || fbuf->size() != flen) { return std::nullopt; }
    return lookup_by_footer_bytes(fbuf->data(), flen);
  }
};

/// Thread-safe cache of `parquet::FileMetaData` keyed by `(filepath, byte_size)`.
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
