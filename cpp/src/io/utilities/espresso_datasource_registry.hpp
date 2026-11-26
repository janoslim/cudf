/*
 * Thread-safe `datasource* -> filepath` registry, shared GPU-backed dataset footer-hash index, and GPU-warmed
 * `FileMetaData` / `GpuHostParsedFooter` cache for unmodified `Velox CudfSplitReader` and `Sirius`
 * (`hybrid_scan_reader` / `hybrid_scan_multifile` / `detail::reader`) SmartSSD and View-Free late materialization.
 *
 * Inputs / Outputs:
 *   - Input : Live `cudf::io::datasource const*` pointers, local POSIX file paths, raw Parquet footer byte spans,
 *             `parquet::FileMetaData`, and `espresso::csd::GpuHostParsedFooter`.
 *   - Output: Canonical `.parquet` file paths resolved via `_espresso_filepath` (`EspressoDatasourceRegistry` or
 *             `EspressoDatasetFooterIndex`), plus GPU-parsed `FileMetaData` and `GpuHostParsedFooter` cache entries
 *             keyed by `(filepath, size)`, `filepath`, and `(footer_len, fnv1a_64)` (`0` host `open`/`pread`/Thrift
 *             re-parse on SmartSSD paths).
 */

#pragma once

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <espresso/csd_gpu_footer.cuh>
#include <espresso/csd_gpu_runtime.hpp>
#include <espresso/cudf_io.hpp>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cudf::io::espresso_detail {

inline constexpr char const* ESPRESSO_FILEPATH_META_KEY = "_espresso_filepath";
inline constexpr std::size_t MAX_CACHED_FOOTERS         = 32768;
inline constexpr std::size_t GPU_FOOTER_BATCH_CHUNK_MAX = 256;

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

/// Resolve symlinked view paths to canonical POSIX paths so `GpuNvmeSubsystem::file_id_of` matches FIEMAP entries.
/// Memoized in a process-wide read-mostly cache to avoid repeated `::realpath` syscalls on hot scan paths.
[[nodiscard]] inline std::string resolve_canonical_local_path(std::string_view raw)
{
  std::string norm = normalize_local_filepath(raw);
  if (norm.empty()) { return norm; }

  struct CanonicalPathCache {
    std::shared_mutex mtx;
    std::unordered_map<std::string, std::string> map;
  };
  // Leaky singleton avoids static-destruction-order hazards across shared libraries at process exit.
  static auto* cache = new CanonicalPathCache();

  {
    std::shared_lock const rlock(cache->mtx);
    if (auto const it = cache->map.find(norm); it != cache->map.end()) { return it->second; }
  }

  char resolved[PATH_MAX];
  std::string canonical = (::realpath(norm.c_str(), resolved) != nullptr) ? std::string(resolved) : norm;

  {
    std::unique_lock const wlock(cache->mtx);
    if (cache->map.size() >= MAX_CACHED_FOOTERS) { cache->map.clear(); }
    cache->map.emplace(norm, canonical);
  }
  return canonical;
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

/// Attach `_espresso_filepath` to `fmd.key_value_metadata` if not already present.
inline void ensure_espresso_metadata_filepath(parquet::FileMetaData& fmd, std::string const& filepath)
{
  if (filepath.empty()) { return; }
  for (auto const& kv : fmd.key_value_metadata) {
    if (kv.key == ESPRESSO_FILEPATH_META_KEY) { return; }
  }
  fmd.key_value_metadata.push_back(parquet::KeyValue{ESPRESSO_FILEPATH_META_KEY, filepath});
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

/// Encode signed 32-bit or 64-bit integer statistics in little-endian Parquet `Statistics` byte format.
[[nodiscard]] inline std::optional<std::vector<std::uint8_t>> encode_le_signed_stat(
  std::int64_t val,
  std::int32_t physical_type)
{
  if (physical_type == static_cast<std::int32_t>(parquet::Type::INT32)) {
    std::int32_t const v32 = static_cast<std::int32_t>(val);
    std::vector<std::uint8_t> bytes(sizeof(std::int32_t));
    std::memcpy(bytes.data(), &v32, sizeof(std::int32_t));
    return bytes;
  }
  if (physical_type == static_cast<std::int32_t>(parquet::Type::INT64)) {
    std::vector<std::uint8_t> bytes(sizeof(std::int64_t));
    std::memcpy(bytes.data(), &val, sizeof(std::int64_t));
    return bytes;
  }
  return std::nullopt;
}

/// Convert a GPU-parsed `GpuHostParsedFooter` into an initialized cuDF `parquet::FileMetaData`
/// (`schema` tree walked, `schema_idx` populated, `_espresso_filepath` attached).
/// Symmetric inverse of `espresso::cudf_io::try_convert_file_metadata` for flat scalar Parquet files.
[[nodiscard]] inline std::optional<parquet::FileMetaData> convert_gpu_footer_to_file_metadata(
  espresso::csd::GpuHostParsedFooter const& gf,
  std::string const& filepath)
{
  if (gf.status != espresso::csd::GPU_FOOTER_OK || gf.num_leaf_cols == 0 ||
      gf.leaves.size() != gf.num_leaf_cols ||
      gf.chunks.size() != static_cast<std::size_t>(gf.num_row_groups) * gf.num_leaf_cols) {
    return std::nullopt;
  }

  std::size_t const num_leaves = gf.num_leaf_cols;
  parquet::FileMetaData fmd{};
  fmd.version  = static_cast<std::int32_t>(gf.version);
  fmd.num_rows = static_cast<std::int64_t>(gf.num_rows);

  // Step 1: Populate flattened schema tree (`schema[0]` root + `schema[1..N]` flat leaves).
  fmd.schema.resize(num_leaves + 1u);
  auto& root                = fmd.schema[0];
  root.name                 = "schema";
  root.type                 = parquet::Type::UNDEFINED;
  root.repetition_type      = parquet::FieldRepetitionType::REQUIRED;
  root.num_children         = static_cast<std::int32_t>(num_leaves);
  root.parent_idx           = 0;
  root.max_definition_level = 0;
  root.max_repetition_level = 0;
  root.children_idx.reserve(num_leaves);

  for (std::size_t c = 0; c < num_leaves; ++c) {
    auto const& leaf       = gf.leaves[c];
    auto& se               = fmd.schema[c + 1u];
    std::size_t const nlen = ::strnlen(leaf.name, espresso::csd::GPU_FOOTER_MAX_NAME_BYTES);
    se.name.assign(leaf.name, nlen);
    se.type                 = static_cast<parquet::Type>(leaf.physical_type);
    se.type_length          = leaf.type_length;
    se.repetition_type      = (leaf.max_definition_level > 0) ? parquet::FieldRepetitionType::OPTIONAL
                                                              : parquet::FieldRepetitionType::REQUIRED;
    se.num_children         = 0;
    se.max_definition_level = static_cast<int>(leaf.max_definition_level);
    se.max_repetition_level = 0;
    se.parent_idx           = 0;
    se.decimal_scale        = leaf.decimal_scale;
    se.decimal_precision    = leaf.decimal_precision;
    if (leaf.has_field_id != 0u) { se.field_id = leaf.field_id; }
    if (leaf.converted_type >= 0) { se.converted_type = static_cast<parquet::ConvertedType>(leaf.converted_type); }

    if (leaf.has_logical_annotation != 0u) {
      switch (leaf.logical_kind) {
        case espresso::csd::GPU_LOGICAL_UTF8:
          se.logical_type = parquet::LogicalType{parquet::LogicalType::STRING};
          break;
        case espresso::csd::GPU_LOGICAL_DECIMAL:
          se.logical_type = parquet::LogicalType{parquet::DecimalType{leaf.decimal_scale, leaf.decimal_precision}};
          break;
        case espresso::csd::GPU_LOGICAL_DATE:
          se.logical_type = parquet::LogicalType{parquet::LogicalType::DATE};
          break;
        case espresso::csd::GPU_LOGICAL_TIMESTAMP: {
          parquet::TimeUnit::Type const unit =
            (leaf.ts_unit == 1u)
              ? parquet::TimeUnit::MILLIS
              : ((leaf.ts_unit == 3u) ? parquet::TimeUnit::NANOS : parquet::TimeUnit::MICROS);
          se.logical_type = parquet::LogicalType{parquet::TimestampType{leaf.ts_is_utc != 0u, parquet::TimeUnit{unit}}};
          break;
        }
        case espresso::csd::GPU_LOGICAL_INTEGER:
          se.logical_type = parquet::LogicalType{parquet::IntType{leaf.int_bit_width, leaf.int_is_signed != 0u}};
          break;
        default: break;
      }
    }
    root.children_idx.push_back(static_cast<cudf::size_type>(c + 1u));
  }

  // Mark standard Parquet type-order sorting so column chunk min/max statistics remain valid for cuDF consumers.
  fmd.column_orders =
    std::vector<parquet::ColumnOrder>(num_leaves, parquet::ColumnOrder{parquet::ColumnOrder::TYPE_ORDER});

  // Step 2: Populate row groups, column chunks, encodings, size statistics, and min/max statistics.
  fmd.row_groups.resize(gf.num_row_groups);
  constexpr std::uint32_t max_enc = static_cast<std::uint32_t>(parquet::Encoding::NUM_ENCODINGS);
  for (std::size_t rg = 0; rg < gf.num_row_groups; ++rg) {
    auto& r           = fmd.row_groups[rg];
    r.num_rows        = static_cast<std::int64_t>(gf.rg_rows[rg]);
    r.total_byte_size = static_cast<std::int64_t>(gf.rg_total_byte_size[rg]);
    r.ordinal         = static_cast<std::int16_t>(rg);
    r.columns.resize(num_leaves);

    std::int64_t rg_min_offset = std::numeric_limits<std::int64_t>::max();
    std::int64_t rg_comp_total = 0;
    for (std::size_t c = 0; c < num_leaves; ++c) {
      auto const& cm = gf.chunks[rg * num_leaves + c];
      auto& cc       = r.columns[c];
      cc.schema_idx  = static_cast<int>(c + 1u);
      cc.file_offset = static_cast<std::int64_t>(cm.file_offset);
      rg_min_offset  = std::min(rg_min_offset, cc.file_offset);
      rg_comp_total += static_cast<std::int64_t>(cm.compressed_bytes);

      auto& md                   = cc.meta_data;
      md.type                    = static_cast<parquet::Type>(cm.physical_type);
      md.path_in_schema          = {fmd.schema[c + 1u].name};
      md.codec                   = static_cast<parquet::Compression>(cm.codec);
      md.num_values              = static_cast<std::int64_t>(cm.num_values);
      md.total_uncompressed_size = static_cast<std::int64_t>(cm.uncompressed_bytes);
      md.total_compressed_size   = static_cast<std::int64_t>(cm.compressed_bytes);
      md.data_page_offset        = cm.data_page_offset;
      md.dictionary_page_offset  = cm.dictionary_page_offset;

      for (std::uint32_t bit = 0; bit < max_enc; ++bit) {
        if ((cm.encodings_mask & (1u << bit)) != 0u) { md.encodings.push_back(static_cast<parquet::Encoding>(bit)); }
      }
      if (cm.pages_total > 0u && cm.data_page_encodings_mask != 0u) {
        std::vector<parquet::Encoding> data_encs;
        for (std::uint32_t bit = 0; bit < max_enc; ++bit) {
          if ((cm.data_page_encodings_mask & (1u << bit)) != 0u) {
            data_encs.push_back(static_cast<parquet::Encoding>(bit));
          }
        }
        if (!data_encs.empty()) {
          std::vector<parquet::PageEncodingStats> estats;
          estats.reserve(data_encs.size());
          // Preserve both `pages_total` sum and every active encoding bit across `try_convert_file_metadata`.
          std::uint32_t remaining = cm.pages_total;
          for (std::size_t ei = 0; ei < data_encs.size(); ++ei) {
            std::uint32_t const extra = static_cast<std::uint32_t>(data_encs.size() - 1u - ei);
            std::int32_t const cnt    = (ei + 1u == data_encs.size())
                                          ? static_cast<std::int32_t>(std::max<std::uint32_t>(1u, remaining))
                                          : static_cast<std::int32_t>(remaining > extra ? 1u : 0u);
            if (cnt > 0) {
              remaining = (remaining >= static_cast<std::uint32_t>(cnt)) ? (remaining - static_cast<std::uint32_t>(cnt)) : 0u;
              estats.push_back(parquet::PageEncodingStats{parquet::PageType::DATA_PAGE, data_encs[ei], cnt});
            }
          }
          if (!estats.empty()) { md.encoding_stats = std::move(estats); }
        }
      }
      if (cm.unencoded_byte_array_len > 0u) {
        parquet::SizeStatistics sz_stats{};
        sz_stats.unencoded_byte_array_data_bytes = static_cast<std::int64_t>(cm.unencoded_byte_array_len);
        md.size_statistics                       = std::move(sz_stats);
      }
      if (cm.has_null_count != 0u) { md.statistics.null_count = cm.null_count; }
      if (cm.has_min_max != 0u) {
        auto lo = encode_le_signed_stat(cm.min_i64, cm.physical_type);
        auto hi = encode_le_signed_stat(cm.max_i64, cm.physical_type);
        if (lo.has_value() && hi.has_value()) {
          md.statistics.min_value = *lo;
          md.statistics.max_value = *hi;
          md.statistics.min       = std::move(*lo);
          md.statistics.max       = std::move(*hi);
        }
      }
    }
    if (rg_min_offset != std::numeric_limits<std::int64_t>::max()) { r.file_offset = rg_min_offset; }
    r.total_compressed_size = rg_comp_total;
  }

  // Step 3: Attach `_espresso_filepath` so `hybrid_scan_reader` recovers the source path without KvikIO handles.
  fmd.key_value_metadata.push_back(parquet::KeyValue{ESPRESSO_FILEPATH_META_KEY, filepath});
  return fmd;
}

/// Thread-safe map tracking live `datasource*` to file paths.
class EspressoDatasourceRegistry {
 public:
  static EspressoDatasourceRegistry& instance()
  {
    static auto* reg = new EspressoDatasourceRegistry();
    return *reg;
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
class EspressoDatasetFooterIndex {
 public:
  static EspressoDatasetFooterIndex& instance()
  {
    static auto* idx = new EspressoDatasetFooterIndex();
    return *idx;
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

/// Cached triple returned by `EspressoFooterCache` so `hybrid_scan_reader(footer_bytes, opts)` skips both CPU Thrift
/// deserialization and `try_convert_file_metadata` conversion in `O(1)`.
struct EspressoCachedFooterBundle {
  std::string filepath;
  parquet::FileMetaData file_metadata;
  std::optional<espresso::csd::GpuHostParsedFooter> gpu_footer;
};

/// Thread-safe cache of `parquet::FileMetaData` (keyed by `(filepath, byte_size)`, `filepath`, and
/// `(footer_len, fnv1a_64)`) and GPU-parsed `GpuHostParsedFooter` (keyed by path), with directory-level GPU NVMe
/// batch warming (`plan_files_on_gpu`).
class EspressoFooterCache {
 public:
  static EspressoFooterCache& instance()
  {
    static auto* cache = new EspressoFooterCache();
    return *cache;
  }

  [[nodiscard]] std::optional<parquet::FileMetaData> lookup(std::string const& filepath,
                                                            std::size_t file_size) const
  {
    std::string const norm = normalize_local_filepath(filepath);
    std::string const key  = make_key(norm, file_size);
    std::lock_guard const lock(_mtx);
    if (auto const it = _entries.find(key); it != _entries.end()) { return it->second; }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<espresso::csd::GpuHostParsedFooter> lookup_gpu_footer(
    std::string const& filepath) const
  {
    std::string const norm = normalize_local_filepath(filepath);
    {
      std::lock_guard const lock(_mtx);
      if (auto const it = _gpu_footers.find(norm); it != _gpu_footers.end()) { return it->second; }
    }
    std::string const real = resolve_canonical_local_path(norm);
    if (real == norm) { return std::nullopt; }
    std::lock_guard const lock(_mtx);
    if (auto const it = _gpu_footers.find(real); it != _gpu_footers.end()) { return it->second; }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<EspressoCachedFooterBundle> lookup_bundle_by_path(
    std::string const& filepath) const
  {
    std::string const norm = normalize_local_filepath(filepath);
    auto find_locked       = [this](std::string const& path_key) -> std::optional<EspressoCachedFooterBundle> {
      auto const fit = _fmd_by_path.find(path_key);
      if (fit == _fmd_by_path.end()) { return std::nullopt; }
      EspressoCachedFooterBundle bundle{path_key, fit->second, std::nullopt};
      if (auto const git = _gpu_footers.find(path_key); git != _gpu_footers.end()) {
        bundle.gpu_footer = git->second;
      }
      return bundle;
    };
    {
      std::lock_guard const lock(_mtx);
      if (auto hit = find_locked(norm); hit.has_value()) { return hit; }
    }
    std::string const real = resolve_canonical_local_path(norm);
    if (real == norm) { return std::nullopt; }
    std::lock_guard const lock(_mtx);
    return find_locked(real);
  }

  [[nodiscard]] std::optional<EspressoCachedFooterBundle> lookup_by_footer_hash(
    std::uint8_t const* data,
    std::size_t len) const
  {
    if (data == nullptr || len == 0 || len > std::numeric_limits<std::uint32_t>::max()) { return std::nullopt; }
    FooterHashKey const key{static_cast<std::uint32_t>(len), hash_parquet_footer_bytes(data, len)};
    std::lock_guard const lock(_mtx);
    auto const it = _by_footer_hash.find(key);
    if (it != _by_footer_hash.end()) { return it->second; }
    return std::nullopt;
  }

  void store(std::string const& filepath,
             std::size_t file_size,
             parquet::FileMetaData const& meta)
  {
    std::string const norm = normalize_local_filepath(filepath);
    std::string const key  = make_key(norm, file_size);
    std::lock_guard const lock(_mtx);
    if (_entries.size() >= MAX_CACHED_FOOTERS) { _entries.clear(); }
    _entries[key] = meta;
    if (_fmd_by_path.size() >= MAX_CACHED_FOOTERS) { _fmd_by_path.clear(); }
    _fmd_by_path[norm] = meta;
  }

  void store_by_footer_hash(std::uint8_t const* data,
                            std::size_t len,
                            std::string const& filepath,
                            parquet::FileMetaData const& meta,
                            std::optional<espresso::csd::GpuHostParsedFooter> const& gpu_footer)
  {
    if (data == nullptr || len == 0 || len > std::numeric_limits<std::uint32_t>::max()) { return; }
    std::string const norm           = normalize_local_filepath(filepath);
    parquet::FileMetaData stored_fmd = meta;
    ensure_espresso_metadata_filepath(stored_fmd, norm);
    FooterHashKey const key{static_cast<std::uint32_t>(len), hash_parquet_footer_bytes(data, len)};

    std::lock_guard const lock(_mtx);
    if (_by_footer_hash.size() >= MAX_CACHED_FOOTERS) { _by_footer_hash.clear(); }
    _by_footer_hash[key] = EspressoCachedFooterBundle{norm, stored_fmd, gpu_footer};
    if (_fmd_by_path.size() >= MAX_CACHED_FOOTERS) { _fmd_by_path.clear(); }
    _fmd_by_path[norm] = std::move(stored_fmd);
    if (gpu_footer.has_value()) {
      if (_gpu_footers.size() >= MAX_CACHED_FOOTERS) { _gpu_footers.clear(); }
      _gpu_footers[norm] = *gpu_footer;
    }
  }

  /// Batch-parse every SmartSSD `.parquet` footer in `sample_filepath`'s parent directory on the GPU
  /// via `GpuRuntime::plan_files_on_gpu` (`<= 256` files/chunk, `0` host `open`/`pread`/Thrift parse).
  /// Mutex-deduplicated so concurrent driver threads warm a table directory exactly once.
  bool warm_directory_on_gpu(std::string const& sample_filepath)
  {
    if (!espresso::csd::GpuRuntime::is_open()) { return false; }

    namespace fs = std::filesystem;
    std::string const norm_sample = normalize_local_filepath(sample_filepath);
    std::string const real_sample = resolve_canonical_local_path(norm_sample);
    if (!espresso::cudf_io::is_smartssd_parquet_path(real_sample)) { return false; }

    fs::path const dir_path   = fs::path(norm_sample).parent_path();
    std::string const dir_key = dir_path.string();

    std::lock_guard const warm_lock(_warm_mtx);
    if (_warmed_dirs.count(dir_key) != 0u) { return true; }

    // Step 1: Enumerate `.parquet` files in the table directory (sorted for deterministic chunking).
    std::vector<std::pair<std::string, std::string>> logical_and_real;
    std::error_code ec;
    if (!dir_path.empty() && fs::is_directory(dir_path, ec)) {
      for (auto const& ent : fs::directory_iterator(dir_path, ec)) {
        std::string const fname = ent.path().filename().string();
        if (fname.empty() || fname.front() == '.' || ent.path().extension() != ".parquet") { continue; }
        std::string const lpath = ent.path().string();
        std::string const rpath = resolve_canonical_local_path(lpath);
        if (espresso::cudf_io::is_smartssd_parquet_path(rpath)) {
          logical_and_real.emplace_back(lpath, rpath);
        }
      }
    }
    if (logical_and_real.empty()) { logical_and_real.emplace_back(norm_sample, real_sample); }
    std::sort(logical_and_real.begin(), logical_and_real.end());

    // Step 2: Run GPU NVMe tail read + Thrift parse in chunks of `<= GPU_FOOTER_BATCH_CHUNK_MAX`.
    auto& gpu_rt = espresso::csd::GpuRuntime::get();
    espresso::csd::GpuStatPruneSpec const no_prune{};
    for (std::size_t off = 0; off < logical_and_real.size(); off += GPU_FOOTER_BATCH_CHUNK_MAX) {
      std::size_t const end = std::min(off + GPU_FOOTER_BATCH_CHUNK_MAX, logical_and_real.size());
      std::vector<std::string> chunk_real;
      chunk_real.reserve(end - off);
      for (std::size_t i = off; i < end; ++i) { chunk_real.push_back(logical_and_real[i].second); }

      auto parsed = gpu_rt.plan_files_on_gpu(chunk_real, no_prune);
      if (parsed.size() != chunk_real.size()) {
        throw std::runtime_error("[espresso::footer_cache] plan_files_on_gpu count mismatch for dir " + dir_key);
      }

      // Step 3: Convert each `GpuHostParsedFooter` -> `FileMetaData` and index by path, `(path, size)`, and hash.
      struct WarmedRow {
        std::string lpath;
        std::string rpath;
        std::size_t fsize;
        parquet::FileMetaData fmd;
        espresso::csd::GpuHostParsedFooter gf;
        std::optional<std::pair<std::uint32_t, std::uint64_t>> fp;
      };
      std::vector<WarmedRow> batch_rows;
      batch_rows.reserve(parsed.size());
      for (std::size_t k = 0; k < parsed.size(); ++k) {
        auto const& lpath = logical_and_real[off + k].first;
        auto const& rpath = logical_and_real[off + k].second;
        auto const& gf    = parsed[k];
        if (gf.status != espresso::csd::GPU_FOOTER_OK) {
          throw std::runtime_error("[espresso::footer_cache] GPU Parquet footer parse failed (status=" +
                                   std::to_string(gf.status) + ") for SmartSSD file: " + rpath);
        }
        std::size_t const fsize = static_cast<std::size_t>(gpu_rt.cached_file_bytes(rpath));
        auto fmd                = convert_gpu_footer_to_file_metadata(gf, lpath);
        if (!fmd.has_value()) {
          throw std::runtime_error("[espresso::footer_cache] Failed to convert GpuHostParsedFooter to FileMetaData: " +
                                   rpath);
        }
        auto fp = espresso::csd::lookup_cached_file_fingerprint(lpath);
        if (!fp.has_value() && rpath != lpath) {
          fp = espresso::csd::lookup_cached_file_fingerprint(rpath);
        }
        batch_rows.push_back(WarmedRow{lpath, rpath, fsize, std::move(*fmd), std::move(parsed[k]), fp});
      }

      std::lock_guard const lock(_mtx);
      if (_entries.size() + batch_rows.size() * 2u >= MAX_CACHED_FOOTERS) { _entries.clear(); }
      if (_gpu_footers.size() + batch_rows.size() * 2u >= MAX_CACHED_FOOTERS) { _gpu_footers.clear(); }
      if (_fmd_by_path.size() + batch_rows.size() * 2u >= MAX_CACHED_FOOTERS) { _fmd_by_path.clear(); }
      if (_by_footer_hash.size() + batch_rows.size() >= MAX_CACHED_FOOTERS) { _by_footer_hash.clear(); }
      for (auto& row : batch_rows) {
        if (row.fp.has_value()) {
          _by_footer_hash[FooterHashKey{row.fp->first, row.fp->second}] =
            EspressoCachedFooterBundle{row.lpath, row.fmd, row.gf};
        }
        _gpu_footers[row.lpath] = row.gf;
        if (row.rpath != row.lpath) { _gpu_footers[row.rpath] = std::move(row.gf); }
        _fmd_by_path[row.lpath] = row.fmd;
        if (row.rpath != row.lpath) { _fmd_by_path[row.rpath] = row.fmd; }
        _entries[make_key(row.lpath, row.fsize)] = row.fmd;
        if (row.rpath != row.lpath) { _entries[make_key(row.rpath, row.fsize)] = std::move(row.fmd); }
      }
    }

    _warmed_dirs.insert(dir_key);
    return true;
  }

 private:
  struct FooterHashKey {
    std::uint32_t footer_len = 0;
    std::uint64_t fnv1a_hash = 0;
    bool operator==(FooterHashKey const& other) const noexcept = default;
  };
  struct FooterHashKeyHasher {
    std::size_t operator()(FooterHashKey const& k) const noexcept
    {
      return static_cast<std::size_t>(k.fnv1a_hash ^ (static_cast<std::uint64_t>(k.footer_len) * 0x9E3779B97F4A7C15ULL));
    }
  };

  [[nodiscard]] static std::string make_key(std::string const& filepath, std::size_t file_size)
  {
    return filepath + "|" + std::to_string(file_size);
  }

  mutable std::mutex _mtx;
  std::mutex _warm_mtx;
  std::unordered_set<std::string> _warmed_dirs;
  std::unordered_map<std::string, parquet::FileMetaData> _entries;
  std::unordered_map<std::string, parquet::FileMetaData> _fmd_by_path;
  std::unordered_map<std::string, espresso::csd::GpuHostParsedFooter> _gpu_footers;
  std::unordered_map<FooterHashKey, EspressoCachedFooterBundle, FooterHashKeyHasher> _by_footer_hash;
};

}  // namespace cudf::io::espresso_detail
