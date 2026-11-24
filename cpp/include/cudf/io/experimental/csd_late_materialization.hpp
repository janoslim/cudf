/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Espresso Authors & NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Public experimental cuDF API for SmartSSD computational storage Parquet scans and two-pass
 * GPU-bitmap late materialization (`espresso::cudf_io` / `espresso::late_mat`).
 *
 * Inputs / Outputs:
 *   - Inputs : `cudf::io::parquet_reader_options`, `cudf::column_view` of 64-bit physical rowids,
 *              `cuda::stream_ref`, and `cudf::memory_resources`.
 *   - Outputs: `cudf::io::table_with_metadata` and `late_materialization_bridge_stats`.
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace CUDF_EXPORT cudf {
namespace io::experimental {

/// Positive 8-bit magic tag (`0x5E` = `0101_1110b`) for zero-engine-change INT64 `_espresso_rowid` join keys.
inline constexpr std::uint8_t ESPRESSO_ROWID_MAGIC = 0x5Eu;
inline constexpr char const* ESPRESSO_VIRTUAL_ROWID_COLUMN = "_espresso_rowid";

/// Pack an untagged 64-bit physical SmartSSD row address (`UINT64`):
/// `(card_idx << 56) | (file_idx << 40) | (rg_idx << 32) | row_in_rg`.
[[nodiscard]] constexpr std::uint64_t pack_csd_rowid(std::uint32_t card_idx,
                                                     std::uint32_t file_idx,
                                                     std::uint32_t rg_idx,
                                                     std::uint32_t row_in_rg) noexcept
{
  return (std::uint64_t(card_idx & 0xFFu) << 56) | (std::uint64_t(file_idx & 0xFFFFu) << 40) |
         (std::uint64_t(rg_idx & 0xFFu) << 32) | std::uint64_t(row_in_rg);
}

/// Pack a magic-tagged positive signed `INT64` `_espresso_rowid` (`8b 0x5E | 3b card | 13b file | 8b rg | 32b row`).
[[nodiscard]] constexpr std::int64_t pack_espresso_rowid(std::uint32_t card_idx,
                                                         std::uint32_t file_table_idx,
                                                         std::uint32_t rg_idx,
                                                         std::uint32_t row_in_rg) noexcept
{
  std::uint64_t const u = (std::uint64_t(ESPRESSO_ROWID_MAGIC) << 56) |
                          (std::uint64_t(card_idx & 0x7u) << 53) |
                          (std::uint64_t(file_table_idx & 0x1FFFu) << 40) |
                          (std::uint64_t(rg_idx & 0xFFu) << 32) |
                          std::uint64_t(row_in_rg);
  return static_cast<std::int64_t>(u);
}

[[nodiscard]] constexpr bool is_espresso_magic_rowid(std::uint64_t raw) noexcept
{
  return ((raw >> 56) & 0xFFULL) == ESPRESSO_ROWID_MAGIC;
}

/// Options controlling Pass-1 SmartSSD Parquet scans that emit 64-bit physical row addresses.
struct csd_scan_options {
  bool emit_rowids                  = true;
  bool magic_tag_rowid              = false;
  bool prepend_rowid_column         = false;
  std::string rowid_column_name     = "_csd_rowid";
  bool strict_csd                   = true;
};

/// Options controlling Pass-2 post-join payload materialization driven by 64-bit physical `rowids`.
struct csd_late_materialization_options {
  bool rowids_are_sorted_unique = true;
  bool emit_rowids              = false;
  std::string rowid_column_name = "_csd_rowid";
};

/// Telemetry snapshot of the virtual-table `cudf::hash_join -> cudf::io::read_parquet` survival-bitmap bridge.
struct late_materialization_bridge_stats {
  std::string table_key;
  std::size_t total_table_row_groups   = 0;
  std::size_t touched_row_groups       = 0;
  std::size_t skipped_zero_row_groups  = 0;
  std::uint64_t unique_surviving_rows  = 0;
  std::size_t bitmap_bytes             = 0;
  std::size_t max_single_rg_bitmap_b   = 0;
  std::uint64_t payload_scans_executed = 0;
};

/// True when `libcudf.so` was compiled with `-DCUDF_USE_ESPRESSO_CSD=ON`.
[[nodiscard]] bool is_espresso_csd_compiled() noexcept;

/// Pass-1 SmartSSD Parquet scan that emits 64-bit physical `rowid`s alongside projected filter/join-key columns.
[[nodiscard]] table_with_metadata read_parquet_with_csd_rowids(
  parquet_reader_options const& options,
  csd_scan_options const& csd_options = {},
  cuda::stream_ref stream             = cudf::get_default_stream(),
  cudf::memory_resources mr           = cudf::get_current_device_resource_ref());

/// Pass-2 post-join payload scan: builds `<= 128 KiB` GPU row-group survival bitmaps from `rowids`, skips `0`-survivor
/// row groups, and scans only touched row groups in `BITMAP_LOAD = 1` mode.
[[nodiscard]] table_with_metadata read_parquet_by_rowids(
  parquet_reader_options const& options,
  column_view const& rowids,
  csd_late_materialization_options const& late_options = {},
  cuda::stream_ref stream                              = cudf::get_default_stream(),
  cudf::memory_resources mr                            = cudf::get_current_device_resource_ref());

/// Chunked Pass-1 SmartSSD Parquet reader with optional 64-bit physical `rowid` emission.
class csd_chunked_parquet_reader {
 public:
  csd_chunked_parquet_reader(
    std::size_t chunk_read_limit,
    std::size_t pass_read_limit,
    parquet_reader_options const& options,
    csd_scan_options const& csd_options = {},
    cuda::stream_ref stream             = cudf::get_default_stream(),
    cudf::memory_resources mr           = cudf::get_current_device_resource_ref());

  ~csd_chunked_parquet_reader();
  csd_chunked_parquet_reader(csd_chunked_parquet_reader const&)            = delete;
  csd_chunked_parquet_reader& operator=(csd_chunked_parquet_reader const&) = delete;

  [[nodiscard]] bool has_next() const;
  [[nodiscard]] table_with_metadata read_chunk() const;

 private:
  struct impl;
  std::unique_ptr<impl> _impl;
};

/// Inspect survival-bitmap statistics from the most recent `cudf::hash_join` build on `_espresso_rowid`.
[[nodiscard]] std::optional<late_materialization_bridge_stats> latest_late_materialization_bridge_stats(
  std::string_view table_key = "");

/// Clear cached virtual-table catalogs and survival bitmaps in `LateMatBridgeRegistry`.
void clear_late_materialization_bridge();

}  // namespace io::experimental

namespace io::parquet::experimental {
using cudf::io::experimental::clear_late_materialization_bridge;
using cudf::io::experimental::csd_chunked_parquet_reader;
using cudf::io::experimental::csd_late_materialization_options;
using cudf::io::experimental::csd_scan_options;
using cudf::io::experimental::ESPRESSO_ROWID_MAGIC;
using cudf::io::experimental::ESPRESSO_VIRTUAL_ROWID_COLUMN;
using cudf::io::experimental::is_espresso_csd_compiled;
using cudf::io::experimental::is_espresso_magic_rowid;
using cudf::io::experimental::late_materialization_bridge_stats;
using cudf::io::experimental::latest_late_materialization_bridge_stats;
using cudf::io::experimental::pack_csd_rowid;
using cudf::io::experimental::pack_espresso_rowid;
using cudf::io::experimental::read_parquet_by_rowids;
using cudf::io::experimental::read_parquet_with_csd_rowids;
}  // namespace io::parquet::experimental

}  // namespace CUDF_EXPORT cudf
