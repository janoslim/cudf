/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Espresso Authors & NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Implementation of `cudf/io/experimental/csd_late_materialization.hpp` delegating to the embedded
 * `espresso::cudf_io` and `espresso::late_mat` modules when `CUDF_HAS_ESPRESSO_CSD` is enabled.
 */

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/experimental/csd_late_materialization.hpp>
#include <cudf/utilities/error.hpp>

#ifdef CUDF_HAS_ESPRESSO_CSD
#include <espresso/cudf_io.hpp>
#include <espresso/late_mat_bridge.hpp>
#endif

namespace cudf::io::experimental {

bool is_espresso_csd_compiled() noexcept
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  return true;
#else
  return false;
#endif
}

table_with_metadata read_parquet_with_csd_rowids(parquet_reader_options const& options,
                                                 csd_scan_options const& csd_options,
                                                 cuda::stream_ref stream,
                                                 cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
#ifdef CUDF_HAS_ESPRESSO_CSD
  espresso::cudf_io::csd_scan_options opts{};
  opts.emit_rowids          = csd_options.emit_rowids;
  opts.magic_tag_rowid      = csd_options.magic_tag_rowid;
  opts.prepend_rowid_column = csd_options.prepend_rowid_column;
  opts.rowid_column_name    = csd_options.rowid_column_name;
  opts.strict_csd           = csd_options.strict_csd;
  return espresso::cudf_io::read_parquet_with_csd_rowids(options, opts, stream, mr.get_output_mr());
#else
  (void)options;
  (void)csd_options;
  (void)stream;
  (void)mr;
  CUDF_FAIL("libcudf was built without -DCUDF_USE_ESPRESSO_CSD=ON");
#endif
}

table_with_metadata read_parquet_by_rowids(parquet_reader_options const& options,
                                           column_view const& rowids,
                                           csd_late_materialization_options const& late_options,
                                           cuda::stream_ref stream,
                                           cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
#ifdef CUDF_HAS_ESPRESSO_CSD
  espresso::cudf_io::csd_late_materialization_options opts{};
  opts.rowids_are_sorted_unique = late_options.rowids_are_sorted_unique;
  opts.emit_rowids              = late_options.emit_rowids;
  opts.rowid_column_name        = late_options.rowid_column_name;
  return espresso::cudf_io::read_parquet_by_rowids(options, rowids, opts, stream, mr.get_output_mr());
#else
  (void)options;
  (void)rowids;
  (void)late_options;
  (void)stream;
  (void)mr;
  CUDF_FAIL("libcudf was built without -DCUDF_USE_ESPRESSO_CSD=ON");
#endif
}

struct csd_chunked_parquet_reader::impl {
#ifdef CUDF_HAS_ESPRESSO_CSD
  mutable std::unique_ptr<espresso::cudf_io::csd_chunked_scan_state> state;
#endif
};

csd_chunked_parquet_reader::csd_chunked_parquet_reader(std::size_t chunk_read_limit,
                                                       std::size_t pass_read_limit,
                                                       parquet_reader_options const& options,
                                                       csd_scan_options const& csd_options,
                                                       cuda::stream_ref stream,
                                                       cudf::memory_resources mr)
  : _impl(std::make_unique<impl>())
{
  CUDF_FUNC_RANGE();
#ifdef CUDF_HAS_ESPRESSO_CSD
  espresso::cudf_io::csd_scan_options opts{};
  opts.emit_rowids          = csd_options.emit_rowids;
  opts.magic_tag_rowid      = csd_options.magic_tag_rowid;
  opts.prepend_rowid_column = csd_options.prepend_rowid_column;
  opts.rowid_column_name    = csd_options.rowid_column_name;
  opts.strict_csd           = csd_options.strict_csd;
  _impl->state = espresso::cudf_io::csd_chunked_scan_state::create_with_rowids(
    chunk_read_limit, pass_read_limit, options, opts, stream, mr.get_output_mr());
#else
  (void)chunk_read_limit;
  (void)pass_read_limit;
  (void)options;
  (void)csd_options;
  (void)stream;
  (void)mr;
  CUDF_FAIL("libcudf was built without -DCUDF_USE_ESPRESSO_CSD=ON");
#endif
}

csd_chunked_parquet_reader::~csd_chunked_parquet_reader() = default;

bool csd_chunked_parquet_reader::has_next() const
{
  CUDF_FUNC_RANGE();
#ifdef CUDF_HAS_ESPRESSO_CSD
  return _impl != nullptr && _impl->state != nullptr && _impl->state->has_next();
#else
  CUDF_FAIL("libcudf was built without -DCUDF_USE_ESPRESSO_CSD=ON");
#endif
}

table_with_metadata csd_chunked_parquet_reader::read_chunk() const
{
  CUDF_FUNC_RANGE();
#ifdef CUDF_HAS_ESPRESSO_CSD
  CUDF_EXPECTS(_impl != nullptr && _impl->state != nullptr, "csd_chunked_parquet_reader is null");
  return _impl->state->read_chunk();
#else
  CUDF_FAIL("libcudf was built without -DCUDF_USE_ESPRESSO_CSD=ON");
#endif
}

std::optional<late_materialization_bridge_stats> latest_late_materialization_bridge_stats(
  std::string_view table_key)
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  auto const st = espresso::late_mat::LateMatBridgeRegistry::instance().latest_stats(table_key);
  if (!st.has_value()) { return std::nullopt; }
  late_materialization_bridge_stats out;
  out.table_key                = st->table_key;
  out.total_table_row_groups   = st->total_table_row_groups;
  out.touched_row_groups       = st->touched_row_groups;
  out.skipped_zero_row_groups  = st->skipped_zero_row_groups;
  out.unique_surviving_rows    = st->unique_surviving_rows;
  out.bitmap_bytes             = st->bitmap_bytes;
  out.max_single_rg_bitmap_b   = st->max_single_rg_bitmap_b;
  out.payload_scans_executed   = st->payload_scans_executed;
  return out;
#else
  (void)table_key;
  return std::nullopt;
#endif
}

void clear_late_materialization_bridge()
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  espresso::late_mat::LateMatBridgeRegistry::instance().clear();
#endif
}

}  // namespace cudf::io::experimental
