/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_impl.hpp"
#ifdef CUDF_HAS_ESPRESSO_CSD
#include "io/utilities/espresso_datasource_registry.hpp"
#include <espresso/cudf_io.hpp>
#include <espresso/late_mat_bridge.hpp>
#include <algorithm>
#endif

#include "cudf/io/text/byte_range_info.hpp"
#include "hybrid_scan_helpers.hpp"
#include "io/parquet/reader_impl_chunking_utils.cuh"

#include <cudf/copying.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/filling.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/iterator>
#include <thrust/host_vector.h>

#include <iterator>
#include <numeric>
#include <utility>

namespace cudf::io::parquet::experimental::detail {

using io::detail::inline_column_buffer;
using parquet::detail::ColumnChunkDesc;
using parquet::detail::decode_kernel_mask;
using parquet::detail::file_intermediate_data;
using parquet::detail::PageInfo;
using parquet::detail::PageNestingDecodeInfo;
using text::byte_range_info;

namespace {

/**
 * @brief Tests the logical type for a fixed length byte array column to see if it should be
 * treated as a string.
 *
 * Currently the only logical type that has special handling is DECIMAL. Other valid types in the
 * future would be UUID (still treated as string) and FLOAT16 (which for now would also be treated
 * as a string).
 *
 * @param logical_type The logical type to test
 * @return Boolean indicating if the logical type should be treated as a string
 */
[[maybe_unused]] inline bool is_treat_fixed_length_as_string(
  cuda::std::optional<LogicalType> const& logical_type)
{
  if (!logical_type.has_value()) { return true; }
  return logical_type->type != LogicalType::DECIMAL;
}

/**
 * @brief Get the output types from the output buffer template
 *
 * @param output_buffer_template Output buffer template
 * @return Output types
 */
[[nodiscard]] std::vector<cudf::data_type> get_output_types(
  std::span<inline_column_buffer const> output_buffer_template)
{
  std::vector<cudf::data_type> output_dtypes;
  output_dtypes.reserve(output_buffer_template.size());
  std::transform(output_buffer_template.begin(),
                 output_buffer_template.end(),
                 std::back_inserter(output_dtypes),
                 [](auto const& col) { return col.type; });
  return output_dtypes;
}

/**
 * @brief Count the number of row groups in the input
 *
 * @param row_group_indices Row group indices
 * @return Number of row groups
 */
[[nodiscard]] inline size_type count_row_groups(
  std::span<std::vector<size_type> const> row_group_indices)
{
  return std::accumulate(
    row_group_indices.begin(),
    row_group_indices.end(),
    size_type{0},
    [](auto sum, auto const& rgs) { return sum + static_cast<size_type>(rgs.size()); });
}

}  // namespace

#ifdef CUDF_HAS_ESPRESSO_CSD
struct espresso_hybrid_state {
  std::vector<std::string> filepaths;
  std::vector<espresso::csd::GpuHostParsedFooter> pre_parsed_footers;
  std::size_t all_cols_bypass_passes{0};
  bool pass_active{false};
  std::optional<table_with_metadata> single_pass_table;
  std::unique_ptr<espresso::cudf_io::csd_chunked_scan_state> csd_chunked;
  std::unique_ptr<espresso::late_mat::SelfJoinChunkedScanState> sj_chunked;

  [[nodiscard]] espresso::cudf_io::csd_scan_options make_csd_scan_options() const
  {
    espresso::cudf_io::csd_scan_options opts{};
    opts.emit_rowids = false;
    opts.strict_csd  = true;
    if (pre_parsed_footers.size() == filepaths.size()) {
      opts.pre_parsed_footers = pre_parsed_footers;
    }
    return opts;
  }

  /// Convert already-parsed cuDF `FileMetaData` into `GpuHostParsedFooter` so Espresso skips NVMe footer
  /// re-reads and GPU Thrift re-parsing; returns empty on any unsupported file to trigger safe fallback.
  [[nodiscard]] static std::vector<espresso::csd::GpuHostParsedFooter> convert_pre_parsed_footers(
    std::vector<FileMetaData> const& metadatas)
  {
    std::vector<espresso::csd::GpuHostParsedFooter> out;
    out.reserve(metadatas.size());
    for (auto const& fmd : metadatas) {
      auto converted = espresso::cudf_io::try_convert_file_metadata(fmd);
      if (!converted.has_value()) { return {}; }
      out.push_back(std::move(*converted));
    }
    return out;
  }

  void reset_pass()
  {
    pass_active = false;
    single_pass_table.reset();
    csd_chunked.reset();
    sj_chunked.reset();
  }

  [[nodiscard]] bool has_next() const noexcept
  {
    if (!pass_active) { return false; }
    if (single_pass_table.has_value()) { return true; }
    if (sj_chunked != nullptr) { return sj_chunked->has_next(); }
    if (csd_chunked != nullptr) { return csd_chunked->has_next(); }
    return false;
  }

  [[nodiscard]] table_with_metadata next_chunk()
  {
    if (single_pass_table.has_value()) {
      auto out = std::move(*single_pass_table);
      single_pass_table.reset();
      pass_active = false;
      return out;
    }
    if (sj_chunked != nullptr) {
      auto out = sj_chunked->read_chunk();
      if (!sj_chunked->has_next()) { pass_active = false; }
      return out;
    }
    auto out = csd_chunked->read_chunk();
    if (!csd_chunked->has_next()) { pass_active = false; }
    return out;
  }
};

hybrid_scan_reader_impl::~hybrid_scan_reader_impl() = default;

void hybrid_scan_reader_impl::init_espresso_hybrid_state()
{
  if (_extended_metadata == nullptr) { return; }
  auto const metadatas = _extended_metadata->parquet_metadatas();
  if (metadatas.empty()) { return; }

  std::vector<std::string> extracted;
  extracted.reserve(metadatas.size());
  for (auto const& fmd : metadatas) {
    std::string found = cudf::io::espresso_detail::extract_espresso_metadata_filepath(fmd);
    if (found.empty()) { return; }
    extracted.push_back(std::move(found));
  }

  _espresso                     = std::make_unique<espresso_hybrid_state>();
  _espresso->filepaths          = std::move(extracted);
  _espresso->pre_parsed_footers = espresso_hybrid_state::convert_pre_parsed_footers(metadatas);
}

bool hybrid_scan_reader_impl::can_espresso_offload_all_columns(
  std::span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options) const
{
  if (_espresso == nullptr || _espresso->filepaths.empty() ||
      _espresso->filepaths.size() != _extended_metadata->get_num_sources() ||
      row_group_indices.size() != _espresso->filepaths.size()) {
    return false;
  }
  if (espresso::late_mat::is_sql_rewrite_bridge_enabled()) {
    auto const probe_opts = build_espresso_pass_options(row_group_indices, options);
    if (espresso::late_mat::LateMatBridgeRegistry::instance().should_intercept_scan(probe_opts)) {
      return true;
    }
  }
  if (!espresso::cudf_io::is_csd_dropin_enabled()) { return false; }
  if (espresso::cudf_io::is_csd_strict_mode()) { return true; }

  for (auto const& fp : _espresso->filepaths) {
    if (!espresso::cudf_io::is_smartssd_parquet_path(fp)) { return false; }
  }
  if (options.get_skip_rows() != 0 || options.get_num_rows().has_value() ||
      options.is_enabled_convert_strings_to_categories() ||
      options.is_enabled_prepend_source_index_column() ||
      options.is_enabled_prepend_row_index_column() ||
      options.get_decimal_width() == type_id::DECIMAL128) {
    return false;
  }

  std::size_t const projected_cols =
    _input_columns.size() >= _num_filter_only_columns ? (_input_columns.size() - _num_filter_only_columns) : 0;
  for (std::size_t col_i = 0; col_i < _input_columns.size(); ++col_i) {
    auto const& ic = _input_columns[col_i];
    if (ic.nesting.size() != 1) { return false; }
    auto const& sch = _extended_metadata->get_schema(ic.schema_idx);
    if (sch.type != Type::INT32 && sch.type != Type::INT64 && sch.type != Type::BYTE_ARRAY) {
      return false;
    }
    bool const is_decimal =
      sch.converted_type == ConvertedType::DECIMAL ||
      (sch.logical_type.has_value() && sch.logical_type->type == LogicalType::DECIMAL);
    if (is_decimal && sch.type == Type::BYTE_ARRAY) { return false; }

    bool const needs_null_free =
      (col_i < projected_cols) && (sch.repetition_type == FieldRepetitionType::OPTIONAL);
    for (std::size_t src_i = 0; src_i < row_group_indices.size(); ++src_i) {
      for (size_type const rg_idx : row_group_indices[src_i]) {
        auto const& cm = _extended_metadata->get_column_metadata(rg_idx, src_i, ic.schema_idx);
        if (cm.codec != Compression::UNCOMPRESSED && cm.codec != Compression::SNAPPY) {
          return false;
        }
        if (needs_null_free) {
          if (!cm.statistics.null_count.has_value() || *cm.statistics.null_count > 0) {
            return false;
          }
        }
      }
    }
  }
   // Mirror Espresso per-card 64 MiB main / 16 MiB chars compact-tile staging guard BEFORE
  // `all_column_chunks_byte_ranges()` returns 0 B, so unfiltered high-entropy wide-string row
  // groups (e.g., synthetic random base62 comments in Zipf datasets) fall back cleanly to stock
  // `libcudf` with populated `column_chunk_data` instead of overflowing the per-card staging arena.
  if (!options.get_filter().has_value()) {
    for (std::size_t src_i = 0; src_i < row_group_indices.size(); ++src_i) {
      for (size_type const rg_idx : row_group_indices[src_i]) {
        std::uint64_t min_incomp_main = 0;
        std::uint64_t min_raw_chars   = 0;
        bool has_incomp_str           = false;
        auto const& rg_meta           = _extended_metadata->get_row_group(rg_idx, src_i);
        std::uint64_t const rg_rows   = rg_meta.num_rows > 0 ? static_cast<std::uint64_t>(rg_meta.num_rows) : 0ULL;
        for (std::size_t col_i = 0; col_i < projected_cols; ++col_i) {
          auto const& ic  = _input_columns[col_i];
          auto const& sch = _extended_metadata->get_schema(ic.schema_idx);
          auto const& cm  = _extended_metadata->get_column_metadata(rg_idx, src_i, ic.schema_idx);
          if (sch.type == Type::BYTE_ARRAY) {
            std::uint64_t const unc =
              cm.total_uncompressed_size > 0 ? static_cast<std::uint64_t>(cm.total_uncompressed_size) : 0ULL;
            std::uint64_t const comp =
              cm.total_compressed_size > 0 ? static_cast<std::uint64_t>(cm.total_compressed_size) : 0ULL;
            std::uint64_t const unenc_len =
              (cm.size_statistics.has_value() && cm.size_statistics->unencoded_byte_array_data_bytes.has_value() &&
               *cm.size_statistics->unencoded_byte_array_data_bytes > 0)
                ? static_cast<std::uint64_t>(*cm.size_statistics->unencoded_byte_array_data_bytes)
                : 0ULL;
            bool const force_raw = (unenc_len > 0 && rg_rows > 0 && (unenc_len / rg_rows) > 96ULL);
            if (force_raw) {
              min_raw_chars += unc;
              min_incomp_main += rg_rows * 4ULL;
            } else if (unc > 0 && comp * 100ULL >= unc * 85ULL) {
              has_incomp_str = true;
              min_incomp_main += unc + rg_rows * 4ULL;
            }
          } else {
            min_incomp_main += rg_rows * (sch.type == Type::INT64 ? 8ULL : 4ULL);
          }
        }
        if (min_raw_chars > (15ULL << 20) || (has_incomp_str && min_incomp_main > (56ULL << 20))) {
          return false;
        }
      }
    }
  }
  return true;
}

parquet_reader_options hybrid_scan_reader_impl::build_espresso_pass_options(
  std::span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options) const
{
  auto builder = parquet_reader_options::builder(source_info{_espresso->filepaths})
                   .row_groups(std::vector<std::vector<size_type>>{row_group_indices.begin(),
                                                                   row_group_indices.end()})
                   .use_pandas_metadata(options.is_enabled_use_pandas_metadata())
                   .use_arrow_schema(options.is_enabled_use_arrow_schema())
                   .allow_mismatched_pq_schemas(false)
                   .case_sensitive_names(options.is_enabled_case_sensitive_names())
                   .convert_strings_to_categories(options.is_enabled_convert_strings_to_categories())
                   .timestamp_type(options.get_timestamp_type())
                   .decimal_width(options.get_decimal_width());
  if (options.get_skip_rows() != 0) { builder.skip_rows(options.get_skip_rows()); }
  if (options.get_num_rows().has_value()) { builder.num_rows(*options.get_num_rows()); }
  if (options.is_enabled_prepend_source_index_column()) {
    builder.prepend_source_index_column(true);
  }
  if (options.is_enabled_prepend_row_index_column()) {
    builder.prepend_row_index_column(true);
  }
  if (auto const& names = options.get_column_names(); names.has_value()) {
    builder.column_names(*names);
  } else if (auto const& indices = options.get_column_indices(); indices.has_value()) {
    builder.column_indices(*indices);
  } else if (auto const& fids = options.get_column_field_ids(); fids.has_value()) {
    builder.column_field_ids(*fids);
  }
  auto pass_opts = builder.build();
  if (options.get_filter().has_value()) { pass_opts.set_filter(options.get_filter()->get()); }
  return pass_opts;
}
#endif

hybrid_scan_reader_impl::hybrid_scan_reader_impl(
  cudf::host_span<cudf::host_span<uint8_t const> const> footer_bytes,
  parquet_reader_options const& options)
{
  _metadata = std::make_unique<aggregate_reader_metadata>(
    footer_bytes, options.is_enabled_use_arrow_schema(), has_cols_from_mismatched_sources(options));

  _extended_metadata = static_cast<aggregate_reader_metadata*>(_metadata.get());
#ifdef CUDF_HAS_ESPRESSO_CSD
  init_espresso_hybrid_state();
  if (_espresso == nullptr && !footer_bytes.empty() &&
      (espresso::cudf_io::is_csd_dropin_enabled() || espresso::late_mat::is_sql_rewrite_bridge_enabled())) {
    std::vector<std::string> resolved;
    resolved.reserve(footer_bytes.size());
    bool all_found = true;
    for (auto const& fb : footer_bytes) {
      auto hit = cudf::io::espresso_detail::EspressoDatasetFooterIndex::instance().lookup_by_footer_bytes(
        fb.data(), fb.size());
      if (!hit.has_value()) {
        all_found = false;
        break;
      }
      resolved.push_back(std::move(*hit));
    }
    if (all_found) {
      _espresso                     = std::make_unique<espresso_hybrid_state>();
      _espresso->filepaths          = std::move(resolved);
      _espresso->pre_parsed_footers = espresso_hybrid_state::convert_pre_parsed_footers(_extended_metadata->parquet_metadatas());
    }
  }
#endif
}

hybrid_scan_reader_impl::hybrid_scan_reader_impl(
  cudf::host_span<FileMetaData const> parquet_metadatas, parquet_reader_options const& options)
{
  _metadata =
    std::make_unique<aggregate_reader_metadata>(parquet_metadatas,
                                                options.is_enabled_use_arrow_schema(),
                                                has_cols_from_mismatched_sources(options));
  _extended_metadata = static_cast<aggregate_reader_metadata*>(_metadata.get());
#ifdef CUDF_HAS_ESPRESSO_CSD
  init_espresso_hybrid_state();
#endif
}

std::vector<FileMetaData> hybrid_scan_reader_impl::parquet_metadatas() const
{
  auto out = _extended_metadata->parquet_metadatas();
#ifdef CUDF_HAS_ESPRESSO_CSD
  if (_espresso != nullptr && _espresso->filepaths.size() == out.size()) {
    for (std::size_t i = 0; i < out.size(); ++i) {
      if (cudf::io::espresso_detail::extract_espresso_metadata_filepath(out[i]).empty()) {
        out[i].key_value_metadata.push_back(
          KeyValue{std::string(cudf::io::espresso_detail::ESPRESSO_FILEPATH_META_KEY),
                   _espresso->filepaths[i]});
      }
    }
  }
#endif
  return out;
}

std::vector<byte_range_info> hybrid_scan_reader_impl::page_index_byte_ranges() const
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  if (_espresso != nullptr && !_espresso->filepaths.empty() &&
      (espresso::late_mat::is_sql_rewrite_bridge_enabled() ||
       (espresso::cudf_io::is_csd_dropin_enabled() &&
        (espresso::cudf_io::is_csd_strict_mode() ||
         std::all_of(_espresso->filepaths.begin(),
                     _espresso->filepaths.end(),
                     espresso::cudf_io::is_smartssd_parquet_path))))) {
    return std::vector<byte_range_info>(_extended_metadata->get_num_sources(), byte_range_info{0, 0});
  }
#endif
  return _extended_metadata->page_index_byte_ranges();
}

void hybrid_scan_reader_impl::setup_page_indexes(
  cudf::host_span<cudf::host_span<uint8_t const> const> page_index_bytes) const
{
  _extended_metadata->setup_page_indexes(page_index_bytes);
}

void hybrid_scan_reader_impl::select_columns(read_columns_mode read_columns_mode,
                                             parquet_reader_options const& options)
{
  // Initialize reader configuration.
  initialize_reader_config(options);

  // Build column selection options directly from the user options.
  auto selection_options = make_column_selection_options(options);

  if (read_columns_mode == read_columns_mode::ALL_COLUMNS) {
    if (_is_all_columns_selected) { return; }

    // Select only columns required by the options and filter
    auto const select_column_names = get_column_projection(options);

    // Select only columns required by the options and filter.
    // Using as is from:
    // https://github.com/rapidsai/cudf/blob/a8b25cd205dc5d04b9918dcb0b3abd6b8c4e4a74/cpp/src/io/parquet/reader_impl.cpp#L556-L569
    std::optional<std::vector<std::string>> filter_only_columns_names;
    if (options.get_filter().has_value() and select_column_names.has_value()) {
      filter_only_columns_names = parquet::detail::get_column_names_in_expression(
        options.get_filter(), *select_column_names, options, _extended_metadata->get_schema_tree());
      _num_filter_only_columns = filter_only_columns_names->size();
    }
    std::tie(_input_columns, _output_buffers, _output_column_schemas) =
      _metadata->select_columns(select_column_names, filter_only_columns_names, selection_options);

    _is_all_columns_selected     = true;
    _is_filter_columns_selected  = false;
    _is_payload_columns_selected = false;
  } else if (read_columns_mode == read_columns_mode::FILTER_COLUMNS) {
    if (_is_filter_columns_selected) { return; }
    // Must not ignore missing filter columns
    selection_options.ignore_missing_columns = false;

    _filter_columns_names = cudf::io::parquet::detail::get_column_names_in_expression(
      options.get_filter(), {}, options, _extended_metadata->get_schema_tree());
    // Select only filter columns using the base `select_columns` method
    std::tie(_input_columns, _output_buffers, _output_column_schemas) =
      _extended_metadata->select_columns(_filter_columns_names, {}, selection_options);

    _is_filter_columns_selected  = true;
    _is_payload_columns_selected = false;
    _is_all_columns_selected     = false;
  } else {
    if (_is_payload_columns_selected) { return; }

    auto select_column_names = get_column_projection(options);
    std::tie(_input_columns, _output_buffers, _output_column_schemas) =
      _extended_metadata->select_payload_columns(
        select_column_names, _filter_columns_names, selection_options);

    _is_payload_columns_selected = true;
    _is_filter_columns_selected  = false;
    _is_all_columns_selected     = false;
  }

  // Reset the materialization step flag
  _output_chunk_produced = false;

  CUDF_EXPECTS(_input_columns.size() > 0 and _output_buffers.size() > 0, "No columns selected");

  // Clear the output buffers templates
  _output_buffers_template.clear();

  // Save the states of the output buffers for reuse.
  std::transform(_output_buffers.begin(),
                 _output_buffers.end(),
                 std::back_inserter(_output_buffers_template),
                 [](auto const& buff) { return inline_column_buffer::empty_like(buff); });
}

std::vector<std::vector<size_type>> hybrid_scan_reader_impl::all_row_groups(
  parquet_reader_options const& options) const
{
  return _extended_metadata->all_row_groups(options);
}

std::size_t hybrid_scan_reader_impl::total_rows_in_row_groups(
  std::span<std::vector<size_type> const> row_group_indices) const
{
  return _extended_metadata->total_rows_in_row_groups(row_group_indices);
}

void hybrid_scan_reader_impl::reset_column_selection()
{
  _is_all_columns_selected     = false;
  _is_filter_columns_selected  = false;
  _is_payload_columns_selected = false;
}

std::pair<named_to_reference_converter, std::vector<cudf::data_type>>
hybrid_scan_reader_impl::prepare_filter_and_output_types(parquet_reader_options const& options)
{
  CUDF_EXPECTS(options.get_filter().has_value(), "Empty input filter expression encountered");

  select_columns(read_columns_mode::FILTER_COLUMNS, options);

  // Convert the input expression (must be done after column selection)
  auto expr_conv     = build_converted_expression(options);
  auto output_dtypes = get_output_types(_output_buffers_template);

  return {std::move(expr_conv), std::move(output_dtypes)};
}

void hybrid_scan_reader_impl::prepare_materialization(read_columns_mode read_columns_mode,
                                                      std::size_t num_sources,
                                                      parquet_reader_options const& options,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  reset_internal_state();
  initialize_options(options, num_sources, stream, mr);
  select_columns(read_columns_mode, options);
}

std::vector<std::vector<cudf::size_type>>
hybrid_scan_reader_impl::filter_row_groups_with_byte_range(
  std::span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options) const
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");

  if (options.get_skip_bytes() == 0 and not options.get_num_bytes().has_value()) {
    return std::vector<std::vector<cudf::size_type>>{row_group_indices.begin(),
                                                     row_group_indices.end()};
  }

  return _extended_metadata->filter_row_groups_with_byte_range(
    row_group_indices, options.get_skip_bytes(), options.get_num_bytes());
}

std::vector<std::vector<size_type>> hybrid_scan_reader_impl::filter_row_groups_with_stats(
  std::span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  auto [expr_conv, output_dtypes] = prepare_filter_and_output_types(options);

  return _extended_metadata->filter_row_groups_with_stats(row_group_indices,
                                                          output_dtypes,
                                                          _output_column_schemas,
                                                          expr_conv.get_converted_expr().value(),
                                                          stream);
}

std::pair<std::vector<byte_range_info>, std::vector<byte_range_info>>
hybrid_scan_reader_impl::secondary_filters_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices, parquet_reader_options const& options)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  auto [expr_conv, output_dtypes] = prepare_filter_and_output_types(options);

  auto const bloom_filter_bytes =
    _extended_metadata->get_bloom_filter_bytes(row_group_indices,
                                               output_dtypes,
                                               _output_column_schemas,
                                               expr_conv.get_converted_expr().value());
  auto const dictionary_page_bytes =
    _extended_metadata
      ->dictionary_pages_byte_ranges(row_group_indices,
                                     output_dtypes,
                                     _output_column_schemas,
                                     expr_conv.get_converted_expr().value())
      .first;

  return {bloom_filter_bytes, dictionary_page_bytes};
}

std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
hybrid_scan_reader_impl::dictionary_pages_byte_ranges(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  auto [expr_conv, output_dtypes] = prepare_filter_and_output_types(options);

  return _extended_metadata->dictionary_pages_byte_ranges(row_group_indices,
                                                          output_dtypes,
                                                          _output_column_schemas,
                                                          expr_conv.get_converted_expr().value());
}

std::vector<std::vector<size_type>>
hybrid_scan_reader_impl::filter_row_groups_with_dictionary_pages(
  std::span<cudf::device_span<uint8_t const> const> dictionary_page_data,
  std::span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  auto [expr_conv, output_dtypes] = prepare_filter_and_output_types(options);

  // Collect literal and operator pairs for each input column with an (in)equality predicate
  auto const [literals, operators] =
    dictionary_literals_collector{expr_conv.get_converted_expr().value().get(), output_dtypes}
      .get_literals_and_operators();

  // Return all row groups if no dictionary page filtering is needed
  if (literals.empty() or std::all_of(literals.begin(), literals.end(), [](auto& col_literals) {
        return col_literals.empty();
      })) {
    return std::vector<std::vector<size_type>>(row_group_indices.begin(), row_group_indices.end());
  }

  // Collect schema indices of input columns with a non-empty (in)equality literal/operator vector
  std::vector<cudf::size_type> dictionary_col_schemas;
  thrust::copy_if(thrust::host,
                  _output_column_schemas.begin(),
                  _output_column_schemas.end(),
                  literals.begin(),
                  std::back_inserter(dictionary_col_schemas),
                  [](auto& dict_literals) { return not dict_literals.empty(); });

  // Prepare dictionary column chunks and decode page headers
  auto [has_compressed_data, chunks, pages] = prepare_dictionaries(
    row_group_indices, dictionary_page_data, dictionary_col_schemas, options, stream);

  // Decompress dictionary pages if needed and store uncompressed buffers here
  auto const mr                          = cudf::get_current_device_resource_ref();
  auto decompressed_dictionary_page_data = std::optional<rmm::device_buffer>{};
  if (has_compressed_data) {
    // Use the `decompress_page_data` utility to decompress dictionary pages (passed as pass_pages)
    decompressed_dictionary_page_data =
      std::get<0>(parquet::detail::decompress_page_data(chunks, pages, {}, {}, stream, mr));
    pages.host_to_device_async(stream);
  }

  // Filter row groups using dictionary pages
  return _extended_metadata->filter_row_groups_with_dictionary_pages(
    chunks,
    pages,
    row_group_indices,
    literals,
    operators,
    output_dtypes,
    dictionary_col_schemas,
    expr_conv.get_converted_expr().value(),
    stream);
}

std::vector<std::vector<size_type>> hybrid_scan_reader_impl::filter_row_groups_with_bloom_filters(
  std::span<cudf::device_span<uint8_t const> const> bloom_filter_data,
  std::span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  auto [expr_conv, output_dtypes] = prepare_filter_and_output_types(options);

  return _extended_metadata->filter_row_groups_with_bloom_filters(
    bloom_filter_data,
    row_group_indices,
    output_dtypes,
    _output_column_schemas,
    expr_conv.get_converted_expr().value(),
    stream);
}

std::unique_ptr<cudf::column> hybrid_scan_reader_impl::build_all_true_row_mask(
  std::span<std::vector<size_type> const> row_group_indices,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");

  auto const num_rows = total_rows_in_row_groups(row_group_indices);
  CUDF_EXPECTS(num_rows < std::numeric_limits<cudf::size_type>::max(),
               "Total rows in row groups exceed the cudf's column size limit. Retry with a smaller "
               "set of row groups",
               std::invalid_argument);
  auto true_scalar =
    cudf::numeric_scalar<bool>(true, true, stream, cudf::get_current_device_resource_ref());
  return cudf::make_column_from_scalar(true_scalar, num_rows, stream, mr);
}

std::unique_ptr<cudf::column> hybrid_scan_reader_impl::build_row_mask_with_page_index_stats(
  std::span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
  auto [expr_conv, output_dtypes] = prepare_filter_and_output_types(options);

  return _extended_metadata->build_row_mask_with_page_index_stats(
    row_group_indices,
    output_dtypes,
    _output_column_schemas,
    expr_conv.get_converted_expr().value(),
    stream,
    mr);
}

std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
hybrid_scan_reader_impl::get_input_column_chunk_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices) const
{
  // Descriptors for all the chunks that make up the selected columns
  auto const num_input_columns = _input_columns.size();
  auto const num_row_groups    = std::accumulate(
    row_group_indices.begin(),
    row_group_indices.end(),
    std::size_t{0},
    [](std::size_t sum, auto const& row_groups) { return sum + row_groups.size(); });
  auto const num_chunks = num_row_groups * num_input_columns;

  // Association between each column chunk and its source
  auto chunk_source_map = std::vector<size_type>{};
  chunk_source_map.reserve(num_chunks);

  // Keep track of column chunk byte ranges
  auto column_chunk_byte_ranges = std::vector<byte_range_info>{};
  column_chunk_byte_ranges.reserve(num_chunks);

  std::for_each(cuda::counting_iterator<std::size_t>{0},
                cuda::counting_iterator{row_group_indices.size()},
                [&](auto const source_idx) {
                  auto const& row_groups = row_group_indices[source_idx];
                  for (auto const row_group_index : row_groups) {
                    // generate ColumnChunkDesc objects for everything to be decoded (all input
                    // columns)
                    for (auto const& col : _input_columns) {
                      // look up metadata
                      auto const& col_meta = _extended_metadata->get_column_metadata(
                        row_group_index, source_idx, col.schema_idx);

                      auto const chunk_offset =
                        (col_meta.dictionary_page_offset != 0)
                          ? std::min(col_meta.data_page_offset, col_meta.dictionary_page_offset)
                          : col_meta.data_page_offset;

                      auto const chunk_size = col_meta.total_compressed_size;
                      column_chunk_byte_ranges.emplace_back(chunk_offset, chunk_size);

                      // Map each column chunk to its column index and its source index
                      chunk_source_map.emplace_back(static_cast<size_type>(source_idx));
                    }
                  }
                });

  return {std::move(column_chunk_byte_ranges), std::move(chunk_source_map)};
}

std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
hybrid_scan_reader_impl::filter_column_chunks_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices, parquet_reader_options const& options)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");

  select_columns(read_columns_mode::FILTER_COLUMNS, options);
  return get_input_column_chunk_byte_ranges(row_group_indices);
}

std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
hybrid_scan_reader_impl::payload_column_chunks_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices, parquet_reader_options const& options)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");

  select_columns(read_columns_mode::PAYLOAD_COLUMNS, options);
  return get_input_column_chunk_byte_ranges(row_group_indices);
}

std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
hybrid_scan_reader_impl::all_column_chunks_byte_ranges(
  std::span<std::vector<size_type> const> row_group_indices, parquet_reader_options const& options)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");

  select_columns(read_columns_mode::ALL_COLUMNS, options);
#ifdef CUDF_HAS_ESPRESSO_CSD
  if (can_espresso_offload_all_columns(row_group_indices, options)) {
    ++_espresso->all_cols_bypass_passes;
    // Return empty byte ranges (0 B host/GDS I/O) so callers such as Sirius `locate_chunks`
    // that validate every returned range against FileMetaData offsets do not fail on `{0, 0}`.
    return {{}, {}};
  }
#endif
  return get_input_column_chunk_byte_ranges(row_group_indices);
}

table_with_metadata hybrid_scan_reader_impl::materialize_filter_columns(
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  cudf::mutable_column_view& row_mask,
  use_data_page_mask mask_data_pages,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(std::cmp_equal(row_mask.size(), total_rows_in_row_groups(row_group_indices)),
               "Row mask must span across all input row groups");
  CUDF_EXPECTS(options.get_filter().has_value(), "Empty input filter expression encountered");

  prepare_materialization(
    read_columns_mode::FILTER_COLUMNS, row_group_indices.size(), options, stream, mr);

  // Convert the input expression (must be done after prepare_materialization)
  _expr_conv = build_converted_expression(options);

  // Return early if all rows are pruned
  if (are_all_rows_pruned(row_mask, stream)) {
    auto const empty_row_groups =
      std::vector<std::vector<size_type>>(row_group_indices.size(), std::vector<size_type>{});
    prepare_data(read_mode::READ_ALL, empty_row_groups, {}, {});
    // Set correct number of input row groups to the output metadata
    _file_itm_data.num_input_row_groups = count_row_groups(row_group_indices);
    return read_chunk_internal(read_mode::READ_ALL, read_columns_mode::FILTER_COLUMNS, row_mask);
  }

  auto data_page_mask = thrust::host_vector<bool>{};
  if (mask_data_pages == use_data_page_mask::YES) {
    data_page_mask = _extended_metadata->compute_data_page_mask(
      row_mask, row_group_indices, _input_columns, _row_mask_offset, stream);
  }

  prepare_data(read_mode::READ_ALL, row_group_indices, column_chunk_data, data_page_mask);

  return read_chunk_internal(read_mode::READ_ALL, read_columns_mode::FILTER_COLUMNS, row_mask);
}

table_with_metadata hybrid_scan_reader_impl::materialize_payload_columns(
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(std::cmp_equal(row_mask.size(), total_rows_in_row_groups(row_group_indices)),
               "Row mask must span across all input row groups");
  CUDF_EXPECTS(row_mask.null_count() == 0,
               "Row mask must not have any nulls when materializing payload column");

  prepare_materialization(
    read_columns_mode::PAYLOAD_COLUMNS, row_group_indices.size(), options, stream, mr);

  // Return early if all rows are pruned
  if (are_all_rows_pruned(row_mask, stream)) {
    auto const empty_row_groups =
      std::vector<std::vector<size_type>>(row_group_indices.size(), std::vector<size_type>{});
    prepare_data(read_mode::READ_ALL, empty_row_groups, {}, {});
    // Set correct number of input row groups to the output metadata
    _file_itm_data.num_input_row_groups = count_row_groups(row_group_indices);
    return read_chunk_internal(read_mode::READ_ALL, read_columns_mode::PAYLOAD_COLUMNS, row_mask);
  }

  auto data_page_mask = thrust::host_vector<bool>{};
  if (not row_mask.is_empty() and mask_data_pages == use_data_page_mask::YES) {
    data_page_mask = _extended_metadata->compute_data_page_mask(
      row_mask, row_group_indices, _input_columns, _row_mask_offset, stream);
  }

  prepare_data(read_mode::READ_ALL, row_group_indices, column_chunk_data, data_page_mask);

  return read_chunk_internal(read_mode::READ_ALL, read_columns_mode::PAYLOAD_COLUMNS, row_mask);
}

table_with_metadata hybrid_scan_reader_impl::materialize_all_columns(
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
#ifdef CUDF_HAS_ESPRESSO_CSD
  select_columns(read_columns_mode::ALL_COLUMNS, options);
  if ((_espresso != nullptr && _espresso->all_cols_bypass_passes > 0) ||
      can_espresso_offload_all_columns(row_group_indices, options)) {
    auto const pass_opts = build_espresso_pass_options(row_group_indices, options);
    if (auto sj = espresso::late_mat::LateMatBridgeRegistry::instance().try_read_table(
          pass_opts, stream, mr);
        sj.has_value()) {
      return std::move(*sj);
    }
    return espresso::cudf_io::read_parquet(
      pass_opts, _espresso ? _espresso->make_csd_scan_options() : espresso::cudf_io::csd_scan_options{.emit_rowids = false, .strict_csd = true}, stream, mr);
  }
#endif

  prepare_materialization(
    read_columns_mode::ALL_COLUMNS, row_group_indices.size(), options, stream, mr);

  // Convert the input expression (must be done after prepare_materialization)
  _expr_conv = build_converted_expression(options);

  prepare_data(read_mode::READ_ALL, row_group_indices, column_chunk_data, {});

  // Use the main reader's function
  auto result = reader_impl::read_chunk_internal(read_mode::READ_ALL);

  // base read_chunk_internal() does not update the output chunk produced flag
  _output_chunk_produced = true;

  return result;
}

void hybrid_scan_reader_impl::setup_chunking_for_filter_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  std::span<std::vector<size_type> const> row_group_indices,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(options.get_filter().has_value(), "Empty input filter expression encountered");
  CUDF_EXPECTS(std::cmp_equal(row_mask.size(), total_rows_in_row_groups(row_group_indices)),
               "Row mask must span across all input row groups");

  prepare_materialization(
    read_columns_mode::FILTER_COLUMNS, row_group_indices.size(), options, stream, mr);

  _input_pass_read_limit   = pass_read_limit;
  _output_chunk_read_limit = chunk_read_limit;

  // Convert the input expression (must be done after prepare_materialization)
  _expr_conv = build_converted_expression(options);

  // Return early if all rows are pruned
  if (are_all_rows_pruned(row_mask, stream)) {
    auto const empty_row_groups =
      std::vector<std::vector<size_type>>(row_group_indices.size(), std::vector<size_type>{});
    prepare_data(read_mode::CHUNKED_READ, empty_row_groups, {}, {});
    // Set correct number of input row groups to the output metadata
    _file_itm_data.num_input_row_groups = count_row_groups(row_group_indices);
    return;
  }

  auto data_page_mask = thrust::host_vector<bool>{};
  if (mask_data_pages == use_data_page_mask::YES) {
    data_page_mask = _extended_metadata->compute_data_page_mask(
      row_mask, row_group_indices, _input_columns, _row_mask_offset, stream);
  }

  prepare_data(read_mode::CHUNKED_READ, row_group_indices, column_chunk_data, data_page_mask);
}

table_with_metadata hybrid_scan_reader_impl::materialize_filter_columns_chunk(
  cudf::mutable_column_view& row_mask)
{
  CUDF_EXPECTS(_file_preprocessed, "Chunking for filter columns not yet setup");

  // Reset the output buffers to their original states (right after reader construction).
  // Don't need to do it if we read the file all at once.
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes() and
      not is_first_output_chunk()) {
    _output_buffers.resize(0);
    for (auto const& buff : _output_buffers_template) {
      _output_buffers.emplace_back(cudf::io::detail::inline_column_buffer::empty_like(buff));
    }
  }

  prepare_data(read_mode::CHUNKED_READ, {}, {}, {});
  return read_chunk_internal(read_mode::CHUNKED_READ, read_columns_mode::FILTER_COLUMNS, row_mask);
}

void hybrid_scan_reader_impl::setup_chunking_for_payload_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  std::span<std::vector<size_type> const> row_group_indices,
  cudf::column_view const& row_mask,
  use_data_page_mask mask_data_pages,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(std::cmp_equal(row_mask.size(), total_rows_in_row_groups(row_group_indices)),
               "Row mask must span across all input row groups");
  CUDF_EXPECTS(row_mask.null_count() == 0,
               "Row mask must not have any nulls when materializing payload column");

  prepare_materialization(
    read_columns_mode::PAYLOAD_COLUMNS, row_group_indices.size(), options, stream, mr);

  _input_pass_read_limit   = pass_read_limit;
  _output_chunk_read_limit = chunk_read_limit;

  // Return early if all rows are pruned
  if (are_all_rows_pruned(row_mask, stream)) {
    auto const empty_row_groups =
      std::vector<std::vector<size_type>>(row_group_indices.size(), std::vector<size_type>{});
    prepare_data(read_mode::CHUNKED_READ, empty_row_groups, {}, {});
    // Set correct number of input row groups to the output metadata
    _file_itm_data.num_input_row_groups = count_row_groups(row_group_indices);
    return;
  }

  auto data_page_mask = thrust::host_vector<bool>{};
  if (not row_mask.is_empty() and mask_data_pages == use_data_page_mask::YES) {
    data_page_mask = _extended_metadata->compute_data_page_mask(
      row_mask, row_group_indices, _input_columns, _row_mask_offset, stream);
  }

  prepare_data(read_mode::CHUNKED_READ, row_group_indices, column_chunk_data, data_page_mask);
}

table_with_metadata hybrid_scan_reader_impl::materialize_payload_columns_chunk(
  cudf::column_view const& row_mask)
{
  CUDF_EXPECTS(_file_preprocessed, "Chunking for payload columns not yet setup");

  // Reset the output buffers to their original states (right after reader construction).
  // Don't need to do it if we read the file all at once.
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes() and
      not is_first_output_chunk()) {
    _output_buffers.resize(0);
    for (auto const& buff : _output_buffers_template) {
      _output_buffers.emplace_back(cudf::io::detail::inline_column_buffer::empty_like(buff));
    }
  }
  prepare_data(read_mode::CHUNKED_READ, {}, {}, {});
  return read_chunk_internal(read_mode::CHUNKED_READ, read_columns_mode::PAYLOAD_COLUMNS, row_mask);
}

void hybrid_scan_reader_impl::setup_chunking_for_all_columns(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not row_group_indices.empty(), "Empty input row group indices encountered");
#ifdef CUDF_HAS_ESPRESSO_CSD
  if (_espresso != nullptr) { _espresso->reset_pass(); }
  select_columns(read_columns_mode::ALL_COLUMNS, options);
  if ((_espresso != nullptr && _espresso->all_cols_bypass_passes > 0) ||
      can_espresso_offload_all_columns(row_group_indices, options)) {
    auto const pass_opts = build_espresso_pass_options(row_group_indices, options);
    if (auto sj = espresso::late_mat::SelfJoinChunkedScanState::try_create(
          chunk_read_limit, pass_read_limit, pass_opts, stream, mr);
        sj != nullptr) {
      _espresso->sj_chunked  = std::move(sj);
      _espresso->pass_active = true;
      return;
    }
    auto const csd_opts = _espresso ? _espresso->make_csd_scan_options()
                                    : espresso::cudf_io::csd_scan_options{.emit_rowids = false, .strict_csd = true};
    if (chunk_read_limit == 0) {
      _espresso->single_pass_table = espresso::cudf_io::read_parquet(
        pass_opts, csd_opts, stream, mr);
    } else {
      _espresso->csd_chunked = espresso::cudf_io::csd_chunked_scan_state::create_with_rowids(
        chunk_read_limit,
        pass_read_limit,
        pass_opts,
        csd_opts,
        stream,
        mr);
    }
    _espresso->pass_active = true;
    return;
  }
#endif

  prepare_materialization(
    read_columns_mode::ALL_COLUMNS, row_group_indices.size(), options, stream, mr);

  _input_pass_read_limit   = pass_read_limit;
  _output_chunk_read_limit = chunk_read_limit;

  // Convert the input expression (must be done after column selection)
  _expr_conv = build_converted_expression(options);

  prepare_data(read_mode::CHUNKED_READ, row_group_indices, column_chunk_data, {});
}

table_with_metadata hybrid_scan_reader_impl::materialize_all_columns_chunk()
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  if (_espresso != nullptr && _espresso->pass_active) {
    return _espresso->next_chunk();
  }
#endif
  CUDF_EXPECTS(_file_preprocessed, "Chunking for all columns not yet setup");

  // Reset the output buffers to their original states (right after reader construction).
  // Don't need to do it if we read the file all at once.
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes() and
      not is_first_output_chunk()) {
    _output_buffers.resize(0);
    for (auto const& buff : _output_buffers_template) {
      _output_buffers.emplace_back(cudf::io::detail::inline_column_buffer::empty_like(buff));
    }
  }
  prepare_data(read_mode::CHUNKED_READ, {}, {}, {});

  // Use the main reader's function for reading all columns
  auto result = reader_impl::read_chunk_internal(read_mode::CHUNKED_READ);

  // base read_chunk_internal() does not update the output chunk produced flag
  _output_chunk_produced = true;

  return result;
}

std::pair<std::vector<std::vector<cudf::size_type>>, std::vector<cudf::size_type>>
hybrid_scan_reader_impl::construct_row_group_passes(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  std::size_t total_row_groups,
  std::size_t pass_read_limit) const
{
  CUDF_EXPECTS(
    total_row_groups > 0, "Empty input row group indices encountered", std::invalid_argument);

  CUDF_EXPECTS(row_group_indices.size() == _extended_metadata->get_num_sources(),
               "Mismatch in the number of row group indices vectors and the number of input "
               "datasources",
               std::invalid_argument);

  if (pass_read_limit == 0) {
    return {
      std::vector<std::vector<cudf::size_type>>{row_group_indices.begin(), row_group_indices.end()},
      std::vector<cudf::size_type>{}};
  }

  CUDF_EXPECTS(
    pass_read_limit > 0, "Pass read limit must be greater than 0", std::invalid_argument);

  auto row_groups_info = std::vector<row_group_info>{};
  row_groups_info.reserve(total_row_groups);
  size_t start_row = 0;
  std::for_each(cuda::counting_iterator<cudf::size_type>(0),
                cuda::counting_iterator<cudf::size_type>(row_group_indices.size()),
                [&](auto const source_index) {
                  auto const& src_row_groups = row_group_indices[source_index];
                  std::transform(
                    src_row_groups.begin(),
                    src_row_groups.end(),
                    std::back_inserter(row_groups_info),
                    [&](auto const rg_index) {
                      auto const& row_group =
                        _extended_metadata->get_row_group(rg_index, source_index);
                      auto const [compressed_size, total_size, num_rows, max_leaf_values] =
                        _extended_metadata->get_row_group_properties(row_group);
                      auto rg_info = row_group_info{.index               = rg_index,
                                                    .start_row           = start_row,
                                                    .unadjusted_num_rows = num_rows,
                                                    .source_index        = source_index,
                                                    .compressed_size     = compressed_size,
                                                    .max_leaf_values     = max_leaf_values};
                      start_row += num_rows;
                      return rg_info;
                    });
                });

  auto const comp_read_limit = static_cast<std::size_t>(
    pass_read_limit * cudf::io::parquet::detail::input_limit_compression_reserve);

  auto const pass_data =
    cudf::io::parquet::detail::compute_row_group_passes(row_groups_info, comp_read_limit, 0);

  // Convert offset-based pass boundaries back to vectors of row group indices
  auto const& offsets = pass_data.pass_row_group_offsets;
  auto passes         = std::vector<std::vector<cudf::size_type>>{};
  passes.reserve(offsets.size() - 1);
  auto row_group_source_map       = std::vector<cudf::size_type>{};
  auto const has_multiple_sources = row_group_indices.size() > 1;
  if (has_multiple_sources) { row_group_source_map.reserve(row_groups_info.size()); }
  std::transform(offsets.begin(),
                 offsets.end() - 1,
                 offsets.begin() + 1,
                 std::back_inserter(passes),
                 [&](auto const start, auto const end) {
                   auto pass = std::vector<cudf::size_type>{};
                   pass.reserve(end - start);
                   std::for_each(row_groups_info.begin() + start,
                                 row_groups_info.begin() + end,
                                 [&](auto const& rg_info) {
                                   pass.emplace_back(rg_info.index);
                                   if (has_multiple_sources) {
                                     row_group_source_map.emplace_back(rg_info.source_index);
                                   }
                                 });
                   return pass;
                 });
  return {std::move(passes), std::move(row_group_source_map)};
}

bool hybrid_scan_reader_impl::has_next_table_chunk()
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  if (_espresso != nullptr && (_espresso->pass_active || _espresso->all_cols_bypass_passes > 0)) {
    if (_espresso->pass_active) { return _espresso->has_next(); }
    if (!_file_preprocessed) { return false; }
  }
#endif
  CUDF_EXPECTS(_file_preprocessed, "Chunking not yet setup");
  prepare_data(read_mode::CHUNKED_READ, {}, {}, {});

  // current_input_pass will only be incremented to be == num_passes after
  // the last chunk in the last subpass in the last pass has been returned
  // if not has_more_work then check if this is the first pass in an empty
  // table and return true so it could be read once.
  return has_more_work() or is_first_output_chunk();
}

void hybrid_scan_reader_impl::reset_internal_state()
{
  _row_mask_offset   = 0;
  _file_itm_data     = file_intermediate_data{};
  _file_preprocessed = false;
  _has_page_index    = false;
  _pass_itm_data.reset();
  _pass_page_mask.clear();
  _subpass_page_mask.reset();
  _output_metadata.reset();

  _options.timestamp_type = cudf::data_type{};
  _options.decimal_width  = type_id::EMPTY;
  _options.num_rows       = std::nullopt;
  _options.row_group_indices.clear();
  _options.use_jit_filter              = false;
  _options.case_sensitive_names        = true;
  _options.prepend_source_index_column = false;
  _options.prepend_row_index_column    = false;

  _num_sources             = 0;
  _input_pass_read_limit   = 0;
  _output_chunk_read_limit = 0;
  _strings_to_categorical  = false;
  _reader_column_schema.reset();
  _expr_conv = named_to_reference_converter{};
  _mr        = cudf::get_current_device_resource_ref();
}

void hybrid_scan_reader_impl::initialize_reader_config(parquet_reader_options const& options)
{
  // Strings may be returned as either string or categorical columns
  _strings_to_categorical = options.is_enabled_convert_strings_to_categories();

  _options.timestamp_type              = cudf::data_type{options.get_timestamp_type().id()};
  _options.decimal_width               = options.get_decimal_width();
  _options.use_jit_filter              = options.is_enabled_use_jit_filter();
  _options.case_sensitive_names        = options.is_enabled_case_sensitive_names();
  _options.prepend_source_index_column = options.is_enabled_prepend_source_index_column();
  _options.prepend_row_index_column    = options.is_enabled_prepend_row_index_column();
}

void hybrid_scan_reader_impl::initialize_options(parquet_reader_options const& options,
                                                 std::size_t num_sources,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  // Binary columns can be read as binary or strings
  _reader_column_schema = options.get_column_schema();

  _num_sources = num_sources;

  // CUDA stream to use for internal operations
  _stream = stream;

  // Device memory resource to use for allocations
  _mr = mr;
}

named_to_reference_converter hybrid_scan_reader_impl::build_converted_expression(
  parquet_reader_options const& options)
{
  if (not options.get_filter().has_value()) { return named_to_reference_converter{}; }

  table_metadata metadata;
  populate_metadata(metadata);
  auto expr_conv = named_to_reference_converter(options.get_filter(),
                                                metadata,
                                                _extended_metadata->get_schema_tree(),
                                                options,
                                                options.is_enabled_case_sensitive_names());
  CUDF_EXPECTS(expr_conv.get_converted_expr().has_value(),
               "Columns names in filter expression must be convertible to index references");
  return expr_conv;
}

void hybrid_scan_reader_impl::prepare_data(
  read_mode mode,
  std::span<std::vector<size_type> const> row_group_indices,
  std::span<cudf::device_span<uint8_t const> const> column_chunk_data,
  host_span<bool const> data_page_mask)
{
  // if we have not preprocessed at the whole-file level, do that now
  if (not _file_preprocessed) {
    // setup file level information
    // - read row group information
    // - setup information on (parquet) chunks
    // - compute schedule of input passes
    prepare_row_groups(read_mode::READ_ALL, row_group_indices);
  }

  // handle any chunking work (ratcheting through the subpasses and chunks within
  // our current pass) if in bounds
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes()) {
    handle_chunking(mode, column_chunk_data, data_page_mask);
  }
}

template <typename RowMaskView>
table_with_metadata hybrid_scan_reader_impl::read_chunk_internal(
  read_mode mode, read_columns_mode read_columns_mode, RowMaskView row_mask)
{
  // If `_output_metadata` has been constructed, just copy it over.
  auto out_metadata = _output_metadata ? table_metadata{*_output_metadata} : table_metadata{};
  out_metadata.schema_info.resize(_output_buffers.size());

  // output cudf columns as determined by the top level schema
  auto out_columns = std::vector<std::unique_ptr<column>>{};
  out_columns.reserve(_output_buffers.size());

  // Copy number of total input row groups and number of surviving row groups from predicate
  // pushdown.
  out_metadata.num_input_row_groups = _file_itm_data.num_input_row_groups;
  // Copy the number surviving row groups from each predicate pushdown only if the filter has value
  if (_expr_conv.get_converted_expr().has_value()) {
    out_metadata.num_row_groups_after_stats_filter =
      _file_itm_data.surviving_row_groups.after_stats_filter;
    out_metadata.num_row_groups_after_bloom_filter =
      _file_itm_data.surviving_row_groups.after_bloom_filter;
  }

  // no work to do (this can happen on the first pass if we have no rows to read)
  if (!has_more_work()) {
    // Empty dataframe case: Simply initialize to a list of zeros
    out_metadata.num_rows_per_source =
      std::vector<std::size_t>(_file_itm_data.num_rows_per_source.size(), 0);

    // Finalize output
    return finalize_output(read_columns_mode, out_metadata, out_columns, row_mask);
  }

  auto& pass            = *_pass_itm_data;
  auto& subpass         = *pass.subpass;
  auto const& read_info = subpass.output_chunk_read_info[subpass.current_output_chunk];

  // computes:
  // PageNestingInfo::batch_size for each level of nesting, for each page, taking row bounds into
  // account. PageInfo::skipped_values, which tells us where to start decoding in the input to
  // respect the user bounds. It is only necessary to do this second pass if uses_custom_row_bounds
  // is set (if the user has specified artificial bounds).
  if (uses_custom_row_bounds(mode)) {
    compute_page_sizes(subpass.pages,
                       pass.chunks,
                       subpass_page_mask_span(),
                       read_info.skip_rows,
                       read_info.num_rows,
                       false,  // num_rows is already computed
                       pass.level_type_size,
                       _stream);
  }

  // preprocess strings
  preprocess_chunk_strings(mode, read_info);

  // Allocate memory buffers for the output columns.
  allocate_columns(mode, read_info.skip_rows, read_info.num_rows);

  // Parse data into the output buffers.
  decode_page_data(mode, read_info.skip_rows, read_info.num_rows);

  // Create the final output cudf columns.
  for (std::size_t i = 0; i < _output_buffers.size(); ++i) {
    auto metadata           = _reader_column_schema.has_value()
                                ? std::make_optional<reader_column_schema>((*_reader_column_schema)[i])
                                : std::nullopt;
    auto const& schema      = _extended_metadata->get_schema(_output_column_schemas[i]);
    auto const logical_type = schema.logical_type.value_or(LogicalType{});
    // FIXED_LEN_BYTE_ARRAY never read as string.
    // TODO: if we ever decide that the default reader behavior is to treat unannotated BINARY
    // as binary and not strings, this test needs to change.
    if (schema.type == Type::FIXED_LEN_BYTE_ARRAY and logical_type.type != LogicalType::DECIMAL) {
      metadata = std::make_optional<reader_column_schema>();
      metadata->set_convert_binary_to_strings(false);
      metadata->set_type_length(schema.type_length);
    }
    // Only construct `out_metadata` if `_output_metadata` has not been cached.
    if (!_output_metadata) {
      column_name_info& col_name = out_metadata.schema_info[i];
      out_columns.emplace_back(make_column(_output_buffers[i], &col_name, metadata, _stream));
    } else {
      out_columns.emplace_back(make_column(_output_buffers[i], nullptr, metadata, _stream));
    }
  }

  out_columns =
    cudf::structs::detail::enforce_null_consistency(std::move(out_columns), _stream, _mr);

  // Compute the output number of rows per source
  if (mode == read_mode::CHUNKED_READ) {
    out_metadata.num_rows_per_source =
      calculate_output_num_rows_per_source(read_info.skip_rows, read_info.num_rows);
  } else {
    // Move is okay here as we are reading in one go.
    out_metadata.num_rows_per_source = std::move(_file_itm_data.num_rows_per_source);
  }

  // Add empty columns if needed. Filter output columns based on filter.
  return finalize_output(read_columns_mode, out_metadata, out_columns, row_mask);
}

template <typename RowMaskView>
table_with_metadata hybrid_scan_reader_impl::finalize_output(
  read_columns_mode read_columns_mode,
  table_metadata& out_metadata,
  std::vector<std::unique_ptr<column>>& out_columns,
  RowMaskView row_mask)
{
  // Create empty columns as needed (this can happen if we've ended up with no actual data to
  // read)
  for (std::size_t i = out_columns.size(); i < _output_buffers.size(); ++i) {
    if (!_output_metadata) {
      column_name_info& col_name = out_metadata.schema_info[i];
      out_columns.emplace_back(io::detail::empty_like(_output_buffers[i], &col_name, _stream, _mr));
    } else {
      out_columns.emplace_back(io::detail::empty_like(_output_buffers[i], nullptr, _stream, _mr));
    }
  }

  if (!_output_metadata) {
    populate_metadata(out_metadata);
    // Finally, save the output table metadata into `_output_metadata` for reuse next time.
    _output_metadata = std::make_unique<table_metadata>(out_metadata);
  }

  // Row-range of the current output chunk relative to the current row group selection.
  auto const read_info =
    (_file_itm_data._current_input_pass < _file_itm_data.num_passes())
      ? _pass_itm_data->subpass
          ->output_chunk_read_info[_pass_itm_data->subpass->current_output_chunk]
      : cudf::io::parquet::detail::row_range{0, 0};

  // advance output chunk/subpass/pass info for non-empty tables if and only if we are in bounds
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes()) {
    auto& pass    = *_pass_itm_data;
    auto& subpass = *pass.subpass;
    subpass.current_output_chunk++;
  }

  // increment the output chunk count
  _file_itm_data._output_chunk_count++;

  apply_decimal_width_cast(out_columns);

  // Prepend the source and row index columns to filter columns only
  if (read_columns_mode == read_columns_mode::FILTER_COLUMNS) {
    if (_options.prepend_row_index_column) {
      out_columns.emplace(out_columns.begin(),
                          synthesize_row_index_column(read_info, _stream, _mr));
      out_metadata.schema_info.emplace(out_metadata.schema_info.begin(),
                                       column_name_info{.name = "row_index", .is_nullable = false});
    }
    if (_options.prepend_source_index_column) {
      out_columns.emplace(
        out_columns.begin(),
        synthesize_source_index_column(out_metadata.num_rows_per_source, _stream, _mr));
      out_metadata.schema_info.emplace(
        out_metadata.schema_info.begin(),
        column_name_info{.name = "source_index", .is_nullable = false});
    }
  }

  // Create a table from the output columns.
  auto read_table = std::make_unique<table>(std::move(out_columns));

  // If the input row mask is empty, all rows are pruned anyway.
  if (row_mask.is_empty()) {
    _output_chunk_produced = true;
    return {std::move(read_table), std::move(out_metadata)};
  }

  CUDF_EXPECTS(row_mask.type().id() == type_id::BOOL8, "Input row mask must be a boolean column");

  // Get the current row mask offset
  auto const mask_offset = _row_mask_offset;
  // Update the row mask offset and the output chunk produced flag
  _row_mask_offset += read_table->num_rows();
  _output_chunk_produced = true;

  // Clear the number of rows per source as it is not valid after filtering
  out_metadata.num_rows_per_source.clear();

  // For filter columns, apply the filter expression and update the input row mask
  if constexpr (std::is_same_v<RowMaskView, cudf::mutable_column_view>) {
    CUDF_EXPECTS(read_columns_mode == read_columns_mode::FILTER_COLUMNS, "Invalid read mode");

    // Compute the final filter expression incorporating any column reference offsets in _expr_conv
    auto const final_filter      = compute_offset_filter();
    auto const final_filter_expr = final_filter.get_converted_expr();

    auto final_row_mask = cudf::detail::compute_column(*read_table,
                                                       final_filter_expr.value().get(),
                                                       _stream,
                                                       cudf::get_current_device_resource_ref());
    CUDF_EXPECTS(final_row_mask->view().type().id() == type_id::BOOL8,
                 "Predicate filter should return a boolean");

    // Apply the final row mask to get the final output table
    auto output_table = cudf::detail::apply_mask(
      read_table->view(), *final_row_mask, cudf::detail::mask_type::RETENTION, _stream, _mr);

    // Update the input row mask to reflect the final row mask.
    update_row_mask(final_row_mask->view(), row_mask, mask_offset, _stream);

    // Return the final output table and metadata
    return {std::move(output_table), std::move(out_metadata)};
  }
  // For payload columns, simply apply the input row mask to the table.
  else {
    CUDF_EXPECTS(read_columns_mode == read_columns_mode::PAYLOAD_COLUMNS, "Invalid read mode");

    CUDF_EXPECTS(mask_offset + read_table->num_rows() <= row_mask.size(),
                 "Encountered invalid sized row mask to apply");
    auto effective_row_mask =
      (read_table->num_rows() == row_mask.size())
        ? row_mask
        : cudf::split(row_mask, {mask_offset, mask_offset + read_table->num_rows()}, _stream)[1];
    auto output_table = cudf::detail::apply_mask(
      read_table->view(), effective_row_mask, cudf::detail::mask_type::RETENTION, _stream, _mr);
    return {std::move(output_table), std::move(out_metadata)};
  }
}

void hybrid_scan_reader_impl::set_pass_page_mask(std::span<bool const> data_page_mask)
{
  auto const& pass   = _pass_itm_data;
  auto const& chunks = pass->chunks;

  _pass_page_mask        = cudf::detail::make_empty_host_vector<bool>(pass->pages.size(), _stream);
  auto const num_columns = _input_columns.size();

  // Handle the empty page mask case
  if (data_page_mask.empty()) {
    std::fill(_pass_page_mask.begin(), _pass_page_mask.end(), true);
    return;
  }

  std::size_t num_inserted_data_pages = 0;
  std::for_each(
    cuda::counting_iterator<std::size_t>{0},
    cuda::counting_iterator{_input_columns.size()},
    [&](auto col_idx) {
      for (std::size_t chunk_idx = col_idx; chunk_idx < chunks.size(); chunk_idx += num_columns) {
        // Insert a true value for each dictionary page
        if (chunks[chunk_idx].num_dict_pages > 0) { _pass_page_mask.push_back(true); }

        // Number of data pages in this column chunk
        auto const num_data_pages_this_col_chunk = chunks[chunk_idx].num_data_pages;

        // Make sure we have enough page mask for this column chunk
        CUDF_EXPECTS(
          data_page_mask.size() >= num_inserted_data_pages + num_data_pages_this_col_chunk,
          "Encountered invalid data page mask size");

        // Insert page mask for this column chunk
        _pass_page_mask.insert(
          _pass_page_mask.end(),
          data_page_mask.begin() + num_inserted_data_pages,
          data_page_mask.begin() + num_inserted_data_pages + num_data_pages_this_col_chunk);
        // Update the number of inserted data pages
        num_inserted_data_pages += num_data_pages_this_col_chunk;
      }
    });

  // Make sure we inserted exactly the number of pages for this pass
  CUDF_EXPECTS(_pass_page_mask.size() == pass->pages.size(),
               "Encountered mismatch in number of pass pages and page mask size");
}

}  // namespace cudf::io::parquet::experimental::detail
