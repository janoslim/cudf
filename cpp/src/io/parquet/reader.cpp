/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reader_impl.hpp"

#ifdef CUDF_HAS_ESPRESSO_CSD
#include "io/utilities/espresso_datasource_registry.hpp"
#include <cudf/concatenate.hpp>
#include <espresso/cudf_io.hpp>
#include <espresso/late_mat_bridge.hpp>
#endif

namespace cudf::io::parquet::detail {

#ifdef CUDF_HAS_ESPRESSO_CSD
struct espresso_reader_bridge {
  std::unique_ptr<espresso::late_mat::SelfJoinChunkedScanState> sj_state;
  std::unique_ptr<espresso::cudf_io::csd_chunked_scan_state> csd_state;
  cuda::stream_ref stream;
  rmm::device_async_resource_ref mr;

  espresso_reader_bridge(cuda::stream_ref s, rmm::device_async_resource_ref m)
    : stream(s), mr(m)
  {
  }

  static std::shared_ptr<espresso_reader_bridge> try_open(
    std::size_t chunk_read_limit,
    std::size_t pass_read_limit,
    std::span<FileMetaData const> parquet_metadatas,
    parquet_reader_options const& options,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr)
  {
    auto const meta_opts =
      cudf::io::espresso_detail::try_options_with_metadata_filepaths(options, parquet_metadatas);
    parquet_reader_options const& eff_opts = meta_opts.has_value() ? *meta_opts : options;
    if (auto sj = espresso::late_mat::SelfJoinChunkedScanState::try_create(
          chunk_read_limit, pass_read_limit, eff_opts, stream, mr);
        sj != nullptr) {
      auto b      = std::make_shared<espresso_reader_bridge>(stream, mr);
      b->sj_state = std::move(sj);
      return b;
    }
    if (auto csd = espresso::cudf_io::csd_chunked_scan_state::try_create(
          chunk_read_limit, pass_read_limit, eff_opts, stream, mr);
        csd != nullptr) {
      auto b       = std::make_shared<espresso_reader_bridge>(stream, mr);
      b->csd_state = std::move(csd);
      return b;
    }
    return nullptr;
  }

  [[nodiscard]] bool has_next() const
  {
    return sj_state ? sj_state->has_next() : (csd_state && csd_state->has_next());
  }

  [[nodiscard]] table_with_metadata read_chunk()
  {
    return sj_state ? sj_state->read_chunk() : csd_state->read_chunk();
  }

  [[nodiscard]] table_with_metadata read_all()
  {
    auto first = read_chunk();
    if (!has_next()) { return first; }
    std::vector<std::unique_ptr<cudf::table>> parts;
    std::vector<cudf::table_view> views;
    auto meta = std::move(first.metadata);
    views.push_back(first.tbl->view());
    parts.push_back(std::move(first.tbl));
    while (has_next()) {
      auto nxt = read_chunk();
      views.push_back(nxt.tbl->view());
      parts.push_back(std::move(nxt.tbl));
    }
    return table_with_metadata{cudf::concatenate(views, stream, mr), std::move(meta)};
  }
};

reader_impl::reader_impl(std::shared_ptr<espresso_reader_bridge> bridge)
  : reader_impl()
{
  _espresso_bridge = std::move(bridge);
}

bool reader_impl::espresso_has_next() const { return _espresso_bridge->has_next(); }
table_with_metadata reader_impl::espresso_read_chunk() { return _espresso_bridge->read_chunk(); }
table_with_metadata reader_impl::espresso_read_all() { return _espresso_bridge->read_all(); }
#endif

reader::reader() = default;

reader::reader(std::vector<std::unique_ptr<datasource>>&& sources,
               std::vector<FileMetaData>&& parquet_metadatas,
               parquet_reader_options const& options,
               cuda::stream_ref stream,
               cudf::memory_resources mr)
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  if (auto b =
        espresso_reader_bridge::try_open(0, 0, parquet_metadatas, options, stream, mr.get_output_mr());
      b != nullptr) {
    _impl = std::make_unique<reader_impl>(std::move(b));
    return;
  }
#endif
  _impl = std::make_unique<reader_impl>(
    std::move(sources), std::move(parquet_metadatas), options, stream, mr);
}

reader::~reader() = default;

table_with_metadata reader::read()
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  if (_impl->has_espresso_bridge()) { return _impl->espresso_read_all(); }
#endif
  return _impl->read();
}

chunked_reader::chunked_reader(std::size_t chunk_read_limit,
                               std::size_t pass_read_limit,
                               std::vector<std::unique_ptr<datasource>>&& sources,
                               std::vector<FileMetaData>&& parquet_metadatas,
                               parquet_reader_options const& options,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr)
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  if (auto b = espresso_reader_bridge::try_open(
        chunk_read_limit, pass_read_limit, parquet_metadatas, options, stream, mr.get_output_mr());
      b != nullptr) {
    _impl = std::make_unique<reader_impl>(std::move(b));
    return;
  }
#endif
  _impl = std::make_unique<reader_impl>(chunk_read_limit,
                                        pass_read_limit,
                                        std::move(sources),
                                        std::move(parquet_metadatas),
                                        options,
                                        stream,
                                        mr);
}

chunked_reader::~chunked_reader() = default;

bool chunked_reader::has_next() const
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  if (_impl->has_espresso_bridge()) { return _impl->espresso_has_next(); }
#endif
  return _impl->has_next();
}

table_with_metadata chunked_reader::read_chunk() const
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  if (_impl->has_espresso_bridge()) { return _impl->espresso_read_chunk(); }
#endif
  return _impl->read_chunk();
}

}  // namespace cudf::io::parquet::detail
