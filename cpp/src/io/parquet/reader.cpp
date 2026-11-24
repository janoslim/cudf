/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reader_impl.hpp"

#ifdef CUDF_HAS_ESPRESSO_CSD
#include <cudf/concatenate.hpp>
#include <espresso/cudf_io.hpp>
#include <espresso/late_mat_bridge.hpp>
#endif

namespace cudf::io::parquet::detail {

struct espresso_reader_bridge {
#ifdef CUDF_HAS_ESPRESSO_CSD
  std::unique_ptr<espresso::late_mat::VirtualChunkedScanState> virtual_state;
  std::unique_ptr<espresso::cudf_io::csd_chunked_scan_state> csd_state;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  espresso_reader_bridge(rmm::cuda_stream_view s, rmm::device_async_resource_ref m)
    : stream(s), mr(m)
  {
  }

  static std::unique_ptr<espresso_reader_bridge> try_open(std::size_t chunk_read_limit,
                                                          std::size_t pass_read_limit,
                                                          parquet_reader_options const& options,
                                                          rmm::cuda_stream_view stream,
                                                          rmm::device_async_resource_ref mr)
  {
    if (auto virt = espresso::late_mat::VirtualChunkedScanState::try_create(
          chunk_read_limit, pass_read_limit, options, stream, mr);
        virt != nullptr) {
      auto b           = std::make_unique<espresso_reader_bridge>(stream, mr);
      b->virtual_state = std::move(virt);
      return b;
    }
    if (auto csd = espresso::cudf_io::csd_chunked_scan_state::try_create(
          chunk_read_limit, pass_read_limit, options, stream, mr);
        csd != nullptr) {
      auto b       = std::make_unique<espresso_reader_bridge>(stream, mr);
      b->csd_state = std::move(csd);
      return b;
    }
    return nullptr;
  }

  [[nodiscard]] bool has_next() const
  {
    return virtual_state ? virtual_state->has_next() : (csd_state && csd_state->has_next());
  }

  [[nodiscard]] table_with_metadata read_chunk()
  {
    return virtual_state ? virtual_state->read_chunk() : csd_state->read_chunk();
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
#endif
};

reader::reader() = default;

reader::reader(std::vector<std::unique_ptr<datasource>>&& sources,
               std::vector<FileMetaData>&& parquet_metadatas,
               parquet_reader_options const& options,
               rmm::cuda_stream_view stream,
               rmm::device_async_resource_ref mr)
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  _espresso_bridge = espresso_reader_bridge::try_open(0, 0, options, stream, mr);
  if (_espresso_bridge != nullptr) { return; }
#endif
  _impl = std::make_unique<reader_impl>(
    std::move(sources), std::move(parquet_metadatas), options, stream, mr);
}

reader::~reader() = default;

table_with_metadata reader::read()
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  if (_espresso_bridge != nullptr) { return _espresso_bridge->read_all(); }
#endif
  return _impl->read();
}

chunked_reader::chunked_reader(std::size_t chunk_read_limit,
                               std::size_t pass_read_limit,
                               std::vector<std::unique_ptr<datasource>>&& sources,
                               std::vector<FileMetaData>&& parquet_metadatas,
                               parquet_reader_options const& options,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  _espresso_bridge =
    espresso_reader_bridge::try_open(chunk_read_limit, pass_read_limit, options, stream, mr);
  if (_espresso_bridge != nullptr) { return; }
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
  if (_espresso_bridge != nullptr) { return _espresso_bridge->has_next(); }
#endif
  return _impl->has_next();
}

table_with_metadata chunked_reader::read_chunk() const
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  if (_espresso_bridge != nullptr) { return _espresso_bridge->read_chunk(); }
#endif
  return _impl->read_chunk();
}

}  // namespace cudf::io::parquet::detail
