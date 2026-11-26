/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <string>
#ifdef CUDF_HAS_ESPRESSO_CSD
#include <espresso/cudf_io.hpp>
#include <espresso/late_mat_bridge.hpp>
#include <limits>
#include <type_traits>
#endif

namespace cudf {

#ifdef CUDF_HAS_ESPRESSO_CSD
namespace {
[[nodiscard]] inline bool espresso_scalar_mirror_active() noexcept
{
  return espresso::cudf_io::is_csd_dropin_enabled() || espresso::late_mat::is_sql_rewrite_bridge_enabled();
}

template <typename T>
inline void mirror_fixed_width_host_scalar(void const* validity_ptr,
                                           void const* data_ptr,
                                           data_type dtype,
                                           T value,
                                           bool is_valid) noexcept
{
  if (!espresso_scalar_mirror_active()) { return; }
  espresso::cudf_io::csd_host_literal_value lit{dtype, is_valid, 0, 0, 0.0};
  if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t> ||
                std::is_same_v<T, int64_t>) {
    lit.signed_v = static_cast<std::int64_t>(value);
  } else if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, uint32_t> ||
                       std::is_same_v<T, uint64_t>) {
    lit.uint_v = static_cast<std::uint64_t>(value);
  } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    lit.float_v = static_cast<double>(value);
  } else if constexpr (std::is_same_v<T, timestamp_D> || std::is_same_v<T, timestamp_s> ||
                       std::is_same_v<T, timestamp_ms> || std::is_same_v<T, timestamp_us> ||
                       std::is_same_v<T, timestamp_ns>) {
    lit.signed_v = static_cast<std::int64_t>(value.time_since_epoch().count());
  } else {
    espresso::cudf_io::erase_host_scalar_mirror(validity_ptr);
    return;
  }
  espresso::cudf_io::record_host_scalar_mirror(validity_ptr, data_ptr, lit);
}

template <typename Rep>
inline void mirror_fixed_point_host_scalar(void const* validity_ptr,
                                           void const* data_ptr,
                                           data_type dtype,
                                           Rep raw_value,
                                           bool is_valid) noexcept
{
  if (!espresso_scalar_mirror_active()) { return; }
  if constexpr (sizeof(Rep) > sizeof(std::int64_t)) {
    if (raw_value < static_cast<Rep>(std::numeric_limits<std::int64_t>::min()) ||
        raw_value > static_cast<Rep>(std::numeric_limits<std::int64_t>::max())) {
      espresso::cudf_io::erase_host_scalar_mirror(validity_ptr);
      return;
    }
  }
  espresso::cudf_io::csd_host_literal_value lit{dtype, is_valid, static_cast<std::int64_t>(raw_value), 0, 0.0};
  espresso::cudf_io::record_host_scalar_mirror(validity_ptr, data_ptr, lit);
}
}  // namespace
#endif

scalar::scalar(data_type type,
               bool is_valid,
               rmm::cuda_stream_view stream,
               rmm::device_async_resource_ref mr)
  : _type(type), _is_valid(is_valid, stream, mr)
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  if (espresso_scalar_mirror_active()) { espresso::cudf_io::erase_host_scalar_mirror(_is_valid.data()); }
#endif
}

scalar::scalar(scalar const& other, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
  : _type(other.type()), _is_valid(other._is_valid, stream, mr)
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  if (espresso_scalar_mirror_active()) { espresso::cudf_io::erase_host_scalar_mirror(_is_valid.data()); }
#endif
}

data_type scalar::type() const noexcept { return _type; }

void scalar::set_valid_async(bool is_valid, rmm::cuda_stream_view stream)
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  if (espresso_scalar_mirror_active()) { espresso::cudf_io::erase_host_scalar_mirror(_is_valid.data()); }
#endif
  _is_valid.set_value_async(is_valid, stream);
}

bool scalar::is_valid(rmm::cuda_stream_view stream) const { return _is_valid.value(stream); }

bool* scalar::validity_data()
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  if (espresso_scalar_mirror_active()) { espresso::cudf_io::erase_host_scalar_mirror(_is_valid.data()); }
#endif
  return _is_valid.data();
}

bool const* scalar::validity_data() const { return _is_valid.data(); }

string_scalar::string_scalar(std::string_view string,
                             bool is_valid,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
  : scalar(data_type(type_id::STRING), is_valid, stream, mr),
    _data(string.data(), string.size(), stream, mr)
{
  CUDF_EXPECTS(
    string.size() <= static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
    "Data exceeds the string size limit",
    std::overflow_error);
}

string_scalar::string_scalar(string_scalar const& other,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
  : scalar(other, stream, mr), _data(other._data, stream, mr)
{
}

string_scalar::string_scalar(rmm::device_scalar<value_type>& data,
                             bool is_valid,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
  : string_scalar(data.value(stream), is_valid, stream, mr)
{
}

string_scalar::string_scalar(value_type const& source,
                             bool is_valid,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
  : scalar(data_type(type_id::STRING), is_valid, stream, mr),
    _data(source.data(), source.size_bytes(), stream, mr)
{
}

string_scalar::string_scalar(rmm::device_buffer&& data,
                             bool is_valid,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
  : scalar(data_type(type_id::STRING), is_valid, stream, mr), _data(std::move(data))
{
}

string_scalar::value_type string_scalar::value(rmm::cuda_stream_view stream) const
{
  return value_type{data(), size()};
}

size_type string_scalar::size() const { return _data.size(); }

char const* string_scalar::data() const { return static_cast<char const*>(_data.data()); }

std::string string_scalar::to_string(rmm::cuda_stream_view stream) const
{
  std::string result(size(), '\0');
  detail::cuda_memcpy(host_span<char>{result.data(), result.size()},
                      device_span<char const>{data(), _data.size()},
                      stream);
  return result;
}

template <typename T>
fixed_point_scalar<T>::fixed_point_scalar(rep_type value,
                                          numeric::scale_type scale,
                                          bool is_valid,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
  : scalar{data_type{type_to_id<T>(), static_cast<int32_t>(scale)}, is_valid, stream, mr},
    _data{value, stream, mr}
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  mirror_fixed_point_host_scalar(static_cast<scalar const*>(this)->validity_data(), _data.data(), this->type(), value, is_valid);
#endif
}

template <typename T>
fixed_point_scalar<T>::fixed_point_scalar(rep_type value,
                                          bool is_valid,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
  : scalar{data_type{type_to_id<T>(), 0}, is_valid, stream, mr}, _data{value, stream, mr}
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  mirror_fixed_point_host_scalar(static_cast<scalar const*>(this)->validity_data(), _data.data(), this->type(), value, is_valid);
#endif
}

template <typename T>
fixed_point_scalar<T>::fixed_point_scalar(T value,
                                          bool is_valid,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
  : scalar{data_type{type_to_id<T>(), value.scale()}, is_valid, stream, mr},
    _data{value.value(), stream, mr}
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  mirror_fixed_point_host_scalar(static_cast<scalar const*>(this)->validity_data(), _data.data(), this->type(), value.value(), is_valid);
#endif
}

template <typename T>
fixed_point_scalar<T>::fixed_point_scalar(rmm::device_scalar<rep_type>&& data,
                                          numeric::scale_type scale,
                                          bool is_valid,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
  : scalar{data_type{type_to_id<T>(), scale}, is_valid, stream, mr}, _data{std::move(data)}
{
}

template <typename T>
fixed_point_scalar<T>::fixed_point_scalar(fixed_point_scalar<T> const& other,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
  : scalar{other, stream, mr}, _data(other._data, stream, mr)
{
}

template <typename T>
typename fixed_point_scalar<T>::rep_type fixed_point_scalar<T>::value(
  rmm::cuda_stream_view stream) const
{
  return _data.value(stream);
}

template <typename T>
T fixed_point_scalar<T>::fixed_point_value(rmm::cuda_stream_view stream) const
{
  return value_type{
    numeric::scaled_integer<rep_type>{_data.value(stream), numeric::scale_type{type().scale()}}};
}

template <typename T>
typename fixed_point_scalar<T>::rep_type* fixed_point_scalar<T>::data()
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  if (espresso_scalar_mirror_active()) { espresso::cudf_io::erase_host_scalar_mirror(static_cast<scalar const*>(this)->validity_data()); }
#endif
  return _data.data();
}

template <typename T>
typename fixed_point_scalar<T>::rep_type const* fixed_point_scalar<T>::data() const
{
  return _data.data();
}

/**
 * @brief These define the valid fixed-point scalar types.
 *
 * See `is_fixed_point` in @see cudf/utilities/traits.hpp
 *
 * Adding a new supported type only requires adding the appropriate line here
 * and does not require updating the scalar.hpp file.
 */
template class fixed_point_scalar<numeric::decimal32>;
template class fixed_point_scalar<numeric::decimal64>;
template class fixed_point_scalar<numeric::decimal128>;

namespace CUDF_HIDDEN detail {

template <typename T>
fixed_width_scalar<T>::fixed_width_scalar(T value,
                                          bool is_valid,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
  : scalar(data_type(type_to_id<T>()), is_valid, stream, mr), _data(value, stream, mr)
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  mirror_fixed_width_host_scalar(static_cast<scalar const*>(this)->validity_data(), _data.data(), this->type(), value, is_valid);
#endif
}

template <typename T>
fixed_width_scalar<T>::fixed_width_scalar(rmm::device_scalar<T>&& data,
                                          bool is_valid,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
  : scalar(data_type(type_to_id<T>()), is_valid, stream, mr), _data{std::move(data)}
{
}

template <typename T>
fixed_width_scalar<T>::fixed_width_scalar(fixed_width_scalar<T> const& other,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
  : scalar{other, stream, mr}, _data(other._data, stream, mr)
{
}

template <typename T>
void fixed_width_scalar<T>::set_value(T value, rmm::cuda_stream_view stream)
{
  _data.set_value_async(value, stream);
  this->set_valid_async(true, stream);
#ifdef CUDF_HAS_ESPRESSO_CSD
  mirror_fixed_width_host_scalar(static_cast<scalar const*>(this)->validity_data(), _data.data(), this->type(), value, true);
#endif
}

template <typename T>
T fixed_width_scalar<T>::value(rmm::cuda_stream_view stream) const
{
  return _data.value(stream);
}

template <typename T>
T* fixed_width_scalar<T>::data()
{
#ifdef CUDF_HAS_ESPRESSO_CSD
  if (espresso_scalar_mirror_active()) { espresso::cudf_io::erase_host_scalar_mirror(static_cast<scalar const*>(this)->validity_data()); }
#endif
  return _data.data();
}

template <typename T>
T const* fixed_width_scalar<T>::data() const
{
  return _data.data();
}

/**
 * @brief These define the valid fixed-width scalar types.
 *
 * See `is_fixed_width` in @see cudf/utilities/traits.hpp
 *
 * Adding a new supported type only requires adding the appropriate line here
 * and does not require updating the scalar.hpp file.
 */
template class fixed_width_scalar<bool>;
template class fixed_width_scalar<int8_t>;
template class fixed_width_scalar<int16_t>;
template class fixed_width_scalar<int32_t>;
template class fixed_width_scalar<int64_t>;
template class fixed_width_scalar<__int128_t>;
template class fixed_width_scalar<uint8_t>;
template class fixed_width_scalar<uint16_t>;
template class fixed_width_scalar<uint32_t>;
template class fixed_width_scalar<uint64_t>;
template class fixed_width_scalar<float>;
template class fixed_width_scalar<double>;
template class fixed_width_scalar<timestamp_D>;
template class fixed_width_scalar<timestamp_s>;
template class fixed_width_scalar<timestamp_ms>;
template class fixed_width_scalar<timestamp_us>;
template class fixed_width_scalar<timestamp_ns>;
template class fixed_width_scalar<duration_D>;
template class fixed_width_scalar<duration_s>;
template class fixed_width_scalar<duration_ms>;
template class fixed_width_scalar<duration_us>;
template class fixed_width_scalar<duration_ns>;

}  // namespace CUDF_HIDDEN detail

template <typename T>
numeric_scalar<T>::numeric_scalar(T value,
                                  bool is_valid,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
  : detail::fixed_width_scalar<T>(value, is_valid, stream, mr)
{
}

template <typename T>
numeric_scalar<T>::numeric_scalar(rmm::device_scalar<T>&& data,
                                  bool is_valid,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
  : detail::fixed_width_scalar<T>(std::forward<rmm::device_scalar<T>>(data), is_valid, stream, mr)
{
}

template <typename T>
numeric_scalar<T>::numeric_scalar(numeric_scalar<T> const& other,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
  : detail::fixed_width_scalar<T>{other, stream, mr}
{
}

/**
 * @brief These define the valid numeric scalar types.
 *
 * See `is_numeric` in @see cudf/utilities/traits.hpp
 *
 * Adding a new supported type only requires adding the appropriate line here
 * and does not require updating the scalar.hpp file.
 */
template class numeric_scalar<bool>;
template class numeric_scalar<int8_t>;
template class numeric_scalar<int16_t>;
template class numeric_scalar<int32_t>;
template class numeric_scalar<int64_t>;
template class numeric_scalar<__int128_t>;
template class numeric_scalar<uint8_t>;
template class numeric_scalar<uint16_t>;
template class numeric_scalar<uint32_t>;
template class numeric_scalar<uint64_t>;
template class numeric_scalar<float>;
template class numeric_scalar<double>;

template <typename T>
chrono_scalar<T>::chrono_scalar(T value,
                                bool is_valid,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
  : detail::fixed_width_scalar<T>(value, is_valid, stream, mr)
{
}

template <typename T>
chrono_scalar<T>::chrono_scalar(rmm::device_scalar<T>&& data,
                                bool is_valid,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
  : detail::fixed_width_scalar<T>(std::forward<rmm::device_scalar<T>>(data), is_valid, stream, mr)
{
}

template <typename T>
chrono_scalar<T>::chrono_scalar(chrono_scalar<T> const& other,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
  : detail::fixed_width_scalar<T>{other, stream, mr}
{
}

/**
 * @brief These define the valid chrono scalar types.
 *
 * See `is_chrono` in @see cudf/utilities/traits.hpp
 *
 * Adding a new supported type only requires adding the appropriate line here
 * and does not require updating the scalar.hpp file.
 */
template class chrono_scalar<timestamp_D>;
template class chrono_scalar<timestamp_s>;
template class chrono_scalar<timestamp_ms>;
template class chrono_scalar<timestamp_us>;
template class chrono_scalar<timestamp_ns>;
template class chrono_scalar<duration_D>;
template class chrono_scalar<duration_s>;
template class chrono_scalar<duration_ms>;
template class chrono_scalar<duration_us>;
template class chrono_scalar<duration_ns>;

template <typename T>
duration_scalar<T>::duration_scalar(rep_type value,
                                    bool is_valid,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
  : chrono_scalar<T>(T{value}, is_valid, stream, mr)
{
}

template <typename T>
duration_scalar<T>::duration_scalar(duration_scalar<T> const& other,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
  : chrono_scalar<T>{other, stream, mr}
{
}

template <typename T>
typename duration_scalar<T>::rep_type duration_scalar<T>::count(rmm::cuda_stream_view stream)
{
  return this->value(stream).count();
}

/**
 * @brief These define the valid duration scalar types.
 *
 * See `is_duration` in @see cudf/utilities/traits.hpp
 *
 * Adding a new supported type only requires adding the appropriate line here
 * and does not require updating the scalar.hpp file.
 */
template class duration_scalar<duration_D>;
template class duration_scalar<duration_s>;
template class duration_scalar<duration_ms>;
template class duration_scalar<duration_us>;
template class duration_scalar<duration_ns>;

template <typename T>
typename timestamp_scalar<T>::rep_type timestamp_scalar<T>::ticks_since_epoch(
  rmm::cuda_stream_view stream)
{
  return this->value(stream).time_since_epoch().count();
}

/**
 * @brief These define the valid timestamp scalar types.
 *
 * See `is_timestamp` in @see cudf/utilities/traits.hpp
 *
 * Adding a new supported type only requires adding the appropriate line here
 * and does not require updating the scalar.hpp file.
 */
template class timestamp_scalar<timestamp_D>;
template class timestamp_scalar<timestamp_s>;
template class timestamp_scalar<timestamp_ms>;
template class timestamp_scalar<timestamp_us>;
template class timestamp_scalar<timestamp_ns>;

template <typename T>
template <typename D>
timestamp_scalar<T>::timestamp_scalar(D const& value,
                                      bool is_valid,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
  : chrono_scalar<T>(T{typename T::duration{value}}, is_valid, stream, mr)
{
}

template <typename T>
timestamp_scalar<T>::timestamp_scalar(timestamp_scalar<T> const& other,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
  : chrono_scalar<T>{other, stream, mr}
{
}

#define TS_CTOR(TimestampType, DurationType)                  \
  template timestamp_scalar<TimestampType>::timestamp_scalar( \
    DurationType const&, bool, rmm::cuda_stream_view, rmm::device_async_resource_ref);

/**
 * @brief These are the valid combinations of duration types to timestamp types.
 */
TS_CTOR(timestamp_D, duration_D)
TS_CTOR(timestamp_D, int32_t)
TS_CTOR(timestamp_s, duration_D)
TS_CTOR(timestamp_s, duration_s)
TS_CTOR(timestamp_s, int64_t)
TS_CTOR(timestamp_ms, duration_D)
TS_CTOR(timestamp_ms, duration_s)
TS_CTOR(timestamp_ms, duration_ms)
TS_CTOR(timestamp_ms, int64_t)
TS_CTOR(timestamp_us, duration_D)
TS_CTOR(timestamp_us, duration_s)
TS_CTOR(timestamp_us, duration_ms)
TS_CTOR(timestamp_us, duration_us)
TS_CTOR(timestamp_us, int64_t)
TS_CTOR(timestamp_ns, duration_D)
TS_CTOR(timestamp_ns, duration_s)
TS_CTOR(timestamp_ns, duration_ms)
TS_CTOR(timestamp_ns, duration_us)
TS_CTOR(timestamp_ns, duration_ns)
TS_CTOR(timestamp_ns, int64_t)

list_scalar::list_scalar(cudf::column_view const& data,
                         bool is_valid,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr)
  : scalar(data_type(type_id::LIST), is_valid, stream, mr), _data(data, stream, mr)
{
}

list_scalar::list_scalar(cudf::column&& data,
                         bool is_valid,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr)
  : scalar(data_type(type_id::LIST), is_valid, stream, mr), _data(std::move(data))
{
}

list_scalar::list_scalar(list_scalar const& other,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr)
  : scalar{other, stream, mr}, _data(other._data, stream, mr)
{
}

column_view list_scalar::view() const { return _data.view(); }

struct_scalar::struct_scalar(struct_scalar const& other,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
  : scalar{other, stream, mr}, _data(other._data, stream, mr)
{
}

struct_scalar::struct_scalar(table_view const& data,
                             bool is_valid,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
  : scalar(data_type(type_id::STRUCT), is_valid, stream, mr),
    _data{init_data(table{data, stream, mr}, is_valid, stream, mr)}
{
  assert_valid_size();
}

struct_scalar::struct_scalar(std::span<column_view const> data,
                             bool is_valid,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
  : scalar(data_type(type_id::STRUCT), is_valid, stream, mr),
    _data{
      init_data(table{table_view{std::vector<column_view>{data.begin(), data.end()}}, stream, mr},
                is_valid,
                stream,
                mr)}
{
  assert_valid_size();
}

struct_scalar::struct_scalar(table&& data,
                             bool is_valid,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
  : scalar(data_type(type_id::STRUCT), is_valid, stream, mr),
    _data{init_data(std::move(data), is_valid, stream, mr)}
{
  assert_valid_size();
}

table_view struct_scalar::view() const { return _data.view(); }

void struct_scalar::assert_valid_size()
{
  auto const tv = _data.view();
  CUDF_EXPECTS(
    std::all_of(tv.begin(), tv.end(), [](column_view const& col) { return col.size() == 1; }),
    "Struct scalar inputs must have exactly 1 row");
}

table struct_scalar::init_data(table&& data,
                               bool is_valid,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  if (is_valid) { return std::move(data); }

  auto data_cols = data.release();

  // push validity mask down
  auto const validity = cudf::detail::create_null_mask(
    1, mask_state::ALL_NULL, stream, cudf::get_current_device_resource_ref());
  for (auto& col : data_cols) {
    col = cudf::structs::detail::superimpose_and_sanitize_nulls(
      static_cast<bitmask_type const*>(validity.data()), 1, std::move(col), stream, mr);
  }

  return table{std::move(data_cols)};
}

}  // namespace cudf
