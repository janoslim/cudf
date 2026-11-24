# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# This function finds rtcx
function(find_and_configure_rtcx VERSION)

  # Ensure rtcx installs its targets unconditionally. In shared builds the static library is
  # absorbed into libcudf.so, but CMake still requires the target to be in an export set for
  # install(EXPORT) validation. In static builds consumers need to link librtcx.a directly.
  set(RTCX_INSTALL ON)

  rapids_cpm_find(
    rtcx ${VERSION}
    GLOBAL_TARGETS rtcx::rtcx
    CPM_ARGS
    GIT_REPOSITORY https://github.com/rapidsai/librtcx.git
    GIT_TAG efad266c1fd9de6d8486c6ba71bfa74df063eb1f
    GIT_SHALLOW FALSE
    EXCLUDE_FROM_ALL ${CUDF_EXCLUDE_DEPS_FROM_ALL}
  )

  # When CPM fetches from source (add_subdirectory), embed.cmake is not auto-included. Include it
  # explicitly so add_embed/embed_includes/embed functions are available.
  if(rtcx_ADDED OR DEFINED CPM_rtcx_SOURCE)
    # GCC 15+ removed transitive <variant> include from <filesystem>; ensure embed.hpp includes it.
    if(EXISTS "${rtcx_SOURCE_DIR}/embed.hpp")
      file(READ "${rtcx_SOURCE_DIR}/embed.hpp" _rtcx_embed_hdr)
      string(FIND "${_rtcx_embed_hdr}" "#include <variant>" _rtcx_has_variant)
      if(_rtcx_has_variant EQUAL -1)
        string(REPLACE "#include <vector>" "#include <variant>\n#include <vector>" _rtcx_embed_hdr "${_rtcx_embed_hdr}")
        file(WRITE "${rtcx_SOURCE_DIR}/embed.hpp" "${_rtcx_embed_hdr}")
      endif()
    endif()
    include("${rtcx_SOURCE_DIR}/embed.cmake")
  endif()
endfunction()

set(RTCX_MIN_VERSION_cudf "0.1")
find_and_configure_rtcx(${RTCX_MIN_VERSION_cudf})
