// Copyright © 2023-2025 Apple Inc.

#include <nanobind/nanobind.h>

#include "mlx/backend/cuda/cuda.h"

namespace mx = mlx::core;
namespace nb = nanobind;

void init_cuda(nb::module_& m) {
  nb::module_ cuda = m.def_submodule("cuda", "mlx.cuda");

  cuda.def(
      "is_available",
      &mx::cu::is_available,
      R"pbdoc(
      Check if the CUDA back-end is available.
      )pbdoc");

  cuda.def(
      "clear_graph_caches",
      &mx::cu::clear_graph_caches,
      R"pbdoc(
      Clear all CUDA graph caches across all devices and streams.

      This frees GPU workspace memory pinned by cached CUDA graphs.
      Unlike :func:`mlx.core.clear_cache` which only clears buffer cache,
      this clears the compiled CUDA graph executables that accumulate
      when running kernels with different configurations.
      )pbdoc");
}
