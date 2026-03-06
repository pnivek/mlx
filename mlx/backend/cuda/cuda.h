// Copyright © 2025 Apple Inc.

#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <variant>

#include "mlx/api.h"

namespace mlx::core::cu {

/* Check if the CUDA backend is available. */
MLX_API bool is_available();

/* Get information about a CUDA device. */
MLX_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info(int device_index = 0);

/* Clear all CUDA graph caches across all devices and streams. */
MLX_API void clear_graph_caches();

/* Get CUDA memory pool stats: (reserved, used) in bytes. */
MLX_API std::pair<size_t, size_t> get_pool_memory();

} // namespace mlx::core::cu
