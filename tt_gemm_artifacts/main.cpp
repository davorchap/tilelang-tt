// Generated TT Host Metadata Program
// Partition-aware host summary derived from layout-aware metadata

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

struct TensorAccessorArgs {
  bool initialized;
  const char* buffer;
  const char* memory;
  const char* layout;
  uint32_t tile_rows;
  uint32_t tile_cols;
  uint32_t shard_grid_y;
  uint32_t shard_grid_x;
  TensorAccessorArgs()
      : initialized(false), buffer(nullptr), memory(nullptr), layout(nullptr),
        tile_rows(0), tile_cols(0), shard_grid_y(0), shard_grid_x(0) {}
  static TensorAccessorArgs Create(const char* buffer,
                                    const char* memory,
                                    const char* layout,
                                    uint32_t tile_rows,
                                    uint32_t tile_cols,
                                    uint32_t shard_grid_y,
                                    uint32_t shard_grid_x) {
    TensorAccessorArgs args;
    args.initialized = true;
    args.buffer = buffer;
    args.memory = memory;
    args.layout = layout;
    args.tile_rows = tile_rows;
    args.tile_cols = tile_cols;
    args.shard_grid_y = shard_grid_y;
    args.shard_grid_x = shard_grid_x;
    return args;
  }
};

inline void GuardTensorAccessor(const TensorAccessorArgs& args) {
  if (!args.initialized) {
    throw std::runtime_error("TensorAccessorArgs must be created via TensorAccessorArgs::Create");
  }
}

struct RuntimeConstant {
  const char* name;
  uint32_t value;
};

constexpr const char* kPartitionMode = "global";
constexpr uint32_t kMt = 8;
constexpr uint32_t kKt = 8;
constexpr uint32_t kNt = 8;
constexpr uint32_t kM = 256;
constexpr uint32_t kK = 256;
constexpr uint32_t kN = 256;

constexpr std::array<const char*, 5> kRuntimeArgNames = {{"tt_start_tile", "tt_tile_count", "Mt", "Kt", "Nt"}};
constexpr std::array<RuntimeConstant, 3> kRuntimeConstants = {{
    {"Kt", 1},
    {"Mt", 8},
    {"Nt", 8}
  }};
constexpr std::array<std::array<uint32_t, 5>, 64> kCoreRuntimeArgs = {
    {{0, 1, 0, 0, 0}},
    {{1, 1, 0, 0, 0}},
    {{2, 1, 0, 0, 0}},
    {{3, 1, 0, 0, 0}},
    {{4, 1, 0, 0, 0}},
    {{5, 1, 0, 0, 0}},
    {{6, 1, 0, 0, 0}},
    {{7, 1, 0, 0, 0}},
    {{8, 1, 0, 0, 0}},
    {{9, 1, 0, 0, 0}},
    {{10, 1, 0, 0, 0}},
    {{11, 1, 0, 0, 0}},
    {{12, 1, 0, 0, 0}},
    {{13, 1, 0, 0, 0}},
    {{14, 1, 0, 0, 0}},
    {{15, 1, 0, 0, 0}},
    {{16, 1, 0, 0, 0}},
    {{17, 1, 0, 0, 0}},
    {{18, 1, 0, 0, 0}},
    {{19, 1, 0, 0, 0}},
    {{20, 1, 0, 0, 0}},
    {{21, 1, 0, 0, 0}},
    {{22, 1, 0, 0, 0}},
    {{23, 1, 0, 0, 0}},
    {{24, 1, 0, 0, 0}},
    {{25, 1, 0, 0, 0}},
    {{26, 1, 0, 0, 0}},
    {{27, 1, 0, 0, 0}},
    {{28, 1, 0, 0, 0}},
    {{29, 1, 0, 0, 0}},
    {{30, 1, 0, 0, 0}},
    {{31, 1, 0, 0, 0}},
    {{32, 1, 0, 0, 0}},
    {{33, 1, 0, 0, 0}},
    {{34, 1, 0, 0, 0}},
    {{35, 1, 0, 0, 0}},
    {{36, 1, 0, 0, 0}},
    {{37, 1, 0, 0, 0}},
    {{38, 1, 0, 0, 0}},
    {{39, 1, 0, 0, 0}},
    {{40, 1, 0, 0, 0}},
    {{41, 1, 0, 0, 0}},
    {{42, 1, 0, 0, 0}},
    {{43, 1, 0, 0, 0}},
    {{44, 1, 0, 0, 0}},
    {{45, 1, 0, 0, 0}},
    {{46, 1, 0, 0, 0}},
    {{47, 1, 0, 0, 0}},
    {{48, 1, 0, 0, 0}},
    {{49, 1, 0, 0, 0}},
    {{50, 1, 0, 0, 0}},
    {{51, 1, 0, 0, 0}},
    {{52, 1, 0, 0, 0}},
    {{53, 1, 0, 0, 0}},
    {{54, 1, 0, 0, 0}},
    {{55, 1, 0, 0, 0}},
    {{56, 1, 0, 0, 0}},
    {{57, 1, 0, 0, 0}},
    {{58, 1, 0, 0, 0}},
    {{59, 1, 0, 0, 0}},
    {{60, 1, 0, 0, 0}},
    {{61, 1, 0, 0, 0}},
    {{62, 1, 0, 0, 0}},
    {{63, 1, 0, 0, 0}}
};

static_assert(kCoreRuntimeArgs.size() >= 1, "At least one core required");
static_assert(kRuntimeArgNames.size() == 0 || kRuntimeArgNames.size() == kCoreRuntimeArgs[0].size(),
              "Runtime argument schema mismatch");

TensorAccessorArgs tensor_accessors[] = {
    TensorAccessorArgs::Create("A", "DRAM", "interleaved", 32, 32, 1, 1),
    TensorAccessorArgs::Create("B", "DRAM", "interleaved", 32, 32, 1, 1),
    TensorAccessorArgs::Create("C", "DRAM", "interleaved", 32, 32, 1, 1)
};

int main() {
  std::cout << "Tenstorrent Host Metadata Summary" << std::endl;
  std::cout << "Partition mode: " << kPartitionMode << std::endl;
  std::cout << "Tiled dims (Mt,Kt,Nt): " << kMt << ", " << kKt << ", " << kNt << std::endl;
  std::cout << "Element dims (M,K,N): " << kM << ", " << kK << ", " << kN << std::endl;

  for (const auto& ta : tensor_accessors) {
    GuardTensorAccessor(ta);
    std::cout << "  buffer=" << ta.buffer
              << ", memory=" << ta.memory
              << ", layout=" << ta.layout
              << ", tile=" << ta.tile_rows << "x" << ta.tile_cols
              << ", shard=" << ta.shard_grid_y << "x" << ta.shard_grid_x
              << std::endl;
  }

  std::cout << "Runtime constants:" << std::endl;
  if (kRuntimeConstants.size() == 0) {
    std::cout << "  (none)" << std::endl;
  } else {
    for (const auto& constant : kRuntimeConstants) {
      std::cout << "  " << constant.name << " = " << constant.value << std::endl;
    }
  }

  std::cout << "Runtime args per core (" << kCoreRuntimeArgs.size() << " cores)" << std::endl;
  for (size_t core = 0; core < kCoreRuntimeArgs.size(); ++core) {
    const auto& args = kCoreRuntimeArgs[core];
    std::cout << "  core " << core;
    if (args.size() == 0) {
      std::cout << ": (no args)" << std::endl;
      continue;
    }
    std::cout << ": ";
    for (size_t idx = 0; idx < args.size(); ++idx) {
      std::cout << kRuntimeArgNames[idx] << "=" << args[idx];
      if (idx + 1 < args.size()) {
        std::cout << ", ";
      }
    }
    std::cout << std::endl;
  }

  return 0;
}
