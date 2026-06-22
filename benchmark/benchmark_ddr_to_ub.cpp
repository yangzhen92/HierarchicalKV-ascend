/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "debug.h"
#include "kernels/utils_kernel/utils_kernel.h"
#include "tiling/platform/platform_ascendc.h"

using V = float;

namespace {

constexpr uint64_t kBytesPerKiB = 1024ULL;
constexpr uint64_t kBytesPerMiB = 1024ULL * 1024ULL;
constexpr uint32_t kDefaultDataMiB = 8192;
constexpr uint32_t kDefaultRepeat = 10;
constexpr uint32_t kDefaultWarmup = 3;
constexpr uint32_t kDefaultTileKiB = 128;

enum class BenchmarkMode {
  kDdrToUb,
  kMemcpyH2D,
};

uint64_t parse_u64_arg(char** argv, int argc, int index, uint64_t fallback) {
  if (argc <= index) {
    return fallback;
  }
  char* end = nullptr;
  uint64_t value = std::strtoull(argv[index], &end, 10);
  if (end == argv[index] || *end != '\0') {
    return fallback;
  }
  return value;
}

bool is_u64_arg(const char* arg) {
  if (arg == nullptr || *arg == '\0') {
    return false;
  }
  char* end = nullptr;
  (void)std::strtoull(arg, &end, 10);
  return end != arg && *end == '\0';
}

BenchmarkMode parse_mode(char** argv, int argc, int& first_numeric_arg) {
  first_numeric_arg = 1;
  if (argc <= 1 || is_u64_arg(argv[1])) {
    return BenchmarkMode::kDdrToUb;
  }

  std::string mode = argv[1];
  first_numeric_arg = 2;
  if (mode == "ddr_to_ub") {
    return BenchmarkMode::kDdrToUb;
  }
  if (mode == "memcpy_h2d") {
    return BenchmarkMode::kMemcpyH2D;
  }

  HKV_CHECK(false,
            "Unknown mode. Supported modes: ddr_to_ub, memcpy_h2d.");
  return BenchmarkMode::kDdrToUb;
}

const char* mode_name(BenchmarkMode mode) {
  switch (mode) {
    case BenchmarkMode::kDdrToUb:
      return "ddr_to_ub";
    case BenchmarkMode::kMemcpyH2D:
      return "memcpy_h2d";
  }
  return "unknown";
}

void print_usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " [mode=ddr_to_ub] [data_mib=8192] [repeat=10] [tile_kib=128]"
         " [warmup=3] [aiv_num=all]\n"
      << "Modes:\n"
      << "  ddr_to_ub   mapped host DDR -> UB via DataCopyPad kernel\n"
      << "  memcpy_h2d  mapped host DDR -> HBM via aclrtMemcpyAsync\n";
}

void query_memory() {
  size_t free = 0;
  size_t total = 0;
  NPU_CHECK(aclrtGetMemInfo(ACL_DDR_MEM, &free, &total));
  std::cout << "DDR free memory: " << free / (1ULL << 30)
            << " GiB, DDR total memory: " << total / (1ULL << 30)
            << " GiB\n";

  NPU_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, &free, &total));
  std::cout << "HBM free memory: " << free / (1ULL << 30)
            << " GiB, HBM total memory: " << total / (1ULL << 30)
            << " GiB\n";
}

uint64_t get_total_ub_bytes() {
  auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  HKV_CHECK(platform != nullptr, "Get ascendc platform info failed!");
  uint64_t ub_total_size = 0;
  platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_total_size);
  HKV_CHECK(ub_total_size > 0, "UB size should be greater than 0.");
  return ub_total_size;
}

void* malloc_mapped_host(size_t bytes) {
  aclrtMallocAttrValue attr_value;
  attr_value.vaFlag = 1;

  aclrtMallocAttribute attribute[1];
  attribute[0].attr = ACL_RT_MEM_ATTR_VA_FLAG;
  attribute[0].value = attr_value;

  aclrtMallocConfig cfg;
  cfg.numAttrs = 1;
  cfg.attrs = attribute;

  void* ptr = nullptr;
  NPU_CHECK(aclrtMallocHostWithCfg(&ptr, bytes, &cfg));
  void* stub_ptr = nullptr;
  NPU_CHECK(aclrtHostRegister(ptr, bytes, ACL_HOST_REGISTER_MAPPED, &stub_ptr));
  return ptr;
}

void free_mapped_host(void* ptr) {
  if (ptr == nullptr) {
    return;
  }
  NPU_CHECK(aclrtHostUnregister(ptr));
  NPU_CHECK(aclrtFreeHost(ptr));
}

V source_pattern(uint64_t element_idx) {
  return static_cast<V>((element_idx * 131ULL + 17ULL) & 0xFFULL);
}

std::vector<uint64_t> make_sample_indices(uint64_t total_elements) {
  std::vector<uint64_t> samples;
  if (total_elements == 0) {
    return samples;
  }

  auto add = [&](uint64_t idx) {
    idx = std::min(idx, total_elements - 1);
    if (std::find(samples.begin(), samples.end(), idx) == samples.end()) {
      samples.push_back(idx);
    }
  };

  add(0);
  add(total_elements / 4);
  add(total_elements / 2);
  add((total_elements * 3) / 4);
  add(total_elements - 1);

  constexpr uint64_t kMaxExtraSamples = 1024;
  uint64_t stride = std::max<uint64_t>(1, total_elements / kMaxExtraSamples);
  for (uint64_t idx = 0; idx < total_elements; idx += stride) {
    add(idx);
  }
  add(total_elements - 1);

  return samples;
}

void seed_source_for_samples(V* h_src, const std::vector<uint64_t>& samples) {
  for (uint64_t element_idx : samples) {
    h_src[element_idx] = source_pattern(element_idx);
  }
}

void run_kernel(uint32_t block_dim, uint64_t total_tiles, uint32_t tile_elements,
                V* src, aclrtStream stream) {
  uint64_t tail_core_tile_num = total_tiles / block_dim;
  uint64_t former_core_tile_num = tail_core_tile_num + 1;
  uint64_t former_num = total_tiles - tail_core_tile_num * block_dim;
  uint32_t ub_bytes = tile_elements * sizeof(V) * 2;

  npu::hkv::ddr_to_ub_read_kernel<V><<<block_dim, ub_bytes, stream>>>(
      former_num, former_core_tile_num, tail_core_tile_num, tile_elements, src);
}

bool verify_memcpy_h2d(V* h_src, V* d_hbm,
                       const std::vector<uint64_t>& samples) {
  for (uint64_t element_idx : samples) {
    V actual = static_cast<V>(0);
    NPU_CHECK(aclrtMemcpy(&actual, sizeof(V), d_hbm + element_idx, sizeof(V),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    V expected = h_src[element_idx];
    if (actual != expected) {
      std::cout << "verify mismatch element " << element_idx
                << " expected: " << expected << " actual: " << actual
                << "\n";
      return false;
    }
  }

  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--help") {
    print_usage(argv[0]);
    return 0;
  }

  int first_numeric_arg = 1;
  BenchmarkMode mode = parse_mode(argv, argc, first_numeric_arg);
  uint64_t data_mib =
      parse_u64_arg(argv, argc, first_numeric_arg, kDefaultDataMiB);
  uint64_t repeat =
      parse_u64_arg(argv, argc, first_numeric_arg + 1, kDefaultRepeat);
  uint64_t tile_kib =
      parse_u64_arg(argv, argc, first_numeric_arg + 2, kDefaultTileKiB);
  uint64_t warmup =
      parse_u64_arg(argv, argc, first_numeric_arg + 3, kDefaultWarmup);

  HKV_CHECK(data_mib > 0, "data_mib must be greater than 0.");
  HKV_CHECK(repeat > 0, "repeat must be greater than 0.");
  HKV_CHECK(tile_kib > 0, "tile_kib must be greater than 0.");

  NPU_CHECK(aclInit(nullptr));
  auto device_id_env = std::getenv("HKV_TEST_DEVICE");
  int32_t device_id = 0;
  try {
    device_id = (device_id_env != nullptr) ? std::stoi(device_id_env) : 0;
  } catch (...) {
    device_id = 0;
    std::cout << "set env HKV_TEST_DEVICE error, using default device_id 0\n";
  }
  NPU_CHECK(aclrtSetDevice(device_id));

  auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  HKV_CHECK(platform != nullptr, "Get ascendc platform info failed!");
  uint32_t device_aiv_num = platform->GetCoreNumAiv();
  uint64_t requested_aiv_num =
      parse_u64_arg(argv, argc, first_numeric_arg + 4, device_aiv_num);
  uint32_t block_dim =
      static_cast<uint32_t>(std::min<uint64_t>(requested_aiv_num,
                                               device_aiv_num));
  HKV_CHECK(block_dim > 0, "AIV core number should be greater than 0.");
  std::cout << "Soc version: " << aclrtGetSocName() << " device_id: "
            << device_id << " aiv_num: " << block_dim
            << " device_aiv_num: " << device_aiv_num << "\n";
  query_memory();

  uint64_t requested_bytes = data_mib * kBytesPerMiB;
  uint64_t tile_bytes = tile_kib * kBytesPerKiB;
  tile_bytes = std::max<uint64_t>(sizeof(V), tile_bytes);
  tile_bytes = (tile_bytes / sizeof(V)) * sizeof(V);
  HKV_CHECK(tile_bytes > 0, "tile bytes should be greater than 0.");
  if (mode == BenchmarkMode::kDdrToUb) {
    uint64_t ub_total_bytes = get_total_ub_bytes();
    uint64_t requested_ub_bytes = tile_bytes * 2;
    HKV_CHECK(requested_ub_bytes <= ub_total_bytes,
              log_format(
                  "tile_kib %lu requires %lu KiB UB with double buffer, "
                  "but this device reports %lu KiB UB. Try tile_kib <= "
                  "%lu.",
                  tile_kib, requested_ub_bytes / kBytesPerKiB,
                  ub_total_bytes / kBytesPerKiB,
                  ub_total_bytes / (2 * kBytesPerKiB)));
  }

  uint32_t tile_elements = static_cast<uint32_t>(tile_bytes / sizeof(V));
  uint64_t total_tiles = (requested_bytes + tile_bytes - 1) / tile_bytes;
  HKV_CHECK(total_tiles > 0, "total tile count should be greater than 0.");
  uint64_t effective_bytes = total_tiles * tile_bytes;
  uint64_t total_elements = effective_bytes / sizeof(V);

  V* h_src = reinterpret_cast<V*>(malloc_mapped_host(effective_bytes));
  std::vector<uint64_t> samples = make_sample_indices(total_elements);
  seed_source_for_samples(h_src, samples);

  V* d_hbm = nullptr;
  if (mode == BenchmarkMode::kMemcpyH2D) {
    NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_hbm), effective_bytes,
                          ACL_MEM_MALLOC_HUGE_FIRST));
  }

  aclrtStream stream;
  NPU_CHECK(aclrtCreateStream(&stream));

  for (uint64_t i = 0; i < warmup; i++) {
    if (mode == BenchmarkMode::kDdrToUb) {
      run_kernel(block_dim, total_tiles, tile_elements, h_src, stream);
    } else {
      NPU_CHECK(aclrtMemcpyAsync(d_hbm, effective_bytes, h_src,
                                 effective_bytes, ACL_MEMCPY_HOST_TO_DEVICE,
                                 stream));
    }
  }
  NPU_CHECK(aclrtSynchronizeStream(stream));

  auto start = std::chrono::steady_clock::now();
  for (uint64_t i = 0; i < repeat; i++) {
    if (mode == BenchmarkMode::kDdrToUb) {
      run_kernel(block_dim, total_tiles, tile_elements, h_src, stream);
    } else {
      NPU_CHECK(aclrtMemcpyAsync(d_hbm, effective_bytes, h_src,
                                 effective_bytes, ACL_MEMCPY_HOST_TO_DEVICE,
                                 stream));
    }
  }
  NPU_CHECK(aclrtSynchronizeStream(stream));
  auto end = std::chrono::steady_clock::now();

  bool verify_ok = true;
  if (mode == BenchmarkMode::kDdrToUb) {
    std::cout << "verify: SKIP (ddr_to_ub has no GM output)\n";
  } else {
    verify_ok = verify_memcpy_h2d(h_src, d_hbm, samples);
  }

  double elapsed_us =
      std::chrono::duration<double, std::micro>(end - start).count();
  double elapsed_s = elapsed_us / 1000000.0;
  double total_bytes = static_cast<double>(effective_bytes) * repeat;
  double bandwidth_gib = total_bytes / elapsed_s / (1024.0 * 1024.0 * 1024.0);
  double bandwidth_gb = total_bytes / elapsed_s / 1000000000.0;

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "mode: " << mode_name(mode) << "\n";
  std::cout << "data_mib: " << data_mib << "\n";
  std::cout << "effective_read_bytes_per_iter: " << effective_bytes << "\n";
  std::cout << "repeat: " << repeat << "\n";
  std::cout << "tile_kib: " << tile_kib << "\n";
  if (mode == BenchmarkMode::kMemcpyH2D) {
    std::cout << "verify: " << (verify_ok ? "PASS" : "FAIL") << "\n";
  }
  std::cout << "elapsed_us: " << elapsed_us << "\n";
  std::cout << "bandwidth_GBps: " << bandwidth_gb << "\n";
  std::cout << "bandwidth_GiBps: " << bandwidth_gib << "\n";

  NPU_CHECK(aclrtDestroyStream(stream));
  if (d_hbm != nullptr) {
    NPU_CHECK(aclrtFree(d_hbm));
  }
  free_mapped_host(h_src);
  NPU_CHECK(aclrtResetDevice(device_id));
  NPU_CHECK(aclFinalize());
  return verify_ok ? 0 : 1;
}
