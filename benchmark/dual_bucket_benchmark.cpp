/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file dual_bucket_benchmark.cpp
 * @brief Single-bucket vs dual-bucket `insert_or_assign` comparison.
 *
 *        The dual-bucket layout exists to remove the *premature eviction* of
 * the single-bucket layout: when one bucket of a cluster fills up while the
 *        table still has room elsewhere, the single-bucket path has nowhere to
 *        put the key and has to drop an entry. The dual-bucket path lets the
 *        key fall back to its second candidate, so at the same capacity and the
 *        same key count it should retain more entries -- and it should do so
 *        without giving up insert throughput.
 *
 *        Both properties are measured here, over a sweep of load factors:
 *          - `throughput` (Billion KV/s) of `insert_or_assign`, and
 *          - `retained`, the number of live entries after the sweep, relative
 *            to the capacity.
 *
 *        The two modes are built from *identical* options -- same capacity,
 *        same `max_bucket_size`, same dim, pure HBM -- so the only difference
 * is `enable_dual_bucket`. Both tables insert the same key stream.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>
#include "acl/acl.h"
#include "benchmark_util.h"
#include "debug.h"
#include "hkv_hashtable.h"
#include "tiling/platform/platform_ascendc.h"

using K = uint64_t;
using V = float;
using S = uint64_t;
using EvictStrategy = npu::hkv::EvictStrategy;
using TableOptions = npu::hkv::HashTableOptions;

namespace {

/// 4Mi slots is 32768 buckets of 128, small enough to sweep several load
/// factors but large enough that the per-launch overhead is amortised away.
constexpr size_t CAPACITY = 4UL * 1024 * 1024;
/// The dual-bucket mode is tuned for, and restricted to, this bucket size.
constexpr size_t BUCKET_MAX_SIZE = 128;
/// One batch of `insert_or_assign` calls; the sweep is chunked into these.
constexpr size_t KEY_NUM_PER_OP = 1UL * 1024 * 1024;

/// Reserved bits of the key space; the generated key streams stay below this so
/// that no key is silently dropped.
constexpr K RESERVED_KEY_LOW = UINT64_C(0xFFFFFFFFFFFFFFFC);

struct ModeResult {
  /// Billion KV/s, matching the column unit of `hkv_hashtable_benchmark`.
  double throughput = 0.0;
  /// Live entries after the inserts.
  size_t retained = 0;
};

/// Build a table whose options differ from the reference only in the
/// dual-bucket flag.
template <class Table>
std::shared_ptr<Table> make_table(size_t dim, size_t hbm_for_vectors_gb,
                                  bool dual_bucket) {
  TableOptions options;
  // `init_capacity == max_capacity` is a precondition of the dual-bucket mode,
  // so the reference table uses a fixed capacity too and the comparison stays
  // apples to apples.
  options.init_capacity = CAPACITY;
  options.max_capacity = CAPACITY;
  options.max_bucket_size = BUCKET_MAX_SIZE;
  options.dim = dim;
  options.max_hbm_for_vectors = npu::hkv::GB(hbm_for_vectors_gb);
  options.io_by_cpu = false;
  options.enable_dual_bucket = dual_bucket;

  auto table = std::make_shared<Table>();
  table->init(options);
  return table;
}

/// Clear the table, insert `key_num` distinct keys and report how long that
/// took and how many entries survived.
template <class Table>
ModeResult run_case(std::shared_ptr<Table>& table, size_t key_num, size_t dim,
                    aclrtStream stream) {
  std::vector<K> h_keys(KEY_NUM_PER_OP);
  std::vector<V> h_values(KEY_NUM_PER_OP * dim);
  std::vector<S> h_scores(KEY_NUM_PER_OP);

  K* d_keys = nullptr;
  V* d_values = nullptr;
  S* d_scores = nullptr;
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_keys),
                        KEY_NUM_PER_OP * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_values),
                        KEY_NUM_PER_OP * sizeof(V) * dim,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_scores),
                        KEY_NUM_PER_OP * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST));

  table->clear(stream);
  NPU_CHECK(aclrtSynchronizeStream(stream));

  auto timer = benchmark::Timer<double>();
  K start = 0;
  size_t remaining = key_num;
  uint64_t global_epoch = 0;
  timer.start();
  while (remaining > 0) {
    const size_t n = std::min(remaining, KEY_NUM_PER_OP);
    benchmark::create_continuous_keys<K, S>(h_keys.data(), h_scores.data(), n,
                                            start);
    benchmark::init_value_using_key<K, V>(h_keys.data(), h_values.data(), n,
                                          dim);
    NPU_CHECK(aclrtMemcpy(d_keys, n * sizeof(K), h_keys.data(), n * sizeof(K),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    NPU_CHECK(aclrtMemcpy(d_values, n * sizeof(V) * dim, h_values.data(),
                          n * sizeof(V) * dim, ACL_MEMCPY_HOST_TO_DEVICE));
    NPU_CHECK(aclrtMemcpy(d_scores, n * sizeof(S), h_scores.data(),
                          n * sizeof(S), ACL_MEMCPY_HOST_TO_DEVICE));

    table->set_global_epoch(global_epoch++);
    table->insert_or_assign(n, d_keys, d_values, d_scores, stream);
    NPU_CHECK(aclrtSynchronizeStream(stream));

    start += static_cast<K>(n);
    remaining -= n;
  }
  timer.end();

  ModeResult result;
  const double seconds = timer.getResult();
  HKV_CHECK(seconds > 0.0, "timer returned a non-positive duration");
  result.throughput = key_num / seconds / (1024.0 * 1024.0 * 1024.0);
  result.retained = table->size(stream);
  NPU_CHECK(aclrtSynchronizeStream(stream));

  NPU_CHECK(aclrtFree(d_keys));
  NPU_CHECK(aclrtFree(d_values));
  NPU_CHECK(aclrtFree(d_scores));
  NpuCheckError();
  return result;
}

std::string rep(int n) { return std::string(n, ' '); }

void print_title() {
  std::cout << "\n"
            << "| " << rep(4) << "λ "
            << "| " << rep(4) << "dim "
            << "| " << rep(6) << "single(BK/s) "
            << "| " << rep(6) << "dual(BK/s) "
            << "| " << rep(8) << "dual/single "
            << "| " << rep(6) << "single ret. "
            << "| " << rep(6) << "dual ret. "
            << "| " << rep(4) << "keys/cap "
            << "|\n"
            << "|------:"
            << "|------:"
            << "|-------------:"
            << "|-----------:"
            << "|------------:"
            << "|------------:"
            << "|----------:"
            << "|---------:"
            << "|\n";
}

void print_row(double load_factor, size_t dim, const ModeResult& single,
               const ModeResult& dual, double key_ratio) {
  const double throughput_ratio =
      single.throughput > 0.0 ? dual.throughput / single.throughput : 0.0;
  std::cout << "|" << rep(1) << std::fixed << std::setprecision(2)
            << load_factor << " " << "|" << std::setw(6) << dim << " " << "|"
            << std::setw(13) << std::setprecision(2) << single.throughput << " "
            << "|" << std::setw(11) << dual.throughput << " " << "|"
            << std::setw(12) << std::setprecision(3) << throughput_ratio << " "
            << "|" << std::setw(12) << std::setprecision(4)
            << static_cast<double>(single.retained) / CAPACITY << " " << "|"
            << std::setw(10) << static_cast<double>(dual.retained) / CAPACITY
            << " " << "|" << std::setw(9) << std::setprecision(2) << key_ratio
            << " " << "|\n";
}

void benchmark_dual_bucket() {
  /// `kCustomized` takes the supplied scores verbatim, which keeps the eviction
  /// order identical across the two modes; the LRU strategies reject explicit
  /// scores.
  using Table = npu::hkv::HashTable<K, V, S, EvictStrategy::kCustomized>;

  /// Through 1.0 the two modes only differ in how well they pack; past 1.0 the
  /// table is genuinely over-subscribed and every mode has to evict, which is
  /// where the "premature eviction" difference shows up as a retention gap.
  const std::vector<double> key_ratios = {0.50, 0.75, 0.90, 0.95, 1.00, 1.05};
  const std::vector<size_t> dims = {8, 64};

  aclrtStream stream = nullptr;
  NPU_CHECK(aclrtCreateStream(&stream));

  print_title();
  for (const size_t dim : dims) {
    const size_t hbm_gb = (dim <= 8) ? 8 : 16;
    for (const double key_ratio : key_ratios) {
      const size_t key_num =
          static_cast<size_t>(static_cast<double>(CAPACITY) * key_ratio);

      ModeResult single;
      {
        auto table = make_table<Table>(dim, hbm_gb, /* dual_bucket */ false);
        single = run_case<Table>(table, key_num, dim, stream);
      }
      ModeResult dual;
      {
        auto table = make_table<Table>(dim, hbm_gb, /* dual_bucket */ true);
        dual = run_case<Table>(table, key_num, dim, stream);
      }

      print_row(key_ratio, dim, single, dual, key_ratio);
      std::cout.flush();
    }
  }

  NPU_CHECK(aclrtDestroyStream(stream));
}

void query_memory() {
  size_t free = 0;
  size_t total = 0;
  NPU_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, &free, &total));
  std::cout << "HBM free memory:" << free / (1 << 30)
            << " GB, HBM total memory:" << total / (1 << 30) << " GB"
            << std::endl;
}

}  // namespace

int32_t main(int32_t argc, char* argv[]) {
  NPU_CHECK(aclInit(nullptr));
  const auto device_id_env = std::getenv("HKV_TEST_DEVICE");
  int32_t device_id = 0;
  try {
    device_id = (device_id_env != nullptr) ? std::stoi(device_id_env) : 0;
  } catch (...) {
    device_id = 0;
    std::cout << "set env HKV_TEST_DEVICE error, using default device_id 0"
              << std::endl;
  }
  NPU_CHECK(aclrtSetDevice(device_id));
  auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  HKV_CHECK(ascendc_platform != nullptr, "Get ascendc platform info failed!");
  std::cout << "Soc version:" << aclrtGetSocName() << " device_id:" << device_id
            << " aiv_num: " << ascendc_platform->GetCoreNumAiv()
            << " capacity: " << CAPACITY
            << " max_bucket_size: " << BUCKET_MAX_SIZE
            << " key_num_per_op: " << KEY_NUM_PER_OP << std::endl;
  query_memory();

  try {
    benchmark_dual_bucket();
  } catch (const npu::hkv::NpuException& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  NPU_CHECK(aclrtSynchronizeDevice());
  NPU_CHECK(aclrtResetDevice(device_id));
  NPU_CHECK(aclFinalize());
  return 0;
}
