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
 * @brief Sweep single-bucket against dual-bucket `insert_or_assign` over load
 *        factors, comparing insert throughput and retained entries, then sweep
 *        the value-copy `find` over the tables that sweep built, comparing
 *        lookup throughput and hit rate. Both modes are built from identical
 *        options, so the only difference is `enable_dual_bucket`.
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
/// 4Mi slots: 32768 buckets of 128, small enough to sweep load factors.
constexpr size_t CAPACITY = 4UL * 1024 * 1024;
/// The bucket size the dual-bucket mode is restricted to.
constexpr size_t BUCKET_MAX_SIZE = 128;
/// One batch of `insert_or_assign`; the sweep is chunked into these.
constexpr size_t KEY_NUM_PER_OP = 1UL * 1024 * 1024;
/// One batch of `find`; the lookup sweep is measured over batches of this size.
constexpr size_t QUERY_NUM_PER_OP = 1UL * 1024 * 1024;
/// Lookup batches per measured point, so the per-launch cost is amortized.
constexpr size_t FIND_REPEAT = 10;
/// Reserved bits of the key space; generated keys stay below this.
constexpr K RESERVED_KEY_LOW = UINT64_C(0xFFFFFFFFFFFFFFFC);
/// First key of the "never written" tail of a query batch. Far above the
/// continuous block of inserted keys and well below the reserved range, so a
/// miss query really does probe the table instead of short-circuiting on a
/// sentinel key.
constexpr K FIND_MISS_BASE = UINT64_C(0x0001000000000000);

struct ModeResult {
  /// Billion KV/s, matching the column unit of `hkv_hashtable_benchmark`.
  double throughput = 0.0;
  /// Live entries after the inserts.
  size_t retained = 0;
};

struct FindResult {
  /// Million queries/s.
  double qps = 0.0;
  /// Share of the batch the kernel reported as found.
  double hit_rate = 0.0;
};

/// One row of either sweep. An insert row carries the two `ModeResult`s, a
/// lookup row the two `FindResult`s of the tables that row produced; `is_find`
/// says which of the two it is.
struct Point {
  bool is_find = false;
  size_t dim = 0;
  double key_ratio = 0.0;
  double hit_ratio = 0.0;
  ModeResult single_insert;
  ModeResult dual_insert;
  FindResult single_find;
  FindResult dual_find;
};
/// Options differ from the reference only in the dual-bucket flag.
template <class Table>
std::shared_ptr<Table> make_table(size_t dim, size_t hbm_for_vectors_gb,
                                  bool dual_bucket) {
  TableOptions options;
  // `init_capacity == max_capacity` is a dual-bucket precondition, so the
  // reference uses a fixed capacity too and the comparison stays apples to
  // apples.
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

/// Clear, insert `key_num` keys, report the duration and survivor count.
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
/// A query batch of `QUERY_NUM_PER_OP` keys: a `hit_ratio` fraction of them are
/// drawn from the `inserted_num` keys that were written, the rest are fresh
/// keys that never were. The draw is seeded so both modes see the same batch.
void make_query_keys(K* h_keys, size_t inserted_num, double hit_ratio) {
  const size_t hit_num =
      static_cast<size_t>(static_cast<double>(QUERY_NUM_PER_OP) * hit_ratio);
  std::mt19937_64 eng(20261016);
  std::uniform_int_distribution<K> distr(0, static_cast<K>(inserted_num) - 1);
  for (size_t i = 0; i < hit_num; ++i) {
    h_keys[i] = distr(eng);
  }
  for (size_t i = hit_num; i < QUERY_NUM_PER_OP; ++i) {
    h_keys[i] = FIND_MISS_BASE + static_cast<K>(i - hit_num);
  }
}

/// Time `FIND_REPEAT` runs of the value-copy `find` against an already
/// populated table. The batch is uploaded once, outside the timed region, so
/// the duration covers the lookup and nothing else.
template <class Table>
FindResult run_find_case(std::shared_ptr<Table>& table, size_t inserted_num,
                         size_t dim, double hit_ratio, aclrtStream stream) {
  std::vector<K> h_keys(QUERY_NUM_PER_OP);
  std::vector<V> h_values(QUERY_NUM_PER_OP * dim);
  std::vector<S> h_scores(QUERY_NUM_PER_OP, 0);
  std::vector<char> h_founds(QUERY_NUM_PER_OP, 0);
  make_query_keys(h_keys.data(), inserted_num, hit_ratio);
  benchmark::init_value_using_key<K, V>(h_keys.data(), h_values.data(),
                                        QUERY_NUM_PER_OP, dim);

  K* d_keys = nullptr;
  V* d_values = nullptr;
  bool* d_founds = nullptr;
  S* d_scores = nullptr;
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_keys),
                        QUERY_NUM_PER_OP * sizeof(K),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_values),
                        QUERY_NUM_PER_OP * sizeof(V) * dim,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_founds),
                        QUERY_NUM_PER_OP * sizeof(bool),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_scores),
                        QUERY_NUM_PER_OP * sizeof(S),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMemcpy(d_keys, QUERY_NUM_PER_OP * sizeof(K), h_keys.data(),
                        QUERY_NUM_PER_OP * sizeof(K),
                        ACL_MEMCPY_HOST_TO_DEVICE));
  NPU_CHECK(aclrtMemcpy(d_values, QUERY_NUM_PER_OP * sizeof(V) * dim,
                        h_values.data(), QUERY_NUM_PER_OP * sizeof(V) * dim,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  NPU_CHECK(aclrtMemcpy(d_scores, QUERY_NUM_PER_OP * sizeof(S), h_scores.data(),
                        QUERY_NUM_PER_OP * sizeof(S),
                        ACL_MEMCPY_HOST_TO_DEVICE));

  /// One warm-up run, which also yields the measured hit rate. It is reported
  /// rather than assumed: at a load factor of 1.0 a bucket can overflow and the
  /// two modes then retain different subsets, so the requested hit ratio is not
  /// necessarily the one the lookup actually sees.
  table->find(QUERY_NUM_PER_OP, d_keys, d_values, d_founds, d_scores, stream);
  NPU_CHECK(aclrtSynchronizeStream(stream));
  NPU_CHECK(aclrtMemcpy(h_founds.data(), QUERY_NUM_PER_OP * sizeof(char),
                        d_founds, QUERY_NUM_PER_OP * sizeof(bool),
                        ACL_MEMCPY_DEVICE_TO_HOST));
  size_t hit_num = 0;
  for (size_t i = 0; i < QUERY_NUM_PER_OP; ++i) {
    hit_num += (h_founds[i] != 0) ? 1 : 0;
  }

  auto timer = benchmark::Timer<double>();
  timer.start();
  for (size_t i = 0; i < FIND_REPEAT; ++i) {
    table->find(QUERY_NUM_PER_OP, d_keys, d_values, d_founds, d_scores, stream);
  }
  NPU_CHECK(aclrtSynchronizeStream(stream));
  timer.end();

  FindResult result;
  const double seconds = timer.getResult();
  HKV_CHECK(seconds > 0.0, "timer returned a non-positive duration");
  result.qps = static_cast<double>(FIND_REPEAT * QUERY_NUM_PER_OP) / seconds /
               (1024.0 * 1024.0);
  result.hit_rate =
      static_cast<double>(hit_num) / static_cast<double>(QUERY_NUM_PER_OP);
  NPU_CHECK(aclrtFree(d_keys));
  NPU_CHECK(aclrtFree(d_values));
  NPU_CHECK(aclrtFree(d_founds));
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

void print_find_title() {
  std::cout << "\n"
            << "| " << rep(4) << "λ "
            << "| " << rep(4) << "dim "
            << "| " << rep(6) << "hit target "
            << "| " << rep(6) << "single(MQ/s) "
            << "| " << rep(6) << "dual(MQ/s) "
            << "| " << rep(8) << "dual/single "
            << "| " << rep(6) << "single hit "
            << "| " << rep(6) << "dual hit "
            << "|\n"
            << "|------:"
            << "|------:"
            << "|-----------:"
            << "|-------------:"
            << "|-----------:"
            << "|------------:"
            << "|-----------:"
            << "|---------:"
            << "|\n";
}
void print_find_row(double load_factor, size_t dim, double hit_ratio,
                    const FindResult& single, const FindResult& dual) {
  const double qps_ratio = single.qps > 0.0 ? dual.qps / single.qps : 0.0;
  std::cout << "|" << rep(1) << std::fixed << std::setprecision(2)
            << load_factor << " " << "|" << std::setw(6) << dim << " " << "|"
            << std::setw(11) << std::setprecision(2) << hit_ratio << " " << "|"
            << std::setw(13) << single.qps << " " << "|" << std::setw(11)
            << dual.qps << " " << "|" << std::setw(12) << std::setprecision(3)
            << qps_ratio << " " << "|" << std::setw(11) << std::setprecision(4)
            << single.hit_rate << " " << "|" << std::setw(9) << dual.hit_rate
            << " " << "|\n";
}

void benchmark_dual_bucket() {
  /// `kCustomized` takes the supplied scores verbatim, keeping the eviction
  /// order identical across both modes.
  using Table = npu::hkv::HashTable<K, V, S, EvictStrategy::kCustomized>;

  /// Past 1.0 every mode must evict, which is where the gap shows up.
  const std::vector<double> key_ratios = {0.50, 0.75, 0.90, 0.95, 1.00, 1.05};
  /// The lookup sweep runs on the tables the insert sweep built, so it drops
  /// the over-capacity point, where which keys survived is decided by the
  /// eviction order. A hit ratio of 1.0 asks the whole batch to hit.
  const std::vector<double> hit_ratios = {0.50, 0.90, 1.00};
  const std::vector<size_t> dims = {8, 64};
  aclrtStream stream = nullptr;
  NPU_CHECK(aclrtCreateStream(&stream));

  std::vector<Point> points;
  for (const size_t dim : dims) {
    const size_t hbm_gb = (dim <= 8) ? 8 : 16;
    for (const double key_ratio : key_ratios) {
      const size_t key_num =
          static_cast<size_t>(static_cast<double>(CAPACITY) * key_ratio);
      Point point;
      point.dim = dim;
      point.key_ratio = key_ratio;
      std::vector<FindResult> single_finds;
      std::vector<FindResult> dual_finds;
      {
        auto table = make_table<Table>(dim, hbm_gb, /* dual_bucket */ false);
        point.single_insert = run_case<Table>(table, key_num, dim, stream);
        if (key_ratio <= 1.0) {
          for (const double hit_ratio : hit_ratios) {
            single_finds.push_back(
                run_find_case<Table>(table, key_num, dim, hit_ratio, stream));
          }
        }
      }
      {
        auto table = make_table<Table>(dim, hbm_gb, /* dual_bucket */ true);
        point.dual_insert = run_case<Table>(table, key_num, dim, stream);
        if (key_ratio <= 1.0) {
          for (const double hit_ratio : hit_ratios) {
            dual_finds.push_back(
                run_find_case<Table>(table, key_num, dim, hit_ratio, stream));
          }
        }
      }
      points.push_back(point);
      /// One lookup row per requested hit ratio, carrying the results the two
      /// tables produced while they were still alive.
      for (size_t i = 0; i < single_finds.size(); ++i) {
        Point find_point;
        find_point.is_find = true;
        find_point.dim = dim;
        find_point.key_ratio = key_ratio;
        find_point.hit_ratio = hit_ratios[i];
        find_point.single_find = single_finds[i];
        find_point.dual_find = dual_finds[i];
        points.push_back(find_point);
      }
    }
  }

  print_title();
  for (const Point& point : points) {
    if (!point.is_find) {
      print_row(point.key_ratio, point.dim, point.single_insert,
                point.dual_insert, point.key_ratio);
      std::cout.flush();
    }
  }
  print_find_title();
  for (const Point& point : points) {
    if (point.is_find) {
      print_find_row(point.key_ratio, point.dim, point.hit_ratio,
                     point.single_find, point.dual_find);
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
            << " key_num_per_op: " << KEY_NUM_PER_OP
            << " query_num_per_op: " << QUERY_NUM_PER_OP
            << " find_repeat: " << FIND_REPEAT << std::endl;
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
