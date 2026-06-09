/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
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
uint32_t g_aiv_num = 0;
}

// 表头各列宽度（与 print_w 参数一致，便于对齐 Markdown 风格输出）
void print_title() {
  std::cout << std::endl
            << "|    \u03BB " << "| capacity " << "| max_hbm_for_vectors "
            << "| max_bucket_size " << "| dim " << "| missed_ratio "
            << "| throughput(BillionKV/secs)";
  std::cout << "|\n";

  std::cout << "|------"
            << "|----------"
            << "|---------------------"
            << "|-----------------"
            << "|-----"
            << "|--------------"
            << "|---------------------------";
  std::cout << "|\n";
}

template <typename T>
void print_w(const T& t, size_t width) {
  std::cout << "|" << std::setw(width) << t;
}

// 各列域宽与 print_title
// 中分隔线长度一致：λ(6)、capacity(10)、max_hbm_for_vectors(21)、
// max_bucket_size(17)、dim(5)、missed_ratio(14)、througput(27)；改表头时需同步改此处。
void print_result(double load_factor, size_t capacity,
                  size_t max_hbm_for_vectors_mb, size_t max_bucket_size,
                  size_t dim, double missed_ratio, float throughput) {
  print_w(load_factor, 6);
  print_w(capacity, 10);
  print_w(max_hbm_for_vectors_mb, 21);
  print_w(max_bucket_size, 17);
  print_w(dim, 5);
  print_w(missed_ratio, 14);
  print_w(throughput, 27);
  std::cout << "|\n";
}

template <typename K>
void create_find_keys_balanced_by_core(K* h_keys, size_t n,
                                       double missed_ratio) {
  HKV_CHECK(h_keys != nullptr, "h_keys should not be nullptr");
  HKV_CHECK(missed_ratio >= 0.0 && missed_ratio <= 1.0,
            "Invalid `missed_ratio`");

  if (n == 0) {
    return;
  }

  HKV_CHECK(g_aiv_num > 0, "g_aiv_num should be greater than 0");
  uint32_t active_core_num = g_aiv_num;
  size_t hit_count =
      static_cast<size_t>(static_cast<double>(n) * (1.0 - missed_ratio));
  size_t hit_base = hit_count / active_core_num;
  size_t hit_remain = hit_count % active_core_num;
  size_t range_base = n / active_core_num;
  size_t range_remain = n % active_core_num;

  for (size_t i = 0; i < n; i++) {
    h_keys[i] = static_cast<K>(n + i);
  }

  size_t global_hit_idx = 0;
  size_t range_start = 0;
  for (uint32_t core_idx = 0; core_idx < active_core_num; core_idx++) {
    size_t range_size = range_base + (core_idx < range_remain ? 1 : 0);
    size_t core_hit_count = hit_base + (core_idx < hit_remain ? 1 : 0);

    for (size_t hit_idx = 0; hit_idx < core_hit_count; hit_idx++) {
      // Place hit keys evenly inside this core's contiguous range.
      size_t local_pos = ((hit_idx + 1) * range_size) / (core_hit_count + 1);
      h_keys[range_start + local_pos] = static_cast<K>(global_hit_idx++);
    }

    range_start += range_size;
  }
}

void test_find(size_t capacity, size_t dim, size_t max_hbm_for_vectors_mb,
               double load_factor, size_t max_bucket_size,
               double missed_ratio) {
  HKV_CHECK(load_factor >= 0.0 && load_factor <= 1.0, "Invalid `load_factor`");
  K* h_keys;
  S* h_scores;
  V* h_vectors;

  TableOptions options;
  options.init_capacity = capacity;
  options.max_capacity = capacity;
  options.dim = dim;
  // HashTableOptions::max_hbm_for_vectors 为字节；入参为 MB
  options.max_hbm_for_vectors = max_hbm_for_vectors_mb * 1024UL * 1024UL;
  options.max_bucket_size = max_bucket_size;

  size_t key_num = capacity;
  NPU_CHECK(aclrtMallocHost((void**)&h_keys, key_num * sizeof(K)));
  NPU_CHECK(aclrtMallocHost((void**)&h_scores, key_num * sizeof(S)));
  NPU_CHECK(
      aclrtMallocHost((void**)&h_vectors, key_num * options.dim * sizeof(V)));

  K* d_keys;
  S* d_scores;
  V* d_vectors;
  K* d_missed_keys;
  int* d_missed_indices;
  int* d_missed_size;

  NPU_CHECK(aclrtMalloc((void**)&d_keys, key_num * sizeof(K),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc((void**)&d_scores, key_num * sizeof(S),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc((void**)&d_vectors, key_num * sizeof(V) * options.dim,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc((void**)&d_missed_keys, key_num * sizeof(K),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc((void**)&d_missed_indices, key_num * sizeof(int),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc((void**)&d_missed_size, sizeof(int),
                        ACL_MEM_MALLOC_HUGE_FIRST));

  aclrtStream stream;
  NPU_CHECK(aclrtCreateStream(&stream));

  size_t insert_num =
      static_cast<size_t>(static_cast<double>(key_num) * load_factor);

  // 从 key=0 起连续插入，与后续 find 的命中区间一致
  benchmark::create_continuous_keys<K, S>(h_keys, h_scores, insert_num, 0);
  benchmark::init_value_using_key<K, V>(h_keys, h_vectors, insert_num,
                                        options.dim);
  NPU_CHECK(aclrtMemcpy(d_keys, insert_num * sizeof(K), h_keys,
                        insert_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE));
  NPU_CHECK(aclrtMemcpy(d_scores, insert_num * sizeof(S), h_scores,
                        insert_num * sizeof(S), ACL_MEMCPY_HOST_TO_DEVICE));
  NPU_CHECK(aclrtMemcpy(d_vectors, insert_num * sizeof(V) * options.dim,
                        h_vectors, insert_num * sizeof(V) * options.dim,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  using Table = npu::hkv::HashTable<K, V, S, EvictStrategy::kCustomized>;
  Table table;
  table.init(options);
  table.insert_or_assign(insert_num, d_keys, d_vectors, d_scores, stream);
  NPU_CHECK(aclrtSynchronizeStream(stream));

  // Build find keys so each AIV core gets a balanced number of hit keys.
  create_find_keys_balanced_by_core<K>(h_keys, insert_num, missed_ratio);
  NPU_CHECK(aclrtMemcpy(d_keys, insert_num * sizeof(K), h_keys,
                        insert_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE));

  NPU_CHECK(aclrtMemset(d_missed_size, sizeof(int), 0, sizeof(int)));
  auto timer = benchmark::Timer<double>();
  timer.start();
  table.find(insert_num, d_keys, d_vectors, d_missed_keys, d_missed_indices,
             d_missed_size, d_scores, stream);
  NPU_CHECK(aclrtSynchronizeStream(stream));
  timer.end();

  NPU_CHECK(aclrtFreeHost(h_keys));
  NPU_CHECK(aclrtFreeHost(h_scores));
  NPU_CHECK(aclrtFreeHost(h_vectors));
  NPU_CHECK(aclrtFree(d_keys));
  NPU_CHECK(aclrtFree(d_scores));
  NPU_CHECK(aclrtFree(d_vectors));
  NPU_CHECK(aclrtFree(d_missed_keys));
  NPU_CHECK(aclrtFree(d_missed_indices));
  NPU_CHECK(aclrtFree(d_missed_size));
  NPU_CHECK(aclrtDestroyStream(stream));

  NpuCheckError();
  // 与 hkv_hashtable_benchmark::test_one_api 一致：Billion-KV/s（除以 2^30）
  float throughput = insert_num / timer.getResult() / (1024 * 1024 * 1024.0f);
  print_result(load_factor, capacity, max_hbm_for_vectors_mb, max_bucket_size,
               dim, missed_ratio, throughput);
}

void test_main(double load_factor, double missed_ratio) {
  print_title();
  // pure HBM：容量 / dim / 向量占用 HBM 上限与 hkv_hashtable_benchmark 中纯 HBM
  // 多组配置同量级 （如 128M KV & dim=8、64M & dim=64 …）；首项 100M
  // 为额外大表压力场景。
  std::vector<size_t> capacities_hbm = {100000000UL,        128 * 1024 * 1024UL,
                                        64 * 1024 * 1024UL, 32 * 1024 * 1024UL,
                                        16 * 1024 * 1024UL, 8 * 1024 * 1024UL,
                                        4 * 1024 * 1024UL};
  // 向量维度（float 个数），与主 benchmark 扫的 dim 序列一致
  std::vector<size_t> dim_hbm = {8, 32, 64, 128, 256, 512, 1024};
  // max_hbm_for_vectors（MB）：dim=8 时用 8GB，其余行 16GB，与主 benchmark 各档
  // HBM 规模对齐
  std::vector<size_t> max_hbm_for_vectors_mb_hbm = {
      8 * 1024UL,  16 * 1024UL, 16 * 1024UL, 16 * 1024UL,
      16 * 1024UL, 16 * 1024UL, 16 * 1024UL};
  for (size_t i = 0; i < capacities_hbm.size(); i++) {
    // max_bucket_size：256 / 128 两种常见分桶上限，覆盖不同 kernel/tiling 路径
    test_find(capacities_hbm[i], dim_hbm[i], max_hbm_for_vectors_mb_hbm[i],
              load_factor, 256, missed_ratio);
    test_find(capacities_hbm[i], dim_hbm[i], max_hbm_for_vectors_mb_hbm[i],
              load_factor, 128, missed_ratio);
  }

  std::vector<size_t> capacities_hybrid = {
      256 * 1024 * 1024UL, 128 * 1024 * 1024UL, 64 * 1024 * 1024UL,
      32 * 1024 * 1024UL,  16 * 1024 * 1024UL,  8 * 1024 * 1024UL,
      4 * 1024 * 1024UL};
  std::vector<size_t> dim_hybrid = {16, 32, 64, 128, 256, 512, 1024};
  for (size_t i = 0; i < capacities_hybrid.size(); i++) {
    // hybrid mode
    test_find(capacities_hybrid[i], dim_hybrid[i], 8 * 1024UL, load_factor, 256,
              missed_ratio);
    test_find(capacities_hybrid[i], dim_hybrid[i], 8 * 1024UL, load_factor, 128,
              missed_ratio);
    // pure HMEM mode
    test_find(capacities_hybrid[i], dim_hybrid[i], 0, load_factor, 256,
              missed_ratio);
    test_find(capacities_hybrid[i], dim_hybrid[i], 0, load_factor, 128,
              missed_ratio);
  }
}

void query_memory() {
  size_t free = 0;
  size_t total = 0;
  NPU_CHECK(aclrtGetMemInfo(ACL_DDR_MEM, &free, &total));
  // 字节 -> GB（与 hkv_hashtable_benchmark::query_memory 相同：除以 2^30）
  std::cout << "DDR free memory:" << free / (1 << 30)
            << " GB, DDR total memory:" << total / (1 << 30) << " GB"
            << std::endl;

  NPU_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, &free, &total));
  std::cout << "HBM free memory:" << free / (1 << 30)
            << " GB, HBM total memory:" << total / (1 << 30) << " GB"
            << std::endl;
}

int main() {
  NPU_CHECK(aclInit(nullptr));
  auto device_id_env = std::getenv("HKV_TEST_DEVICE");
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
  g_aiv_num = ascendc_platform->GetCoreNumAiv();
  std::cout << "Soc version:" << aclrtGetSocName() << " device_id:" << device_id
            << " aiv_num: " << g_aiv_num << std::endl;
  query_memory();

  // 第一维：表装载率 λ（init 后键数 / capacity）；第二维：find 未命中比例
  // missed_ratio
  test_main(0.2, 0);
  test_main(0.2, 0.5);
  test_main(0.2, 1.0);
  test_main(0.5, 0);
  test_main(0.5, 0.5);
  test_main(0.5, 1.0);
  test_main(1.0, 0);
  test_main(1.0, 0.5);
  test_main(1.0, 1.0);

  NPU_CHECK(aclrtResetDevice(device_id));
  NPU_CHECK(aclFinalize());
  return 0;
}
