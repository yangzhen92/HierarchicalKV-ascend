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

#include <assert.h>
#include <algorithm>
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
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "benchmark_util.h"
#include "debug.h"
#include "tiling/platform/platform_ascendc.h"

using K = uint64_t;
using V = float;
using S = uint64_t;
using EvictStrategy = npu::hkv::EvictStrategy;
using TableOptions = npu::hkv::HashTableOptions;

static uint32_t g_core_num_aiv = 0;

void print_title() {
  std::cout << std::endl
            << "|    \u03BB " << "| capacity " << "| max_hbm_for_vectors "
            << "| max_bucket_size " << "| dim " << "| missed_ratio "
            << "| througput(BillionKV/secs) ";
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

void print_result(double load_factor, size_t capacity,
                  size_t max_hbm_for_vectors_mb, size_t max_bucket_size,
                  size_t dim, double missed_ratio, float througput) {
  print_w(load_factor, 6);
  print_w(capacity, 10);
  print_w(max_hbm_for_vectors_mb, 21);
  print_w(max_bucket_size, 17);
  print_w(dim, 5);
  print_w(missed_ratio, 14);
  print_w(througput, 27);
  std::cout << "|\n";
}

void test_find(size_t capacity, size_t dim, size_t max_hbm_for_vectors_mb,
               double load_factor, size_t max_bucket_size,
               double missed_ratio) {
  HKV_CHECK(load_factor >= 0.0 && load_factor <= 1.0,
            "Invalid `load_factor`");
  K* h_keys;
  S* h_scores;
  V* h_vectors;

  TableOptions options;
  options.init_capacity = capacity;
  options.max_capacity = capacity;
  options.dim = dim;
  options.max_hbm_for_vectors = max_hbm_for_vectors_mb * 1024UL * 1024UL;
  options.max_bucket_size = max_bucket_size;

  size_t key_num = capacity;
  NPU_CHECK(aclrtMallocHost((void**)&h_keys, key_num * sizeof(K)));
  NPU_CHECK(aclrtMallocHost((void**)&h_scores, key_num * sizeof(S)));
  NPU_CHECK(aclrtMallocHost((void**)&h_vectors, key_num * options.dim * sizeof(V)));

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

  size_t find_num = static_cast<size_t>(
      static_cast<double>(insert_num) * (1.0 - missed_ratio));
  benchmark::create_continuous_keys<K, S>(h_keys, nullptr, find_num, 0);
  benchmark::create_continuous_keys<K, S>(
      h_keys + find_num, nullptr, insert_num - find_num, insert_num);
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
  float througput = insert_num / timer.getResult() / (1024 * 1024 * 1024.0f);
  print_result(load_factor, capacity, max_hbm_for_vectors_mb, max_bucket_size,
               dim, missed_ratio, througput);
}

void test_main(double load_factor, double missed_ratio) {
  constexpr size_t CAPACITY = 100000000UL;
  print_title();
  // pure HBM
  test_find(CAPACITY, 8, 8 * 1024UL, load_factor, 256, missed_ratio);
  test_find(CAPACITY, 8, 8 * 1024UL, load_factor, 128, missed_ratio);
  // TODO: hybrid mode
  // test_find(CAPACITY, 8, 1 * 1024UL, load_factor, 256, missed_ratio);
  // test_find(CAPACITY, 8, 1 * 1024UL, load_factor, 128, missed_ratio);
  // TODO: pure HMEM mode
  // test_find(CAPACITY, 8, 0, load_factor, 256, missed_ratio);
  // test_find(CAPACITY, 8, 0, load_factor, 128, missed_ratio);
}

void query_memory() {
  size_t free = 0;
  size_t total = 0;
  NPU_CHECK(aclrtGetMemInfo(ACL_DDR_MEM, &free, &total));
  std::cout << "DDR free memory:" << free / (1 << 30)
            << " GB, DDR total memory:" << total / (1 << 30) << " GB"
            << std::endl;

  NPU_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, &free, &total));
  std::cout << "HBM free memory:" << free / (1 << 30)
            << " GB, HBM total memory:" << total / (1 << 30) << " GB"
            << std::endl;
}

int main() {
  const char* socVersion = SOC_VERSION;
  auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
  HKV_CHECK(ascendc_platform != nullptr,
            "Get ascendc platform info failed, please check SOC_VERSION!");
  g_core_num_aiv = ascendc_platform->GetCoreNumAiv();
  std::cout << "Soc version: " << socVersion
            << " aiv_num: " << g_core_num_aiv << std::endl;

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
  std::cout << "aclrtGetSocName:" << aclrtGetSocName()
            << " device_id:" << device_id << std::endl;
  query_memory();

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
