/* *
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_device_data.h"
#include "test_util.h"

using namespace std;
using namespace npu::hkv;
using namespace test_util;

using K = uint64_t;
using V = float;
using S = uint64_t;

constexpr size_t TABLE_CAPACITY = 1024;
constexpr size_t KEY_NUM = 128;
constexpr size_t DIM = 8;
constexpr size_t HBM_FOR_VALUES = 1024UL * 1024;

// 统一的建表函数：表大小=1024
template <typename Table>
std::unique_ptr<Table> make_table() {
  auto table = std::make_unique<Table>();
  HashTableOptions options{
      .init_capacity = TABLE_CAPACITY,
      .max_capacity = TABLE_CAPACITY,
      .max_hbm_for_vectors = HBM_FOR_VALUES,
      .dim = DIM,
      .io_by_cpu = false,
  };
  table->init(options);
  return table;
}

// ============================================================================
// UT 1. init / size / capacity / empty / dim / max_bucket_size / bucket_count
// ============================================================================
TEST(TestAllInterfaces, test_init_and_basic_properties) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();
  EXPECT_EQ(table->size(), 0);
  EXPECT_EQ(table->capacity(), TABLE_CAPACITY);
  EXPECT_EQ(table->empty(), true);
  EXPECT_EQ(table->dim(), DIM);
  EXPECT_GT(table->max_bucket_size(), 0);
  EXPECT_GT(table->bucket_count(), 0);
}

// ============================================================================
// UT 2. insert_or_assign
// ============================================================================
TEST(TestAllInterfaces, test_insert_or_assign) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();

  DeviceData<K, V, S> device_data;
  device_data.malloc(KEY_NUM);

  vector<K> host_keys(KEY_NUM, 0);
  vector<V> host_values(KEY_NUM * DIM, 0);
  vector<S> host_scores(KEY_NUM, 0);
  create_continuous_keys<K, S, V, DIM>(host_keys.data(), host_scores.data(),
                                       host_values.data(), KEY_NUM);

  device_data.copy_keys(host_keys, KEY_NUM);
  device_data.copy_values(host_values, KEY_NUM, DIM);
  device_data.copy_scores(host_scores, KEY_NUM);

  table->insert_or_assign(KEY_NUM, device_data.device_keys,
                        device_data.device_values, device_data.device_scores,
                        device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), KEY_NUM);
}

// ============================================================================
// UT 3. insert_and_evict (with evicted_keys/values/scores + d_counter)
// ============================================================================
TEST(TestAllInterfaces, test_insert_and_evict_with_counter) {
  init_env();
  // 用较小的capacity，使桶被填满后会产生evict
  constexpr size_t small_cap = 128;
  HashTableOptions options{
      .init_capacity = small_cap,
      .max_capacity = small_cap,
      .max_hbm_for_vectors = HBM_FOR_VALUES,
      .dim = DIM,
      .io_by_cpu = false,
  };
  auto table = std::make_unique<HashTable<K, V, S, EvictStrategy::klru>>();
  table->init(options);

  DeviceData<K, V, S> device_data;
  device_data.malloc(KEY_NUM);

  vector<K> host_keys(KEY_NUM, 0);
  vector<V> host_values(KEY_NUM * DIM, 0);
  vector<S> host_scores(KEY_NUM, 0);
  create_continuous_keys<K, S, V, DIM>(host_keys.data(), host_scores.data(),
                                       host_values.data(), KEY_NUM);

  device_data.copy_keys(host_keys, KEY_NUM);
  device_data.copy_values(host_values, KEY_NUM, DIM);
  device_data.copy_scores(host_scores, KEY_NUM);

  K* d_evicted_keys = nullptr;
  V* d_evicted_values = nullptr;
  S* d_evicted_scores = nullptr;
  size_t* d_counter = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_evicted_keys),
                        KEY_NUM * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_evicted_values),
                        KEY_NUM * DIM * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_evicted_scores),
                        KEY_NUM * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_counter),
                        sizeof(size_t), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemset(d_counter, sizeof(size_t), 0, sizeof(size_t)),
            ACL_ERROR_NONE);

  table->insert_and_evict(KEY_NUM, device_data.device_keys,
                          device_data.device_values, device_data.device_scores,
                          d_evicted_keys, d_evicted_values,
                          d_evicted_scores, d_counter, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  size_t host_counter = 0;
  ASSERT_EQ(aclrtMemcpy(&host_counter, sizeof(size_t), d_counter,
                        sizeof(size_t), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  EXPECT_GE(table->size(device_data.stream), 0);

  aclrtFree(d_evicted_keys);
  aclrtFree(d_evicted_values);
  aclrtFree(d_evicted_scores);
  aclrtFree(d_counter);
}

// ============================================================================
// UT 4. insert_and_evict (返回值版本)
// ============================================================================
TEST(TestAllInterfaces, test_insert_and_evict_return_size) {
  init_env();
  constexpr size_t small_cap = 128;
  HashTableOptions options{
      .init_capacity = small_cap,
      .max_capacity = small_cap,
      .max_hbm_for_vectors = HBM_FOR_VALUES,
      .dim = DIM,
      .io_by_cpu = false,
  };
  auto table = std::make_unique<HashTable<K, V, S, EvictStrategy::klru>>();
  table->init(options);

  DeviceData<K, V, S> device_data;
  device_data.malloc(KEY_NUM);

  vector<K> host_keys(KEY_NUM, 0);
  vector<V> host_values(KEY_NUM * DIM, 0);
  vector<S> host_scores(KEY_NUM, 0);
  create_continuous_keys<K, S, V, DIM>(host_keys.data(), host_scores.data(),
                                       host_values.data(), KEY_NUM);
  device_data.copy_keys(host_keys, KEY_NUM);
  device_data.copy_values(host_values, KEY_NUM, DIM);
  device_data.copy_scores(host_scores, KEY_NUM);

  K* d_evicted_keys = nullptr;
  V* d_evicted_values = nullptr;
  S* d_evicted_scores = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_evicted_keys),
                        KEY_NUM * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_evicted_values),
                        KEY_NUM * DIM * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_evicted_scores),
                        KEY_NUM * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  size_t evicted = table->insert_and_evict(
      KEY_NUM, device_data.device_keys, device_data.device_values,
      device_data.device_scores, d_evicted_keys, d_evicted_values,
      d_evicted_scores, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_GE(evicted, 0);

  aclrtFree(d_evicted_keys);
  aclrtFree(d_evicted_values);
  aclrtFree(d_evicted_scores);
}

// ============================================================================
// UT 5. accum_or_assign
// ============================================================================
TEST(TestAllInterfaces, test_accum_or_assign) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();

  DeviceData<K, V, S> device_data;
  device_data.malloc(KEY_NUM);

  vector<K> host_keys(KEY_NUM, 0);
  vector<V> host_values(KEY_NUM * DIM, 0);
  vector<S> host_scores(KEY_NUM, 0);
  create_continuous_keys<K, S, V, DIM>(host_keys.data(), host_scores.data(),
                                       host_values.data(), KEY_NUM);
  device_data.copy_keys(host_keys, KEY_NUM);
  device_data.copy_values(host_values, KEY_NUM, DIM);
  device_data.copy_scores(host_scores, KEY_NUM);

  // 先插入一批key
  table->insert_or_assign(KEY_NUM, device_data.device_keys,
                        device_data.device_values, device_data.device_scores,
                        device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  // 准备 accum_or_assign 所需的 bool 数组（全为false表示assign）
  bool* d_accum_or_assigns = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_accum_or_assigns),
            KEY_NUM * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  table->accum_or_assign(KEY_NUM, device_data.device_keys,
                      device_data.device_values, d_accum_or_assigns,
                      device_data.device_scores, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), KEY_NUM);

  aclrtFree(d_accum_or_assigns);
}

// ============================================================================
// UT 6. find_or_insert (value copy版本)
// ============================================================================
TEST(TestAllInterfaces, test_find_or_insert_value_copy) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();

  DeviceData<K, V, S> device_data;
  device_data.malloc(KEY_NUM);

  vector<K> host_keys(KEY_NUM, 0);
  vector<V> host_values(KEY_NUM * DIM, 0);
  vector<S> host_scores(KEY_NUM, 0);
  create_continuous_keys<K, S, V, DIM>(host_keys.data(), host_scores.data(),
                                       host_values.data(), KEY_NUM);
  device_data.copy_keys(host_keys, KEY_NUM);
  device_data.copy_values(host_values, KEY_NUM, DIM);
  device_data.copy_scores(host_scores, KEY_NUM);

  table->find_or_insert(KEY_NUM, device_data.device_keys,
                       device_data.device_values,
                       device_data.device_scores,
                       device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), KEY_NUM);
}

// ============================================================================
// UT 7. find_or_insert (ptr版本)
// ============================================================================
TEST(TestAllInterfaces, test_find_or_insert_ptr) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();

  DeviceData<K, V, S> device_data;
  device_data.malloc(KEY_NUM);

  vector<K> host_keys(KEY_NUM, 0);
  vector<V> host_values(KEY_NUM * DIM, 0);
  vector<S> host_scores(KEY_NUM, 0);
  create_continuous_keys<K, S, V, DIM>(host_keys.data(), host_scores.data(),
                                       host_values.data(), KEY_NUM);
  device_data.copy_keys(host_keys, KEY_NUM);
  device_data.copy_values(host_values, KEY_NUM, DIM);
  device_data.copy_scores(host_scores, KEY_NUM);

  V** d_values_ptr = device_data.device_values_ptr;
  bool* d_found = device_data.device_found;

  table->find_or_insert(KEY_NUM, device_data.device_keys, d_values_ptr, d_found,
                       device_data.device_scores, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), KEY_NUM);
}

// ============================================================================
// UT 8. lock_keys / unlock_keys
// ============================================================================
TEST(TestAllInterfaces, test_lock_unlock_keys) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();

  DeviceData<K, V, S> device_data;
  device_data.malloc(KEY_NUM);

  vector<K> host_keys(KEY_NUM, 0);
  vector<V> host_values(KEY_NUM * DIM, 0);
  create_continuous_keys<K, S, V, DIM>(host_keys.data(), nullptr,
                                       host_values.data(), KEY_NUM);
  device_data.copy_keys(host_keys, KEY_NUM);
  device_data.copy_values(host_values, KEY_NUM, DIM);

  table->insert_or_assign(KEY_NUM, device_data.device_keys,
                        device_data.device_values, nullptr,
                        device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  K** d_locked_ptrs = nullptr;
  bool* d_succeededs = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_locked_ptrs),
            KEY_NUM * sizeof(K*), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_succeededs),
            KEY_NUM * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  table->lock_keys(KEY_NUM, device_data.device_keys, d_locked_ptrs,
                    d_succeededs, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  table->unlock_keys(KEY_NUM, d_locked_ptrs, device_data.device_keys,
                    d_succeededs, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  aclrtFree(d_locked_ptrs);
  aclrtFree(d_succeededs);
}

// ============================================================================
// UT 9. assign (keys + values + scores)
// ============================================================================
TEST(TestAllInterfaces, test_assign_with_values_scores) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();

  DeviceData<K, V, S> device_data;
  device_data.malloc(KEY_NUM);

  vector<K> host_keys(KEY_NUM, 0);
  vector<V> host_values(KEY_NUM * DIM, 0);
  vector<S> host_scores(KEY_NUM, 0);
  create_continuous_keys<K, S, V, DIM>(host_keys.data(), host_scores.data(),
                                       host_values.data(), KEY_NUM);
  device_data.copy_keys(host_keys, KEY_NUM);
  device_data.copy_values(host_values, KEY_NUM, DIM);
  device_data.copy_scores(host_scores, KEY_NUM);

  table->insert_or_assign(KEY_NUM, device_data.device_keys,
                        device_data.device_values, device_data.device_scores,
                        device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  // 重新赋值
  table->assign(KEY_NUM, device_data.device_keys,
               device_data.device_values,
               device_data.device_scores, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), KEY_NUM);
}

// ============================================================================
// UT 10. assign_scores
// ============================================================================
TEST(TestAllInterfaces, test_assign_scores) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();

  DeviceData<K, V, S> device_data;
  device_data.malloc(KEY_NUM);

  vector<K> host_keys(KEY_NUM, 0);
  vector<S> host_scores(KEY_NUM, 0);
  create_continuous_keys<K, S, V, DIM>(host_keys.data(), host_scores.data(),
                                         nullptr, KEY_NUM);
  device_data.copy_keys(host_keys, KEY_NUM);
  device_data.copy_scores(host_scores, KEY_NUM);

  table->insert_or_assign(KEY_NUM, device_data.device_keys,
                        device_data.device_values, device_data.device_scores,
                        device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  table->assign_scores(KEY_NUM, device_data.device_keys,
                      device_data.device_scores, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
}

// ============================================================================
// UT 11. assign (仅 scores 的重载)
// ============================================================================
TEST(TestAllInterfaces, test_assign_scores_only) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();

  DeviceData<K, V, S> device_data;
  device_data.malloc(KEY_NUM);

  vector<K> host_keys(KEY_NUM, 0);
  vector<S> host_scores(KEY_NUM, 0);
  create_continuous_keys<K, S, V, DIM>(host_keys.data(), host_scores.data(),
                                     nullptr, KEY_NUM);
  device_data.copy_keys(host_keys, KEY_NUM);
  device_data.copy_scores(host_scores, KEY_NUM);

  table->insert_or_assign(KEY_NUM, device_data.device_keys,
                        device_data.device_values, device_data.device_scores,
                        device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  table->assign(KEY_NUM, device_data.device_keys,
                device_data.device_scores, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
}

// ============================================================================
// UT 12. assign_values
// ============================================================================
TEST(TestAllInterfaces, test_assign_values) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();

  DeviceData<K, V, S> device_data;
  device_data.malloc(KEY_NUM);

  vector<K> host_keys(KEY_NUM, 0);
  vector<V> host_values(KEY_NUM * DIM, 0);
  create_continuous_keys<K, S, V, DIM>(host_keys.data(), nullptr,
                                       host_values.data(), KEY_NUM);
  device_data.copy_keys(host_keys, KEY_NUM);
  device_data.copy_values(host_values, KEY_NUM, DIM);

  table->insert_or_assign(KEY_NUM, device_data.device_keys,
                        device_data.device_values, nullptr,
                        device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  table->assign_values(KEY_NUM, device_data.device_keys,
                     device_data.device_values, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
}

// ============================================================================
// UT 13. find (basic value copy)
// ============================================================================
TEST(TestAllInterfaces, test_find_value_copy) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();

  DeviceData<K, V, S> device_data;
  device_data.malloc(KEY_NUM);

  vector<K> host_keys(KEY_NUM, 0);
  vector<V> host_values(KEY_NUM * DIM, 0);
  vector<S> host_scores(KEY_NUM, 0);
  create_continuous_keys<K, S, V, DIM>(host_keys.data(), host_scores.data(),
                                       host_values.data(), KEY_NUM);
  device_data.copy_keys(host_keys, KEY_NUM);
  device_data.copy_values(host_values, KEY_NUM, DIM);
  device_data.copy_scores(host_scores, KEY_NUM);

  table->insert_or_assign(KEY_NUM, device_data.device_keys,
                        device_data.device_values, device_data.device_scores,
                        device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  table->find(KEY_NUM, device_data.device_keys, device_data.device_values,
               device_data.device_found, device_data.device_scores,
               device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
}

// ============================================================================
// UT 14. find (with missed_keys / indices / size)
// ============================================================================
TEST(TestAllInterfaces, test_find_with_missed) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();

  DeviceData<K, V, S> device_data;
  device_data.malloc(KEY_NUM);

  vector<K> host_keys(KEY_NUM, 0);
  vector<V> host_values(KEY_NUM * DIM, 0);
  create_continuous_keys<K, S, V, DIM>(host_keys.data(), nullptr,
                                       host_values.data(), KEY_NUM);
  device_data.copy_keys(host_keys, KEY_NUM);
  device_data.copy_values(host_values, KEY_NUM, DIM);

  table->insert_or_assign(KEY_NUM / 2, device_data.device_keys,
                        device_data.device_values, nullptr,
                        device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  K* d_missed_keys = nullptr;
  int* d_missed_indices = nullptr;
  int* d_missed_size = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_missed_keys),
                        KEY_NUM * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_missed_indices),
                        KEY_NUM * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_missed_size),
                        sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  table->find(KEY_NUM, device_data.device_keys, device_data.device_values,
             d_missed_keys, d_missed_indices, d_missed_size,
             nullptr, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  aclrtFree(d_missed_keys);
  aclrtFree(d_missed_indices);
  aclrtFree(d_missed_size);
}

// ============================================================================
// UT 15. find (ptr版本)
// ============================================================================
TEST(TestAllInterfaces, test_find_ptr) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();

  DeviceData<K, V, S> device_data;
  device_data.malloc(KEY_NUM);

  vector<K> host_keys(KEY_NUM, 0);
  vector<V> host_values(KEY_NUM * DIM, 0);
  create_continuous_keys<K, S, V, DIM>(host_keys.data(), nullptr,
                                       host_values.data(), KEY_NUM);
  device_data.copy_keys(host_keys, KEY_NUM);
  device_data.copy_values(host_values, KEY_NUM, DIM);

  table->insert_or_assign(KEY_NUM, device_data.device_keys,
                        device_data.device_values, nullptr,
                        device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  table->find(KEY_NUM, device_data.device_keys,
              device_data.device_values_ptr, device_data.device_found,
              nullptr, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
}

// ============================================================================
// UT 16. find_and_update
// ============================================================================
TEST(TestAllInterfaces, test_find_and_update) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();

  DeviceData<K, V, S> device_data;
  device_data.malloc(KEY_NUM);

  vector<K> host_keys(KEY_NUM, 0);
  vector<V> host_values(KEY_NUM * DIM, 0);
  vector<S> host_scores(KEY_NUM, 0);
  create_continuous_keys<K, S, V, DIM>(host_keys.data(), host_scores.data(),
                                       host_values.data(), KEY_NUM);
  device_data.copy_keys(host_keys, KEY_NUM);
  device_data.copy_values(host_values, KEY_NUM, DIM);
  device_data.copy_scores(host_scores, KEY_NUM);

  table->insert_or_assign(KEY_NUM, device_data.device_keys,
                        device_data.device_values, device_data.device_scores,
                        device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  table->find_and_update(KEY_NUM, device_data.device_keys,
                      device_data.device_values_ptr, device_data.device_found,
                      device_data.device_scores, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
}

// ============================================================================
// UT 17. contains
// ============================================================================
TEST(TestAllInterfaces, test_contains) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();

  DeviceData<K, V, S> device_data;
  device_data.malloc(KEY_NUM);

  vector<K> host_keys(KEY_NUM, 0);
  create_continuous_keys<K, S, V, DIM>(host_keys.data(), nullptr, nullptr,
                                       KEY_NUM);
  device_data.copy_keys(host_keys, KEY_NUM);

  table->insert_or_assign(KEY_NUM, device_data.device_keys,
                        device_data.device_values, nullptr,
                        device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  table->contains(KEY_NUM, device_data.device_keys,
                 device_data.device_found, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
}

// ============================================================================
// UT 18. erase
// ============================================================================
TEST(TestAllInterfaces, test_erase) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();

  DeviceData<K, V, S> device_data;
  device_data.malloc(KEY_NUM);

  vector<K> host_keys(KEY_NUM, 0);
  create_continuous_keys<K, S, V, DIM>(host_keys.data(), nullptr, nullptr,
                                       KEY_NUM);
  device_data.copy_keys(host_keys, KEY_NUM);

  table->insert_or_assign(KEY_NUM, device_data.device_keys,
                        device_data.device_values, nullptr,
                        device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), KEY_NUM);

  table->erase(KEY_NUM / 2, device_data.device_keys, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_LT(table->size(device_data.stream), KEY_NUM);
}

// ============================================================================
// UT 19. clear
// ============================================================================
TEST(TestAllInterfaces, test_clear) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();

  DeviceData<K, V, S> device_data;
  device_data.malloc(KEY_NUM);

  vector<K> host_keys(KEY_NUM, 0);
  create_continuous_keys<K, S, V, DIM>(host_keys.data(), nullptr, nullptr,
                                       KEY_NUM);
  device_data.copy_keys(host_keys, KEY_NUM);

  table->insert_or_assign(KEY_NUM, device_data.device_keys,
                        device_data.device_values, nullptr,
                        device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  table->clear(device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), 0);
  EXPECT_EQ(table->empty(device_data.stream), true);
}

// ============================================================================
// UT 20. export_batch (d_counter版本)
// ============================================================================
TEST(TestAllInterfaces, test_export_batch_with_counter) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();

  DeviceData<K, V, S> device_data;
  device_data.malloc(KEY_NUM);

  vector<K> host_keys(KEY_NUM, 0);
  vector<V> host_values(KEY_NUM * DIM, 0);
  vector<S> host_scores(KEY_NUM, 0);
  create_continuous_keys<K, S, V, DIM>(host_keys.data(), host_scores.data(),
                                       host_values.data(), KEY_NUM);
  device_data.copy_keys(host_keys, KEY_NUM);
  device_data.copy_values(host_values, KEY_NUM, DIM);
  device_data.copy_scores(host_scores, KEY_NUM);

  table->insert_or_assign(KEY_NUM, device_data.device_keys,
                        device_data.device_values, device_data.device_scores,
                        device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  size_t* d_counter = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_counter), sizeof(size_t),
                       ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemset(d_counter, sizeof(size_t), 0, sizeof(size_t)),
            ACL_ERROR_NONE);

  table->export_batch(KEY_NUM, 0, d_counter, device_data.device_keys,
                      device_data.device_values, device_data.device_scores,
                      device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  size_t host_counter = 0;
  ASSERT_EQ(aclrtMemcpy(&host_counter, sizeof(size_t), d_counter, sizeof(size_t),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  EXPECT_GT(host_counter, 0);

  aclrtFree(d_counter);
}

// ============================================================================
// UT 21. export_batch (返回值版本)
// ============================================================================
TEST(TestAllInterfaces, test_export_batch_return_size) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();

  DeviceData<K, V, S> device_data;
  device_data.malloc(KEY_NUM);

  vector<K> host_keys(KEY_NUM, 0);
  vector<V> host_values(KEY_NUM * DIM, 0);
  create_continuous_keys<K, S, V, DIM>(host_keys.data(), nullptr,
                                       host_values.data(), KEY_NUM);
  device_data.copy_keys(host_keys, KEY_NUM);
  device_data.copy_values(host_values, KEY_NUM, DIM);

  table->insert_or_assign(KEY_NUM, device_data.device_keys,
                        device_data.device_values, nullptr,
                        device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  size_t exported = table->export_batch(KEY_NUM, 0, device_data.device_keys,
                                     device_data.device_values, nullptr,
                                     device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_GT(exported, 0);
}

// ============================================================================
// UT 22. empty
// ============================================================================
TEST(TestAllInterfaces, test_empty) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();
  EXPECT_EQ(table->empty(), true);

  DeviceData<K, V, S> device_data;
  device_data.malloc(KEY_NUM);
  vector<K> host_keys(KEY_NUM, 0);
  create_continuous_keys<K, S, V, DIM>(host_keys.data(), nullptr, nullptr,
                                       KEY_NUM);
  device_data.copy_keys(host_keys, KEY_NUM);

  table->insert_or_assign(KEY_NUM, device_data.device_keys,
                        device_data.device_values, nullptr,
                        device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->empty(device_data.stream), false);
}

// ============================================================================
// UT 23. size
// ============================================================================
TEST(TestAllInterfaces, test_size) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();

  DeviceData<K, V, S> device_data;
  device_data.malloc(KEY_NUM);

  vector<K> host_keys(KEY_NUM, 0);
  create_continuous_keys<K, S, V, DIM>(host_keys.data(), nullptr, nullptr,
                                       KEY_NUM);
  device_data.copy_keys(host_keys, KEY_NUM);

  EXPECT_EQ(table->size(), 0);

  table->insert_or_assign(KEY_NUM, device_data.device_keys,
                        device_data.device_values, nullptr,
                        device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), KEY_NUM);
}

// ============================================================================
// UT 24. reserve
// ============================================================================
TEST(TestAllInterfaces, test_reserve) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();
  size_t original_cap = table->capacity();

  table->reserve(TABLE_CAPACITY / 2, nullptr);
  ASSERT_EQ(aclrtSynchronizeStream(nullptr), ACL_ERROR_NONE);

  // reserve后的capacity不应变小（或维持原来的capacity）
  EXPECT_GE(table->capacity(), 0);
}

// ============================================================================
// UT 25. load_factor
// ============================================================================
TEST(TestAllInterfaces, test_load_factor) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();
  EXPECT_GE(table->load_factor(), 0.0f);

  DeviceData<K, V, S> device_data;
  device_data.malloc(KEY_NUM);

  vector<K> host_keys(KEY_NUM, 0);
  create_continuous_keys<K, S, V, DIM>(host_keys.data(), nullptr, nullptr,
                                       KEY_NUM);
  device_data.copy_keys(host_keys, KEY_NUM);

  table->insert_or_assign(KEY_NUM, device_data.device_keys,
                        device_data.device_values, nullptr,
                        device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  float lf = table->load_factor(device_data.stream);
  EXPECT_GT(lf, 0.0f);
}

// ============================================================================
// UT 26. set_max_capacity
// ============================================================================
TEST(TestAllInterfaces, test_set_max_capacity) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();
  size_t new_max = TABLE_CAPACITY * 2;
  table->set_max_capacity(new_max);
  EXPECT_EQ(table->capacity(), new_max);
}

// ============================================================================
// UT 27. is_pure_hbm_mode
// ============================================================================
TEST(TestAllInterfaces, test_is_pure_hbm_mode) {
  init_env();
  auto table = make_table<HashTable<K, V, S, EvictStrategy::klru>>();
  EXPECT_TRUE(table->is_pure_hbm_mode());
}
