/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#include "test_util.h"

using namespace std;
using namespace npu::hkv;
using namespace test_util;

using K = uint64_t;
using V = float;
using S = uint64_t;

TEST(test_reserve, test_reserve_all_hbm) {
  // 1. 初始化
  init_env();

  size_t total_mem = 0;
  size_t free_mem = 0;
  constexpr size_t hbm_for_values = 1UL << 30;
  ASSERT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem),
            ACL_ERROR_NONE);
  ASSERT_GT(free_mem, hbm_for_values)
      << "free HBM is not enough free:" << free_mem
      << "need:" << hbm_for_values;

  constexpr size_t dim = 8;
  constexpr size_t key_num = 1UL * 1024;
  constexpr size_t init_capacity = 128UL * 1024;
  constexpr size_t max_capacity = init_capacity * 128;

  size_t each_key_size = sizeof(K);
  size_t each_score_size = sizeof(S);
  size_t each_value_size = sizeof(V);

  // 2. 建表
  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = max_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  using Table = HashTable<K, V>;

  Table table;
  table.init(options);
  EXPECT_EQ(table.size(), 0);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);

  // 3.2 申请hbm内存
  K* device_keys = nullptr;
  V* device_values = nullptr;
  V** device_values_ptr = nullptr;
  bool* device_found = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        key_num * each_key_size, ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(
      aclrtMalloc(reinterpret_cast<void**>(&device_values),
                  key_num * each_value_size * dim, ACL_MEM_MALLOC_HUGE_FIRST),
      ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values_ptr),
                        key_num * sizeof(V*), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                        key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 4. 插值
  // 4.1 生产连续值
  create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr, nullptr, key_num);
  ASSERT_EQ(aclrtMemcpy(device_keys, key_num * each_key_size, host_keys.data(),
                        key_num * each_key_size, ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_values, key_num * each_value_size * dim,
                        host_values.data(), key_num * each_value_size * dim,
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  // 4.2 插值
  table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(stream), key_num);

  // 5. 验证reserve
  auto cur_capacity = table.capacity();
  auto load_factor = table.load_factor(stream);
  EXPECT_EQ(cur_capacity, init_capacity);
  EXPECT_EQ(load_factor, (key_num * 1.0) / (init_capacity * 1.0));
  EXPECT_EQ(table.size(stream), key_num);
  // 5.1 不能缩小
  table.reserve(cur_capacity / 2, stream);
  load_factor = table.load_factor(stream);
  cur_capacity = table.capacity();
  EXPECT_EQ(cur_capacity, init_capacity);
  EXPECT_EQ(table.size(stream), key_num);
  EXPECT_EQ(load_factor, (key_num * 1.0) / (init_capacity * 1.0));
  // 5.2 容量满足不扩容
  table.reserve(cur_capacity);
  load_factor = table.load_factor(stream);
  cur_capacity = table.capacity();
  EXPECT_EQ(cur_capacity, init_capacity);
  EXPECT_EQ(table.size(stream), key_num);
  EXPECT_EQ(load_factor, (key_num * 1.0) / (init_capacity * 1.0));
  // 5.3 按2倍扩容
  table.reserve(cur_capacity + 1, stream);
  cur_capacity = table.capacity();
  load_factor = table.load_factor(stream);
  EXPECT_EQ(cur_capacity, init_capacity * 2);
  EXPECT_EQ(load_factor, (key_num * 1.0) / (init_capacity * 2.0));
  EXPECT_EQ(table.size(stream), key_num);
  // 5.4 按最大容量扩容
  table.reserve(max_capacity);
  cur_capacity = table.capacity();
  load_factor = table.load_factor(stream);
  EXPECT_EQ(cur_capacity, max_capacity);
  EXPECT_EQ(load_factor, (key_num * 1.0) / (max_capacity * 1.0));
  EXPECT_EQ(table.size(stream), key_num);

  // 6. 校验扩容后，value不变
  table.find(key_num, device_keys, device_values_ptr, device_found,
             nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  bool* host_found = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_found, key_num * sizeof(bool), device_found,
                        key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  vector<void*> real_values_ptr(key_num, nullptr);
  ASSERT_EQ(aclrtMemcpy(real_values_ptr.data(), key_num * sizeof(void*),
                        device_values_ptr, key_num * sizeof(void*),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  size_t found_num = 0;
  for (size_t i = 0; i < key_num; i++) {
    if (host_found[i]) {
      ASSERT_NE(real_values_ptr[i], nullptr);
      found_num++;

      vector<V> real_values(dim, 0);
      ASSERT_EQ(aclrtMemcpy(real_values.data(), dim * each_value_size,
                            real_values_ptr[i], dim * each_value_size,
                            ACL_MEMCPY_DEVICE_TO_HOST),
                ACL_ERROR_NONE);
      vector<V> expect_values(host_values.begin() + i * dim,
                              host_values.begin() + i * dim + dim);
      EXPECT_EQ(expect_values, real_values);
    } else {
      EXPECT_EQ(real_values_ptr[i], nullptr);
    }
  }
  EXPECT_EQ(found_num, key_num);

  table.clear(stream);
  EXPECT_EQ(table.size(stream), 0);

  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values_ptr), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);

  ASSERT_EQ(aclrtFreeHost(host_found), ACL_ERROR_NONE);
}

void test_dynamic_max_capacity_expand() {
  constexpr size_t dim = 64;
  constexpr size_t len = 10000;
  size_t max_capacity = 1 << 14;
  constexpr size_t init_capacity = 1 << 12;
  constexpr size_t uplimit = 1 << 20;
  constexpr float load_factor_threshold = 0.98f;

  init_env();

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = max_capacity,
      .max_hbm_for_vectors = uplimit * dim * sizeof(V),
      .dim = dim,
  };
  using Table = HashTable<K, V>;

  Table table;
  table.init(options);

  std::map<K, std::vector<V>> ref_map;

  DeviceData<K, V, S> insert_data;
  insert_data.malloc(len, dim);
  DeviceData<K, V, S> evict_data;
  evict_data.malloc(len, dim);

  size_t offset = 0;
  size_t total_len = 0;

  while (true) {
    // Generate sequential keys starting from offset
    std::vector<K> host_keys(len);
    std::vector<V> host_values(len * dim);
    for (size_t i = 0; i < len; ++i) {
      host_keys[i] = offset + i;
      for (size_t j = 0; j < dim; ++j) {
        host_values[i * dim + j] = static_cast<V>(host_keys[i]);
      }
    }
    insert_data.copy_keys(host_keys, len);
    insert_data.copy_values(host_values, len, dim);

    size_t n_evicted = table.insert_and_evict(
        len, insert_data.device_keys, insert_data.device_values, nullptr,
        evict_data.device_keys, evict_data.device_values, nullptr,
        insert_data.stream);
    offset += len;
    total_len += len;

    if (n_evicted > 0) {
      std::vector<K> evicted_keys(n_evicted);
      std::vector<V> evicted_values(n_evicted * dim);
      ASSERT_EQ(aclrtMemcpy(evicted_keys.data(), n_evicted * sizeof(K),
                            evict_data.device_keys, n_evicted * sizeof(K),
                            ACL_MEMCPY_DEVICE_TO_HOST),
                ACL_ERROR_NONE);
      ASSERT_EQ(aclrtMemcpy(evicted_values.data(), n_evicted * dim * sizeof(V),
                            evict_data.device_values, n_evicted * dim * sizeof(V),
                            ACL_MEMCPY_DEVICE_TO_HOST),
                ACL_ERROR_NONE);
      for (size_t i = 0; i < n_evicted; ++i) {
        std::vector<V> vec(dim);
        for (size_t j = 0; j < dim; ++j) {
          vec[j] = evicted_values[i * dim + j];
        }
        ref_map[evicted_keys[i]] = vec;
      }
    }

    ASSERT_EQ(aclrtSynchronizeStream(insert_data.stream), ACL_ERROR_NONE);

    if (table.load_factor(insert_data.stream) >= load_factor_threshold) {
      ASSERT_GE(table.size(insert_data.stream),
                static_cast<size_t>(static_cast<float>(max_capacity) *
                                    load_factor_threshold));
      max_capacity *= 2;
      if (max_capacity > uplimit) {
        break;
      }
      
      table.set_max_capacity(max_capacity);
      table.reserve(max_capacity, insert_data.stream);
      ASSERT_EQ(max_capacity, table.capacity());
      ASSERT_LE(table.load_factor(insert_data.stream), 0.5f);
    }

    if (total_len > uplimit * 2) {
      throw std::runtime_error("Traverse too much keys but not finish test.");
    }
  }

  // Export all keys and verify
  DeviceData<K, V, S> export_data;
  export_data.malloc(table.capacity(), dim);

  for (size_t export_offset = 0; export_offset < table.capacity();
       export_offset += len) {
    size_t search_len = len;
    if (export_offset + search_len > table.capacity()) {
      search_len = table.capacity() - export_offset;
    }
    size_t n_exported = table.export_batch(
        search_len, export_offset, export_data.device_keys,
        export_data.device_values, nullptr, export_data.stream);
    ASSERT_EQ(aclrtSynchronizeStream(export_data.stream), ACL_ERROR_NONE);

    std::vector<K> exported_keys(n_exported);
    std::vector<V> exported_values(n_exported * dim);
    ASSERT_EQ(aclrtMemcpy(exported_keys.data(), n_exported * sizeof(K),
                          export_data.device_keys, n_exported * sizeof(K),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMemcpy(exported_values.data(), n_exported * dim * sizeof(V),
                          export_data.device_values, n_exported * dim * sizeof(V),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);

    for (size_t i = 0; i < n_exported; ++i) {
      std::vector<V> vec(dim);
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = exported_values[i * dim + j];
      }
      ref_map[exported_keys[i]] = vec;
    }
  }

  for (const auto& it : ref_map) {
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_EQ(static_cast<V>(it.first), it.second[j]);
    }
  }

  ASSERT_EQ(table.capacity() * 2, max_capacity);
  ASSERT_GE(static_cast<float>(ref_map.size()),
            static_cast<float>(table.capacity()) * load_factor_threshold);
}

TEST(test_reserve, test_dynamic_max_capacity_expand) {
  test_dynamic_max_capacity_expand();
}