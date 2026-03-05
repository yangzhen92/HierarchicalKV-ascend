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
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_device_data.h"
#include "test_util.h"

using namespace std;
using namespace npu::hkv;
using namespace test_util;

template <typename K, typename V, typename S>
void test_erase_basic() {
  // 1. 初始化
  init_env();
  constexpr size_t key_num = 1UL * 1024;

  // 2. 建表
  auto table = get_default_table<K, V, S>();
  EXPECT_EQ(table->size(), 0);

  // 3. 申请hbm内存
  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num);

  // 4. 空桶插值
  vector<K> host_keys(key_num, 0);
  create_continuous_keys<K, S, V, DEFAULT_DIM>(host_keys.data(), nullptr,
                                               nullptr, key_num);
  device_data.copy_keys(host_keys, key_num);
  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, nullptr,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), key_num);

  // 5. 删除部分key
  size_t erase_num = key_num / 2;
  table->erase(erase_num, device_data.device_keys, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), key_num - erase_num);

  // 7. 再插入相同的key，确保size不变
  table->insert_or_assign(erase_num, device_data.device_keys,
                          device_data.device_values, nullptr,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), key_num);
}

TEST(test_erase, test_erase_basic) {
  test_erase_basic<uint64_t, float, uint64_t>();
  test_erase_basic<uint64_t, double, uint64_t>();
}

TEST(test_erase, test_erase_in_full_bucket) {
  using K = int64_t;
  using V = float;
  using S = uint64_t;

  // 1. 初始化
  init_env();
  constexpr size_t key_num = 1UL * 1024;

  // 2. 建表，使用capacity为128是为了让1024个key插入后填满整个桶
  constexpr size_t capacity = 128;
  auto table = get_table<K, V, S>(DEFAULT_DIM, capacity, 1);
  EXPECT_EQ(table->size(), 0);

  // 3. 申请hbm内存
  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num);

  // 4. 空桶插值，确保最后一个值在桶内
  vector<K> host_keys(key_num, 0);
  create_continuous_keys<K, S, V, DEFAULT_DIM>(host_keys.data(), nullptr,
                                               nullptr, key_num);
  device_data.copy_keys(host_keys, key_num);
  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, nullptr,
                          device_data.stream);
  table->insert_or_assign(1, device_data.device_keys + key_num - 1,
                          device_data.device_values, nullptr,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), capacity);

  // 5. 记录当前表内key的存在情况
  table->find(key_num, device_data.device_keys, device_data.device_values_ptr,
              device_data.device_found, nullptr, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  bool *host_found = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            key_num * sizeof(bool)), ACL_ERROR_NONE);
  ASSERT_EQ(
      aclrtMemcpy(host_found, key_num * sizeof(bool), device_data.device_found,
                  key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_found, key_num * sizeof(bool), device_data.device_found, sizeof(bool) * key_num,
                        ACL_MEMCPY_DEVICE_TO_HOST), ACL_ERROR_NONE);

  // 6. 在满桶中删除最后一个key
  table->erase(1, device_data.device_keys + key_num - 1, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), capacity - 1);

  // 7. 再插入相同的key，确保size不变，find结果不变
  table->insert_or_assign(1, device_data.device_keys + key_num - 1,
                          device_data.device_values, nullptr,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  table->find(key_num, device_data.device_keys, device_data.device_values_ptr,
              device_data.device_found, nullptr, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  bool *host_found_again = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found_again),
                            key_num * sizeof(bool)), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_found_again, key_num * sizeof(bool), device_data.device_found, sizeof(bool) * key_num,
                        ACL_MEMCPY_DEVICE_TO_HOST), ACL_ERROR_NONE);
  for (size_t i = 0; i < key_num; ++i) {
    EXPECT_EQ(host_found_again[i], host_found[i]);
  }
  aclrtFree(host_found);
  aclrtFree(host_found_again);
}