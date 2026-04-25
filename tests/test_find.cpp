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

#include <gtest/gtest.h>

#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

using namespace npu::hkv;
using namespace test_util;

namespace {

using K = uint64_t;
using V = float;
using S = uint64_t;

constexpr size_t kDefaultDim = 8;
constexpr size_t kDefaultInitCapacity = 128UL * 1024;
constexpr size_t kDefaultHbmForValues = 1UL << 30;

class FindValueTest : public ::testing::Test {
 public:
  void SetUp() override { init_env(); }

  template <typename Table>
  void InitTable(Table& table, size_t dim,
                 size_t init_capacity = kDefaultInitCapacity) {
    HashTableOptions options;
    options.init_capacity = init_capacity;
    options.max_capacity = init_capacity;
    options.max_hbm_for_vectors = kDefaultHbmForValues;
    options.dim = dim;
    options.io_by_cpu = false;
    table.init(options);
  }

  template <typename Table, size_t DIM>
  void InsertContinuousKeys(Table& table, size_t key_num,
                            std::vector<K>& host_keys,
                            std::vector<V>& host_values,
                            std::vector<S>& host_scores, bool with_scores,
                            aclrtStream stream, K start = 1) {
    host_keys.resize(key_num);
    host_values.resize(key_num * DIM);
    host_scores.resize(with_scores ? key_num : 0);
    create_continuous_keys<K, S, V, DIM>(
        host_keys.data(), with_scores ? host_scores.data() : nullptr,
        host_values.data(), static_cast<int>(key_num), start);

    K* device_keys = nullptr;
    V* device_values = nullptr;
    S* device_scores = nullptr;
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                          key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                          key_num * DIM * sizeof(V),
                          ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMemcpy(device_keys, key_num * sizeof(K), host_keys.data(),
                          key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMemcpy(device_values, key_num * DIM * sizeof(V),
                          host_values.data(), key_num * DIM * sizeof(V),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);

    if (with_scores) {
      ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_scores),
                            key_num * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
                ACL_ERROR_NONE);
      ASSERT_EQ(aclrtMemcpy(device_scores, key_num * sizeof(S),
                            host_scores.data(), key_num * sizeof(S),
                            ACL_MEMCPY_HOST_TO_DEVICE),
                ACL_ERROR_NONE);
    }

    table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                           stream);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

    ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
    if (device_scores != nullptr) {
      ASSERT_EQ(aclrtFree(device_scores), ACL_ERROR_NONE);
    }
  }

  template <size_t DIM>
  void VerifyFindResults(
      const std::vector<K>& query_keys, const std::vector<V>& result_values,
      const std::vector<char>& found_bytes,
      const std::vector<V>& expected_values,
      const std::unordered_map<K, size_t>& key_to_insert_idx,
      size_t expected_found_num, size_t expected_not_found_num,
      const std::vector<S>* result_scores = nullptr,
      const std::vector<S>* expected_scores = nullptr) {
    size_t found_num = 0;
    size_t not_found_num = 0;
    for (size_t i = 0; i < query_keys.size(); ++i) {
      auto it = key_to_insert_idx.find(query_keys[i]);
      bool found = static_cast<bool>(found_bytes[i]);
      if (it != key_to_insert_idx.end()) {
        EXPECT_TRUE(found) << "key=" << query_keys[i] << " should be found";
        if (!found) {
          continue;
        }
        ++found_num;
        size_t insert_idx = it->second;
        for (size_t j = 0; j < DIM; ++j) {
          EXPECT_EQ(result_values[i * DIM + j],
                    expected_values[insert_idx * DIM + j])
              << "key=" << query_keys[i] << " dim_idx=" << j;
        }
        if (result_scores != nullptr && expected_scores != nullptr) {
          EXPECT_EQ((*result_scores)[i], (*expected_scores)[insert_idx])
              << "key=" << query_keys[i] << " score mismatch";
        }
      } else {
        EXPECT_FALSE(found)
            << "key=" << query_keys[i] << " should not be found";
        if (!found) {
          ++not_found_num;
        }
      }
    }
    EXPECT_EQ(found_num, expected_found_num);
    EXPECT_EQ(not_found_num, expected_not_found_num);
  }

  template <size_t DIM>
  void RunAllExistNoScores(size_t key_num) {
    HashTable<K, V> table;
    InitTable(table, DIM);

    aclrtStream stream = nullptr;
    ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

    std::vector<K> host_keys;
    std::vector<V> host_values;
    std::vector<S> host_scores;
    InsertContinuousKeys<decltype(table), DIM>(
        table, key_num, host_keys, host_values, host_scores, false, stream);

    std::unordered_map<K, size_t> key_to_idx;
    for (size_t i = 0; i < key_num; ++i) {
      key_to_idx[host_keys[i]] = i;
    }

    K* device_keys = nullptr;
    V* device_values = nullptr;
    bool* device_found = nullptr;
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                          key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                          key_num * DIM * sizeof(V),
                          ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                          key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMemcpy(device_keys, key_num * sizeof(K), host_keys.data(),
                          key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);

    table.find(key_num, device_keys, device_values, device_found, nullptr,
               stream);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

    std::vector<V> result_values(key_num * DIM);
    std::vector<char> found_bytes(key_num);
    ASSERT_EQ(aclrtMemcpy(result_values.data(), key_num * DIM * sizeof(V),
                          device_values, key_num * DIM * sizeof(V),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMemcpy(found_bytes.data(), key_num * sizeof(bool),
                          device_found, key_num * sizeof(bool),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);

    VerifyFindResults<DIM>(host_keys, result_values, found_bytes, host_values,
                           key_to_idx, key_num, 0);

    ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
  }

  template <size_t DIM>
  void RunPartialExistWithScores(size_t insert_num, size_t query_num,
                                 K missing_start) {
    HashTable<K, V, S, EvictStrategy::kCustomized> table;
    InitTable(table, DIM);

    aclrtStream stream = nullptr;
    ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

    std::vector<K> insert_keys;
    std::vector<V> insert_values;
    std::vector<S> insert_scores;
    InsertContinuousKeys<decltype(table), DIM>(
        table, insert_num, insert_keys, insert_values, insert_scores, true,
        stream);

    std::unordered_map<K, size_t> key_to_idx;
    for (size_t i = 0; i < insert_num; ++i) {
      key_to_idx[insert_keys[i]] = i;
    }

    std::vector<K> query_keys(query_num);
    for (size_t i = 0; i < insert_num; ++i) {
      query_keys[i] = insert_keys[i];
    }
    create_continuous_keys<K, S, V, DIM>(
        query_keys.data() + insert_num, nullptr, nullptr,
        static_cast<int>(query_num - insert_num), missing_start);

    K* device_keys = nullptr;
    V* device_values = nullptr;
    S* device_scores = nullptr;
    bool* device_found = nullptr;
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                          query_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                          query_num * DIM * sizeof(V),
                          ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_scores),
                          query_num * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                          query_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMemcpy(device_keys, query_num * sizeof(K),
                          query_keys.data(), query_num * sizeof(K),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);

    table.find(query_num, device_keys, device_values, device_found,
               device_scores, stream);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

    std::vector<V> result_values(query_num * DIM);
    std::vector<S> result_scores(query_num);
    std::vector<char> found_bytes(query_num);
    ASSERT_EQ(aclrtMemcpy(result_values.data(), query_num * DIM * sizeof(V),
                          device_values, query_num * DIM * sizeof(V),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMemcpy(result_scores.data(), query_num * sizeof(S),
                          device_scores, query_num * sizeof(S),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMemcpy(found_bytes.data(), query_num * sizeof(bool),
                          device_found, query_num * sizeof(bool),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);

    VerifyFindResults<DIM>(query_keys, result_values, found_bytes,
                           insert_values, key_to_idx, insert_num,
                           query_num - insert_num, &result_scores,
                           &insert_scores);

    ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtFree(device_scores), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
  }

  template <size_t DIM>
  void RunPartialExistNoScores(size_t insert_num, size_t query_num,
                               K missing_start) {
    HashTable<K, V, S, EvictStrategy::kCustomized> table;
    InitTable(table, DIM);

    aclrtStream stream = nullptr;
    ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

    std::vector<K> insert_keys;
    std::vector<V> insert_values;
    std::vector<S> insert_scores;
    InsertContinuousKeys<decltype(table), DIM>(
        table, insert_num, insert_keys, insert_values, insert_scores, true,
        stream);

    std::unordered_map<K, size_t> key_to_idx;
    for (size_t i = 0; i < insert_num; ++i) {
      key_to_idx[insert_keys[i]] = i;
    }

    std::vector<K> query_keys(query_num);
    for (size_t i = 0; i < insert_num; ++i) {
      query_keys[i] = insert_keys[i];
    }
    create_continuous_keys<K, S, V, DIM>(
        query_keys.data() + insert_num, nullptr, nullptr,
        static_cast<int>(query_num - insert_num), missing_start);

    K* device_keys = nullptr;
    V* device_values = nullptr;
    bool* device_found = nullptr;
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                          query_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                          query_num * DIM * sizeof(V),
                          ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                          query_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMemcpy(device_keys, query_num * sizeof(K),
                          query_keys.data(), query_num * sizeof(K),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);

    table.find(query_num, device_keys, device_values, device_found, nullptr,
               stream);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

    std::vector<V> result_values(query_num * DIM);
    std::vector<char> found_bytes(query_num);
    ASSERT_EQ(aclrtMemcpy(result_values.data(), query_num * DIM * sizeof(V),
                          device_values, query_num * DIM * sizeof(V),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMemcpy(found_bytes.data(), query_num * sizeof(bool),
                          device_found, query_num * sizeof(bool),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);

    VerifyFindResults<DIM>(query_keys, result_values, found_bytes,
                           insert_values, key_to_idx, insert_num,
                           query_num - insert_num);

    ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
  }
};

TEST_F(FindValueTest, BoundaryTest_EmptyQuery) {
  HashTable<K, V> table;
  InitTable(table, kDefaultDim);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  table.find(0, static_cast<K*>(nullptr), static_cast<V*>(nullptr),
             static_cast<bool*>(nullptr), nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

TEST_F(FindValueTest, BasicFunction_SmallScaleAllExist_NoScores) {
  RunAllExistNoScores<kDefaultDim>(1024);
}

TEST_F(FindValueTest, BasicFunction_SmallScaleNoneExist) {
  constexpr size_t key_num = 1024;
  constexpr V magic_value = -123.0F;

  HashTable<K, V> table;
  InitTable(table, kDefaultDim);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  std::vector<K> insert_keys;
  std::vector<V> insert_values;
  std::vector<S> insert_scores;
  InsertContinuousKeys<decltype(table), kDefaultDim>(
      table, key_num, insert_keys, insert_values, insert_scores, false,
      stream);

  std::vector<K> query_keys(key_num);
  create_continuous_keys<K, S, V, kDefaultDim>(
      query_keys.data(), nullptr, nullptr, static_cast<int>(key_num),
      static_cast<K>(100000));

  K* device_keys = nullptr;
  V* device_values = nullptr;
  bool* device_found = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                        key_num * kDefaultDim * sizeof(V),
                        ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                        key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_keys, key_num * sizeof(K), query_keys.data(),
                        key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  std::vector<V> initial_values(key_num * kDefaultDim, magic_value);
  ASSERT_EQ(aclrtMemcpy(device_values, key_num * kDefaultDim * sizeof(V),
                        initial_values.data(),
                        key_num * kDefaultDim * sizeof(V),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  table.find(key_num, device_keys, device_values, device_found, nullptr,
             stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  std::vector<V> result_values(key_num * kDefaultDim);
  std::vector<char> found_bytes(key_num);
  ASSERT_EQ(aclrtMemcpy(result_values.data(),
                        key_num * kDefaultDim * sizeof(V), device_values,
                        key_num * kDefaultDim * sizeof(V),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(found_bytes.data(), key_num * sizeof(bool),
                        device_found, key_num * sizeof(bool),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  for (size_t i = 0; i < key_num; ++i) {
    EXPECT_FALSE(static_cast<bool>(found_bytes[i]));
    for (size_t j = 0; j < kDefaultDim; ++j) {
      EXPECT_EQ(result_values[i * kDefaultDim + j], magic_value);
    }
  }

  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

TEST_F(FindValueTest, MixedScenario_PartialExist_WithScores) {
  RunPartialExistWithScores<kDefaultDim>(1024, 2048, 10000);
}

TEST_F(FindValueTest, MediumScale_PartialExist) {
  RunPartialExistNoScores<kDefaultDim>(5 * 1024, 10 * 1024, 20000);
}

TEST_F(FindValueTest, LargeScale_AllExist) {
  RunAllExistNoScores<kDefaultDim>(64 * 1024);
}

TEST_F(FindValueTest, DifferentDim_128) {
  RunAllExistNoScores<128>(1024);
}

TEST_F(FindValueTest, DifferentDim_4) {
  RunAllExistNoScores<4>(2048);
}

TEST_F(FindValueTest, DuplicateKeys) {
  constexpr size_t unique_key_num = 200;
  constexpr size_t query_num = 1000;

  HashTable<K, V, S, EvictStrategy::kCustomized> table;
  InitTable(table, kDefaultDim);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  std::vector<K> insert_keys;
  std::vector<V> insert_values;
  std::vector<S> insert_scores;
  InsertContinuousKeys<decltype(table), kDefaultDim>(
      table, unique_key_num, insert_keys, insert_values, insert_scores, true,
      stream);

  std::unordered_map<K, size_t> key_to_idx;
  for (size_t i = 0; i < unique_key_num; ++i) {
    key_to_idx[insert_keys[i]] = i;
  }

  std::vector<K> query_keys(query_num);
  for (size_t i = 0; i < query_num; ++i) {
    query_keys[i] = insert_keys[i % unique_key_num];
  }

  K* device_keys = nullptr;
  V* device_values = nullptr;
  S* device_scores = nullptr;
  bool* device_found = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        query_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                        query_num * kDefaultDim * sizeof(V),
                        ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_scores),
                        query_num * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                        query_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_keys, query_num * sizeof(K),
                        query_keys.data(), query_num * sizeof(K),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  table.find(query_num, device_keys, device_values, device_found,
             device_scores, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  std::vector<V> result_values(query_num * kDefaultDim);
  std::vector<S> result_scores(query_num);
  std::vector<char> found_bytes(query_num);
  ASSERT_EQ(aclrtMemcpy(result_values.data(),
                        query_num * kDefaultDim * sizeof(V), device_values,
                        query_num * kDefaultDim * sizeof(V),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(result_scores.data(), query_num * sizeof(S),
                        device_scores, query_num * sizeof(S),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(found_bytes.data(), query_num * sizeof(bool),
                        device_found, query_num * sizeof(bool),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  VerifyFindResults<kDefaultDim>(query_keys, result_values, found_bytes,
                                 insert_values, key_to_idx, query_num, 0,
                                 &result_scores, &insert_scores);

  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_scores), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

TEST_F(FindValueTest, NotFound_InitialValuePreserved) {
  constexpr size_t insert_num = 100;
  constexpr size_t query_num = 100;
  constexpr V magic_value = -999.0F;
  constexpr S magic_score = UINT64_C(0xDEADBEEFDEADBEEF);

  HashTable<K, V, S, EvictStrategy::kCustomized> table;
  InitTable(table, kDefaultDim);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  std::vector<K> insert_keys;
  std::vector<V> insert_values;
  std::vector<S> insert_scores;
  InsertContinuousKeys<decltype(table), kDefaultDim>(
      table, insert_num, insert_keys, insert_values, insert_scores, true,
      stream);

  std::vector<K> query_keys(query_num);
  create_continuous_keys<K, S, V, kDefaultDim>(
      query_keys.data(), nullptr, nullptr, static_cast<int>(query_num),
      static_cast<K>(50000));

  K* device_keys = nullptr;
  V* device_values = nullptr;
  S* device_scores = nullptr;
  bool* device_found = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        query_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                        query_num * kDefaultDim * sizeof(V),
                        ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_scores),
                        query_num * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                        query_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_keys, query_num * sizeof(K), query_keys.data(),
                        query_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  std::vector<V> initial_values(query_num * kDefaultDim, magic_value);
  std::vector<S> initial_scores(query_num, magic_score);
  ASSERT_EQ(aclrtMemcpy(device_values,
                        query_num * kDefaultDim * sizeof(V),
                        initial_values.data(),
                        query_num * kDefaultDim * sizeof(V),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_scores, query_num * sizeof(S),
                        initial_scores.data(), query_num * sizeof(S),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  table.find(query_num, device_keys, device_values, device_found,
             device_scores, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  std::vector<V> result_values(query_num * kDefaultDim);
  std::vector<S> result_scores(query_num);
  std::vector<char> found_bytes(query_num);
  ASSERT_EQ(aclrtMemcpy(result_values.data(),
                        query_num * kDefaultDim * sizeof(V), device_values,
                        query_num * kDefaultDim * sizeof(V),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(result_scores.data(), query_num * sizeof(S),
                        device_scores, query_num * sizeof(S),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(found_bytes.data(), query_num * sizeof(bool),
                        device_found, query_num * sizeof(bool),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  for (size_t i = 0; i < query_num; ++i) {
    EXPECT_FALSE(static_cast<bool>(found_bytes[i]));
    EXPECT_EQ(result_scores[i], magic_score);
    for (size_t j = 0; j < kDefaultDim; ++j) {
      EXPECT_EQ(result_values[i * kDefaultDim + j], magic_value);
    }
  }

  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_scores), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

TEST_F(FindValueTest, ReservedKeys_MixedWithNormalKeys) {
  constexpr size_t normal_key_num = 1024;
  const std::vector<K> reserved_keys = {
      UINT64_C(0xFFFFFFFFFFFFFFFC), UINT64_C(0xFFFFFFFFFFFFFFFD),
      UINT64_C(0xFFFFFFFFFFFFFFFE), UINT64_C(0xFFFFFFFFFFFFFFFF)};
  const size_t query_num = normal_key_num + reserved_keys.size();
  constexpr V magic_value = -999.0F;
  constexpr S magic_score = UINT64_C(0xDEADBEEFDEADBEEF);

  HashTable<K, V, S, EvictStrategy::kCustomized> table;
  InitTable(table, kDefaultDim);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  std::vector<K> insert_keys;
  std::vector<V> insert_values;
  std::vector<S> insert_scores;
  InsertContinuousKeys<decltype(table), kDefaultDim>(
      table, normal_key_num, insert_keys, insert_values, insert_scores, true,
      stream);

  std::vector<K> query_keys(query_num);
  for (size_t i = 0; i < normal_key_num; ++i) {
    query_keys[i] = insert_keys[i];
  }
  for (size_t i = 0; i < reserved_keys.size(); ++i) {
    query_keys[normal_key_num + i] = reserved_keys[i];
  }

  K* device_keys = nullptr;
  V* device_values = nullptr;
  S* device_scores = nullptr;
  bool* device_found = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        query_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                        query_num * kDefaultDim * sizeof(V),
                        ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_scores),
                        query_num * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                        query_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_keys, query_num * sizeof(K), query_keys.data(),
                        query_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  std::vector<V> initial_values(query_num * kDefaultDim, magic_value);
  std::vector<S> initial_scores(query_num, magic_score);
  ASSERT_EQ(aclrtMemcpy(device_values,
                        query_num * kDefaultDim * sizeof(V),
                        initial_values.data(),
                        query_num * kDefaultDim * sizeof(V),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_scores, query_num * sizeof(S),
                        initial_scores.data(), query_num * sizeof(S),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  table.find(query_num, device_keys, device_values, device_found,
             device_scores, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  std::vector<V> result_values(query_num * kDefaultDim);
  std::vector<S> result_scores(query_num);
  std::vector<char> found_bytes(query_num);
  ASSERT_EQ(aclrtMemcpy(result_values.data(),
                        query_num * kDefaultDim * sizeof(V), device_values,
                        query_num * kDefaultDim * sizeof(V),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(result_scores.data(), query_num * sizeof(S),
                        device_scores, query_num * sizeof(S),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(found_bytes.data(), query_num * sizeof(bool),
                        device_found, query_num * sizeof(bool),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  for (size_t i = 0; i < normal_key_num; ++i) {
    EXPECT_TRUE(static_cast<bool>(found_bytes[i]));
    EXPECT_EQ(result_scores[i], insert_scores[i]);
    for (size_t j = 0; j < kDefaultDim; ++j) {
      EXPECT_EQ(result_values[i * kDefaultDim + j],
                insert_values[i * kDefaultDim + j]);
    }
  }
  for (size_t i = normal_key_num; i < query_num; ++i) {
    EXPECT_FALSE(static_cast<bool>(found_bytes[i]));
    EXPECT_EQ(result_scores[i], magic_score);
    for (size_t j = 0; j < kDefaultDim; ++j) {
      EXPECT_EQ(result_values[i * kDefaultDim + j], magic_value);
    }
  }

  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_scores), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

}  // namespace
