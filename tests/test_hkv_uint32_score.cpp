/* *
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <unordered_map>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

using namespace npu::hkv;
using namespace test_util;

constexpr size_t DIM = 8;
constexpr uint64_t CAPACITY = 1024;
constexpr uint64_t KEY_NUM = 256;

using K = uint64_t;
using V = float;
using S = uint32_t;
using TableOptions = HashTableOptions;
using Uint32ScoreTable = HashTable<K, V, S, EvictStrategy::kCustomized>;

namespace {

TableOptions default_options() {
  TableOptions options;
  options.init_capacity = CAPACITY;
  options.max_capacity = CAPACITY;
  options.dim = DIM;
  options.max_bucket_size = 128;
  options.max_hbm_for_vectors = GB(1);
  return options;
}

void fill_sequential(K* h_keys, S* h_scores, V* h_vectors, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    const K key = static_cast<K>(i + 1);
    h_keys[i] = key;
    h_scores[i] = static_cast<S>(key);
    for (size_t j = 0; j < DIM; ++j) {
      h_vectors[i * DIM + j] = static_cast<V>(key * 0.00001f);
    }
  }
}

}  // namespace

TEST(Uint32ScoreTest, FindOrInsertAndFind) {
  init_env();

  aclrtStream stream = nullptr;
  NPU_CHECK(aclrtCreateStream(&stream));

  std::unique_ptr<Uint32ScoreTable> table = std::make_unique<Uint32ScoreTable>();
  table->init(default_options());

  K* h_keys = nullptr;
  S* h_scores = nullptr;
  V* h_vectors = nullptr;
  bool* h_found = nullptr;
  NPU_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&h_keys),
                            KEY_NUM * sizeof(K)));
  NPU_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&h_scores),
                            KEY_NUM * sizeof(S)));
  NPU_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&h_vectors),
                            KEY_NUM * sizeof(V) * DIM));
  NPU_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&h_found),
                            KEY_NUM * sizeof(bool)));

  fill_sequential(h_keys, h_scores, h_vectors, KEY_NUM);

  K* d_keys = nullptr;
  S* d_scores = nullptr;
  V* d_vectors = nullptr;
  bool* d_found = nullptr;
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_keys), KEY_NUM * sizeof(K),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_scores),
                        KEY_NUM * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_vectors),
                        KEY_NUM * sizeof(V) * DIM,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_found),
                        KEY_NUM * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST));

  NPU_CHECK(aclrtMemcpy(d_keys, KEY_NUM * sizeof(K), h_keys,
                        KEY_NUM * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE));
  NPU_CHECK(aclrtMemcpy(d_scores, KEY_NUM * sizeof(S), h_scores,
                        KEY_NUM * sizeof(S), ACL_MEMCPY_HOST_TO_DEVICE));
  NPU_CHECK(aclrtMemcpy(d_vectors, KEY_NUM * sizeof(V) * DIM, h_vectors,
                        KEY_NUM * sizeof(V) * DIM,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  NPU_CHECK(aclrtMemset(d_found, KEY_NUM * sizeof(bool), 0,
                        KEY_NUM * sizeof(bool)));

  table->find_or_insert(KEY_NUM, d_keys, d_vectors, d_scores, stream);
  NPU_CHECK(aclrtSynchronizeStream(stream));

  const size_t size = table->size(stream);
  NPU_CHECK(aclrtSynchronizeStream(stream));
  ASSERT_EQ(size, KEY_NUM);

  NPU_CHECK(aclrtMemset(d_found, KEY_NUM * sizeof(bool), 0,
                        KEY_NUM * sizeof(bool)));
  NPU_CHECK(aclrtMemset(d_vectors, KEY_NUM * sizeof(V) * DIM, 0,
                        KEY_NUM * sizeof(V) * DIM));
  NPU_CHECK(aclrtMemset(d_scores, KEY_NUM * sizeof(S), 0,
                        KEY_NUM * sizeof(S)));

  table->find(KEY_NUM, d_keys, d_vectors, d_found, d_scores, stream);
  NPU_CHECK(aclrtSynchronizeStream(stream));

  NPU_CHECK(aclrtMemcpy(h_found, KEY_NUM * sizeof(bool), d_found,
                        KEY_NUM * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST));
  NPU_CHECK(aclrtMemcpy(h_scores, KEY_NUM * sizeof(S), d_scores,
                        KEY_NUM * sizeof(S), ACL_MEMCPY_DEVICE_TO_HOST));
  NPU_CHECK(aclrtMemcpy(h_vectors, KEY_NUM * sizeof(V) * DIM, d_vectors,
                        KEY_NUM * sizeof(V) * DIM,
                        ACL_MEMCPY_DEVICE_TO_HOST));

  for (size_t i = 0; i < KEY_NUM; ++i) {
    ASSERT_TRUE(h_found[i]);
    ASSERT_EQ(h_scores[i], static_cast<S>(i + 1));
    for (size_t j = 0; j < DIM; ++j) {
      ASSERT_EQ(h_vectors[i * DIM + j],
                static_cast<V>((i + 1) * 0.00001f));
    }
  }

  constexpr size_t MISSING_NUM = 16;
  K* h_missing_keys = nullptr;
  bool* h_missing_found = nullptr;
  NPU_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&h_missing_keys),
                            MISSING_NUM * sizeof(K)));
  NPU_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&h_missing_found),
                            MISSING_NUM * sizeof(bool)));
  for (size_t i = 0; i < MISSING_NUM; ++i) {
    h_missing_keys[i] = static_cast<K>(KEY_NUM + 1000 + i);
  }

  K* d_missing_keys = nullptr;
  V* d_missing_vectors = nullptr;
  bool* d_missing_found = nullptr;
  S* d_missing_scores = nullptr;
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_missing_keys),
                        MISSING_NUM * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_missing_vectors),
                        MISSING_NUM * sizeof(V) * DIM,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_missing_found),
                        MISSING_NUM * sizeof(bool),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_missing_scores),
                        MISSING_NUM * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST));

  NPU_CHECK(aclrtMemcpy(d_missing_keys, MISSING_NUM * sizeof(K),
                        h_missing_keys, MISSING_NUM * sizeof(K),
                        ACL_MEMCPY_HOST_TO_DEVICE));
  NPU_CHECK(aclrtMemset(d_missing_found, MISSING_NUM * sizeof(bool), 0,
                        MISSING_NUM * sizeof(bool)));

  table->find(MISSING_NUM, d_missing_keys, d_missing_vectors, d_missing_found,
              d_missing_scores, stream);
  NPU_CHECK(aclrtSynchronizeStream(stream));

  NPU_CHECK(aclrtMemcpy(h_missing_found, MISSING_NUM * sizeof(bool),
                        d_missing_found, MISSING_NUM * sizeof(bool),
                        ACL_MEMCPY_DEVICE_TO_HOST));

  for (size_t i = 0; i < MISSING_NUM; ++i) {
    ASSERT_FALSE(h_missing_found[i]);
  }

  NPU_CHECK(aclrtDestroyStream(stream));

  NPU_CHECK(aclrtFreeHost(h_keys));
  NPU_CHECK(aclrtFreeHost(h_scores));
  NPU_CHECK(aclrtFreeHost(h_vectors));
  NPU_CHECK(aclrtFreeHost(h_found));
  NPU_CHECK(aclrtFreeHost(h_missing_keys));
  NPU_CHECK(aclrtFreeHost(h_missing_found));

  NPU_CHECK(aclrtFree(d_keys));
  NPU_CHECK(aclrtFree(d_scores));
  NPU_CHECK(aclrtFree(d_vectors));
  NPU_CHECK(aclrtFree(d_found));
  NPU_CHECK(aclrtFree(d_missing_keys));
  NPU_CHECK(aclrtFree(d_missing_vectors));
  NPU_CHECK(aclrtFree(d_missing_found));
  NPU_CHECK(aclrtFree(d_missing_scores));

  NPU_CHECK(aclrtSynchronizeDevice());
  NpuCheckError();
}

TEST(Uint32ScoreTest, AssignScoresAndExport) {
  init_env();

  aclrtStream stream = nullptr;
  NPU_CHECK(aclrtCreateStream(&stream));

  std::unique_ptr<Uint32ScoreTable> table = std::make_unique<Uint32ScoreTable>();
  table->init(default_options());

  K* h_keys = nullptr;
  S* h_scores = nullptr;
  V* h_vectors = nullptr;
  bool* h_found = nullptr;
  NPU_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&h_keys),
                            KEY_NUM * sizeof(K)));
  NPU_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&h_scores),
                            KEY_NUM * sizeof(S)));
  NPU_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&h_vectors),
                            KEY_NUM * sizeof(V) * DIM));
  NPU_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&h_found),
                            KEY_NUM * sizeof(bool)));

  fill_sequential(h_keys, h_scores, h_vectors, KEY_NUM);

  K* d_keys = nullptr;
  S* d_scores = nullptr;
  V* d_vectors = nullptr;
  bool* d_found = nullptr;
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_keys), KEY_NUM * sizeof(K),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_scores),
                        KEY_NUM * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_vectors),
                        KEY_NUM * sizeof(V) * DIM,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_found),
                        KEY_NUM * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST));

  NPU_CHECK(aclrtMemcpy(d_keys, KEY_NUM * sizeof(K), h_keys,
                        KEY_NUM * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE));
  NPU_CHECK(aclrtMemcpy(d_scores, KEY_NUM * sizeof(S), h_scores,
                        KEY_NUM * sizeof(S), ACL_MEMCPY_HOST_TO_DEVICE));
  NPU_CHECK(aclrtMemcpy(d_vectors, KEY_NUM * sizeof(V) * DIM, h_vectors,
                        KEY_NUM * sizeof(V) * DIM,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  table->find_or_insert(KEY_NUM, d_keys, d_vectors, d_scores, stream);
  NPU_CHECK(aclrtSynchronizeStream(stream));

  S expected_scores_host[KEY_NUM];
  for (size_t i = 0; i < KEY_NUM; ++i) {
    h_scores[i] = static_cast<S>(1000 + i);
  }
  h_scores[0] = static_cast<S>(0);
  h_scores[1] = std::numeric_limits<S>::max();
  h_scores[2] = static_cast<S>(1);
  h_scores[3] = std::numeric_limits<S>::max() - 1;
  for (size_t i = 0; i < KEY_NUM; ++i) {
    expected_scores_host[i] = h_scores[i];
  }

  NPU_CHECK(aclrtMemcpy(d_scores, KEY_NUM * sizeof(S), h_scores,
                        KEY_NUM * sizeof(S), ACL_MEMCPY_HOST_TO_DEVICE));

  table->assign_scores(KEY_NUM, d_keys, d_scores, stream);
  NPU_CHECK(aclrtSynchronizeStream(stream));

  NPU_CHECK(aclrtMemset(d_found, KEY_NUM * sizeof(bool), 0,
                        KEY_NUM * sizeof(bool)));
  table->find(KEY_NUM, d_keys, d_vectors, d_found, d_scores, stream);
  NPU_CHECK(aclrtSynchronizeStream(stream));

  NPU_CHECK(aclrtMemcpy(h_found, KEY_NUM * sizeof(bool), d_found,
                        KEY_NUM * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST));
  NPU_CHECK(aclrtMemcpy(h_scores, KEY_NUM * sizeof(S), d_scores,
                        KEY_NUM * sizeof(S), ACL_MEMCPY_DEVICE_TO_HOST));

  for (size_t i = 0; i < KEY_NUM; ++i) {
    ASSERT_TRUE(h_found[i]);
    ASSERT_EQ(h_scores[i], expected_scores_host[i]);
  }

  const size_t capacity = table->capacity();
  K* h_export_keys = nullptr;
  S* h_export_scores = nullptr;
  V* h_export_vectors = nullptr;
  NPU_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&h_export_keys),
                            capacity * sizeof(K)));
  NPU_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&h_export_scores),
                            capacity * sizeof(S)));
  NPU_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&h_export_vectors),
                            capacity * sizeof(V) * DIM));

  K* d_export_keys = nullptr;
  S* d_export_scores = nullptr;
  V* d_export_vectors = nullptr;
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_export_keys),
                        capacity * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_export_scores),
                        capacity * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_export_vectors),
                        capacity * sizeof(V) * DIM,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  const size_t dumped = table->export_batch(
      capacity, 0, d_export_keys, d_export_vectors, d_export_scores, stream);
  ASSERT_EQ(dumped, KEY_NUM);
  NPU_CHECK(aclrtSynchronizeStream(stream));

  NPU_CHECK(aclrtMemcpy(h_export_keys, capacity * sizeof(K), d_export_keys,
                        capacity * sizeof(K), ACL_MEMCPY_DEVICE_TO_HOST));
  NPU_CHECK(aclrtMemcpy(h_export_scores, capacity * sizeof(S), d_export_scores,
                        capacity * sizeof(S), ACL_MEMCPY_DEVICE_TO_HOST));
  NPU_CHECK(aclrtMemcpy(h_export_vectors, capacity * sizeof(V) * DIM,
                        d_export_vectors, capacity * sizeof(V) * DIM,
                        ACL_MEMCPY_DEVICE_TO_HOST));

  std::unordered_map<K, S> expected_scores;
  expected_scores.reserve(KEY_NUM);
  for (size_t i = 0; i < KEY_NUM; ++i) {
    expected_scores.emplace(static_cast<K>(i + 1), expected_scores_host[i]);
  }

  for (size_t i = 0; i < dumped; ++i) {
    const K key = h_export_keys[i];
    auto it = expected_scores.find(key);
    ASSERT_NE(it, expected_scores.end());
    ASSERT_EQ(h_export_scores[i], it->second);
    expected_scores.erase(it);
    for (size_t j = 0; j < DIM; ++j) {
      ASSERT_EQ(h_export_vectors[i * DIM + j],
                static_cast<V>(key * 0.00001f));
    }
  }
  ASSERT_TRUE(expected_scores.empty());

  NPU_CHECK(aclrtDestroyStream(stream));

  NPU_CHECK(aclrtFreeHost(h_keys));
  NPU_CHECK(aclrtFreeHost(h_scores));
  NPU_CHECK(aclrtFreeHost(h_vectors));
  NPU_CHECK(aclrtFreeHost(h_found));
  NPU_CHECK(aclrtFreeHost(h_export_keys));
  NPU_CHECK(aclrtFreeHost(h_export_scores));
  NPU_CHECK(aclrtFreeHost(h_export_vectors));

  NPU_CHECK(aclrtFree(d_keys));
  NPU_CHECK(aclrtFree(d_scores));
  NPU_CHECK(aclrtFree(d_vectors));
  NPU_CHECK(aclrtFree(d_found));
  NPU_CHECK(aclrtFree(d_export_keys));
  NPU_CHECK(aclrtFree(d_export_scores));
  NPU_CHECK(aclrtFree(d_export_vectors));

  NPU_CHECK(aclrtSynchronizeDevice());
  NpuCheckError();
}

TEST(Uint32ScoreTest, EvictCustomizedCorrectRateFull) {
  init_env();

  constexpr uint64_t BATCH_SIZE = 1024 * 1024UL;
  constexpr uint64_t STEPS = 128;
  constexpr uint64_t MAX_BUCKET_SIZE = 128;
  constexpr uint64_t INIT_CAPACITY = BATCH_SIZE * STEPS;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr float EXPECTED_CORRECT_RATE = 0.964f;
  const int rounds = 6;

  TableOptions options;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.reserved_key_start_bit = 17;
  options.num_of_buckets_per_alloc = 128;
  options.max_bucket_size = MAX_BUCKET_SIZE;
  options.max_hbm_for_vectors = GB(16);

  K* h_keys_base = nullptr;
  S* h_scores_base = nullptr;
  V* h_vectors_base = nullptr;
  K* h_keys_temp = nullptr;
  S* h_scores_temp = nullptr;
  V* h_vectors_temp = nullptr;
  NPU_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&h_keys_base),
                            BATCH_SIZE * sizeof(K)));
  NPU_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&h_scores_base),
                            BATCH_SIZE * sizeof(S)));
  NPU_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&h_vectors_base),
                            BATCH_SIZE * sizeof(V) * options.dim));
  NPU_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&h_keys_temp),
                            MAX_CAPACITY * sizeof(K)));
  NPU_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&h_scores_temp),
                            MAX_CAPACITY * sizeof(S)));
  NPU_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&h_vectors_temp),
                            MAX_CAPACITY * sizeof(V) * options.dim));

  K* d_keys_temp = nullptr;
  S* d_scores_temp = nullptr;
  V* d_vectors_temp = nullptr;
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_keys_temp),
                        MAX_CAPACITY * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_scores_temp),
                        MAX_CAPACITY * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_vectors_temp),
                        MAX_CAPACITY * sizeof(V) * options.dim,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  aclrtStream stream = nullptr;
  NPU_CHECK(aclrtCreateStream(&stream));

  std::unique_ptr<Uint32ScoreTable> table = std::make_unique<Uint32ScoreTable>();
  table->init(options);

  size_t total_size = table->size(stream);
  NPU_CHECK(aclrtSynchronizeStream(stream));
  ASSERT_EQ(total_size, 0);

  size_t global_start_key = 100000;
  size_t start_key = global_start_key;

  for (int r = 0; r < rounds; ++r) {
    const K expected_min_key =
        static_cast<K>(global_start_key + INIT_CAPACITY * r);
    const K expected_max_key =
        static_cast<K>(global_start_key + INIT_CAPACITY * (r + 1) - 1);
    const size_t expected_table_size =
        (r == 0) ? static_cast<size_t>(EXPECTED_CORRECT_RATE * INIT_CAPACITY)
                 : INIT_CAPACITY;

    for (int s = 0; s < static_cast<int>(STEPS); ++s) {
      create_continuous_keys<K, S, V, DIM>(
          h_keys_base, h_scores_base, h_vectors_base, BATCH_SIZE, start_key);
      start_key += BATCH_SIZE;

      NPU_CHECK(aclrtMemcpy(d_keys_temp, BATCH_SIZE * sizeof(K), h_keys_base,
                            BATCH_SIZE * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE));
      NPU_CHECK(aclrtMemcpy(d_scores_temp, BATCH_SIZE * sizeof(S),
                            h_scores_base, BATCH_SIZE * sizeof(S),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      NPU_CHECK(aclrtMemcpy(d_vectors_temp,
                            BATCH_SIZE * sizeof(V) * options.dim, h_vectors_base,
                            BATCH_SIZE * sizeof(V) * options.dim,
                            ACL_MEMCPY_HOST_TO_DEVICE));
      table->insert_or_assign(BATCH_SIZE, d_keys_temp, d_vectors_temp,
                              d_scores_temp, stream);
      NPU_CHECK(aclrtSynchronizeStream(stream));
    }

    total_size = table->size(stream);
    NPU_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_GE(total_size, expected_table_size);
    ASSERT_EQ(MAX_CAPACITY, table->capacity());

    const size_t dump_counter = table->export_batch(
        MAX_CAPACITY, 0, d_keys_temp, d_vectors_temp, d_scores_temp, stream);
    NPU_CHECK(aclrtSynchronizeStream(stream));

    NPU_CHECK(aclrtMemcpy(h_keys_temp, MAX_CAPACITY * sizeof(K), d_keys_temp,
                          MAX_CAPACITY * sizeof(K), ACL_MEMCPY_DEVICE_TO_HOST));
    NPU_CHECK(aclrtMemcpy(h_scores_temp, MAX_CAPACITY * sizeof(S),
                          d_scores_temp, MAX_CAPACITY * sizeof(S),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    NPU_CHECK(aclrtMemcpy(h_vectors_temp,
                          MAX_CAPACITY * sizeof(V) * options.dim, d_vectors_temp,
                          MAX_CAPACITY * sizeof(V) * options.dim,
                          ACL_MEMCPY_DEVICE_TO_HOST));

    ASSERT_EQ(total_size, dump_counter);
    size_t bigger_score_counter = 0;
    K max_key = 0;

    for (size_t i = 0; i < dump_counter; ++i) {
      ASSERT_EQ(h_scores_temp[i], static_cast<S>(h_keys_temp[i]));
      max_key = std::max(max_key, h_keys_temp[i]);
      if (h_scores_temp[i] >= static_cast<S>(expected_min_key)) {
        ++bigger_score_counter;
      }
      for (size_t j = 0; j < options.dim; ++j) {
        const V expected = static_cast<V>(h_keys_temp[i] * 0.00001);
        ASSERT_EQ(h_vectors_temp[i * options.dim + j], expected);
      }
    }

    const float correct_rate =
        (bigger_score_counter * 1.0f) / static_cast<float>(MAX_CAPACITY);
    std::cout << "[Round " << r << "] "
              << "correct_rate=" << correct_rate << std::endl;
    ASSERT_GE(max_key, expected_max_key);
    ASSERT_GE(correct_rate, EXPECTED_CORRECT_RATE);
  }

  NPU_CHECK(aclrtDestroyStream(stream));

  NPU_CHECK(aclrtFreeHost(h_keys_base));
  NPU_CHECK(aclrtFreeHost(h_scores_base));
  NPU_CHECK(aclrtFreeHost(h_vectors_base));
  NPU_CHECK(aclrtFreeHost(h_keys_temp));
  NPU_CHECK(aclrtFreeHost(h_scores_temp));
  NPU_CHECK(aclrtFreeHost(h_vectors_temp));

  NPU_CHECK(aclrtFree(d_keys_temp));
  NPU_CHECK(aclrtFree(d_scores_temp));
  NPU_CHECK(aclrtFree(d_vectors_temp));

  NPU_CHECK(aclrtSynchronizeDevice());
  NpuCheckError();
}
