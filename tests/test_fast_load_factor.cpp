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

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include "acl/acl.h"
#include "async_fast_load_factor.h"
#include "hkv_hashtable.h"
#include "test_device_data.h"
#include "test_util.h"

using namespace npu::hkv;

namespace {

class AsyncFastLoadFactorTest : public testing::Test {
 protected:
  void SetUp() override {
    test_util::init_env();
    ASSERT_EQ(aclrtCreateStream(&stream_), ACL_ERROR_NONE);
  }

  void TearDown() override {
    if (stream_ != nullptr) {
      EXPECT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
    }
    if (buckets_size_ != nullptr) {
      EXPECT_EQ(aclrtFree(buckets_size_), ACL_ERROR_NONE);
    }
    if (stream_ != nullptr) {
      EXPECT_EQ(aclrtDestroyStream(stream_), ACL_ERROR_NONE);
    }
  }

  void set_buckets(const std::vector<int32_t>& buckets) {
    buckets_num_ = buckets.size();
    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&buckets_size_),
                    buckets_num_ * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST),
        ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMemcpy(buckets_size_, buckets_num_ * sizeof(int32_t),
                          buckets.data(), buckets_num_ * sizeof(int32_t),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);
  }

  void copy_buckets(const std::vector<int32_t>& buckets) {
    ASSERT_EQ(buckets.size(), buckets_num_);
    ASSERT_EQ(aclrtMemcpy(buckets_size_, buckets_num_ * sizeof(int32_t),
                          buckets.data(), buckets_num_ * sizeof(int32_t),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);
  }

  aclrtStream stream_ = nullptr;
  int32_t* buckets_size_ = nullptr;
  size_t buckets_num_ = 0;
};

TEST_F(AsyncFastLoadFactorTest, FirstCallReturnsDeltaEstimate) {
  constexpr size_t max_bucket_size = 16;
  constexpr size_t capacity = 64;
  ASSERT_NO_FATAL_FAILURE(set_buckets({2, 4, 6, 8}));

  AsyncFastLoadFactor load_factor;

  EXPECT_FLOAT_EQ(load_factor.calculate(4, buckets_size_, buckets_num_,
                                        max_bucket_size, capacity, stream_),
                  0.125f);
}

TEST_F(AsyncFastLoadFactorTest, CompletedSampleReturnsCachedLoadFactor) {
  constexpr size_t max_bucket_size = 16;
  constexpr size_t capacity = 64;
  ASSERT_NO_FATAL_FAILURE(set_buckets({2, 4, 6, 8}));

  AsyncFastLoadFactor load_factor;

  EXPECT_FLOAT_EQ(load_factor.calculate(0, buckets_size_, buckets_num_,
                                        max_bucket_size, capacity, stream_),
                  0.0f);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  EXPECT_FLOAT_EQ(load_factor.calculate(0, buckets_size_, buckets_num_,
                                        max_bucket_size, capacity, stream_),
                  0.3125f);
}

TEST_F(AsyncFastLoadFactorTest, CompletedSampleScalesWithCurrentCapacity) {
  constexpr size_t max_bucket_size = 16;
  constexpr size_t sampled_capacity = 64;
  constexpr size_t current_capacity = 128;
  ASSERT_NO_FATAL_FAILURE(set_buckets({2, 4, 6, 8}));

  AsyncFastLoadFactor load_factor;

  EXPECT_FLOAT_EQ(
      load_factor.calculate(0, buckets_size_, buckets_num_, max_bucket_size,
                            sampled_capacity, stream_),
      0.0f);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  EXPECT_FLOAT_EQ(
      load_factor.calculate(0, buckets_size_, buckets_num_, max_bucket_size,
                            current_capacity, stream_),
      0.15625f);
}

TEST_F(AsyncFastLoadFactorTest, ZeroSampleOverwritesPreviousCache) {
  constexpr size_t max_bucket_size = 16;
  constexpr size_t capacity = 64;
  ASSERT_NO_FATAL_FAILURE(set_buckets({8, 8, 8, 8}));

  AsyncFastLoadFactor load_factor;

  EXPECT_FLOAT_EQ(load_factor.calculate(0, buckets_size_, buckets_num_,
                                        max_bucket_size, capacity, stream_),
                  0.0f);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  ASSERT_NO_FATAL_FAILURE(copy_buckets({0, 0, 0, 0}));
  EXPECT_FLOAT_EQ(load_factor.calculate(0, buckets_size_, buckets_num_,
                                        max_bucket_size, capacity, stream_),
                  0.5f);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  EXPECT_FLOAT_EQ(load_factor.calculate(0, buckets_size_, buckets_num_,
                                        max_bucket_size, capacity, stream_),
                  0.0f);
}

TEST_F(AsyncFastLoadFactorTest, NullStreamUsesTheDefaultStream) {
  constexpr size_t max_bucket_size = 16;
  constexpr size_t capacity = 64;
  ASSERT_NO_FATAL_FAILURE(set_buckets({8, 8, 8, 8}));

  AsyncFastLoadFactor load_factor;

  EXPECT_FLOAT_EQ(load_factor.calculate(0, buckets_size_, buckets_num_,
                                        max_bucket_size, capacity, nullptr),
                  0.0f);
  ASSERT_EQ(aclrtSynchronizeDevice(), ACL_ERROR_NONE);

  EXPECT_FLOAT_EQ(load_factor.calculate(0, buckets_size_, buckets_num_,
                                        max_bucket_size, capacity, nullptr),
                  0.5f);
}

TEST(AsyncFastLoadFactorIntegrationTest,
     LoadFactorCorrectionStopsAfterThreeExpansions) {
  using Key = uint64_t;
  using Value = float;
  using Score = uint64_t;

  constexpr size_t dim = 8;
  constexpr size_t init_capacity = 4096;
  constexpr size_t max_capacity = 65536;
  constexpr size_t insert_batch_size = 256;
  constexpr size_t insert_batch_num = 8;
  constexpr size_t initial_insert_num = insert_batch_size * insert_batch_num;
  constexpr size_t large_insert_num = 7168;
  constexpr size_t total_key_num = initial_insert_num + large_insert_num;
  constexpr size_t expected_capacity = 32768;
  constexpr size_t expected_expansion_count = 3;

  test_util::init_env();

  DeviceData<Key, Value, Score> device_data;
  ASSERT_NO_FATAL_FAILURE(device_data.malloc(total_key_num, dim));

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = max_capacity,
      .max_hbm_for_vectors = 8UL << 20,
      .dim = dim,
      .max_load_factor = 0.5f,
      .enable_async_fast_load_factor = true,
  };
  HashTable<Key, Value, Score> table;
  table.init(options);

  std::vector<Key> keys(total_key_num);
  std::vector<Value> values(total_key_num * dim);
  test_util::create_continuous_keys<Key, Score, Value>(
      dim, keys.data(), nullptr, values.data(), total_key_num);
  ASSERT_NO_FATAL_FAILURE(device_data.copy_keys(keys, total_key_num));
  ASSERT_NO_FATAL_FAILURE(device_data.copy_values(values, total_key_num, dim));

  for (size_t batch = 0; batch < insert_batch_num; ++batch) {
    const size_t offset = batch * insert_batch_size;
    const size_t expected_size = offset + insert_batch_size;
    table.insert_or_assign(insert_batch_size, device_data.device_keys + offset,
                           device_data.device_values + offset * dim, nullptr,
                           device_data.stream);
    ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

    EXPECT_EQ(table.capacity(), init_capacity);
    EXPECT_EQ(table.size(device_data.stream), expected_size);
    EXPECT_FLOAT_EQ(
        table.load_factor(device_data.stream),
        static_cast<float>(expected_size) / static_cast<float>(init_capacity));
  }

  // Launch a sample of the table at exactly 0.5 load without changing size.
  table.insert_or_assign(1, device_data.device_keys, device_data.device_values,
                         nullptr, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  ASSERT_EQ(table.size(device_data.stream), initial_insert_num);
  ASSERT_FLOAT_EQ(table.load_factor(device_data.stream), 0.5f);

  table.insert_or_assign(large_insert_num,
                         device_data.device_keys + initial_insert_num,
                         device_data.device_values + initial_insert_num * dim,
                         nullptr, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  size_t expansion_count = 0;
  for (size_t capacity = init_capacity; capacity < table.capacity();
       capacity *= 2) {
    ++expansion_count;
  }

  EXPECT_EQ(expansion_count, expected_expansion_count);
  EXPECT_EQ(table.capacity(), expected_capacity);
  EXPECT_NE(table.capacity(), max_capacity);
  EXPECT_EQ(table.size(device_data.stream), total_key_num);
  EXPECT_FLOAT_EQ(table.load_factor(device_data.stream), 0.28125f);
}

}  // namespace
