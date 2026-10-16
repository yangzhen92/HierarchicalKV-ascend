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
 * @file dual_bucket_test_util.h
 * @brief Layout and fixture helpers shared by the dual-bucket acceptance suites
 *        (`test_dual_bucket.cpp` for the insert path,
 * `test_dual_bucket_find.cpp` for the lookup path).
 *
 *        The dual-bucket mapping is replicated on the host so that a test can
 *        state which bucket and slot a key is expected to occupy without asking
 *        any device-side code.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_device_data.h"
#include "test_util.h"

namespace dual_bucket_test {

using K = uint64_t;
using V = float;
using S = uint64_t;

constexpr size_t DIM = 8;
constexpr size_t BUCKET_MAX_SIZE = 128;
/// With `bucket_max_size == 128` the device's `max_bucket_shift` is 7.
constexpr uint32_t BUCKET_MAX_SHIFT = 7;
/// 8 buckets; `init_capacity == max_capacity` is a precondition of the
/// dual-bucket mode, so this is also the final capacity.
constexpr size_t CAPACITY = 1024;
constexpr size_t BUCKETS_NUM = CAPACITY / BUCKET_MAX_SIZE;
constexpr size_t PAIR_SLOTS = 2 * BUCKET_MAX_SIZE;

constexpr K RESERVED_KEY_LOW = UINT64_C(0xFFFFFFFFFFFFFFFC);
constexpr K RESERVED_KEY_HIGH = UINT64_C(0xFFFFFFFFFFFFFFFE);

/// `kCustomized` takes the scores verbatim, which makes the eviction order of
/// the tests exactly predictable.
using DualTable =
    npu::hkv::HashTable<K, V, S, npu::hkv::EvictStrategy::kCustomized>;

// --- host-side replication of the device hash/layout ---------------------

/// Matches `get_global_idx` in the kernel, which computes `x % capacity` by
/// fast division; `CAPACITY` is a power of two, so this is exact.
inline uint64_t host_global_idx(uint64_t half) { return half % CAPACITY; }

/// The candidate slot of @p key coming from half @p sel of its hash (0 = low
/// 32 bits, 1 = high 32 bits), before the "two halves collided" adjustment.
inline std::pair<size_t, uint32_t> host_raw_slot(K key, int sel) {
  const uint64_t hashed = test_util::Murmur3HashHost(key);
  const uint64_t half = (sel == 0) ? static_cast<uint32_t>(hashed)
                                   : static_cast<uint32_t>(hashed >> 32);
  const uint64_t global_idx = host_global_idx(half);
  return {static_cast<size_t>(global_idx >> BUCKET_MAX_SHIFT),
          static_cast<uint32_t>(global_idx & (BUCKET_MAX_SIZE - 1))};
}

/// The two candidate slots of a key, in the order the device builds them, with
/// `bkt1` advanced when both halves land on the same bucket. `{bkt, slot}`.
inline std::pair<std::pair<size_t, uint32_t>, std::pair<size_t, uint32_t>>
host_slots(K key) {
  std::pair<size_t, uint32_t> slot0 = host_raw_slot(key, 0);
  std::pair<size_t, uint32_t> slot1 = host_raw_slot(key, 1);
  if (slot1.first == slot0.first) {
    slot1.first = (slot1.first + 1 == BUCKETS_NUM) ? 0 : slot1.first + 1;
  }
  return {slot0, slot1};
}

/// The two independent candidate buckets of a key, in the order the device
/// builds them.
inline std::pair<size_t, size_t> host_buckets(K key) {
  const auto slots = host_slots(key);
  return {slots.first.first, slots.second.first};
}

inline bool is_reserved_key(K key) {
  return (key & RESERVED_KEY_LOW) == RESERVED_KEY_LOW;
}

// --- small helpers --------------------------------------------------------

inline std::unique_ptr<DualTable> make_dual_table(size_t dim = DIM) {
  npu::hkv::HashTableOptions options{
      .init_capacity = CAPACITY,
      .max_capacity = CAPACITY,
      .max_hbm_for_vectors = 4UL << 30,
      .max_bucket_size = BUCKET_MAX_SIZE,
      .dim = dim,
      .io_by_cpu = false,
      .num_of_buckets_per_alloc = 1,
      .enable_dual_bucket = true,
  };
  auto table = std::make_unique<DualTable>();
  table->init(options);
  return table;
}

/// The value payload of a key; `seed` distinguishes two versions of the same
/// key so that an update can be told apart from the original insert.
inline std::vector<V> expected_value(double seed, size_t dim = DIM) {
  std::vector<V> value(dim);
  for (size_t j = 0; j < dim; ++j) {
    value[j] = static_cast<V>(seed + static_cast<double>(j));
  }
  return value;
}

inline void append_kv(std::vector<K>& keys, std::vector<V>& values,
                      std::vector<S>& scores, K key, S score, double value_seed,
                      size_t dim = DIM) {
  keys.push_back(key);
  scores.push_back(score);
  const std::vector<V> value = expected_value(value_seed, dim);
  values.insert(values.end(), value.begin(), value.end());
}

/// Collect `count` distinct non-reserved keys whose candidate buckets are
/// exactly `bkt0` and `bkt1`, in that order, so that they all compete for the
/// same pair of buckets.
inline std::vector<K> keys_for_pair(size_t bkt0, size_t bkt1, size_t count,
                                    K start = 1000) {
  std::vector<K> keys;
  keys.reserve(count);
  for (K candidate = start; keys.size() < count; ++candidate) {
    if (is_reserved_key(candidate)) {
      continue;
    }
    if (host_buckets(candidate) == std::make_pair(bkt0, bkt1)) {
      keys.push_back(candidate);
    }
  }
  return keys;
}

/// Copy a batch into the device buffers and run `insert_or_assign` on it.
inline void insert_batch(DualTable& table, DeviceData<K, V, S>& dev, size_t n,
                         const std::vector<K>& keys,
                         const std::vector<V>& values,
                         const std::vector<S>& scores, size_t dim = DIM) {
  std::vector<K> host_keys(keys);
  std::vector<V> host_values(values);
  std::vector<S> host_scores(scores);
  dev.copy_keys(host_keys, n);
  dev.copy_values(host_values, n, dim);
  dev.copy_scores(host_scores, n);
  table.insert_or_assign(n, dev.device_keys, dev.device_values,
                         dev.device_scores, dev.stream);
  ASSERT_EQ(aclrtSynchronizeStream(dev.stream), ACL_ERROR_NONE);
}

// --- slot dumps -----------------------------------------------------------

/// A compacted dump of a slot range. `export_batch` walks the flat slot space
/// starting at `offset`, where the flat index of a slot is
/// `bucket * bucket_max_size + key_pos`, and drops empty/LOCKED/RECLAIMED
/// slots. Dumping `[bkt * bucket_max_size, bucket_max_size)` therefore dumps
/// exactly one bucket.
struct DumpResult {
  std::vector<K> keys;
  std::vector<V> values;
  std::vector<S> scores;
  size_t counter = 0;
  /// The value width of the table the dump came from. It is not always `DIM`:
  /// the wide-value cases build their table with a wider value so that the move
  /// takes the MTE path.
  size_t dim = DIM;
};

inline DumpResult dump_range(DualTable& table, aclrtStream stream,
                             size_t offset, size_t n) {
  DumpResult result;
  result.dim = table.dim();
  result.keys.assign(n, 0);
  result.values.assign(n * result.dim, 0);
  result.scores.assign(n, 0);

  /// `export_batch` writes from the device, so its outputs have to live in
  /// device memory; the host vectors above only receive the copy-back.
  K* d_keys = nullptr;
  V* d_values = nullptr;
  S* d_scores = nullptr;
  EXPECT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_keys), n * sizeof(K),
                        ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  EXPECT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_values),
                        n * result.dim * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  EXPECT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_scores), n * sizeof(S),
                        ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  result.counter =
      table.export_batch(n, offset, d_keys, d_values, d_scores, stream);
  EXPECT_LE(result.counter, n);

  EXPECT_EQ(aclrtMemcpy(result.keys.data(), result.counter * sizeof(K), d_keys,
                        result.counter * sizeof(K), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  EXPECT_EQ(
      aclrtMemcpy(result.scores.data(), result.counter * sizeof(S), d_scores,
                  result.counter * sizeof(S), ACL_MEMCPY_DEVICE_TO_HOST),
      ACL_ERROR_NONE);
  EXPECT_EQ(
      aclrtMemcpy(result.values.data(), result.counter * result.dim * sizeof(V),
                  d_values, result.counter * result.dim * sizeof(V),
                  ACL_MEMCPY_DEVICE_TO_HOST),
      ACL_ERROR_NONE);

  EXPECT_EQ(aclrtFree(d_keys), ACL_ERROR_NONE);
  EXPECT_EQ(aclrtFree(d_values), ACL_ERROR_NONE);
  EXPECT_EQ(aclrtFree(d_scores), ACL_ERROR_NONE);
  result.keys.resize(result.counter);
  result.scores.resize(result.counter);
  result.values.resize(result.counter * result.dim);
  return result;
}

inline DumpResult dump_all(DualTable& table, aclrtStream stream) {
  return dump_range(table, stream, 0, CAPACITY);
}

/// Dump both buckets of a candidate pair. The two candidates are independent
/// buckets, so they are generally not adjacent in the flat slot space and
/// cannot be covered by a single `dump_range`.
inline DumpResult dump_two_buckets(DualTable& table, aclrtStream stream,
                                   size_t bkt0, size_t bkt1) {
  DumpResult result =
      dump_range(table, stream, bkt0 * BUCKET_MAX_SIZE, BUCKET_MAX_SIZE);
  const DumpResult other =
      dump_range(table, stream, bkt1 * BUCKET_MAX_SIZE, BUCKET_MAX_SIZE);
  result.keys.insert(result.keys.end(), other.keys.begin(), other.keys.end());
  result.scores.insert(result.scores.end(), other.scores.begin(),
                       other.scores.end());
  result.values.insert(result.values.end(), other.values.begin(),
                       other.values.end());
  result.counter += other.counter;
  return result;
}

/// One entry of a dump, keyed by the key it was stored under.
struct DumpEntry {
  S score = 0;
  std::vector<V> value;
};

/// Index a dump by key. Fails the calling test if a key shows up twice.
inline std::unordered_map<K, DumpEntry> build_map(const DumpResult& dump) {
  std::unordered_map<K, DumpEntry> map;
  for (size_t i = 0; i < dump.counter; ++i) {
    DumpEntry entry;
    entry.score = dump.scores[i];
    entry.value.assign(dump.values.begin() + i * dump.dim,
                       dump.values.begin() + (i + 1) * dump.dim);
    const auto inserted = map.emplace(dump.keys[i], std::move(entry));
    EXPECT_TRUE(inserted.second)
        << "key " << dump.keys[i] << " appears twice in the dump";
  }
  return map;
}

/// The set of keys a dump holds.
inline std::unordered_set<K> key_set(const DumpResult& dump) {
  return std::unordered_set<K>(dump.keys.begin(), dump.keys.end());
}

}  // namespace dual_bucket_test
