/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
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

/**
 * @file test_dual_bucket.cpp
 * @brief Acceptance tests for the dual-bucket `insert_or_assign` path.
 *
 *        The dual-bucket layout is inspected through `export_batch`, which
 *        dumps slots straight out of GM, so the tests never depend on a lookup
 *        path agreeing with the layout. `find` and friends are rejected in
 *        dual-bucket mode on purpose (see `DualBucketUnsupportedApis`).
 *
 *        `export_batch(n, offset, ...)` walks the flat slot space starting at
 *        `offset`, where the flat index of a slot is
 *        `bucket * bucket_max_size + key_pos`. Dumping
 *        `[bkt * bucket_max_size, bucket_max_size)` therefore dumps exactly one
 *        bucket, and the compactor drops empty/LOCKED/RECLAIMED slots. The two
 *        candidates of a key are independent buckets, so probing one key needs
 *        two such dumps merged by `dump_two_buckets`.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_device_data.h"
#include "test_util.h"

using npu::hkv::EvictStrategy;
using npu::hkv::HashTable;
using npu::hkv::HashTableOptions;
using npu::hkv::HkvException;

namespace {

using K = uint64_t;
using V = float;
using S = uint64_t;

constexpr size_t DIM = 8;
constexpr size_t BUCKET_MAX_SIZE = 128;
/// 8 buckets; `init_capacity == max_capacity` is a precondition of the
/// dual-bucket mode, so this is also the final capacity.
constexpr size_t CAPACITY = 1024;
constexpr size_t BUCKETS_NUM = CAPACITY / BUCKET_MAX_SIZE;
constexpr size_t PAIR_SLOTS = 2 * BUCKET_MAX_SIZE;

constexpr K RESERVED_KEY_LOW = UINT64_C(0xFFFFFFFFFFFFFFFC);
constexpr K RESERVED_KEY_HIGH = UINT64_C(0xFFFFFFFFFFFFFFFE);

/// `kCustomized` takes the scores verbatim, which makes the eviction order of
/// the tests below exactly predictable.
using DualTable = HashTable<K, V, S, EvictStrategy::kCustomized>;

// --- host-side replication of the device hash/layout ---------------------

/// Matches `get_global_idx` in the kernel, which computes `x % capacity` by
/// fast division; `CAPACITY` is a power of two, so this is exact.
inline uint64_t host_global_idx(uint64_t half) { return half % CAPACITY; }

/// The two independent candidate buckets of a key, in the order the device
/// builds them: `bkt0` from the low 32 bits of the hash, `bkt1` from the high
/// 32 bits, advanced when the two halves collide.
inline std::pair<size_t, size_t> host_buckets(K key) {
  const uint64_t hashed = test_util::Murmur3HashHost(key);
  const size_t bkt0 =
      static_cast<size_t>(host_global_idx(static_cast<uint32_t>(hashed)) >> 7U);
  const size_t bkt1 = static_cast<size_t>(
      host_global_idx(static_cast<uint32_t>(hashed >> 32)) >> 7U);
  if (bkt0 != bkt1) {
    return {bkt0, bkt1};
  }
  return {bkt0, (bkt0 + 1 == BUCKETS_NUM) ? 0 : bkt0 + 1};
}

inline bool is_reserved_key(K key) {
  return (key & RESERVED_KEY_LOW) == RESERVED_KEY_LOW;
}

// --- small helpers --------------------------------------------------------

std::unique_ptr<DualTable> make_dual_table(size_t dim = DIM) {
  HashTableOptions options{
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
std::vector<V> expected_value(double seed) {
  std::vector<V> value(DIM);
  for (size_t j = 0; j < DIM; ++j) {
    value[j] = static_cast<V>(seed + static_cast<double>(j));
  }
  return value;
}

void append_kv(std::vector<K>& keys, std::vector<V>& values,
               std::vector<S>& scores, K key, S score, double value_seed) {
  keys.push_back(key);
  scores.push_back(score);
  const std::vector<V> value = expected_value(value_seed);
  values.insert(values.end(), value.begin(), value.end());
}

struct DumpResult {
  std::vector<K> keys;
  std::vector<V> values;
  std::vector<S> scores;
  size_t counter = 0;
  /// The value width of the table the dump came from. It is not always `DIM`:
  /// `DualBucketLargeValueUsesMteMove` builds its table with a wider value so
  /// that the move takes the MTE path.
  size_t dim = DIM;
};

/// Dump the slots of `[offset, offset + n)` and compact the result on the host.
/// Slots that are empty, LOCKED or reclaimed are dropped by the kernel, so
/// `counter` is the number of live entries in the range.
DumpResult dump_range(DualTable& table, aclrtStream stream, size_t offset,
                      size_t n) {
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

DumpResult dump_all(DualTable& table, aclrtStream stream) {
  return dump_range(table, stream, 0, CAPACITY);
}

/// Dump both buckets of a candidate pair. The two candidates are independent
/// buckets, so they are generally not adjacent in the flat slot space and
/// cannot be covered by a single `dump_range`.
DumpResult dump_two_buckets(DualTable& table, aclrtStream stream, size_t bkt0,
                            size_t bkt1) {
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

struct Entry {
  S score = 0;
  std::vector<V> value;
};

std::unordered_map<K, Entry> build_map(const DumpResult& dump) {
  std::unordered_map<K, Entry> map;
  for (size_t i = 0; i < dump.counter; ++i) {
    Entry entry;
    entry.score = dump.scores[i];
    entry.value.assign(dump.values.begin() + i * dump.dim,
                       dump.values.begin() + (i + 1) * dump.dim);
    const auto inserted = map.emplace(dump.keys[i], std::move(entry));
    EXPECT_TRUE(inserted.second)
        << "key " << dump.keys[i] << " appears twice in the dump";
  }
  return map;
}

void insert_batch(DualTable& table, DeviceData<K, V, S>& dev, size_t n,
                  const std::vector<K>& keys, const std::vector<V>& values,
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

/// Collect `count` distinct non-reserved keys whose candidate buckets are
/// exactly `bkt0` and `bkt1`, in that order, so that they all compete for the
/// same pair of buckets.
std::vector<K> keys_for_pair(size_t bkt0, size_t bkt1, size_t count,
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

std::unordered_set<K> key_set(const DumpResult& dump) {
  return std::unordered_set<K>(dump.keys.begin(), dump.keys.end());
}

/// Fill the candidate buckets `(bkt0, bkt1)` with all 256 slots and finish with
/// distinct scores so the eviction victim is unique. Every insert during the
/// fill carries the same low score, so none of them can be refused and the pair
/// ends up exactly full. Returns the dump of the pair taken after the scores
/// were reassigned.
DumpResult fill_pair(DualTable& table, DeviceData<K, V, S>& dev, size_t bkt0,
                     size_t bkt1, S score_base) {
  const std::vector<K> candidates = keys_for_pair(bkt0, bkt1, PAIR_SLOTS + 4);
  const size_t fill_num = PAIR_SLOTS;

  std::vector<K> keys;
  std::vector<V> values;
  std::vector<S> scores;
  for (size_t i = 0; i < fill_num; ++i) {
    append_kv(keys, values, scores, candidates[i], static_cast<S>(5),
              static_cast<double>(i));
  }
  insert_batch(table, dev, keys.size(), keys, values, scores);
  EXPECT_EQ(table.size(dev.stream), static_cast<size_t>(fill_num));

  // The fill may have evicted some of the keys above, so re-insert exactly the
  // survivors to hand them distinct scores; all of those are duplicates now and
  // therefore update in place.
  DumpResult after_fill = dump_two_buckets(table, dev.stream, bkt0, bkt1);
  EXPECT_EQ(after_fill.counter, fill_num);

  keys.clear();
  values.clear();
  scores.clear();
  for (size_t i = 0; i < after_fill.keys.size(); ++i) {
    append_kv(keys, values, scores, after_fill.keys[i],
              static_cast<S>(score_base + i), static_cast<double>(1000 + i));
  }
  insert_batch(table, dev, keys.size(), keys, values, scores);
  EXPECT_EQ(table.size(dev.stream), static_cast<size_t>(fill_num));

  DumpResult result = dump_two_buckets(table, dev.stream, bkt0, bkt1);
  EXPECT_EQ(result.counter, fill_num);
  return result;
}

// --- tests ----------------------------------------------------------------

TEST(TestDualBucket, DualBucketBasicInsert) {
  test_util::init_env();
  constexpr size_t key_num = 512;

  auto table = make_dual_table();
  EXPECT_EQ(table->capacity(), CAPACITY);
  EXPECT_EQ(table->size(), 0);

  DeviceData<K, V, S> dev;
  dev.malloc(key_num, DIM);

  std::vector<K> keys;
  std::vector<V> values;
  std::vector<S> scores;
  for (size_t i = 0; i < key_num; ++i) {
    append_kv(keys, values, scores, static_cast<K>(i + 1),
              static_cast<S>(i + 1), static_cast<double>(i));
  }
  insert_batch(*table, dev, key_num, keys, values, scores);
  EXPECT_EQ(table->size(dev.stream), static_cast<size_t>(key_num));

  const DumpResult dump = dump_all(*table, dev.stream);
  ASSERT_EQ(dump.counter, key_num);
  const auto map = build_map(dump);
  ASSERT_EQ(map.size(), key_num);
  for (size_t i = 0; i < key_num; ++i) {
    const auto it = map.find(keys[i]);
    ASSERT_NE(it, map.end()) << "key " << keys[i] << " is missing after insert";
    EXPECT_EQ(it->second.score, scores[i]);
    EXPECT_EQ(it->second.value, expected_value(static_cast<double>(i)));
  }

  // Dumping one bucket at a time attributes every entry to the bucket that
  // physically holds it, so this checks that no key escaped its candidates.
  for (size_t bkt = 0; bkt < BUCKETS_NUM; ++bkt) {
    const DumpResult bucket_dump =
        dump_range(*table, dev.stream, bkt * BUCKET_MAX_SIZE, BUCKET_MAX_SIZE);
    for (const K key : bucket_dump.keys) {
      const auto pair = host_buckets(key);
      EXPECT_TRUE(pair.first == bkt || pair.second == bkt)
          << "key " << key << " was stored outside its candidate buckets";
    }
  }
}

TEST(TestDualBucket, DualBucketLargeValueUsesMteMove) {
  test_util::init_env();
  /// 65 floats = 260 bytes, above `SIMT_RW_PKG_SZ * 2`, so the value move goes
  /// through the MTE task queue rather than the SIMT group copy. The dual
  /// kernel has a separate tail for that, and this is the only case that
  /// reaches it.
  constexpr size_t dim = 65;
  constexpr size_t key_num = 256;

  auto table = make_dual_table(dim);
  DeviceData<K, V, S> dev;
  dev.malloc(key_num, dim);

  std::vector<K> keys(key_num, 0);
  std::vector<V> values(key_num * dim, 0);
  std::vector<S> scores(key_num, 0);
  for (size_t i = 0; i < key_num; ++i) {
    keys[i] = static_cast<K>(i + 1);
    scores[i] = static_cast<S>(i + 1);
    for (size_t j = 0; j < dim; ++j) {
      values[i * dim + j] = static_cast<V>(i * 1000 + j);
    }
  }
  insert_batch(*table, dev, key_num, keys, values, scores, dim);
  EXPECT_EQ(table->size(dev.stream), key_num);

  const DumpResult dump = dump_all(*table, dev.stream);
  ASSERT_EQ(dump.counter, key_num);
  ASSERT_EQ(dump.dim, dim);

  std::unordered_map<K, size_t> position;
  for (size_t i = 0; i < dump.counter; ++i) {
    ASSERT_TRUE(position.emplace(dump.keys[i], i).second);
  }
  for (size_t i = 0; i < key_num; ++i) {
    const auto it = position.find(keys[i]);
    ASSERT_NE(it, position.end()) << "key " << keys[i] << " is missing";
    EXPECT_EQ(dump.scores[it->second], scores[i]);
    const std::vector<V> expect(values.begin() + i * dim,
                                values.begin() + (i + 1) * dim);
    const std::vector<V> got(dump.values.begin() + it->second * dim,
                             dump.values.begin() + (it->second + 1) * dim);
    EXPECT_EQ(got, expect);
  }
}

TEST(TestDualBucket, DualBucketUpdateExisting) {
  test_util::init_env();
  constexpr size_t key_num = 512;

  auto table = make_dual_table();
  DeviceData<K, V, S> dev;
  dev.malloc(key_num, DIM);

  std::vector<K> keys;
  std::vector<V> values;
  std::vector<S> scores;
  for (size_t i = 0; i < key_num; ++i) {
    append_kv(keys, values, scores, static_cast<K>(i + 1),
              static_cast<S>(i + 1), static_cast<double>(i));
  }
  insert_batch(*table, dev, key_num, keys, values, scores);
  ASSERT_EQ(table->size(dev.stream), static_cast<size_t>(key_num));

  // Re-insert the very same keys with a different payload and score.
  std::vector<V> new_values;
  std::vector<S> new_scores;
  for (size_t i = 0; i < key_num; ++i) {
    const std::vector<V> value = expected_value(static_cast<double>(1000 + i));
    new_values.insert(new_values.end(), value.begin(), value.end());
    new_scores.push_back(static_cast<S>(100000 + i));
  }
  insert_batch(*table, dev, key_num, keys, new_values, new_scores);

  // An update must not grow the table nor duplicate the key.
  EXPECT_EQ(table->size(dev.stream), static_cast<size_t>(key_num));
  const DumpResult dump = dump_all(*table, dev.stream);
  ASSERT_EQ(dump.counter, key_num);
  const auto map = build_map(dump);
  ASSERT_EQ(map.size(), key_num);
  for (size_t i = 0; i < key_num; ++i) {
    const auto it = map.find(keys[i]);
    ASSERT_NE(it, map.end()) << "key " << keys[i] << " disappeared on update";
    EXPECT_EQ(it->second.score, new_scores[i]);
    EXPECT_EQ(it->second.value, expected_value(static_cast<double>(1000 + i)));
  }
}

TEST(TestDualBucket, DualBucketPreferLessLoaded) {
  test_util::init_env();
  constexpr size_t bkt0 = 3;
  constexpr size_t bkt1 = 6;
  constexpr size_t seed_key_num = 200;

  auto table = make_dual_table();
  DeviceData<K, V, S> dev;
  dev.malloc(seed_key_num + 1, DIM);

  const std::vector<K> candidates = keys_for_pair(bkt0, bkt1, seed_key_num + 1);
  std::vector<K> keys;
  std::vector<V> values;
  std::vector<S> scores;
  for (size_t i = 0; i < seed_key_num; ++i) {
    append_kv(keys, values, scores, candidates[i], static_cast<S>(i + 1),
              static_cast<double>(i));
  }
  insert_batch(*table, dev, seed_key_num, keys, values, scores);
  ASSERT_EQ(table->size(dev.stream), static_cast<size_t>(seed_key_num));

  const DumpResult before0 =
      dump_range(*table, dev.stream, bkt0 * BUCKET_MAX_SIZE, BUCKET_MAX_SIZE);
  const DumpResult before1 =
      dump_range(*table, dev.stream, bkt1 * BUCKET_MAX_SIZE, BUCKET_MAX_SIZE);
  ASSERT_EQ(before0.counter + before1.counter, seed_key_num);

  // The pair holds `seed_key_num` < 256 entries, so the less loaded bucket must
  // still have room; ties go to the first candidate, `bkt0`.
  const size_t expected_sel = (before0.counter <= before1.counter) ? 0U : 1U;
  const size_t expected_room =
      expected_sel == 0 ? before0.counter : before1.counter;
  ASSERT_LT(expected_room, BUCKET_MAX_SIZE);

  const K new_key = candidates[seed_key_num];
  keys.assign(1, new_key);
  values = expected_value(4242.0);
  scores.assign(1, 999999);

  insert_batch(*table, dev, 1, keys, values, scores);
  EXPECT_EQ(table->size(dev.stream), static_cast<size_t>(seed_key_num + 1));

  const DumpResult after0 =
      dump_range(*table, dev.stream, bkt0 * BUCKET_MAX_SIZE, BUCKET_MAX_SIZE);
  const DumpResult after1 =
      dump_range(*table, dev.stream, bkt1 * BUCKET_MAX_SIZE, BUCKET_MAX_SIZE);
  const auto in_bucket0 = key_set(after0);
  const auto in_bucket1 = key_set(after1);

  EXPECT_EQ(in_bucket0.count(new_key) + in_bucket1.count(new_key), 1U);
  EXPECT_EQ(in_bucket0.count(new_key), expected_sel == 0 ? 1U : 0U);
  EXPECT_EQ(in_bucket1.count(new_key), expected_sel == 1 ? 1U : 0U);
  EXPECT_EQ(after0.counter + after1.counter, seed_key_num + 1);
}

TEST(TestDualBucket, DualBucketEvictGlobalMin) {
  test_util::init_env();
  constexpr size_t bkt0 = 0;
  constexpr size_t bkt1 = 5;
  constexpr S score_base = 1000;

  auto table = make_dual_table();
  DeviceData<K, V, S> dev;
  dev.malloc(PAIR_SLOTS + 4, DIM);

  const DumpResult full = fill_pair(*table, dev, bkt0, bkt1, score_base);
  ASSERT_EQ(full.counter, PAIR_SLOTS);

  const auto before = build_map(full);
  ASSERT_EQ(before.size(), PAIR_SLOTS);
  K victim = 0;
  S min_score = std::numeric_limits<S>::max();
  size_t min_count = 0;
  for (const auto& item : before) {
    if (item.second.score < min_score) {
      min_score = item.second.score;
      victim = item.first;
      min_count = 1;
    } else if (item.second.score == min_score) {
      ++min_count;
    }
  }
  ASSERT_EQ(min_count, 1U) << "the scores of a filled pair must be distinct";

  const std::vector<K> candidates = keys_for_pair(bkt0, bkt1, PAIR_SLOTS + 4);
  const K hot_key = candidates[PAIR_SLOTS];
  ASSERT_EQ(before.count(hot_key), 0U);

  std::vector<K> keys(1, hot_key);
  std::vector<V> values = expected_value(777.0);
  std::vector<S> scores(1, static_cast<S>(min_score + 1000));
  insert_batch(*table, dev, 1, keys, values, scores);

  // The pair was full, so the insert must have evicted exactly the entry with
  // the globally minimum score across the two buckets.
  EXPECT_EQ(table->size(dev.stream), static_cast<size_t>(PAIR_SLOTS));
  const DumpResult after = dump_two_buckets(*table, dev.stream, bkt0, bkt1);
  ASSERT_EQ(after.counter, PAIR_SLOTS);

  const auto post = build_map(after);
  ASSERT_EQ(post.size(), PAIR_SLOTS);
  EXPECT_EQ(post.count(victim), 0U)
      << "the minimum score entry was not evicted";
  ASSERT_EQ(post.count(hot_key), 1U) << "the new key was not inserted";
  EXPECT_EQ(post.at(hot_key).score, static_cast<S>(min_score + 1000));
  EXPECT_EQ(post.at(hot_key).value, expected_value(777.0));

  // Nothing else may have changed.
  for (const auto& item : before) {
    if (item.first == victim) {
      continue;
    }
    ASSERT_EQ(post.count(item.first), 1U)
        << "key " << item.first << " was evicted unexpectedly";
    EXPECT_EQ(post.at(item.first).score, item.second.score);
    EXPECT_EQ(post.at(item.first).value, item.second.value);
  }
}

TEST(TestDualBucket, DualBucketRefuseLowScore) {
  test_util::init_env();
  constexpr size_t bkt0 = 2;
  constexpr size_t bkt1 = 7;
  constexpr S score_base = 2000;

  auto table = make_dual_table();
  DeviceData<K, V, S> dev;
  dev.malloc(PAIR_SLOTS + 4, DIM);

  const DumpResult full = fill_pair(*table, dev, bkt0, bkt1, score_base);
  ASSERT_EQ(full.counter, PAIR_SLOTS);
  const auto before = build_map(full);
  ASSERT_EQ(before.size(), PAIR_SLOTS);
  const auto before_keys = key_set(full);

  // A score below every score in the pair must be refused rather than evicting
  // a more valuable entry.
  const std::vector<K> candidates = keys_for_pair(bkt0, bkt1, PAIR_SLOTS + 4);
  const K cold_key = candidates[PAIR_SLOTS];
  ASSERT_EQ(before.count(cold_key), 0U);

  std::vector<K> keys(1, cold_key);
  std::vector<V> values = expected_value(512.0);
  std::vector<S> scores(1, static_cast<S>(score_base - 1));
  insert_batch(*table, dev, 1, keys, values, scores);

  EXPECT_EQ(table->size(dev.stream), static_cast<size_t>(PAIR_SLOTS));
  const DumpResult after = dump_two_buckets(*table, dev.stream, bkt0, bkt1);
  ASSERT_EQ(after.counter, PAIR_SLOTS);

  const auto post = build_map(after);
  EXPECT_EQ(post.count(cold_key), 0U) << "a too cold key was inserted anyway";
  EXPECT_EQ(key_set(after), before_keys) << "a refusal must not evict anything";
  for (const auto& item : before) {
    ASSERT_EQ(post.count(item.first), 1U);
    EXPECT_EQ(post.at(item.first).score, item.second.score);
    EXPECT_EQ(post.at(item.first).value, item.second.value);
  }
}

TEST(TestDualBucket, DualBucketReservedKey) {
  test_util::init_env();

  auto table = make_dual_table();
  DeviceData<K, V, S> dev;
  dev.malloc(16, DIM);

  std::vector<K> keys;
  std::vector<V> values;
  std::vector<S> scores;
  const std::vector<K> batch = {
      1, 2, RESERVED_KEY_LOW, 3, 4, RESERVED_KEY_HIGH, 5, 6, 7};
  for (size_t i = 0; i < batch.size(); ++i) {
    append_kv(keys, values, scores, batch[i], static_cast<S>(i + 1),
              static_cast<double>(i));
  }
  insert_batch(*table, dev, batch.size(), keys, values, scores);

  // Reserved keys must neither be inserted nor counted; the dump filters them
  // out on its own, so comparing both numbers catches a leaked one.
  constexpr size_t expected_num = 7;
  EXPECT_EQ(table->size(dev.stream), static_cast<size_t>(expected_num));
  const DumpResult dump = dump_all(*table, dev.stream);
  ASSERT_EQ(dump.counter, expected_num);

  std::unordered_set<K> expected;
  for (const K key : batch) {
    if (!is_reserved_key(key)) {
      expected.insert(key);
    }
  }
  EXPECT_EQ(key_set(dump), expected);
  for (const K key : dump.keys) {
    EXPECT_FALSE(is_reserved_key(key));
  }
}

TEST(TestDualBucket, DualBucketClearReinsert) {
  test_util::init_env();
  constexpr size_t key_num = 512;

  auto table = make_dual_table();
  DeviceData<K, V, S> dev;
  dev.malloc(key_num, DIM);

  std::vector<K> keys;
  std::vector<V> values;
  std::vector<S> scores;
  for (size_t i = 0; i < key_num; ++i) {
    append_kv(keys, values, scores, static_cast<K>(i + 1),
              static_cast<S>(i + 1), static_cast<double>(i));
  }
  insert_batch(*table, dev, key_num, keys, values, scores);
  ASSERT_EQ(table->size(dev.stream), static_cast<size_t>(key_num));

  table->clear(dev.stream);
  EXPECT_EQ(table->size(dev.stream), 0);
  const DumpResult after_clear = dump_all(*table, dev.stream);
  EXPECT_EQ(after_clear.counter, 0U);

  // The dual-bucket empty digest used by `clear` must be the one the insert
  // path probes for, otherwise the table could never be refilled.
  std::vector<V> new_values;
  std::vector<S> new_scores;
  for (size_t i = 0; i < key_num; ++i) {
    const std::vector<V> value = expected_value(static_cast<double>(5000 + i));
    new_values.insert(new_values.end(), value.begin(), value.end());
    new_scores.push_back(static_cast<S>(5000 + i));
  }
  insert_batch(*table, dev, key_num, keys, new_values, new_scores);
  EXPECT_EQ(table->size(dev.stream), static_cast<size_t>(key_num));

  const DumpResult dump = dump_all(*table, dev.stream);
  ASSERT_EQ(dump.counter, key_num);
  const auto map = build_map(dump);
  for (size_t i = 0; i < key_num; ++i) {
    const auto it = map.find(keys[i]);
    ASSERT_NE(it, map.end()) << "key " << keys[i] << " is missing after clear";
    EXPECT_EQ(it->second.score, new_scores[i]);
    EXPECT_EQ(it->second.value, expected_value(static_cast<double>(5000 + i)));
  }
}

TEST(TestDualBucket, DualBucketUnsupportedApis) {
  test_util::init_env();

  auto table = make_dual_table();
  DeviceData<K, V, S> dev;
  dev.malloc(8, DIM);

  std::vector<K> keys;
  std::vector<V> values;
  std::vector<S> scores;
  append_kv(keys, values, scores, 1, 1, 0.0);
  insert_batch(*table, dev, 1, keys, values, scores);

  // The single-bucket lookup, pre-insert and remove paths do not know about the
  // two-candidate layout, so they must fail loudly instead of returning wrong
  // answers. The guards run before any argument validation, hence the nulls.
  EXPECT_THROW(table->reserve(CAPACITY * 2, dev.stream), HkvException);
  EXPECT_THROW(table->find_or_insert(1, dev.device_keys, dev.device_values,
                                     dev.device_scores, dev.stream),
               HkvException);
  EXPECT_THROW(table->insert_and_evict(1, dev.device_keys, dev.device_values,
                                       dev.device_scores, nullptr, nullptr,
                                       nullptr, nullptr, dev.stream),
               HkvException);
  EXPECT_THROW(table->find(1, dev.device_keys, dev.device_values,
                           dev.device_found, dev.device_scores, dev.stream),
               HkvException);
  EXPECT_THROW(table->erase(1, dev.device_keys, dev.stream), HkvException);

  // The table itself stays usable.
  EXPECT_EQ(table->size(dev.stream), 1);
}

}  // namespace
