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
 * @file test_dual_bucket_find.cpp
 * @brief Acceptance tests for the dual-bucket value-copy `find`.
 *
 *        The fixtures are not built with the dual-bucket `insert_or_assign`: an
 *        insert-side bug and a lookup-side bug would otherwise cancel out.
 * Every slot is written by `dual_bucket_seed_kernel.h` at a (bucket, slot)
 *        position the host computed from the same hash the device uses, so the
 *        test states where a key is expected to live instead of asking the
 *        device where it put it.
 *
 *        A sparse bucket is not free real estate for the seeder: the probe
 *        walks the bucket from `align_to(key_pos)` and stops at the first free
 *        slot, so a key is only reachable from its own candidate slot. Keys are
 *        therefore seeded at the slot the mapping assigns them (`put_natural`),
 *        which is exactly where an insert would have put them. Cases that need
 *        to place a key elsewhere — or that need the whole bucket walked — fill
 *        the bucket with filler keys first (`Seeder::fill_bucket`).
 *
 *        `find` only writes to `values` on a hit, so the miss cases pre-fill
 *        the output buffer and assert the payload survived.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>
#include "acl/acl.h"
#include "dual_bucket_seed_kernel.h"
#include "dual_bucket_test_util.h"
#include "hkv_hashtable.h"
#include "test_device_data.h"
#include "test_util.h"

using npu::hkv::HkvException;
using namespace dual_bucket_test;

namespace {

/// Filler keys live far away from every queried key, so a filler can never be
/// mistaken for a test key. They are also never queried.
constexpr K FILLER_BASE = UINT64_C(0x5A5A0000);
/// Keys below this are used by the tests themselves; the "missing" keys of the
/// miss cases are searched for above it.
constexpr K MISS_BASE = UINT64_C(0x10000000);
/// A scores entry that was never written keeps this value.
constexpr S SCORE_SENTINEL = UINT64_C(0xDEADBEEFCAFEBABE);

// --- slot seeding ---------------------------------------------------------

/// Accumulates explicit slot writes and hands them to the seed kernel in one
/// launch.
class Seeder {
 public:
  Seeder(DualTable& table, aclrtStream stream)
      : table_(table), stream_(stream), dim_(table.dim()) {}

  /// Write @p key into @p bucket at slot @p pos, with the key's own digest —
  /// i.e. the slot content a real insert would have produced.
  void put(size_t bucket, uint32_t pos, K key, S score, double value_seed) {
    put_raw(bucket, pos, key, score, value_seed,
            static_cast<uint32_t>(test_util::Murmur3HashHost(key) >> 56));
  }

  /// Same, but storing @p digest instead of the key's own, for the cases that
  /// need a deliberate digest/key mismatch.
  void put_raw(size_t bucket, uint32_t pos, K key, S score, double value_seed,
               uint32_t digest) {
    npu::hkv::DualBucketSeedRecord record =
        npu::hkv::make_seed_record(bucket, pos, key, score);
    record.digest = digest;
    records_.push_back(record);
    const std::vector<V> payload = expected_value(value_seed, dim_);
    payloads_.insert(payloads_.end(), payload.begin(), payload.end());
  }

  /// Write @p key at its own candidate slot @p sel of the pair (0 = low half of
  /// the hash, 1 = high half), which is where a real insert would put it.
  void put_natural(K key, S score, double value_seed, int sel = 0) {
    const auto slots = host_slots(key);
    const auto slot = (sel == 0) ? slots.first : slots.second;
    put(slot.first, slot.second, key, score, value_seed);
  }

  /// Fill every slot of @p bucket with a distinct filler key except @p skip. A
  /// bucket filled this way has no free slot, so probing it can never terminate
  /// on the free-slot short cut and the scan really does cover all 128 slots.
  void fill_bucket(size_t bucket, K first_filler,
                   uint32_t skip = std::numeric_limits<uint32_t>::max()) {
    for (uint32_t pos = 0; pos < BUCKET_MAX_SIZE; ++pos) {
      if (pos == skip) {
        continue;
      }
      put(bucket, pos, first_filler + pos, static_cast<S>(pos),
          static_cast<double>(pos));
    }
  }

  void flush() {
    npu::hkv::seed_dual_bucket_slots(table_.buckets(), BUCKET_MAX_SIZE, dim_,
                                     records_, payloads_, stream_);
    records_.clear();
    payloads_.clear();
  }

 private:
  DualTable& table_;
  aclrtStream stream_;
  size_t dim_;
  std::vector<npu::hkv::DualBucketSeedRecord> records_;
  std::vector<V> payloads_;
};

// --- running a find -------------------------------------------------------

struct FindOutcome {
  /// One entry per queried key; `char` because a `bool` vector cannot be
  /// memcpy'd into.
  std::vector<char> found;
  std::vector<V> values;
  /// The whole scores buffer read back, written or not.
  std::vector<S> score_buffer;
};

std::vector<V> value_at(const std::vector<V>& values, size_t index,
                        size_t dim) {
  return std::vector<V>(values.begin() + index * dim,
                        values.begin() + (index + 1) * dim);
}

/// Run the value-copy `find` over @p keys and copy every output back.
/// @param prefill optional `keys.size() * dim` payload to load into the output
///        buffer first, for the cases that assert a miss leaves it alone.
FindOutcome run_find(DualTable& table, DeviceData<K, V, S>& dev,
                     const std::vector<K>& keys, bool want_scores,
                     const std::vector<V>* prefill = nullptr) {
  const size_t n = keys.size();
  const size_t dim = table.dim();
  FindOutcome out;
  out.found.assign(n, 0);
  out.values.assign(n * dim, 0);
  out.score_buffer.assign(n, 0);

  std::vector<K> host_keys(keys);
  std::vector<V> host_values =
      (prefill != nullptr) ? *prefill : std::vector<V>(n * dim, 0);
  std::vector<S> host_scores(n, SCORE_SENTINEL);
  dev.copy_keys(host_keys, n);
  dev.copy_values(host_values, n, dim);
  dev.copy_scores(host_scores, n);

  table.find(n, dev.device_keys, dev.device_values, dev.device_found,
             want_scores ? dev.device_scores : nullptr, dev.stream);
  EXPECT_EQ(aclrtSynchronizeStream(dev.stream), ACL_ERROR_NONE);

  EXPECT_EQ(aclrtMemcpy(out.found.data(), n * sizeof(char), dev.device_found,
                        n * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  EXPECT_EQ(
      aclrtMemcpy(out.values.data(), n * dim * sizeof(V), dev.device_values,
                  n * dim * sizeof(V), ACL_MEMCPY_DEVICE_TO_HOST),
      ACL_ERROR_NONE);
  EXPECT_EQ(
      aclrtMemcpy(out.score_buffer.data(), n * sizeof(S), dev.device_scores,
                  n * sizeof(S), ACL_MEMCPY_DEVICE_TO_HOST),
      ACL_ERROR_NONE);
  return out;
}

void expect_miss(const FindOutcome& out, const std::vector<V>& prefill,
                 size_t index, size_t dim, const char* what) {
  EXPECT_FALSE(out.found[index]) << what << " was found by mistake";
  EXPECT_EQ(value_at(out.values, index, dim), value_at(prefill, index, dim))
      << what << " changed the value buffer on a miss";
}

// --- host-side key searches -----------------------------------------------

using Slot = std::pair<size_t, uint32_t>;

/// @p count non-reserved keys whose candidate slot @p sel (0 = low half of the
/// hash, 1 = high half) is distinct from every other returned key's.
///
/// Two keys on one slot would overwrite each other, and the probe's invariant —
/// every slot from `key_pos` up to the key itself is occupied — would break as
/// well, since a key is only reachable from its own candidate slot.
std::vector<K> keys_with_distinct_slots(size_t count, int sel, K start = 1) {
  std::vector<K> keys;
  std::set<Slot> used;
  for (K key = start; keys.size() < count; ++key) {
    if (is_reserved_key(key)) {
      continue;
    }
    const auto slots = host_slots(key);
    const auto slot = (sel == 0) ? slots.first : slots.second;
    if (!used.insert(slot).second) {
      continue;
    }
    keys.push_back(key);
  }
  return keys;
}

bool same_digest(K a, K b) {
  return (test_util::Murmur3HashHost(a) >> 56) ==
         (test_util::Murmur3HashHost(b) >> 56);
}

/// The first non-reserved key whose candidate slot @p sel (0 = low half of the
/// hash, 1 = high half) falls into @p bucket at a position of at least
/// @p min_pos and not at @p avoid_pos.
K key_in_bucket(size_t bucket, int sel, uint32_t min_pos,
                uint32_t avoid_pos = std::numeric_limits<uint32_t>::max()) {
  for (K key = 1000; key < 1000000; ++key) {
    if (is_reserved_key(key)) {
      continue;
    }
    const auto slots = host_slots(key);
    const auto slot = (sel == 0) ? slots.first : slots.second;
    if (slot.first == bucket && slot.second >= min_pos &&
        slot.second != avoid_pos) {
      return key;
    }
  }
  return 0;
}

/// The first non-reserved key at or above @p start whose digest byte is
/// @p model's. When @p model is a sentinel, such a key is indistinguishable
/// from a free slot to the digest filter.
K key_with_digest_of(K model, K start = 1000) {
  const uint64_t wanted = test_util::Murmur3HashHost(model) >> 56;
  for (K key = start; key < start + 1000000; ++key) {
    if (is_reserved_key(key)) {
      continue;
    }
    if ((test_util::Murmur3HashHost(key) >> 56) == wanted) {
      return key;
    }
  }
  return 0;
}

/// @p count distinct non-reserved keys that share a digest byte *and* a
/// candidate bucket pair, so that a probe of either of their buckets sees the
/// other's digest.
std::vector<K> colliding_keys(size_t count) {
  std::unordered_map<uint64_t, std::vector<K>> groups;
  for (K key = 1; key < 2000000; ++key) {
    if (is_reserved_key(key)) {
      continue;
    }
    const auto pair = host_buckets(key);
    const uint64_t digest = test_util::Murmur3HashHost(key) >> 56;
    const uint64_t group_id =
        ((pair.first * BUCKETS_NUM + pair.second) << 8) | digest;
    std::vector<K>& group = groups[group_id];
    group.push_back(key);
    if (group.size() == count) {
      return group;
    }
  }
  return {};
}

/// A table with the same shape as `make_dual_table` but on the single-bucket
/// layout, for the regression case.
std::unique_ptr<DualTable> make_single_bucket_table(size_t dim = DIM) {
  npu::hkv::HashTableOptions options{
      .init_capacity = CAPACITY,
      .max_capacity = CAPACITY,
      .max_hbm_for_vectors = 4UL << 30,
      .max_bucket_size = BUCKET_MAX_SIZE,
      .dim = dim,
      .io_by_cpu = false,
      .num_of_buckets_per_alloc = 1,
      .enable_dual_bucket = false,
  };
  auto table = std::make_unique<DualTable>();
  table->init(options);
  return table;
}

// --- tests ----------------------------------------------------------------

TEST(TestDualBucketFind, FindInSecondBucket) {
  test_util::init_env();
  constexpr size_t key_num = 48;
  /// The first half is stored in the *second* candidate only, which is the
  /// fallback the dual-bucket probe has to take; the second half sits in the
  /// first candidate, so both orders are covered.
  constexpr size_t second_only_num = key_num / 2;

  auto table = make_dual_table();
  DeviceData<K, V, S> dev;
  dev.malloc(key_num, DIM);

  std::vector<K> keys;
  std::set<Slot> used;
  Seeder seeder(*table, dev.stream);
  for (K candidate = 5000; keys.size() < key_num; ++candidate) {
    if (is_reserved_key(candidate)) {
      continue;
    }
    const int sel = (keys.size() < second_only_num) ? 1 : 0;
    const auto slots = host_slots(candidate);
    ASSERT_NE(slots.first.first, slots.second.first)
        << "the two candidates of a key must be distinct buckets";
    const auto slot = (sel == 0) ? slots.first : slots.second;
    if (!used.insert(slot).second) {
      continue;
    }
    keys.push_back(candidate);
    seeder.put_natural(candidate, static_cast<S>(keys.size()),
                       static_cast<double>(keys.size() - 1), sel);
  }
  seeder.flush();

  const FindOutcome out = run_find(*table, dev, keys, true);
  for (size_t i = 0; i < key_num; ++i) {
    ASSERT_TRUE(out.found[i]) << "key " << keys[i] << " was not found";
    EXPECT_EQ(out.score_buffer[i], static_cast<S>(i + 1));
    EXPECT_EQ(value_at(out.values, i, DIM),
              expected_value(static_cast<double>(i)));
  }
}

TEST(TestDualBucketFind, FindMissKeepsValue) {
  test_util::init_env();
  constexpr size_t stored_num = 32;

  auto table = make_dual_table();
  DeviceData<K, V, S> dev;
  dev.malloc(2 * stored_num, DIM);

  std::vector<double> stored_seeds;
  Seeder seeder(*table, dev.stream);
  const std::vector<K> stored_keys =
      keys_with_distinct_slots(stored_num, 0, 1000);
  ASSERT_EQ(stored_keys.size(), stored_num);
  for (size_t i = 0; i < stored_num; ++i) {
    stored_seeds.push_back(static_cast<double>(i));
    seeder.put_natural(stored_keys[i], static_cast<S>(i + 1),
                       stored_seeds.back());
  }
  seeder.flush();

  /// Alternate a stored key with one that was never seeded.
  std::vector<K> query;
  std::vector<char> is_stored;
  for (size_t i = 0; i < stored_num; ++i) {
    query.push_back(stored_keys[i]);
    is_stored.push_back(1);
    query.push_back(MISS_BASE + i);
    is_stored.push_back(0);
  }
  const size_t batch = query.size();
  std::vector<V> prefill(batch * DIM);
  for (size_t i = 0; i < batch; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      prefill[i * DIM + j] = static_cast<V>(-1000.0 - static_cast<double>(i));
    }
  }

  const FindOutcome out = run_find(*table, dev, query, true, &prefill);
  size_t hit_num = 0;
  for (size_t i = 0; i < batch; ++i) {
    if (is_stored[i] == 0) {
      expect_miss(out, prefill, i, DIM, "a key that was never stored");
      continue;
    }
    ++hit_num;
    const size_t stored_index = i / 2;
    ASSERT_TRUE(out.found[i]) << "key " << query[i] << " was not found";
    EXPECT_EQ(out.score_buffer[i], static_cast<S>(stored_index + 1));
    EXPECT_EQ(value_at(out.values, i, DIM),
              expected_value(stored_seeds[stored_index]));
  }
  EXPECT_EQ(hit_num, stored_num);
}

TEST(TestDualBucketFind, FindScoresNull) {
  test_util::init_env();
  constexpr size_t key_num = 32;

  auto table = make_dual_table();
  DeviceData<K, V, S> dev;
  dev.malloc(key_num, DIM);

  Seeder seeder(*table, dev.stream);
  const std::vector<K> keys = keys_with_distinct_slots(key_num, 0, 2000);
  ASSERT_EQ(keys.size(), key_num);
  for (size_t i = 0; i < key_num; ++i) {
    seeder.put_natural(keys[i], static_cast<S>(i + 7),
                       static_cast<double>(i) / 2.0);
  }
  seeder.flush();

  const FindOutcome out = run_find(*table, dev, keys, false);
  for (size_t i = 0; i < key_num; ++i) {
    ASSERT_TRUE(out.found[i]) << "key " << keys[i] << " was not found";
    EXPECT_EQ(value_at(out.values, i, DIM),
              expected_value(static_cast<double>(i) / 2.0));
    /// With `scores == nullptr` the kernel must leave the buffer alone.
    EXPECT_EQ(out.score_buffer[i], SCORE_SENTINEL);
  }
}

TEST(TestDualBucketFind, FindDigestCollision) {
  test_util::init_env();

  /// Three keys that are indistinguishable to the digest filter and live in the
  /// same pair of buckets.
  const std::vector<K> group = colliding_keys(3);
  ASSERT_EQ(group.size(), 3U) << "no digest-colliding triple was found";
  const K first = group[0];
  const K second = group[1];
  const K absent = group[2];
  ASSERT_NE(first, second);
  ASSERT_TRUE(same_digest(first, second));
  ASSERT_TRUE(same_digest(first, absent));
  ASSERT_EQ(host_buckets(first), host_buckets(second));
  ASSERT_EQ(host_buckets(first), host_buckets(absent));

  const size_t bkt0 = host_buckets(first).first;
  const size_t bkt1 = host_buckets(first).second;

  auto table = make_dual_table();
  DeviceData<K, V, S> dev;
  dev.malloc(3, DIM);

  Seeder seeder(*table, dev.stream);
  /// Both candidates are filled to the brim, so neither probe can stop at a
  /// free slot and the colliding digest is really compared against a full
  /// bucket before the matching key word is reached.
  seeder.fill_bucket(bkt0, FILLER_BASE, host_slots(first).first.second);
  seeder.fill_bucket(bkt1, FILLER_BASE + 1000,
                     host_slots(second).second.second);
  seeder.put_natural(first, 11, 1.0);
  seeder.put_natural(second, 22, 2.0, 1);
  seeder.flush();

  const std::vector<K> query = {first, second, absent};
  const FindOutcome out = run_find(*table, dev, query, true);
  ASSERT_TRUE(out.found[0]) << "the stored key was not found";
  ASSERT_TRUE(out.found[1]) << "the second bucket's key was not found";
  EXPECT_FALSE(out.found[2])
      << "a digest-equal key must not hit without its own key word";
  EXPECT_EQ(out.score_buffer[0], 11);
  EXPECT_EQ(out.score_buffer[1], 22);
  EXPECT_EQ(value_at(out.values, 0, DIM), expected_value(1.0));
  EXPECT_EQ(value_at(out.values, 1, DIM), expected_value(2.0));
}

TEST(TestDualBucketFind, FindEmptyAndSentinelKeys) {
  test_util::init_env();
  constexpr size_t sentinel_bucket = 3;

  /// `live_key` keeps its own candidate slot in the sentinel bucket, far enough
  /// from slot 0 that a scan of it cannot reach the sentinels first.
  const K live_key = key_in_bucket(sentinel_bucket, 0, 8);
  ASSERT_NE(live_key, 0U);
  /// `other_key` is stored in the second candidate of the same bucket, on a
  /// slot of its own.
  const K other_key =
      key_in_bucket(sentinel_bucket, 1, 8, host_slots(live_key).first.second);
  ASSERT_NE(other_key, 0U);
  /// A key the digest filter cannot tell from a free slot. Searched well above
  /// the range the two stored keys come from, and checked against them, so it
  /// can never be one of the keys the test actually seeded.
  const K empty_digest_key =
      key_with_digest_of(npu::hkv::EMPTY_KEY, UINT64_C(0x70000000));
  ASSERT_NE(empty_digest_key, 0U);
  ASSERT_FALSE(is_reserved_key(empty_digest_key));
  ASSERT_NE(empty_digest_key, live_key);
  ASSERT_NE(empty_digest_key, other_key);

  auto table = make_dual_table();
  DeviceData<K, V, S> dev;
  dev.malloc(8, DIM);

  Seeder seeder(*table, dev.stream);
  /// Slots 0..2 are the three sentinel words, slot 3 a reserved key. The empty
  /// slot at 0 carries the free-slot digest the probe short-circuits on.
  seeder.put(sentinel_bucket, 0, npu::hkv::EMPTY_KEY, 7, 0.0);
  seeder.put(sentinel_bucket, 1, npu::hkv::RECLAIM_KEY, 8, 0.0);
  seeder.put(sentinel_bucket, 2, npu::hkv::LOCKED_KEY, 9, 0.0);
  seeder.put(sentinel_bucket, 3, RESERVED_KEY_LOW, 10, 0.0);
  seeder.put_natural(live_key, 42, 3.0);
  seeder.put_natural(other_key, 43, 4.0, 1);
  seeder.flush();

  const std::vector<K> query = {npu::hkv::EMPTY_KEY,
                                npu::hkv::RECLAIM_KEY,
                                npu::hkv::LOCKED_KEY,
                                RESERVED_KEY_LOW,
                                live_key,
                                other_key,
                                empty_digest_key};
  std::vector<V> prefill(query.size() * DIM, static_cast<V>(5.0));
  const FindOutcome out = run_find(*table, dev, query, true, &prefill);

  expect_miss(out, prefill, 0, DIM, "EMPTY_KEY");
  expect_miss(out, prefill, 1, DIM, "RECLAIM_KEY");
  expect_miss(out, prefill, 2, DIM, "LOCKED_KEY");
  expect_miss(out, prefill, 3, DIM, "a reserved key");
  expect_miss(out, prefill, 6, DIM, "a key whose digest matches a free slot");

  ASSERT_TRUE(out.found[4]) << "a key stored next to the sentinels was lost";
  ASSERT_TRUE(out.found[5]) << "a key in the second bucket was lost";
  EXPECT_EQ(out.score_buffer[4], 42);
  EXPECT_EQ(out.score_buffer[5], 43);
  EXPECT_EQ(value_at(out.values, 4, DIM), expected_value(3.0));
  EXPECT_EQ(value_at(out.values, 5, DIM), expected_value(4.0));
}

TEST(TestDualBucketFind, FindFullBucketAllSlots) {
  test_util::init_env();
  constexpr size_t bkt0 = 1;
  constexpr size_t bkt1 = 6;

  const std::vector<K> candidates = keys_for_pair(bkt0, bkt1, 4);
  const K in_first = candidates[0];
  const K in_second = candidates[1];
  const K absent = candidates[2];
  ASSERT_EQ(host_buckets(in_first), std::make_pair(bkt0, bkt1));
  ASSERT_EQ(host_buckets(in_second), std::make_pair(bkt0, bkt1));
  ASSERT_EQ(host_buckets(absent), std::make_pair(bkt0, bkt1));

  auto table = make_dual_table();
  DeviceData<K, V, S> dev;
  dev.malloc(3, DIM);

  Seeder seeder(*table, dev.stream);
  /// Both buckets end up completely full, so the probe of the first one walks
  /// all 128 slots without a free slot to stop on before falling through to
  /// the second.
  seeder.fill_bucket(bkt0, FILLER_BASE, host_slots(in_first).first.second);
  seeder.fill_bucket(bkt1, FILLER_BASE + 1000,
                     host_slots(in_second).second.second);
  seeder.put_natural(in_first, 5, 5.0);
  seeder.put_natural(in_second, 6, 6.0, 1);
  seeder.flush();

  const DumpResult dump = dump_two_buckets(*table, dev.stream, bkt0, bkt1);
  ASSERT_EQ(dump.counter, PAIR_SLOTS) << "the pair is not actually full";

  const std::vector<K> query = {in_first, in_second, absent};
  std::vector<V> prefill(query.size() * DIM, static_cast<V>(1.5));
  const FindOutcome out = run_find(*table, dev, query, true, &prefill);

  ASSERT_TRUE(out.found[0]) << "a key in a full first bucket was not found";
  EXPECT_EQ(out.score_buffer[0], 5);
  EXPECT_EQ(value_at(out.values, 0, DIM), expected_value(5.0));
  ASSERT_TRUE(out.found[1]) << "a key in a full second bucket was not found";
  EXPECT_EQ(out.score_buffer[1], 6);
  EXPECT_EQ(value_at(out.values, 1, DIM), expected_value(6.0));
  expect_miss(out, prefill, 2, DIM,
              "a key of a full pair that was never stored");
}

TEST(TestDualBucketFind, FindAfterInsert) {
  test_util::init_env();
  constexpr size_t bkt0 = 4;
  constexpr size_t bkt1 = 2;
  constexpr size_t seed_num = 16;
  /// Kept well below the pair's 256 slots, so the insert can neither run out of
  /// room nor evict one of the seeded entries, whatever bucket it picks.
  constexpr size_t insert_num = 80;

  const std::vector<K> candidates =
      keys_for_pair(bkt0, bkt1, 2 * (seed_num + insert_num));
  /// Split the candidates into the ones seeded directly and the ones the insert
  /// path publishes. A candidate whose candidate slot is already taken by an
  /// earlier seed goes to the insert batch instead, so the seeded keys always
  /// occupy slots of their own.
  std::vector<K> seed_keys;
  std::vector<K> insert_keys;
  std::set<Slot> used;
  for (const K key : candidates) {
    if (seed_keys.size() < seed_num &&
        used.insert(host_slots(key).first).second) {
      seed_keys.push_back(key);
    } else if (insert_keys.size() < insert_num) {
      insert_keys.push_back(key);
    }
  }
  ASSERT_EQ(seed_keys.size(), seed_num);
  ASSERT_EQ(insert_keys.size(), insert_num);

  auto table = make_dual_table();
  DeviceData<K, V, S> dev;
  dev.malloc(seed_num + insert_num, DIM);

  std::unordered_map<K, double> value_seed;
  std::unordered_map<K, S> score;
  Seeder seeder(*table, dev.stream);
  for (size_t i = 0; i < seed_num; ++i) {
    value_seed[seed_keys[i]] = static_cast<double>(i);
    score[seed_keys[i]] = static_cast<S>(10 + i);
    seeder.put_natural(seed_keys[i], score[seed_keys[i]],
                       value_seed[seed_keys[i]]);
  }
  seeder.flush();

  std::vector<K> ins_keys;
  std::vector<V> ins_values;
  std::vector<S> ins_scores;
  for (size_t i = 0; i < insert_num; ++i) {
    value_seed[insert_keys[i]] = static_cast<double>(1000 + i);
    score[insert_keys[i]] = static_cast<S>(1000 + i);
    append_kv(ins_keys, ins_values, ins_scores, insert_keys[i],
              score[insert_keys[i]], value_seed[insert_keys[i]]);
  }
  insert_batch(*table, dev, insert_num, ins_keys, ins_values, ins_scores);
  /// The seeded slots are invisible to `size()`, which only counts the entries
  /// the insert path published; this is the trade-off the test-only seeder
  /// makes, and it is why the dump below is the source of truth.
  EXPECT_EQ(table->size(dev.stream), insert_num);

  /// The pair holds 96 of its 256 slots, so the insert neither ran out of room
  /// nor evicted a seeded entry.
  const DumpResult dump = dump_two_buckets(*table, dev.stream, bkt0, bkt1);
  ASSERT_EQ(dump.counter, seed_num + insert_num);
  const auto dumped = build_map(dump);

  std::vector<K> all(seed_keys);
  all.insert(all.end(), insert_keys.begin(), insert_keys.end());
  const FindOutcome out = run_find(*table, dev, all, true);
  for (size_t i = 0; i < all.size(); ++i) {
    const K key = all[i];
    ASSERT_TRUE(out.found[i]) << "key " << key << " was not found";
    const auto entry = dumped.find(key);
    ASSERT_NE(entry, dumped.end())
        << "key " << key << " is missing from the dump";
    EXPECT_EQ(out.score_buffer[i], entry->second.score);
    EXPECT_EQ(out.score_buffer[i], score[key]);
    EXPECT_EQ(value_at(out.values, i, DIM), entry->second.value);
    EXPECT_EQ(value_at(out.values, i, DIM), expected_value(value_seed[key]));
  }
}

TEST(TestDualBucketFind, FindLargeValueUsesMte) {
  test_util::init_env();
  /// 65 floats = 260 bytes, above `SIMT_RW_PKG_SZ * 2`, so the value move goes
  /// through the MTE task queue rather than the SIMT group copy. This is the
  /// only case that reaches that tail of the dual-bucket kernel.
  constexpr size_t dim = 65;
  constexpr size_t key_num = 20;

  auto table = make_dual_table(dim);
  ASSERT_EQ(table->dim(), dim);
  DeviceData<K, V, S> dev;
  dev.malloc(key_num + 1, dim);

  std::vector<K> keys;
  std::set<Slot> used;
  Seeder seeder(*table, dev.stream);
  /// Alternate between the two candidates of the pair, so the MTE tail is
  /// exercised on both probe orders.
  for (size_t i = 0; keys.size() < key_num; ++i) {
    const K key = static_cast<K>(7000 + i);
    const int sel = static_cast<int>(keys.size() % 2);
    const auto slots = host_slots(key);
    const auto slot = (sel == 0) ? slots.first : slots.second;
    if (!used.insert(slot).second) {
      continue;
    }
    keys.push_back(key);
    seeder.put_natural(key, static_cast<S>(keys.size()),
                       static_cast<double>(keys.size() - 1) * 3.0, sel);
  }
  seeder.flush();

  keys.push_back(MISS_BASE);
  std::vector<V> prefill(keys.size() * dim, static_cast<V>(-2.0));
  const FindOutcome out = run_find(*table, dev, keys, true, &prefill);
  for (size_t i = 0; i < key_num; ++i) {
    ASSERT_TRUE(out.found[i]) << "key " << keys[i] << " was not found";
    EXPECT_EQ(out.score_buffer[i], static_cast<S>(i + 1));
    EXPECT_EQ(value_at(out.values, i, dim),
              expected_value(static_cast<double>(i) * 3.0, dim));
  }
  expect_miss(out, prefill, key_num, dim, "a key that was never stored");
}

TEST(TestDualBucketFind, FindUnsupportedStillThrows) {
  test_util::init_env();

  auto table = make_dual_table();
  DeviceData<K, V, S> dev;
  dev.malloc(4, DIM);

  const K key = 4242;
  Seeder seeder(*table, dev.stream);
  seeder.put_natural(key, 3, 0.0);
  seeder.flush();

  /// The pointer and the missed-keys flavors pin a single bucket and are still
  /// rejected; only the value-copy overload knows about the layout.
  EXPECT_THROW(table->find(1, dev.device_keys, dev.device_values_ptr,
                           dev.device_found, dev.device_scores, dev.stream),
               HkvException);
  EXPECT_THROW(
      table->find(1, dev.device_keys, dev.device_values,
                  static_cast<K*>(nullptr), static_cast<int*>(nullptr),
                  static_cast<int*>(nullptr), dev.device_scores, dev.stream),
      HkvException);

  const std::vector<K> query(1, key);
  const FindOutcome out = run_find(*table, dev, query, true);
  EXPECT_TRUE(out.found[0]) << "the value-copy find must still work";
}

TEST(TestDualBucketFind, FindSingleBucketUnaffected) {
  test_util::init_env();
  constexpr size_t key_num = 256;

  /// A single-bucket table of the same shape, so the dispatch change is
  /// exercised on the path it has to leave alone.
  auto table = make_single_bucket_table();
  DeviceData<K, V, S> dev;
  dev.malloc(key_num + 1, DIM);

  std::vector<K> keys;
  std::vector<V> values;
  std::vector<S> scores;
  for (size_t i = 0; i < key_num; ++i) {
    append_kv(keys, values, scores, static_cast<K>(i + 1),
              static_cast<S>(i + 1), static_cast<double>(i));
  }
  insert_batch(*table, dev, key_num, keys, values, scores);
  ASSERT_EQ(table->size(dev.stream), key_num);

  keys.push_back(MISS_BASE);
  std::vector<V> prefill(keys.size() * DIM, static_cast<V>(2.5));
  const FindOutcome out = run_find(*table, dev, keys, true, &prefill);
  for (size_t i = 0; i < key_num; ++i) {
    ASSERT_TRUE(out.found[i]) << "key " << keys[i] << " was not found";
    EXPECT_EQ(out.score_buffer[i], scores[i]);
    EXPECT_EQ(value_at(out.values, i, DIM),
              expected_value(static_cast<double>(i)));
  }
  expect_miss(out, prefill, key_num, DIM, "a key that was never inserted");
}

}  // namespace
