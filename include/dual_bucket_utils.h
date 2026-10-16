/*
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

/**
 * @file dual_bucket_utils.h
 * @brief Device-side helpers for the dual-bucket layout, where every key maps
 *        to two buckets taken from the low and the high half of its hash.
 *
 *        Self-contained on purpose: the dual-bucket path has its own digest
 *        bit range instead of the single-bucket `empty_digest` / `get_digest`.
 */
#pragma once

#include <simt_api/device_sync_functions.h>
#include "types.h"
#include "utils.h"

namespace npu {
namespace hkv {

/**
 * @brief The dual-bucket digest of a pre-computed hash: bits [56:63] of the
 *        Murmur3 hash (single-bucket uses bits [32:39]), so the two modes'
 *        fingerprints never mix.
 *
 * @note This range is correlated with `bkt1` (which derives from the whole high
 *       32 bits), just as the single-bucket digest is with its own bucket
 *       index. Harmless: a digest only pre-filters, and whether a slot is
 *       usable or a key matches is always settled against the full key word.
 */
template <typename K>
__forceinline__ __simt_callee__ D
dual_bucket_digest_from_hash(const K& hashed_key) {
  return static_cast<D>(static_cast<uint64_t>(hashed_key) >> 56);
}

/// @brief The dual-bucket digest of a key (bits [56:63] of its hash).
template <typename K>
__forceinline__ __simt_callee__ D dual_bucket_digest(const K& key) {
  return dual_bucket_digest_from_hash<K>(Murmur3HashDevice(key));
}

/// @brief Pack the digest into all 4 bytes for `vcmpeq4` comparison.
template <typename K>
__forceinline__ __simt_callee__ VecD_Comp
dual_bucket_digests_from_hash(const K& hashed_key) {
  const D digest = dual_bucket_digest_from_hash<K>(hashed_key);
  return static_cast<VecD_Comp>(digest) * 0x01010101U;
}

/**
 * @brief The digest of an empty slot: the true hash-derived value of
 *        `EMPTY_KEY`, same convention as the single-bucket `empty_digest<K>()`
 *        (which only reads a different bit range).
 */
template <typename K>
__forceinline__ __simt_callee__ D dual_bucket_empty_digest() {
  return dual_bucket_digest_from_hash<K>(
      Murmur3HashDevice(static_cast<K>(EMPTY_KEY)));
}

template <typename K>
__forceinline__ __simt_callee__ VecD_Comp dual_bucket_empty_digests() {
  const D digest = dual_bucket_empty_digest<K>();
  // Set every byte in VecD_Comp to `digest`.
  return static_cast<VecD_Comp>(digest) * 0x01010101U;
}

/**
 * @brief Locate the two candidate buckets of a hashed key under one hash rule.
 *
 * @note `bkt0` comes from the low 32 bits of the hash and `bkt1` from the high
 *       32 bits, each with its own probe start position — deliberately not an
 *       adjacent bucket pair, which would share positions and degenerate into
 *       two slots per position. The residues are taken modulo `capacity` and
 *       split by `max_bucket_shift`, reusing the fast-division magic already
 *       precomputed for `capacity`; the quotient is the bucket index and the
 *       remainder the in-bucket position. `bkt1` is advanced when the two
 *       halves collide, which is why `buckets_num >= 2` is required.
 *
 * @param[out] key_pos0/key_pos1 The probe start position inside `bkt0`/`bkt1`.
 * @param[out] bkt0/bkt1 The bucket selected by the low/high half of the hash.
 */
template <typename K>
__forceinline__ __simt_callee__ void get_dual_bucket_indices(
    const K& hashed_key, const uint64_t& capacity,
    const uint32_t& bucket_max_size, const uint32_t& max_bucket_shift,
    const uint64_t& capacity_divisor_magic,
    const uint64_t& capacity_divisor_shift, const uint64_t& buckets_num,
    uint32_t& key_pos0, uint32_t& key_pos1, uint64_t& bkt0, uint64_t& bkt1) {
  const uint64_t hashed = static_cast<uint64_t>(hashed_key);
  const uint64_t lo = static_cast<uint32_t>(hashed);
  const uint64_t hi = hashed >> 32;

  const uint64_t global_idx0 = get_global_idx(lo, capacity_divisor_magic,
                                              capacity_divisor_shift, capacity);
  const uint64_t global_idx1 = get_global_idx(hi, capacity_divisor_magic,
                                              capacity_divisor_shift, capacity);

  key_pos0 = static_cast<uint32_t>(global_idx0 & (bucket_max_size - 1));
  key_pos1 = static_cast<uint32_t>(global_idx1 & (bucket_max_size - 1));
  bkt0 = global_idx0 >> max_bucket_shift;
  bkt1 = global_idx1 >> max_bucket_shift;

  if (bkt1 == bkt0) {
    bkt1 = (bkt1 + 1 == buckets_num) ? 0 : bkt1 + 1;
  }
}

/**
 * @brief Probe one bucket of the pair for the desired key.
 *
 * @note The digest word is fetched once per stride and covers the whole 4-wide
 *       comparison group, matching the single-bucket probe.
 * @return DUPLICATE with @p out_pos set when the key was found and locked,
 *         INITIAL otherwise.
 */
template <class K, class V, class S>
__forceinline__ __simt_callee__ OccupyResult probe_dual_bucket_duplicate(
    __gm__ K* __restrict__ bucket_keys, const K& key, const uint32_t key_pos,
    const VecD_Comp& target_digests, const uint32_t bucket_max_size,
    uint32_t& out_pos) {
  using BUCKET = Bucket<K, V, S>;
  constexpr uint32_t STRIDE = sizeof(VecD_Comp) / sizeof(D);
  uint32_t pos_base = key_pos;
  pos_base = align_to<STRIDE>(pos_base);

  for (uint32_t offset = 0; offset < bucket_max_size + STRIDE;
       offset += STRIDE) {
    const uint32_t pos_cur = (pos_base + offset) & (bucket_max_size - 1);
    __gm__ D* digests_ptr =
        BUCKET::digests(bucket_keys, bucket_max_size, pos_cur);
    const VecD_Comp probe_digests =
        *reinterpret_cast<__gm__ VecD_Comp*>(digests_ptr);

    uint32_t cmp_result = vcmpeq4(probe_digests, target_digests) & 0x01010101;
    while (cmp_result != 0) {
      const uint32_t index = (__ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
      cmp_result &= (cmp_result - 1);
      const uint32_t possible_pos = pos_cur + index;

      __gm__ K* current_key_ptr = BUCKET::keys(bucket_keys, possible_pos);
      const K try_key =
          asc_atomic_cas(current_key_ptr, key, static_cast<K>(LOCKED_KEY));
      if (try_key == key) {
        out_pos = possible_pos;
        return OccupyResult::DUPLICATE;
      }
    }
  }
  return OccupyResult::INITIAL;
}

/**
 * @brief Probe one bucket of the pair for an empty slot and claim it.
 * @return OCCUPIED_EMPTY with @p out_pos set when claimed, INITIAL otherwise.
 */
template <class K, class V, class S>
__forceinline__ __simt_callee__ OccupyResult probe_dual_bucket_empty(
    __gm__ K* __restrict__ bucket_keys, const uint32_t key_pos,
    const uint32_t bucket_max_size, uint32_t& out_pos) {
  using BUCKET = Bucket<K, V, S>;
  constexpr uint32_t STRIDE = sizeof(VecD_Comp) / sizeof(D);
  uint32_t pos_base = key_pos;
  pos_base = align_to<STRIDE>(pos_base);
  const VecD_Comp empty_digests_val = dual_bucket_empty_digests<K>();

  for (uint32_t offset = 0; offset < bucket_max_size + STRIDE;
       offset += STRIDE) {
    const uint32_t pos_cur = (pos_base + offset) & (bucket_max_size - 1);
    __gm__ D* digests_ptr =
        BUCKET::digests(bucket_keys, bucket_max_size, pos_cur);
    const VecD_Comp probe_digests =
        *reinterpret_cast<__gm__ VecD_Comp*>(digests_ptr);

    uint32_t cmp_result =
        vcmpeq4(probe_digests, empty_digests_val) & 0x01010101;
    while (cmp_result != 0) {
      const uint32_t index = (__ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
      cmp_result &= (cmp_result - 1);
      const uint32_t possible_pos = pos_cur + index;
      if (offset == 0 && possible_pos < key_pos) {
        continue;
      }

      __gm__ K* current_key_ptr = BUCKET::keys(bucket_keys, possible_pos);
      const K try_key =
          asc_atomic_cas(current_key_ptr, static_cast<K>(EMPTY_KEY),
                         static_cast<K>(LOCKED_KEY));
      if (try_key == static_cast<K>(EMPTY_KEY)) {
        out_pos = possible_pos;
        return OccupyResult::OCCUPIED_EMPTY;
      }
    }
  }
  return OccupyResult::INITIAL;
}

/// @brief `reduce_dual_bucket_min_score`'s "no evictable entry in the pair"
///        sentinel; never collides with a real bucket selector (0 or 1).
constexpr uint32_t DUAL_BUCKET_NONE = 2U;

/**
 * @brief Reduce the minimum score over a whole bucket pair cooperatively.
 *
 * @note Threads of one EVICT_GROUP_SIZE group walk the flattened range
 *       `[0, 2 * bucket_max_size)`; the winning bucket travels through the
 *       `asc_shfl_xor` reduction together with the score and the position, so a
 *       pair whose entries all carry MAX_SCORE still yields a valid victim. If
 *       nothing is evictable, @p min_bucket comes back as DUAL_BUCKET_NONE and
 *       the caller must refuse the insert.
 *
 * @param[out] min_score Minimum score visible to the whole group.
 * @param[out] min_pos Position of the winning entry inside its bucket.
 * @param[out] min_bucket 0 for `bkt0`, 1 for `bkt1`, DUAL_BUCKET_NONE if none.
 */
template <class K, class S, int32_t EVICT_GROUP_SIZE>
__forceinline__ __simt_callee__ void reduce_dual_bucket_min_score(
    __gm__ S* __restrict__ b0_scores, __gm__ K* __restrict__ b0_keys,
    __gm__ S* __restrict__ b1_scores, __gm__ K* __restrict__ b1_keys,
    const uint32_t bucket_max_size, const uint32_t group_rank, S& min_score,
    uint32_t& min_pos, uint32_t& min_bucket) {
  min_score = static_cast<S>(MAX_SCORE);
  min_pos = 0;
  min_bucket = DUAL_BUCKET_NONE;

  const uint32_t flat_max = bucket_max_size * 2;
  for (uint32_t flat = group_rank; flat < flat_max; flat += EVICT_GROUP_SIZE) {
    const uint32_t bucket = flat >= bucket_max_size ? 1U : 0U;
    const uint32_t pos = flat & (bucket_max_size - 1);
    __gm__ K* key_ptr = (bucket == 0 ? b0_keys : b1_keys) + pos;

    const K current_key = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                                L1CacheType::NON_CACHEABLE>(key_ptr);
    if (current_key == static_cast<K>(LOCKED_KEY) ||
        current_key == static_cast<K>(EMPTY_KEY)) {
      continue;
    }

    __gm__ S* score_ptr = (bucket == 0 ? b0_scores : b1_scores) + pos;
    const S current_score = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                                  L1CacheType::NON_CACHEABLE>(score_ptr);
    if (min_bucket == DUAL_BUCKET_NONE || current_score < min_score) {
      min_score = current_score;
      min_pos = pos;
      min_bucket = bucket;
    }
  }

  for (int32_t offset = EVICT_GROUP_SIZE / 2; offset > 0; offset /= 2) {
    const S other_score = asc_shfl_xor(min_score, offset, EVICT_GROUP_SIZE);
    const uint32_t other_pos = asc_shfl_xor(min_pos, offset, EVICT_GROUP_SIZE);
    const uint32_t other_bucket =
        asc_shfl_xor(min_bucket, offset, EVICT_GROUP_SIZE);
    if (other_bucket != DUAL_BUCKET_NONE &&
        (min_bucket == DUAL_BUCKET_NONE || other_score < min_score)) {
      min_score = other_score;
      min_pos = other_pos;
      min_bucket = other_bucket;
    }
  }
}

}  // namespace hkv
}  // namespace npu
