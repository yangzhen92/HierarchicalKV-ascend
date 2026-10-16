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
 * @file dual_bucket_seed_kernel.h
 * @brief Test-only seed kernel: writes a key / digest / score / value straight
 *        into a given (bucket, slot) of a table, bypassing every public API.
 *
 *        The dual-bucket `find` tests must not build their fixtures with the
 *        dual-bucket `insert_or_assign`, otherwise an insert-side bug and a
 *        lookup-side bug could cancel out. Seeding the slots directly also lets
 *        a case place sentinel keys, a deliberately mismatched digest, or an
 *        exactly-full bucket, none of which the insert path can produce.
 *
 *        `buckets_size` is deliberately left alone: `find` never reads it. A
 *        table seeded this way therefore reports a `size()` / `load_factor()`
 *        that does not match the slots actually occupied.
 *
 *        Only scalar arrays cross into the kernel; `DualBucketSeedRecord` is a
 *        host-side convenience that @ref seed_dual_bucket_slots flattens before
 *        uploading.
 */

#pragma once

#include <cstdint>
#include <vector>
#include "acl/acl.h"
#include "dual_bucket_utils.h"
#include "kernel_operator.h"
#include "test_util.h"
#include "types.h"
#include "utils.h"

namespace npu {
namespace hkv {

using namespace AscendC;

/// One slot to write, addressed by the table's own (bucket, slot) coordinates.
struct DualBucketSeedRecord {
  uint64_t bucket = 0;  ///< Index of the target bucket.
  uint32_t pos = 0;     ///< Slot inside that bucket's `bucket_max_size` range.
  uint64_t key = 0;    ///< Key word to store; sentinels are allowed on purpose.
  uint64_t score = 0;  ///< Score to store.
  /// Digest byte to store. Pass the key's own digest for a realistic slot, or
  /// any other value to build a deliberate digest/key mismatch.
  uint32_t digest = 0;
};

/// @brief Build a record carrying the key's own dual-bucket digest, i.e. the
///        slot content a real `insert_or_assign` would have produced.
inline DualBucketSeedRecord make_seed_record(uint64_t bucket, uint32_t pos,
                                             uint64_t key, uint64_t score) {
  return {bucket, pos, key, score,
          static_cast<uint32_t>(test_util::Murmur3HashHost(key) >> 56)};
}

/// @brief Write one (bucket, slot) per thread. Four independent scalar arrays
///        rather than an array of structs, so nothing but builtin types is
///        passed across the SIMT call boundary.
template <typename K, typename V, typename S>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_512) inline void dual_bucket_seed_kernel_vf(
    __gm__ Bucket<K, V, S>* buckets, uint32_t bucket_max_size, uint32_t dim,
    __gm__ const uint64_t* rec_buckets, __gm__ const uint32_t* rec_positions,
    __gm__ const uint64_t* rec_keys, __gm__ const uint64_t* rec_scores,
    __gm__ const uint32_t* rec_digests, uint32_t n, __gm__ const V* payloads,
    uint32_t block_index, uint32_t block_num) {
  using BUCKET = Bucket<K, V, S>;
  for (uint32_t i = block_index * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * block_num) {
    const uint32_t pos = rec_positions[i];
    __gm__ Bucket<K, V, S>* bucket = buckets + rec_buckets[i];
    // Payload first, key word last, like a real insert publishes a slot: a
    // reader that can see the key can also see its value.
    for (uint32_t j = 0; j < dim; ++j) {
      bucket->vectors[pos * dim + j] = payloads[i * dim + j];
    }
    __threadfence();
    bucket->scores_[pos] = static_cast<S>(rec_scores[i]);
    *(BUCKET::digests(bucket->keys_, bucket_max_size, pos)) =
        static_cast<D>(rec_digests[i]);
    bucket->keys_[pos] = static_cast<K>(rec_keys[i]);
  }
}

template <typename K, typename V, typename S>
__global__ __vector__ void dual_bucket_seed_kernel(
    __gm__ Bucket<K, V, S>* buckets, uint32_t bucket_max_size, uint32_t dim,
    __gm__ const uint64_t* rec_buckets, __gm__ const uint32_t* rec_positions,
    __gm__ const uint64_t* rec_keys, __gm__ const uint64_t* rec_scores,
    __gm__ const uint32_t* rec_digests, uint32_t n, __gm__ const V* payloads) {
  asc_vf_call<dual_bucket_seed_kernel_vf<K, V, S>>(
      dim3{static_cast<uint32_t>(THREAD_NUM_512)}, buckets, bucket_max_size,
      dim, rec_buckets, rec_positions, rec_keys, rec_scores, rec_digests, n,
      payloads, GetBlockIdx(), GetBlockNum());
}

/// @brief Upload @p records / @p payloads and write them into @p buckets.
/// @param payloads `n * dim` values, flattened, in the same order as @p
/// records.
template <typename K, typename V, typename S>
void seed_dual_bucket_slots(__gm__ Bucket<K, V, S>* buckets,
                            uint32_t bucket_max_size, uint32_t dim,
                            const std::vector<DualBucketSeedRecord>& records,
                            const std::vector<V>& payloads,
                            aclrtStream stream) {
  const uint32_t n = static_cast<uint32_t>(records.size());
  if (n == 0) {
    return;
  }
  std::vector<uint64_t> rec_buckets(n);
  std::vector<uint32_t> rec_positions(n);
  std::vector<uint64_t> rec_keys(n);
  std::vector<uint64_t> rec_scores(n);
  std::vector<uint32_t> rec_digests(n);
  for (uint32_t i = 0; i < n; ++i) {
    rec_buckets[i] = records[i].bucket;
    rec_positions[i] = records[i].pos;
    rec_keys[i] = records[i].key;
    rec_scores[i] = records[i].score;
    rec_digests[i] = records[i].digest;
  }

  const size_t buckets_bytes = n * sizeof(uint64_t);
  const size_t positions_bytes = n * sizeof(uint32_t);
  const size_t digests_bytes = n * sizeof(uint32_t);
  const size_t payloads_bytes = payloads.size() * sizeof(V);

  void* d_rec_buckets = nullptr;
  void* d_rec_positions = nullptr;
  void* d_rec_keys = nullptr;
  void* d_rec_scores = nullptr;
  void* d_rec_digests = nullptr;
  void* d_payloads = nullptr;
  ASSERT_EQ(
      aclrtMalloc(&d_rec_buckets, buckets_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
      ACL_ERROR_NONE);
  ASSERT_EQ(
      aclrtMalloc(&d_rec_positions, positions_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
      ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(&d_rec_keys, buckets_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(
      aclrtMalloc(&d_rec_scores, buckets_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
      ACL_ERROR_NONE);
  ASSERT_EQ(
      aclrtMalloc(&d_rec_digests, digests_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
      ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(&d_payloads, payloads_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  ASSERT_EQ(aclrtMemcpy(d_rec_buckets, buckets_bytes, rec_buckets.data(),
                        buckets_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(d_rec_positions, positions_bytes, rec_positions.data(),
                        positions_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(d_rec_keys, buckets_bytes, rec_keys.data(),
                        buckets_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(d_rec_scores, buckets_bytes, rec_scores.data(),
                        buckets_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(d_rec_digests, digests_bytes, rec_digests.data(),
                        digests_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(d_payloads, payloads_bytes, payloads.data(),
                        payloads_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  const size_t grid_size =
      (static_cast<size_t>(n) + THREAD_NUM_512 - 1) / THREAD_NUM_512;
  dual_bucket_seed_kernel<K, V, S><<<grid_size, 0, stream>>>(
      buckets, bucket_max_size, dim,
      reinterpret_cast<const uint64_t*>(d_rec_buckets),
      reinterpret_cast<const uint32_t*>(d_rec_positions),
      reinterpret_cast<const uint64_t*>(d_rec_keys),
      reinterpret_cast<const uint64_t*>(d_rec_scores),
      reinterpret_cast<const uint32_t*>(d_rec_digests), n,
      reinterpret_cast<const V*>(d_payloads));
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  ASSERT_EQ(aclrtFree(d_rec_buckets), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(d_rec_positions), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(d_rec_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(d_rec_scores), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(d_rec_digests), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(d_payloads), ACL_ERROR_NONE);
}

}  // namespace hkv
}  // namespace npu
