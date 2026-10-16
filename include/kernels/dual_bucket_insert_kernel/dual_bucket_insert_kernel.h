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

/**
 * @file dual_bucket_insert_kernel.h
 * @brief The dual-bucket `insert_or_assign` kernel.
 *
 *        Every key maps to two buckets — `bkt0` from the low half of its hash,
 *        `bkt1` from the high half — each with its own probe start position
 *        (`get_dual_bucket_indices` in `dual_bucket_utils.h`). Insert path:
 *        probe both buckets for a duplicate, else claim an empty slot preferring
 *        the less loaded bucket, else evict the entry with the globally minimum
 *        score across the pair (refusing the insert if the new score is lower).
 *
 *        The single-bucket kernel in `insert_or_assign_kernel.h` is untouched,
 *        so the existing path carries zero regression risk.
 */

#ifndef ASCENDC_DUAL_BUCKET_INSERT_KERNEL_H_
#define ASCENDC_DUAL_BUCKET_INSERT_KERNEL_H_

#include <cstdint>
#include "kernel_operator.h"
#include "dual_bucket_utils.h"
#include "kernels/mte_task_queue.h"
#include "score_functor.h"
#include "simt_vf_dispatcher.h"
#include "types.h"
#include "utils.h"

namespace npu {
namespace hkv {

using namespace AscendC;

/// @brief Stand-in for the value-move task queue on the non-MTE instantiation,
///        so the tail is selected with `if constexpr` instead of duplicating
///        the whole function.
struct DualBucketDummyTaskQueue {};

template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          typename VecV = int32_t, typename TaskQueue = DualBucketDummyTaskQueue,
          int32_t Strategy = -1, int32_t EVICT_GROUP_SIZE = 16,
          bool USE_MTE = false>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_512) inline void insert_or_assign_dual_bucket_vf(
    __gm__ void* buckets_addr_gm, __gm__ int32_t* buckets_size_addr_gm,
    uint64_t capacity, uint64_t buckets_num, uint32_t bucket_max_size,
    uint32_t dim, __gm__ void* keys_addr_gm, __gm__ void* values_addr_gm,
    __gm__ void* scores_gm, S cur_score, uint64_t n, uint64_t global_epoch,
    uint32_t block_index, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift,
    uint64_t n_align_warp, int32_t group_size, uint32_t thread_all) {
  if (buckets_addr_gm == nullptr) {
    return;
  }
  if (buckets_size_addr_gm == nullptr) {
    return;
  }
  if (keys_addr_gm == nullptr) {
    return;
  }
  if (values_addr_gm == nullptr) {
    return;
  }

  TaskQueue upsert_task_queue;
  (void)upsert_task_queue;

  using BUCKET = Bucket<K, V, S>;
  using ScoreFunctor = hkv::ScoreFunctor<K, V, S, Strategy>;

  __gm__ Bucket<K, V, S>* __restrict__ buckets =
      reinterpret_cast<__gm__ Bucket<K, V, S>*>(buckets_addr_gm);
  __gm__ int32_t* __restrict__ buckets_size =
      reinterpret_cast<__gm__ int32_t*>(buckets_size_addr_gm);
  __gm__ const K* __restrict__ keys =
      reinterpret_cast<__gm__ const K*>(keys_addr_gm);
  __gm__ S* __restrict__ scores = reinterpret_cast<__gm__ S*>(scores_gm);
  S score = static_cast<S>(EMPTY_SCORE);

  uint32_t key_pos = 0;
  K key = 0;
  uint32_t chosen_sel = 0;  // 0 or 1: which of the pair owns the slot.

  for (uint64_t kv_idx = block_index * blockDim.x + threadIdx.x;
       kv_idx < n_align_warp; kv_idx += thread_all) {
    VecD_Comp target_digests{0};
    OccupyResult occupy_result{OccupyResult::INITIAL};

    // Pointers of the candidate bucket pair, filled by the branch below.
    __gm__ K* keys0 = nullptr;
    __gm__ K* keys1 = nullptr;
    __gm__ S* scores0 = nullptr;
    __gm__ S* scores1 = nullptr;
    __gm__ int32_t* size0 = nullptr;
    __gm__ int32_t* size1 = nullptr;
    uint64_t vectors0_uintptr = 0;
    uint64_t vectors1_uintptr = 0;
    int32_t cur_size0 = 0;
    int32_t cur_size1 = 0;
    chosen_sel = 0;

    // 1. Every thread owns exactly one key.
    if (kv_idx < n) {
      key = keys[kv_idx];
      if (IS_RESERVED_KEY<K>(key)) {
        occupy_result = OccupyResult::ILLEGAL;
      }
      score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch,
                                                cur_score);

      // 2. Hash the key and locate the pair: low half -> `bkt0`, high -> `bkt1`.
      const K hashed_key = Murmur3HashDevice(key);
      target_digests = dual_bucket_digests_from_hash<K>(hashed_key);
      uint64_t bkt0 = 0;
      uint64_t bkt1 = 0;
      uint32_t start_pos0 = 0;
      uint32_t start_pos1 = 0;
      get_dual_bucket_indices<K>(
          hashed_key, capacity, bucket_max_size, max_bucket_shift,
          capacity_divisor_magic, capacity_divisor_shift, buckets_num,
          start_pos0, start_pos1, bkt0, bkt1);

      __gm__ Bucket<K, V, S>* bucket0 = buckets + bkt0;
      __gm__ Bucket<K, V, S>* bucket1 = buckets + bkt1;
      keys0 = bucket0->keys_;
      keys1 = bucket1->keys_;
      scores0 = bucket0->scores_;
      scores1 = bucket1->scores_;
      vectors0_uintptr = reinterpret_cast<uint64_t>(bucket0->vectors);
      vectors1_uintptr = reinterpret_cast<uint64_t>(bucket1->vectors);
      size0 = buckets_size + bkt0;
      size1 = buckets_size + bkt1;
      cur_size0 = *size0;
      cur_size1 = *size1;

      // 3. Phase A: dedup. Both buckets must be probed before any empty slot is
      //    claimed, otherwise a key living in bucket 1 could be inserted a
      //    second time into bucket 0. A reserved key already carries ILLEGAL
      //    and must neither probe nor claim, hence the INITIAL checks below.
      uint32_t out_pos = 0;
      if (occupy_result == OccupyResult::INITIAL &&
          probe_dual_bucket_duplicate<K, V, S>(
              keys0, key, start_pos0, target_digests, bucket_max_size,
              out_pos) == OccupyResult::DUPLICATE) {
        chosen_sel = 0;
        key_pos = out_pos;
        occupy_result = OccupyResult::DUPLICATE;
      } else if (occupy_result == OccupyResult::INITIAL &&
                 probe_dual_bucket_duplicate<K, V, S>(
                     keys1, key, start_pos1, target_digests, bucket_max_size,
                     out_pos) == OccupyResult::DUPLICATE) {
        chosen_sel = 1;
        key_pos = out_pos;
        occupy_result = OccupyResult::DUPLICATE;
      }

      if (occupy_result == OccupyResult::DUPLICATE) {
        // The digest is already correct, only the score needs refreshing.
        __gm__ K* chosen_keys = (chosen_sel == 0) ? keys0 : keys1;
        ScoreFunctor::update_score_only(chosen_keys, key_pos, scores, kv_idx,
                                        score, bucket_max_size, false);
      } else if (occupy_result == OccupyResult::INITIAL) {
        // 4. Phase B: claim an empty slot, preferring the less loaded bucket.
        //    `cur_size` is only a heuristic; correctness comes from the CAS
        //    inside the probe plus the fallback to the other bucket.
        const int32_t bmax = static_cast<int32_t>(bucket_max_size);
        const bool full0 = (cur_size0 >= bmax);
        const bool full1 = (cur_size1 >= bmax);
        const uint32_t first_sel = (cur_size0 <= cur_size1) ? 0U : 1U;
        for (uint32_t t = 0; t < 2; ++t) {
          const uint32_t sel = (t == 0) ? first_sel : (first_sel ^ 1U);
          if ((sel == 0 && full0) || (sel == 1 && full1)) {
            continue;
          }
          __gm__ K* cand_keys = (sel == 0) ? keys0 : keys1;
          __gm__ int32_t* cand_size = (sel == 0) ? size0 : size1;
          const uint32_t cand_start_pos = (sel == 0) ? start_pos0 : start_pos1;
          if (probe_dual_bucket_empty<K, V, S>(cand_keys, cand_start_pos,
                                               bucket_max_size,
                                               out_pos) ==
              OccupyResult::OCCUPIED_EMPTY) {
            chosen_sel = sel;
            key_pos = out_pos;
            occupy_result = OccupyResult::OCCUPIED_EMPTY;
            ScoreFunctor::update_with_digest(cand_keys, key_pos, scores, kv_idx,
                                             score, bucket_max_size,
                                             dual_bucket_digest<K>(key), true);
            asc_atomic_add(cand_size, 1);
            break;
          }
        }
      }
    } else {
      occupy_result = OccupyResult::ILLEGAL;
    }

    // 5. Phase C: both buckets are full, evict the entry with the globally
    //    minimum score across the pair. Mirrors the single-bucket kernel: the
    //    group reduces with `asc_shfl_xor`, then lane `i` acts on behalf of the
    //    group using its own (broadcast) bucket pair.
    auto cg_rank_id = threadIdx.x % EVICT_GROUP_SIZE;
    for (int32_t i = 0; i < EVICT_GROUP_SIZE; i++) {
      auto res_sync = asc_shfl(occupy_result, i, EVICT_GROUP_SIZE);
      while (res_sync == OccupyResult::INITIAL) {
        S min_score = static_cast<S>(MAX_SCORE);
        uint32_t min_pos = 0;
        uint32_t min_bucket = DUAL_BUCKET_NONE;
        const uint64_t scores0_sync = asc_shfl(
            reinterpret_cast<uint64_t>(scores0), i, EVICT_GROUP_SIZE);
        const uint64_t keys0_sync = asc_shfl(
            reinterpret_cast<uint64_t>(keys0), i, EVICT_GROUP_SIZE);
        const uint64_t scores1_sync = asc_shfl(
            reinterpret_cast<uint64_t>(scores1), i, EVICT_GROUP_SIZE);
        const uint64_t keys1_sync = asc_shfl(
            reinterpret_cast<uint64_t>(keys1), i, EVICT_GROUP_SIZE);

        reduce_dual_bucket_min_score<K, S, EVICT_GROUP_SIZE>(
            reinterpret_cast<__gm__ S*>(scores0_sync),
            reinterpret_cast<__gm__ K*>(keys0_sync),
            reinterpret_cast<__gm__ S*>(scores1_sync),
            reinterpret_cast<__gm__ K*>(keys1_sync), bucket_max_size, cg_rank_id,
            min_score, min_pos, min_bucket);

        if (cg_rank_id == i) {
          __gm__ K* victim_keys = (min_bucket == 1) ? keys1 : keys0;
          __gm__ S* victim_scores = (min_bucket == 1) ? scores1 : scores0;
          __gm__ int32_t* victim_size = (min_bucket == 1) ? size1 : size0;
          // 5.1 No evictable entry (all slots transiently LOCKED/EMPTY), or
          //     the new key is less valuable than the pair's minimum.
          if (min_bucket == DUAL_BUCKET_NONE || score < min_score) {
            occupy_result = OccupyResult::REFUSED;
          } else {
            // 5.2 Try to claim the victim slot.
            auto current_key_ptr = victim_keys + min_pos;
            auto current_key =
                __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                      L1CacheType::NON_CACHEABLE>(current_key_ptr);
            if (current_key != static_cast<K>(LOCKED_KEY) &&
                current_key != static_cast<K>(EMPTY_KEY)) {
              auto try_key = asc_atomic_cas(current_key_ptr, current_key,
                                            static_cast<K>(LOCKED_KEY));
              if (try_key == current_key) {
                // 5.3 Re-check the score: it may have grown since the scan.
                if (__ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                          L1CacheType::NON_CACHEABLE>(
                        victim_scores + min_pos) <= min_score) {
                  key_pos = min_pos;
                  chosen_sel = min_bucket;
                  ScoreFunctor::update_with_digest(
                      victim_keys, key_pos, scores, kv_idx, score,
                      bucket_max_size, dual_bucket_digest<K>(key), true);
                  if (try_key == static_cast<K>(RECLAIM_KEY)) {
                    occupy_result = OccupyResult::OCCUPIED_RECLAIMED;
                    asc_atomic_add(victim_size, 1);
                  } else {
                    occupy_result = OccupyResult::EVICT;
                  }
                } else {
                  // 5.4 The victim became more valuable, restore it and retry.
                  (void)asc_atomic_exch(current_key_ptr, current_key);
                }
              }
            }
          }
        }
        res_sync = asc_shfl(occupy_result, i, EVICT_GROUP_SIZE);
      }
    }

    // 6. Phase D: move the value and publish the key, which releases the lock.
    __gm__ K* chosen_keys = (chosen_sel == 0) ? keys0 : keys1;
    const uint64_t chosen_vectors_uintptr =
        (chosen_sel == 0) ? vectors0_uintptr : vectors1_uintptr;

    if constexpr (USE_MTE) {
      __gm__ V* __restrict__ values =
          reinterpret_cast<__gm__ V*>(values_addr_gm);
      const bool need_assign = ((occupy_result != OccupyResult::REFUSED) &&
                                (occupy_result != OccupyResult::ILLEGAL));
      uint32_t warp_acquire_mask =
          upsert_task_queue.get_acquire_mask(need_assign);
      if (warp_acquire_mask == 0) {
        continue;
      }
      __gm__ V* chosen_vectors =
          reinterpret_cast<__gm__ V*>(chosen_vectors_uintptr);
      const uint64_t value_base_addr =
          reinterpret_cast<uint64_t>(values + kv_idx * dim);
      const uint64_t bucket_value_base_addr =
          reinterpret_cast<uint64_t>(chosen_vectors + key_pos * dim);

      uint32_t curr_simt_alloc_pos = 0;
      uint32_t new_simt_alloc_pos = 0;
      uint32_t assign_task_id = 0;
      upsert_task_queue.producer_acquire(warp_acquire_mask, curr_simt_alloc_pos,
                                         new_simt_alloc_pos, assign_task_id);
      upsert_task_queue.set_key_async(
          need_assign, assign_task_id,
          reinterpret_cast<void*>(bucket_value_base_addr),
          reinterpret_cast<void*>(value_base_addr), key,
          reinterpret_cast<__gm__ uint64_t*>(chosen_keys + key_pos));
      upsert_task_queue.producer_commit(curr_simt_alloc_pos,
                                        new_simt_alloc_pos);
    } else {
      __gm__ VecV* __restrict__ values =
          reinterpret_cast<__gm__ VecV*>(values_addr_gm);
      cg_rank_id = threadIdx.x % group_size;
      for (int32_t i = 0; i < group_size; i++) {
        auto res_sync = asc_shfl(occupy_result, i, group_size);
        if ((res_sync != OccupyResult::REFUSED &&
             res_sync != OccupyResult::ILLEGAL)) {
          auto kv_idx_sync = asc_shfl(kv_idx, i, group_size);
          auto value_start = kv_idx_sync * dim;

          auto key_pos_sync = asc_shfl(key_pos, i, group_size);
          uint64_t value_ddr_sync =
              asc_shfl(chosen_vectors_uintptr, i, group_size);
          auto bucket_value_start = key_pos_sync * dim;
          for (uint32_t j = cg_rank_id; j < dim; j += group_size) {
            __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                  L1CacheType::NON_CACHEABLE>(
                reinterpret_cast<__gm__ VecV*>(value_ddr_sync) +
                    bucket_value_start + j,
                values[value_start + j]);
          }
        }
      }

      __threadfence();

      if (occupy_result != OccupyResult::REFUSED &&
          occupy_result != OccupyResult::ILLEGAL) {
        // The key word doubles as the slot lock, so every key write is atomic.
        (void)asc_atomic_exch(chosen_keys + key_pos, key);
      }
    }
  }

  if constexpr (USE_MTE) {
    __syncthreads();

    if (threadIdx.x == 0) {
      upsert_task_queue.producer_finish();
    }
    upsert_task_queue.consumer_wait_all();
  }
}

template <class K, class V, class S, int Strategy = -1>
__global__ __vector__ void insert_or_assign_dual_bucket_kernel(
    __gm__ void* buckets, __gm__ int* buckets_size, uint64_t capacity,
    uint64_t buckets_num, uint32_t bucket_max_size, uint32_t dim,
    __gm__ void* keys, __gm__ void* values, __gm__ void* scores, uint64_t n,
    uint64_t global_epoch, uint32_t value_size, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift,
    uint64_t n_align_warp, int32_t group_size) {
  const uint32_t thread_all = THREAD_NUM_512 * GetBlockNum();
  uint64_t cur_score = (Strategy == npu::hkv::EvictStrategyInternal::kLru ||
                        Strategy == npu::hkv::EvictStrategyInternal::kEpochLru)
                           ? static_cast<uint64_t>(GetSystemCycle())
                           : 0;

  DISPATCH_VALUE_SIZE(
      value_size,
      (asc_vf_call<insert_or_assign_dual_bucket_vf<
          K, V, S, DTYPE, DualBucketDummyTaskQueue, Strategy, 16, false>>(
          dim3{THREAD_NUM_512}, buckets, buckets_size, capacity, buckets_num,
          bucket_max_size, dim, keys, values, scores, cur_score, n,
          global_epoch, GetBlockIdx(), max_bucket_shift, capacity_divisor_magic,
          capacity_divisor_shift, n_align_warp, group_size, thread_all)));
}

template <class K, class V, class S, class TaskQueue, int Strategy = -1>
__global__ __vector__ void insert_or_assign_dual_bucket_kernel_with_mte(
    __gm__ void* buckets, __gm__ int* buckets_size, uint64_t capacity,
    uint64_t buckets_num, uint32_t bucket_max_size, uint32_t dim,
    __gm__ void* keys, __gm__ void* values, __gm__ void* scores, uint64_t n,
    uint64_t global_epoch, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift,
    uint64_t n_align_warp) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  if ASCEND_IS_AIV {
    TaskQueue upsert_task_queue;

    // Clear the flags.
    upsert_task_queue.init();

    pipe_barrier(PIPE_ALL);

    const uint32_t thread_all = THREAD_NUM_512 * GetBlockNum();
    uint64_t cur_score =
        (Strategy == npu::hkv::EvictStrategyInternal::kLru ||
         Strategy == npu::hkv::EvictStrategyInternal::kEpochLru)
            ? static_cast<uint64_t>(GetSystemCycle())
            : 0;
    asc_vf_call<insert_or_assign_dual_bucket_vf<
        K, V, S, int32_t, TaskQueue, Strategy, 16, true>>(
        dim3{THREAD_NUM_512}, buckets, buckets_size, capacity, buckets_num,
        bucket_max_size, dim, keys, values, scores, cur_score, n, global_epoch,
        GetBlockIdx(), max_bucket_shift, capacity_divisor_magic,
        capacity_divisor_shift, n_align_warp, /* group_size */ 0, thread_all);

    uint32_t move_len = dim * sizeof(V);
    upsert_task_queue.run_task(move_len);
  }  // ASCEND_IS_AIV
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_DUAL_BUCKET_INSERT_KERNEL_H_
