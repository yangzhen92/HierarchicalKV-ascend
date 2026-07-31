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

#ifndef ASCENDC_FIND_WITH_DIGEST_KERNEL_H_
#define ASCENDC_FIND_WITH_DIGEST_KERNEL_H_

#include <kernel_operator.h>
#include <cstdint>
#include "find_utils.h"
#include "kernels/mte_task_queue.h"
#include "simt_vf_dispatcher.h"

namespace npu {
namespace hkv {
using namespace AscendC;

template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          typename VecV = int32_t>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_512) inline void find_with_digest_kernel_vf(
    __gm__ Bucket<K, V, S>* buckets, uint64_t capacity, uint64_t buckets_num,
    uint32_t bucket_capacity, uint32_t dim, __gm__ K* keys, __gm__ V* values,
    __gm__ S* scores, __gm__ bool* founds, uint64_t n,
    const uint64_t total_thread_num, uint64_t global_epoch, uint32_t block_id,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift, uint64_t n_align_warp,
    uint32_t group_size) {
  if (buckets == nullptr || keys == nullptr || values == nullptr) {
    return;
  }
  using BUCKET = Bucket<K, V, S>;
  __gm__ VecV* values_vec = reinterpret_cast<__gm__ VecV*>(values);

  for (uint64_t kv_idx = block_id * blockDim.x + threadIdx.x;
       kv_idx < n_align_warp; kv_idx += total_thread_num) {
    VecD_Comp target_digests{0};
    bool found = false;
    uint32_t target_pos = INVALID_KEY_POS;
    uint64_t bucket_values_uintptr = 0;

    // 1、每个线程处理一个key
    if (kv_idx < n) {
      const K key = ldg_l2nc_l1c(keys + kv_idx);
      if (!IS_RESERVED_KEY<K>(key)) {
        // 2、计算key哈希 && 定位桶位置
        const K hashed_key = Murmur3HashDevice(key);
        target_digests = digests_from_hashed<K>(hashed_key);
        uint64_t global_idx = get_global_idx(hashed_key, capacity_divisor_magic,
                                             capacity_divisor_shift, capacity);
        uint32_t key_pos = global_idx & (bucket_capacity - 1);
        uint64_t bkt_idx = global_idx >> max_bucket_shift;

        __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
        bucket_values_uintptr = reinterpret_cast<uint64_t>(bucket->vectors);

        // 3、遍历桶查找key
        OccupyResult result = find_without_lock<K, V, S>(
            bucket, key, key_pos, target_digests, target_pos, bucket_capacity);
        found = (result == OccupyResult::DUPLICATE);

        // 4、设置输出元信息
        if (found) {
          if (scores != nullptr) {
            scores[kv_idx] =
                *(BUCKET::scores(bucket->keys_, bucket_capacity, target_pos));
          }
        }
      }
      if (founds != nullptr) {
        founds[kv_idx] = found;
      }
    }

    // 5、协程组协作搬运value（从bucket拷贝到输出缓冲区）
    auto cg_rank_id = threadIdx.x % group_size;
    for (int32_t i = 0; i < group_size; i++) {
      bool found_sync = asc_shfl(found, i, group_size);
      if (found_sync) {
        auto kv_idx_sync = asc_shfl(kv_idx, i, group_size);
        auto target_pos_sync = asc_shfl(target_pos, i, group_size);
        uint64_t bucket_values_sync =
            asc_shfl(bucket_values_uintptr, i, group_size);
        auto value_start = kv_idx_sync * dim;
        auto bucket_value_start = target_pos_sync * dim;
        for (uint32_t j = cg_rank_id; j < dim; j += group_size) {
          values_vec[value_start + j] =
              __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                    L1CacheType::NON_CACHEABLE>(
                  reinterpret_cast<__gm__ VecV*>(bucket_values_sync) +
                  bucket_value_start + j);
        }
      }
    }
  }
}

template <typename K, typename V, typename S>
__global__ __vector__ void find_with_digest_kernel(
    __gm__ Bucket<K, V, S>* buckets, uint64_t capacity, uint64_t buckets_num,
    uint32_t bucket_capacity, uint32_t dim, __gm__ K* keys, __gm__ V* values,
    __gm__ S* scores, __gm__ bool* founds, uint64_t n, uint64_t global_epoch,
    uint32_t value_size, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift,
    uint64_t n_align_warp, uint32_t group_size) {
  const uint64_t thread_all = THREAD_NUM_512 * GetBlockNum();

  DISPATCH_VALUE_SIZE(
      value_size, (asc_vf_call<find_with_digest_kernel_vf<K, V, S, DTYPE>>(
                      dim3{static_cast<uint32_t>(THREAD_NUM_512), 1, 1},
                      buckets, capacity, buckets_num, bucket_capacity, dim,
                      keys, values, scores, founds, n, thread_all, global_epoch,
                      GetBlockIdx(), max_bucket_shift, capacity_divisor_magic,
                      capacity_divisor_shift, n_align_warp, group_size)));
}

template <typename TaskQueue, typename K = uint64_t, typename V = float,
          typename S = uint64_t>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_512) inline void find_with_digest_kernel_with_mte_vf(
    __gm__ Bucket<K, V, S>* buckets, uint64_t capacity,
    uint32_t bucket_capacity, uint32_t dim, __gm__ K* keys, __gm__ V* values,
    __gm__ S* scores, __gm__ bool* founds, uint64_t n,
    const uint64_t total_thread_num, uint32_t block_id,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift, uint64_t n_align_warp) {
  if (buckets == nullptr || keys == nullptr || values == nullptr) {
    return;
  }

  TaskQueue value_task_queue;
  using BUCKET = Bucket<K, V, S>;
  for (uint64_t kv_idx = block_id * blockDim.x + threadIdx.x;
       kv_idx < n_align_warp; kv_idx += total_thread_num) {
    VecD_Comp target_digests{0};
    bool found = false;
    uint32_t target_pos = INVALID_KEY_POS;
    __gm__ V* bucket_values = nullptr;

    if (kv_idx < n) {
      const K key = ldg_l2nc_l1c(keys + kv_idx);
      if (!IS_RESERVED_KEY<K>(key)) {
        const K hashed_key = Murmur3HashDevice(key);
        target_digests = digests_from_hashed<K>(hashed_key);
        uint64_t global_idx = get_global_idx(hashed_key, capacity_divisor_magic,
                                             capacity_divisor_shift, capacity);
        uint32_t key_pos = global_idx & (bucket_capacity - 1);
        uint64_t bkt_idx = global_idx >> max_bucket_shift;

        __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
        bucket_values = bucket->vectors;
        OccupyResult result = find_without_lock<K, V, S>(
            bucket, key, key_pos, target_digests, target_pos, bucket_capacity);
        found = (result == OccupyResult::DUPLICATE);
        if (found && scores != nullptr) {
          scores[kv_idx] =
              *(BUCKET::scores(bucket->keys_, bucket_capacity, target_pos));
        }
      }
      if (founds != nullptr) {
        founds[kv_idx] = found;
      }
    }

    uint32_t warp_acquire_mask = value_task_queue.get_acquire_mask(found);
    if (warp_acquire_mask == 0) {
      continue;
    }

    uint32_t curr_simt_alloc_pos = 0;
    uint32_t new_simt_alloc_pos = 0;
    uint32_t task_id = 0;
    uint64_t output_value_addr =
        reinterpret_cast<uint64_t>(values + kv_idx * dim);
    uint64_t table_value_addr =
        reinterpret_cast<uint64_t>(bucket_values + target_pos * dim);
    value_task_queue.producer_acquire(warp_acquire_mask, curr_simt_alloc_pos,
                                      new_simt_alloc_pos, task_id);
    value_task_queue.memcpy_async(
        found, task_id, reinterpret_cast<void*>(output_value_addr),
        reinterpret_cast<const void*>(table_value_addr), 0);
    value_task_queue.producer_commit(curr_simt_alloc_pos, new_simt_alloc_pos);
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    value_task_queue.producer_finish();
  }
}

template <typename K, typename V, typename S>
__global__ __vector__ void find_with_digest_kernel_with_mte(
    __gm__ Bucket<K, V, S>* buckets, uint64_t capacity,
    uint32_t bucket_capacity, uint32_t dim, __gm__ K* keys, __gm__ V* values,
    __gm__ S* scores, __gm__ bool* founds, uint64_t n,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift, uint64_t n_align_warp) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  using TaskQueue = GM2GMTaskPipelineQueueBase<THREAD_NUM_512>;
  if ASCEND_IS_AIV {
    TaskQueue value_task_queue;
    value_task_queue.init();
    pipe_barrier(PIPE_ALL);

    const uint64_t thread_all = THREAD_NUM_512 * GetBlockNum();
    asc_vf_call<find_with_digest_kernel_with_mte_vf<TaskQueue, K, V, S>>(
        dim3{static_cast<uint32_t>(THREAD_NUM_512), 1, 1}, buckets, capacity,
        bucket_capacity, dim, keys, values, scores, founds, n, thread_all,
        GetBlockIdx(), max_bucket_shift, capacity_divisor_magic,
        capacity_divisor_shift, n_align_warp);

    value_task_queue.run_task(dim * sizeof(V));
  }
}
}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_FIND_WITH_DIGEST_KERNEL_H_
