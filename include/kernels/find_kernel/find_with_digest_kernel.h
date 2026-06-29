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

#include <cstdint>
#include <kernel_operator.h>
#include "find_utils.h"
#include "mte_task_queue.h"
#include "simt_vf_dispatcher.h"

namespace npu {
namespace hkv {
using namespace AscendC;

template <uint32_t RING_BUFFER_SZ = 512>
class FindValueTaskQueue : public MTETaskQueue<RING_BUFFER_SZ> {
public:
  __aicore__ FindValueTaskQueue() {}

  __simt_callee__ inline void drain(const uint32_t& new_task_end_id) {
    while (true) {
      uint32_t mte_finish_id =
          *reinterpret_cast<__ubuf__ volatile uint32_t*>(
              MTETaskQueue<RING_BUFFER_SZ>::MTE_FINISH_ID_ADDR);
      if (new_task_end_id - mte_finish_id <= RING_BUFFER_SZ) {
        break;
      }
    }
  }

  __simt_callee__ inline void assign_warp_task(const uint64_t& in_ddr,
                                               const uint64_t& out_ddr,
                                               const bool& need_assign) {
    uint32_t write_mask = __ballot(need_assign);
    uint32_t warp_write_count = __popc(write_mask);
    if (warp_write_count == 0) {
      return;
    }

    uint32_t curr_simt_alloc_pos = 0;
    if (laneid() == 0) {
      curr_simt_alloc_pos =
          MTETaskQueue<RING_BUFFER_SZ>::apply_ID(warp_write_count);
    }
    curr_simt_alloc_pos = __shfl(curr_simt_alloc_pos, 0);

    uint32_t new_simt_alloc_pos = curr_simt_alloc_pos + warp_write_count;
    drain(new_simt_alloc_pos);

    if (need_assign) {
      uint32_t offset_in_warp = __popc(write_mask & lanemask_lt());
      uint32_t my_slot_id =
          (curr_simt_alloc_pos + offset_in_warp) &
          MTETaskQueue<RING_BUFFER_SZ>::RING_BUFFER_MASK;
      MTETaskQueue<RING_BUFFER_SZ>::fill_task_info(my_slot_id, in_ddr,
                                                   out_ddr);
    }

    __threadfence();

    if (laneid() == 0) {
      MTETaskQueue<RING_BUFFER_SZ>::assign_task(curr_simt_alloc_pos,
                                                new_simt_alloc_pos);
    }
  }
};

template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          typename VecV = int32_t>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_512) inline void find_with_digest_kernel_vf(
    FindValueTaskQueue<THREAD_NUM_512> find_task_queue,
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

    // 5. Queue hit value copy from bucket storage to output buffer.
    uint64_t value_base_addr = 0;
    uint64_t bucket_value_base_addr = 0;
    if (found) {
      value_base_addr = reinterpret_cast<uint64_t>(
          reinterpret_cast<__gm__ VecV*>(values) + kv_idx * dim);
      bucket_value_base_addr =
          bucket_values_uintptr + target_pos * dim * sizeof(VecV);
    }
    find_task_queue.assign_warp_task(bucket_value_base_addr, value_base_addr,
                                     found);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    find_task_queue.assignment_finish();
  }
}

template <typename K, typename V, typename S>
__global__ __vector__ void find_with_digest_kernel(
    __gm__ Bucket<K, V, S>* buckets, uint64_t capacity, uint64_t buckets_num,
    uint32_t bucket_capacity, uint32_t dim, __gm__ K* keys, __gm__ V* values,
    __gm__ S * scores, __gm__ bool* founds, uint64_t n, uint64_t global_epoch, uint32_t value_size,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift,
    uint64_t n_align_warp, uint32_t group_size) {

  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  if ASCEND_IS_AIV {
    if (buckets == nullptr || keys == nullptr || values == nullptr) {
      return;
    }

    FindValueTaskQueue<THREAD_NUM_512> find_task_queue;
    find_task_queue.init();

    pipe_barrier(PIPE_ALL);

    const uint64_t thread_all = THREAD_NUM_512 * GetBlockNum();

    DISPATCH_VALUE_SIZE(
        value_size,
        (Simt::VF_CALL<find_with_digest_kernel_vf<K, V, S, DTYPE>>(
            Simt::Dim3{static_cast<uint32_t>(THREAD_NUM_512)}, find_task_queue,
            buckets, capacity, buckets_num, bucket_capacity, dim, keys, values,
            scores, founds, n, thread_all, global_epoch, GetBlockIdx(),
            max_bucket_shift, capacity_divisor_magic, capacity_divisor_shift,
            n_align_warp, group_size)));

    uint32_t value_dim_size = dim * value_size;
    find_task_queue.run_task(value_dim_size);
  }
}
}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_FIND_WITH_DIGEST_KERNEL_H_
