/**
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

#ifndef ASCENDC_REHASH_KERNEL_H_
#define ASCENDC_REHASH_KERNEL_H_

#include "../../include/score_functor.h"
#include "../../include/types.h"
#include "../../include/utils.h"
#include "kernel_operator.h"

namespace npu {
namespace hkv {
using namespace AscendC;

template <class K, class V, class S, bool IS_FAST_MODE>
__inline__ __device__ void copy_key_to_new_bucket(
    const K& key, const S& score, __gm__ V* __restrict__ vector,
    __gm__ Bucket<K, V, S>* __restrict__ new_bucket,
    __gm__ int32_t* __restrict__ new_bucket_size, const uint32_t& new_key_pos,
    const size_t& bucket_max_size, const size_t& dim,
    const uint64_t& old_bkt_idx, const uint32_t& old_key_pos,
    __gm__ V* __gm__* d_dst_values, __gm__ uint32_t* d_bkt_src_pos, __gm__ uint32_t* d_bkt_dst_pos) {
  // 1. 遍历新桶，找到空key
  for (uint32_t offset = 0; offset < bucket_max_size; offset++) {
    uint32_t cur_pos = (new_key_pos + offset) % bucket_max_size;
    // 2. 复制旧桶key
    if (new_bucket->keys_[cur_pos] == EMPTY_KEY) {
      new_bucket->keys_[cur_pos] = key;
      new_bucket->scores_[cur_pos] = score;
      *Bucket<K, V, S>::digests(new_bucket->keys_, bucket_max_size, cur_pos) =
          get_digest<K>(key);
      (*new_bucket_size)++;
      size_t vector_start_pos = cur_pos * dim;
      if constexpr (IS_FAST_MODE) {
        for (size_t i = 0; i < dim; i++) {
          new_bucket->vectors[vector_start_pos + i] = vector[i];
        }
      } else {
        // old_key_pos -> new_bucket
        uint32_t origin_pos =
            d_bkt_src_pos[old_bkt_idx * bucket_max_size + old_key_pos];
        d_bkt_src_pos[old_bkt_idx * bucket_max_size + old_key_pos] =
            old_key_pos;

        d_bkt_dst_pos[old_bkt_idx * bucket_max_size + origin_pos] = origin_pos;
        d_dst_values[old_bkt_idx * bucket_max_size + origin_pos] = new_bucket->vectors + vector_start_pos;
      }
      break;
    }
  }
}

/* 压缩的目的是让真实存储位置key_pos更靠近理想位置，即find过程中的offset更少
 * 因此，如下3中场景符合压缩条件（其中，*为理想位置，[]为空位，|为当前位置）
 * 1. -----*-----[]-----|-----
 * 2. -----|-----*-----[]-----
 * 3. -----[]-----|-----*-----
 * []与|位置替换后，由*->|的距离缩小
 */
template <class K, class V, class S, bool IS_FAST_MODE>
__inline__ __device__ void defragmentation_for_rehash(
    __gm__ Bucket<K, V, S>* __restrict__ cur_bucket, uint64_t move_offset_pos,
    uint32_t remove_pos, const size_t& bucket_max_size,
    const size_t& old_buckets_num, const size_t& dim,
    __gm__ V* __gm__* d_dst_values, __gm__ uint32_t* d_bkt_src_pos, __gm__ uint32_t* d_bkt_dst_pos, const uint64_t& bkt_idx) {
  // 1. 从空位置的后一个key开始遍历整个桶
  size_t offset = 1;
  while (offset < bucket_max_size) {
    size_t cur_pos = (remove_pos + offset) % bucket_max_size;
    K cur_key = cur_bucket->keys_[cur_pos];
    // 2. 理想位置要在空位置前面，则key一定是连续有效的，不能为空key
    if (cur_key == EMPTY_KEY) {
      break;
    }

    // 3. 计算理想位置
    K hashed_key = Murmur3HashDevice(cur_key);
    uint64_t global_idx =
        static_cast<uint64_t>(hashed_key % (old_buckets_num * bucket_max_size));
    size_t start_pos = global_idx % bucket_max_size;
    if ((start_pos <= remove_pos && remove_pos < cur_pos) ||
        (cur_pos < start_pos && start_pos <= remove_pos) ||
        (remove_pos < cur_pos && cur_pos < start_pos)) {
      // 4. 找到符合场景，将目标key前移，后将目标key位置置空
      cur_bucket->keys_[remove_pos] = cur_key;
      cur_bucket->scores_[remove_pos] = cur_bucket->scores_[cur_pos];
      *Bucket<K, V, S>::digests(cur_bucket->keys_, bucket_max_size,
                                remove_pos) = get_digest<K>(cur_key);
      if constexpr (IS_FAST_MODE) {
        for (size_t i = 0; i < dim; i++) {
          cur_bucket->vectors[remove_pos * dim + i] =
              cur_bucket->vectors[cur_pos * dim + i];
        }
      } else {
        // bucket内会每个key都会位置前移，并可能多次移动，因此记录每个pos的src_pos
        // 本次移动为cur_pos -> remove_pos，cur_pos可能是其他pos的值
        // 实际是origin_pos -> remove_pos
        uint32_t origin_pos = d_bkt_src_pos[move_offset_pos + cur_pos];
        d_bkt_src_pos[move_offset_pos + remove_pos] = origin_pos;
        d_bkt_src_pos[move_offset_pos + cur_pos] = cur_pos;

        d_bkt_dst_pos[move_offset_pos + origin_pos] = remove_pos;
        d_dst_values[move_offset_pos + origin_pos] =
            cur_bucket->vectors + remove_pos * dim;
        // if (origin_pos < remove_pos) {
        //   AscendC::Simt::printf("forward bkt: %lu, cur_pos: %u, remove_pos: %u, origin_pos: %u\r\n", bkt_idx, cur_pos, remove_pos, origin_pos);
        // }
      }

      cur_bucket->keys_[cur_pos] = EMPTY_KEY;
      *Bucket<K, V, S>::digests(cur_bucket->keys_, bucket_max_size, cur_pos) =
          empty_digest<K>();

      // 5. 当前位置变为新的空位置，重新循环
      remove_pos = cur_pos;
      offset = 1;
    } else {
      offset++;
    }
  }
}

template <typename K, typename V, typename S, bool IS_FAST_MODE>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_512) inline void rehash_kernel_vf(
    __gm__ Table<K, V, S>* __restrict__ table, const size_t old_buckets_num,
    __gm__ V* __gm__* d_dst_values, __gm__ uint32_t* d_bkt_src_pos,
    __gm__ uint32_t* d_bkt_dst_pos, const uint64_t thread_all,
    const uint32_t block_index) {
  const size_t new_buckets_num = table->buckets_num;
  const size_t bucket_max_size = table->bucket_max_size;
  const size_t dim = table->dim;

  if constexpr (!IS_FAST_MODE) {
    const size_t old_capacity = table->capacity / 2;
    for (uint64_t tid = block_index * blockDim.x + threadIdx.x;
         tid < old_capacity; tid += thread_all) {
      d_dst_values[tid] = nullptr;
      d_bkt_src_pos[tid] = tid % bucket_max_size;
      d_bkt_dst_pos[tid] = tid % bucket_max_size;
    }
  }

  K cur_key = 0;
  S cur_score = 0;
  K hashed_key = 0;
  uint64_t new_global_idx = 0;
  uint64_t bkt_idx = block_index * blockDim.x + threadIdx.x;
  for (; bkt_idx < old_buckets_num; bkt_idx += thread_all) {
    // 1. 每个线程处理一个桶
    __gm__ Bucket<K, V, S>* cur_bucket = table->buckets + bkt_idx;
    // 2. 遍历桶内key，进行rehash
    uint32_t key_pos = 0;
    while (key_pos < bucket_max_size) {
      cur_key = cur_bucket->keys_[key_pos];
      cur_score = cur_bucket->scores_[key_pos];
      if ((cur_key == EMPTY_KEY) || (cur_key == RECLAIM_KEY)) {
        key_pos++;
        continue;
      }

      // 3. rehash非空key值
      hashed_key = Murmur3HashDevice(cur_key);
      new_global_idx = static_cast<uint64_t>(
          hashed_key % (new_buckets_num * bucket_max_size));
      uint64_t new_bkt_idx = new_global_idx / bucket_max_size;
      if (new_bkt_idx == bkt_idx) {
        key_pos++;
        continue;
      }

      // 4. 搬运key
      uint32_t new_key_pos = new_global_idx % bucket_max_size;
      copy_key_to_new_bucket<K, V, S, IS_FAST_MODE>(
          cur_key, cur_score, (cur_bucket->vectors + key_pos * dim),
          table->buckets + new_bkt_idx, table->buckets_size + new_bkt_idx,
          new_key_pos, bucket_max_size, dim, bkt_idx, key_pos, d_dst_values, d_bkt_src_pos, d_bkt_dst_pos);
      cur_bucket->keys_[key_pos] = EMPTY_KEY;
      *Bucket<K, V, S>::digests(cur_bucket->keys_, bucket_max_size, key_pos) =
          empty_digest<K>();
      table->buckets_size[bkt_idx]--;

      // 5. 压缩碎片化分布的key。由于key被重排列，因此重新遍历桶。
      defragmentation_for_rehash<K, V, S, IS_FAST_MODE>(
          cur_bucket, bkt_idx * bucket_max_size, key_pos, bucket_max_size,
          old_buckets_num, dim, d_dst_values, d_bkt_src_pos, d_bkt_dst_pos, bkt_idx);
      key_pos = 0;
    }
  }
}

template <class K, class V, class S, bool IS_FAST_MODE>
__global__ __vector__ void rehash_kernel(__gm__ Table<K, V, S>* table_gm,
                                         const size_t old_buckets_num,
                                         __gm__ V* __gm__* d_dst_values,
                                         __gm__ uint32_t* d_bkt_src_pos,
                                         __gm__ uint32_t* d_bkt_dst_pos) {
  __gm__ Table<K, V, S>* __restrict__ table =
      reinterpret_cast<__gm__ Table<K, V, S>*>(table_gm);
  if ((table == nullptr) || (table->buckets == nullptr) ||
      (table->buckets_size == nullptr)) {
    return;
  }

  const uint64_t thread_all = THREAD_NUM_512 * GetBlockNum();

  Simt::VF_CALL<rehash_kernel_vf<K, V, S, IS_FAST_MODE>>(
      Simt::Dim3{THREAD_NUM_512}, table, old_buckets_num, d_dst_values,
      d_bkt_src_pos, d_bkt_dst_pos, thread_all, GetBlockIdx());
}

template <class K, class V, class S>
__global__ __vector__ void rehash_write_kernel(
    uint32_t former_num, uint64_t former_core_move_num,
    uint64_t tail_core_move_num, uint32_t tile_size, uint32_t num_tiles,
    uint32_t dim, uint64_t n, __gm__ Bucket<K, V, S>* __restrict__ buckets,
    uint32_t bucket_max_size, __gm__ V* __gm__* d_dst_values,
    __gm__ uint32_t* d_bkt_dst_pos) {
  uint64_t cur_block_idx = GetBlockIdx();
  uint64_t core_start_idx = 0;
  uint64_t core_move_count = 0;

  if (cur_block_idx < former_num) {
    core_start_idx = cur_block_idx * former_core_move_num;
    core_move_count = former_core_move_num;
  } else {
    core_start_idx = former_num * former_core_move_num +
                     (cur_block_idx - former_num) * tail_core_move_num;
    core_move_count = tail_core_move_num;
  }

  AscendC::TPipe pipe;
  AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 2> move_queue;

  pipe.InitBuffer(move_queue, DOUBLE_BUFFER, dim * sizeof(V));
  AscendC::GlobalTensor<V> moving_src_values_gm;
  AscendC::GlobalTensor<V> moving_dst_values_gm;
  AscendC::LocalTensor<V> move_local;
  AscendC::LocalTensor<V> depend_local;
  DataCopyPadExtParams<V> pad_params{true, 0, 0, 0};
  for (uint64_t i = core_start_idx; i < core_start_idx + core_move_count; i++) {
    // 每个核处理若干bucket
    uint64_t offset = i * bucket_max_size;
    auto bucket = buckets + i;
    // 桶内循环
    for (uint32_t key_idx = 0; key_idx < bucket_max_size; key_idx++) {
      __gm__ V* dst_value = reinterpret_cast<__gm__ V*>(
          ReadGmByPassDCache<uint64_t>(reinterpret_cast<__gm__ uint64_t*>(
              d_dst_values + key_idx + offset)));
      if (dst_value == nullptr) {
        continue;
      }
      // 不为空，判断是否有前置依赖
      uint32_t moving_key_idx = key_idx;
      move_local = move_queue.AllocTensor<V>();
      DataCopyExtParams copy_params{1, static_cast<uint32_t>(dim * sizeof(V)),
                                    0, 0, 0};
      moving_src_values_gm.SetGlobalBuffer(bucket->vectors +
                                           moving_key_idx * dim);
      AscendC::DataCopyPad(move_local, moving_src_values_gm, copy_params,
                           pad_params);
      move_queue.EnQue<V>(move_local);

      // 这里depend_key_idx可能为moving_key_idx本身，表明本身是最后一次搬运，搬运至新桶
      uint32_t depend_key_idx =
          ReadGmByPassDCache<uint32_t>(d_bkt_dst_pos + moving_key_idx + offset);
      auto depend_key_dst_value = reinterpret_cast<__gm__ V*>(
          ReadGmByPassDCache<uint64_t>(reinterpret_cast<__gm__ uint64_t*>(
              d_dst_values + depend_key_idx + offset)));
      // 确保当前value被搬进UB
      move_local = move_queue.DeQue<V>();
      while (depend_key_dst_value != nullptr && depend_key_idx != moving_key_idx) {
        // 有前置依赖，则搬运前置依赖
        moving_dst_values_gm.SetGlobalBuffer(dst_value);
        depend_local = move_queue.AllocTensor<V>();
        AscendC::DataCopyPad(depend_local, moving_dst_values_gm, copy_params,
                             pad_params);
        move_queue.EnQue<V>(depend_local);
        // 确保前置依赖value先被搬进UB
        depend_local = move_queue.DeQue<V>();
        // 完成当前搬运，将前置依赖value赋值给当前value
        AscendC::DataCopyPad(moving_dst_values_gm, move_local, copy_params);
        move_queue.FreeTensor(move_local);
        WriteGmByPassDCache<uint64_t>(
            reinterpret_cast<__gm__ uint64_t*>(d_dst_values + moving_key_idx +
                                               offset),
            (uint64_t)0);

        moving_key_idx = depend_key_idx;
        move_local = depend_local;
        dst_value = depend_key_dst_value;
        depend_key_idx = ReadGmByPassDCache<uint32_t>(d_bkt_dst_pos +
                                                      moving_key_idx + offset);
        depend_key_dst_value = reinterpret_cast<__gm__ V*>(
            ReadGmByPassDCache<uint64_t>(reinterpret_cast<__gm__ uint64_t*>(
                d_dst_values + depend_key_idx + offset)));
      }
      // 没有前置依赖了，直接搬运
      moving_dst_values_gm.SetGlobalBuffer(dst_value);
      AscendC::DataCopyPad(moving_dst_values_gm, move_local, copy_params);
      move_queue.FreeTensor(move_local);
      WriteGmByPassDCache<uint64_t>(reinterpret_cast<__gm__ uint64_t*>(
                                        d_dst_values + moving_key_idx + offset),
                                    (uint64_t)0);
    }
  }
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_REHASH_KERNEL_H_
