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

/* !
 * \file find_or_insert_ptr_kernel.h
 * \brief find_or_insert_ptr_kernel
 */

#ifndef ASCENDC_DUMP_KERNEL_H_
#define ASCENDC_DUMP_KERNEL_H_

#include <kernel_operator.h>
#include <cstdint>
#include "simt_api/asc_simt.h"
#include "simt_vf_dispatcher.h"
#include "types.h"
#include "utils.h"
#include "mte_task_queue.h"

namespace npu {
namespace hkv {

using namespace AscendC;

// MTE搬运任务队列，默认仅支持GM->GM任务
// RING_BUFFER_SZ 必须是2的整数次幂，且必须大于等于8，否则内存不按32对齐
template <uint32_t RING_BUFFER_SZ = 512>
class DumpTaskQueue : public MTETaskQueue<RING_BUFFER_SZ> {
public:
  __aicore__ DumpTaskQueue(__gm__ size_t *dump_count_gm,
    __gm__ uint8_t *d_val_gm,
    __gm__ uint8_t *d_score_gm,
    __gm__ uint8_t *d_key_gm) : dump_count_gm(dump_count_gm), d_val_gm(d_val_gm), d_score_gm(d_score_gm), d_key_gm(d_key_gm) {}

  __aicore__ void init() override {
    *reinterpret_cast<__ubuf__ volatile uint32_t*>(MTETaskQueue<RING_BUFFER_SZ>::TASK_FINISH_FLAG_ADDR) = 0;
    *reinterpret_cast<__ubuf__ volatile uint32_t*>(BLOCK_ACC_ADDR) = 0;
    *reinterpret_cast<__ubuf__ volatile uint32_t*>(BLOCK_FINISH_FLAG_ADDR) = 1;
    *reinterpret_cast<__ubuf__ volatile uint32_t*>(BLOCK_TASK_NUM_ADDR) = 0;
    *reinterpret_cast<__ubuf__ volatile size_t*>(BLOCK_TASK_GLOBAL_OFFSET_ADDR) = 0;
  }

  __simt_callee__ inline void assign_warp_task(const uint64_t &in_ddr, const uint64_t &key, const uint64_t &score, const bool &need_assign) {
    // 1. 统计warp任务数
    uint32_t write_mask = __ballot(need_assign);
    uint32_t warp_write_count = __popc(write_mask);

    // 2. 统计block任务数
    if (threadIdx.x == 0) {
      *reinterpret_cast<__ubuf__ volatile uint32_t*>(BLOCK_ACC_ADDR) = 0;
    }

    asc_syncthreads();

    uint32_t block_offset = 0;
    if (warp_write_count > 0 && laneid() == 0) {
      block_offset = asc_atomic_add(reinterpret_cast<__ubuf__ uint32_t*>(BLOCK_ACC_ADDR), warp_write_count);
    }
    block_offset = __shfl(block_offset, 0);

    asc_syncthreads();

    uint32_t block_task_num = *reinterpret_cast<__ubuf__ volatile uint32_t*>(BLOCK_ACC_ADDR);
    // 3. 统计全局任务偏移数
    size_t global_task_offset = 0;
    if (threadIdx.x == 0) {
      global_task_offset = asc_atomic_add(dump_count_gm, block_task_num);
      // 6. 等待任务空闲
      while (*reinterpret_cast<__ubuf__ volatile uint32_t*>(BLOCK_FINISH_FLAG_ADDR) == !MTETaskQueue<RING_BUFFER_SZ>::FINISH_FLAG) {}
    }

    asc_syncthreads();

    if (need_assign) {
      // 4. 统计warp内任务偏移数
      uint32_t warp_offset = __popc(write_mask & lanemask_lt());

      // 5. 获取当前线程任务总ID
      uint32_t cur_task_id = block_offset + warp_offset;

      // 7. 填充任务信息
      cur_task_id &= MTETaskQueue<RING_BUFFER_SZ>::RING_BUFFER_MASK;
      reinterpret_cast<__ubuf__ volatile uint64_t*>(MTETaskQueue<RING_BUFFER_SZ>::RING_BUFFER_IN_ADDR_QUEUE_ADDR)[cur_task_id] = in_ddr;
      reinterpret_cast<__ubuf__ volatile uint64_t*>(RING_BUFFER_KEY_ADDR)[cur_task_id] = key;
      if (d_score_gm != nullptr) {
        reinterpret_cast<__ubuf__ volatile uint64_t*>(RING_BUFFER_SCORE_ADDR)[cur_task_id] = score;
      }
    }

    // 8. 下发任务
    asc_syncthreads();

    if (threadIdx.x == 0) {
      *reinterpret_cast<__ubuf__ volatile uint32_t*>(BLOCK_TASK_NUM_ADDR) = block_task_num;
      *reinterpret_cast<__ubuf__ volatile size_t*>(BLOCK_TASK_GLOBAL_OFFSET_ADDR) = global_task_offset;
      __threadfence();
      *reinterpret_cast<__ubuf__ volatile uint32_t*>(BLOCK_FINISH_FLAG_ADDR) = 0;
    }
  }

  __aicore__ inline void run_task(const uint32_t &value_size) override {
    int value_size_aligned = (value_size + UB_ALIGN_SIZE - 1) / UB_ALIGN_SIZE * UB_ALIGN_SIZE;
    uint32_t max_proc_batch = MTETaskQueue<RING_BUFFER_SZ>::DATA_BUFFER_SZ / (value_size_aligned);
    max_proc_batch = min(max_proc_batch, RING_BUFFER_SZ);

    __ubuf__ volatile uint32_t* block_finish_flag_addr = reinterpret_cast<__ubuf__ volatile uint32_t*>(BLOCK_FINISH_FLAG_ADDR);
    __ubuf__ volatile uint32_t* task_finish_flag_addr = reinterpret_cast<__ubuf__ volatile uint32_t*>(MTETaskQueue<RING_BUFFER_SZ>::TASK_FINISH_FLAG_ADDR);
    __ubuf__ volatile uint64_t* ring_buffer_in_addr_queue_addr = reinterpret_cast<__ubuf__ volatile uint64_t*>(MTETaskQueue<RING_BUFFER_SZ>::RING_BUFFER_IN_ADDR_QUEUE_ADDR);
    __ubuf__ volatile uint32_t* block_task_num_addr = reinterpret_cast<__ubuf__ volatile uint32_t*>(BLOCK_TASK_NUM_ADDR);
    __ubuf__ volatile size_t* block_task_global_offset_addr = reinterpret_cast<__ubuf__ volatile size_t*>(BLOCK_TASK_GLOBAL_OFFSET_ADDR);
    uint32_t mte_finish_id = 0;
    uint32_t block_finish_flag = 0;
    while (true) {
      block_finish_flag = *block_finish_flag_addr;
      if (block_finish_flag == MTETaskQueue<RING_BUFFER_SZ>::FINISH_FLAG) {
        if (*task_finish_flag_addr == MTETaskQueue<RING_BUFFER_SZ>::FINISH_FLAG) {
          return;
        } else {
          continue;
        }
      }

      uint32_t block_task_num = *block_task_num_addr;
      size_t block_task_global_offset = *block_task_global_offset_addr;

      for (uint32_t i = 0; i < block_task_num; i += max_proc_batch) {
        uint32_t cur_batch = min(max_proc_batch, block_task_num - i);
        for (uint32_t j = 0; j < cur_batch; j++) {
          copy_gm_to_ubuf_align_v2(
            reinterpret_cast<__ubuf__ uint8_t*>(MTETaskQueue<RING_BUFFER_SZ>::DATA_BUFFER_START_ADDR) + j * value_size_aligned,
            reinterpret_cast<__gm__ uint8_t*>(ring_buffer_in_addr_queue_addr[i * max_proc_batch + j]),
            0, 1, value_size, 0, 0, 0, 0, 0, 0);
        }

        set_flag(PIPE_MTE2, PIPE_MTE3, 0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, 0);

        copy_ubuf_to_gm_align_v2(
          reinterpret_cast<__gm__ uint8_t*>(d_val_gm + (block_task_global_offset + i * max_proc_batch) * value_size),
          reinterpret_cast<__ubuf__ uint8_t*>(MTETaskQueue<RING_BUFFER_SZ>::DATA_BUFFER_START_ADDR),
          0, cur_batch, value_size, 0, 0, 0);
        
        copy_ubuf_to_gm_align_v2(
          reinterpret_cast<__gm__ uint8_t*>(d_key_gm + (block_task_global_offset + i * max_proc_batch) * sizeof(uint64_t)),
          reinterpret_cast<__ubuf__ uint8_t*>(RING_BUFFER_KEY_ADDR),
          0, 1, cur_batch * sizeof(uint64_t), 0, 0, 0);

        if (d_score_gm != nullptr) {
          copy_ubuf_to_gm_align_v2(
            reinterpret_cast<__gm__ uint8_t*>(d_score_gm + (block_task_global_offset + i * max_proc_batch) * sizeof(uint64_t)),  
            reinterpret_cast<__ubuf__ uint8_t*>(RING_BUFFER_SCORE_ADDR),
            0, 1, cur_batch * sizeof(uint64_t), 0, 0, 0);
        }

        set_flag(PIPE_MTE3, PIPE_S, 0);
        wait_flag(PIPE_MTE3, PIPE_S, 0);
      }

      // 消费完成，累加消费进展
      *block_finish_flag_addr = MTETaskQueue<RING_BUFFER_SZ>::FINISH_FLAG;
    }
  }

  __aicore__ inline void assignment_finish() {
    *reinterpret_cast<__ubuf__ volatile uint32_t*>(MTETaskQueue<RING_BUFFER_SZ>::TASK_FINISH_FLAG_ADDR) = MTETaskQueue<RING_BUFFER_SZ>::FINISH_FLAG;
  }

private:
  // 队列数据
  static constexpr uint32_t RING_BUFFER_KEY_ADDR = MTETaskQueue<RING_BUFFER_SZ>::DATA_BUFFER_END_ADDR;
  static constexpr uint32_t RING_BUFFER_SCORE_ADDR = RING_BUFFER_KEY_ADDR + RING_BUFFER_SZ * sizeof(uint64_t);

  // 核内累加数
  static constexpr uint32_t BLOCK_ACC_ADDR = RING_BUFFER_SCORE_ADDR + RING_BUFFER_SZ * sizeof(uint64_t);;
  // block任务完成标记
  static constexpr uint32_t BLOCK_FINISH_FLAG_ADDR = BLOCK_ACC_ADDR + UB_ALIGN_SIZE;
  // block任务数
  static constexpr uint32_t BLOCK_TASK_NUM_ADDR = BLOCK_FINISH_FLAG_ADDR + UB_ALIGN_SIZE;
  // block任务起始偏移
  static constexpr uint32_t BLOCK_TASK_GLOBAL_OFFSET_ADDR = BLOCK_TASK_NUM_ADDR + UB_ALIGN_SIZE;

  static_assert(BLOCK_TASK_GLOBAL_OFFSET_ADDR + UB_ALIGN_SIZE <= 248 * 1024 - 32 * 1024 - 8 * 1024, "BLOCK_TASK_GLOBAL_OFFSET_ADDR + UB_ALIGN_SIZE must be <= 248KB - 32KB - 8KB");

  // 全局任务ID计数
  __gm__ size_t *dump_count_gm = nullptr;

  __gm__ uint8_t* d_val_gm = nullptr;

  __gm__ uint8_t* d_key_gm = nullptr;

  __gm__ uint8_t* d_score_gm = nullptr;
};

template <class K, class V, class S>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_512) inline void new_dump_kernel_vf(
    DumpTaskQueue<THREAD_NUM_512> dump_task_queue,
    __gm__ Table<K, V, S>* table, __gm__ Bucket<K, V, S>* buckets,
    const size_t offset, const bool dump_score, const uint32_t dim,
    const size_t search_length, const size_t search_length_align,
    const uint64_t total_thread_num, __gm__ size_t* d_dump_counter,
    uint32_t blockIdx) {
  const size_t bucket_max_size{table->bucket_max_size};
  size_t tid{blockIdx * blockDim.x + threadIdx.x};
  K key = 0;
  S score = 0;
  uint64_t value_ddr = 0;
  for (; tid < search_length_align; tid += total_thread_num) {
    bool dump_flag = false;
    if (tid < search_length) {
      __gm__ Bucket<K, V, S>* bucket =
          buckets + (tid + offset) / bucket_max_size;
      __gm__ K* bucket_keys_ptr = bucket->keys_;
      const int key_idx{static_cast<int>((tid + offset) % bucket_max_size)};
      key = bucket_keys_ptr[key_idx];

      if (!IS_RESERVED_KEY<K>(key)) {
        dump_flag = true;
        value_ddr = reinterpret_cast<uint64_t>(bucket->vectors + key_idx * dim);
        if (dump_score) {
          __gm__ S* bucket_scores_ptr = bucket->scores_;
          score = bucket_scores_ptr[key_idx];
        }
      }
    }
    dump_task_queue.assign_warp_task(value_ddr, key, score, dump_flag);
  }

  asc_syncthreads();

  if (threadIdx.x == 0) {
    dump_task_queue.assignment_finish();
  }
}

template <class K, class V, class S>
__global__ __vector__ void new_dump_kernel(
    __gm__ Table<K, V, S>* table, __gm__ Bucket<K, V, S>* buckets,
    __gm__ K* keys, __gm__ V* vals, __gm__ S* scores, const size_t offset,
    const size_t search_length, const size_t search_length_align,
    __gm__ size_t* dump_counter, uint32_t dim) {
  if ((!table) || (!buckets) || (!keys) || (!vals) || (!dump_counter)) {
    return;
  }

  bool dump_score = (scores != nullptr);

  const uint64_t total_thread_num = THREAD_NUM_512 * GetBlockNum();
  DumpTaskQueue<THREAD_NUM_512> dump_task_queue(dump_counter, reinterpret_cast<__gm__ uint8_t*>(vals),
    reinterpret_cast<__gm__ uint8_t*>(scores), reinterpret_cast<__gm__ uint8_t*>(keys));
  dump_task_queue.init();


  Simt::VF_CALL<new_dump_kernel_vf<K, V, S>>(
    Simt::Dim3{static_cast<uint32_t>(THREAD_NUM_512)}, dump_task_queue, table, buckets,
    offset, dump_score, dim, search_length, search_length_align,
    total_thread_num, dump_counter, GetBlockIdx());

  uint32_t value_size = dim * sizeof(V);
  dump_task_queue.run_task(value_size);
}


template <class K, class V, class S, class VecV>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_2048) inline void dump_kernel_vf(
    __gm__ Table<K, V, S>* table, __gm__ Bucket<K, V, S>* buckets,
    __gm__ K* d_key, __gm__ V* d_val, __gm__ S* d_score, const size_t offset,
    const size_t search_length, const size_t search_length_align,
    const uint64_t total_thread_num, __gm__ size_t* d_dump_counter,
    uint64_t ub_shared_kvs_mem, __ubuf__ uint32_t* block_acc,
    __ubuf__ uint32_t* global_acc, int32_t group_size, uint32_t dim_in,
    uint32_t blockIdx) {
  if ((!table) || (!buckets) || (!d_key) || (!d_val) || (!d_dump_counter)) {
    return;
  }

  auto block_tuples = reinterpret_cast<__ubuf__ KVM<K, S>*>(ub_shared_kvs_mem);

  __gm__ VecV* __restrict__ d_vecv_val = reinterpret_cast<__gm__ VecV*>(d_val);

  const size_t bucket_max_size{table->bucket_max_size};

  size_t tid{blockIdx * blockDim.x + threadIdx.x};
  for (; tid < search_length_align; tid += total_thread_num) {
    if (threadIdx.x == 0) {
      block_acc[0] = 0;
    }
    asc_syncthreads();
    uint32_t choose_flag = 0;
    size_t local_index = 0;
    if (tid < search_length) {
      __gm__ Bucket<K, V, S>* bucket =
          buckets + (tid + offset) / bucket_max_size;
      __gm__ K* bucket_keys_ptr = bucket->keys_;
      __gm__ VecV* bucket_values_ptr =
          reinterpret_cast<__gm__ VecV*>(bucket->vectors);
      __gm__ S* bucket_scores_ptr = bucket->scores_;

      const int key_idx{static_cast<int>((tid + offset) % bucket_max_size)};
      const K key{bucket_keys_ptr[key_idx]};

      if (!IS_RESERVED_KEY<K>(key)) {
        choose_flag = 1;
        local_index = atomicAdd(block_acc, 1u);
        block_tuples[local_index].key = key;
        block_tuples[local_index].value =
            reinterpret_cast<uint64_t>(&bucket_values_ptr[key_idx * dim_in]);
        block_tuples[local_index].score = bucket_scores_ptr[key_idx];
      }
      asc_syncthreads();

      if (threadIdx.x == 0) {
        global_acc[0] =
            atomicAdd(d_dump_counter, static_cast<size_t>(block_acc[0]));
      }
      asc_syncthreads();

      if (block_acc[0] == 0) {
        continue;
      }
    }

    auto cg_rank_id = threadIdx.x % group_size;
    auto cg_rank_id_start = threadIdx.x - cg_rank_id;
    uint64_t tuple_value_ptr = 0;
    __ubuf__ const KVM<K, S>& tuple{block_tuples[local_index]};
    const size_t j{global_acc[0] + local_index};
    if (choose_flag == 1) {
      d_key[j] = tuple.key;

      if (d_score != nullptr) {
        d_score[j] = tuple.score;
      }
    }

    for (int32_t i = 0; i < group_size; i++) {
      auto cg_flag = asc_shfl(choose_flag, i, group_size);
      if (cg_flag == 1) {
        // 协程组并行写入向量值
        auto val_start = asc_shfl(j, i, group_size);
        uint64_t value_sync_ptr = asc_shfl(tuple.value, i, group_size);
        auto value_sync = reinterpret_cast<__gm__ VecV*>(value_sync_ptr);
        for (uint32_t k = cg_rank_id; k < dim_in; k += group_size) {
          __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                L1CacheType::NON_CACHEABLE>(d_vecv_val + val_start * dim_in + k,
                                            value_sync[k]);
        }
      }
    }
  }
}

template <class K, class V, class S>
__global__ __vector__ void dump_kernel(
    __gm__ Table<K, V, S>* table, __gm__ Bucket<K, V, S>* buckets,
    __gm__ K* keys, __gm__ V* vals, __gm__ S* scores, const size_t offset,
    const size_t search_length, const size_t search_length_align,
    __gm__ size_t* dump_counter, uint32_t value_size, int32_t group_size,
    uint32_t dim) {
  const uint64_t total_thread_num = THREAD_NUM_2048 * GetBlockNum();

  AscendC::TPipe pipe;

  AscendC::TBuf<AscendC::TPosition::VECCALC> shared_kvs_mem;
  pipe.InitBuffer(shared_kvs_mem, THREAD_NUM_2048 * sizeof(KVM<K, S>));
  AscendC::LocalTensor<KVM<K, S>> shared_kvs_tensor =
      shared_kvs_mem.Get<KVM<K, S>>();
  AscendC::TBuf<AscendC::TPosition::VECCALC> block_acc;
  pipe.InitBuffer(block_acc, sizeof(uint32_t));
  AscendC::LocalTensor<uint32_t> shared_block_acc_tensor =
      block_acc.Get<uint32_t>();
  __ubuf__ uint32_t* ub_shared_block_acc_mem =
      reinterpret_cast<__ubuf__ uint32_t*>(
          shared_block_acc_tensor.GetPhyAddr());

  AscendC::TBuf<AscendC::TPosition::VECCALC> global_acc;
  pipe.InitBuffer(global_acc, sizeof(uint32_t));
  AscendC::LocalTensor<uint32_t> shared_global_acc_tensor =
      global_acc.Get<uint32_t>();
  __ubuf__ uint32_t* ub_shared_global_acc_mem =
      reinterpret_cast<__ubuf__ uint32_t*>(
          shared_global_acc_tensor.GetPhyAddr());

  DISPATCH_VALUE_SIZE(
      value_size,
      (Simt::VF_CALL<dump_kernel_vf<K, V, S, DTYPE>>(
          Simt::Dim3{static_cast<uint32_t>(THREAD_NUM_2048)}, table, buckets,
          keys, vals, scores, offset, search_length, search_length_align,
          total_thread_num, dump_counter, shared_kvs_tensor.GetPhyAddr(),
          ub_shared_block_acc_mem, ub_shared_global_acc_mem, group_size, dim,
          GetBlockIdx())));
}

template <class K, class V, class S>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_2048) inline void dump_kernel_hybrid_vf(
    __gm__ Table<K, V, S>* table, __gm__ Bucket<K, V, S>* buckets,
    __gm__ K* d_key, __gm__ V* __gm__* dst_val, __gm__ S* d_score,
    const size_t offset, const size_t search_length,
    const uint64_t total_thread_num, __gm__ size_t* d_dump_counter,
    __ubuf__ uint32_t* block_acc, __ubuf__ uint32_t* global_acc,
    uint32_t dim_in, uint32_t blockIdx) {
  if ((!table) || (!buckets) || (!d_key) || (!dst_val) || (!d_dump_counter)) {
    return;
  }

  const size_t bucket_max_size{table->bucket_max_size};
  size_t tid{blockIdx * blockDim.x + threadIdx.x};
  for (; tid < search_length; tid += total_thread_num) {
    if (threadIdx.x == 0) {
      block_acc[0] = 0;
    }
    asc_syncthreads();
    uint32_t choose_flag = 0;
    size_t local_index = 0;
    __gm__ Bucket<K, V, S>* bucket = buckets + (tid + offset) / bucket_max_size;
    __gm__ K* bucket_keys_ptr = bucket->keys_;
    __gm__ V* bucket_values_ptr = bucket->vectors;
    __gm__ S* bucket_scores_ptr = bucket->scores_;

    const int key_idx{static_cast<int>((tid + offset) % bucket_max_size)};
    const K key{bucket_keys_ptr[key_idx]};

    if (!IS_RESERVED_KEY<K>(key)) {
      choose_flag = 1;
      local_index = atomicAdd(block_acc, 1u);
    }
    asc_syncthreads();

    if (threadIdx.x == 0) {
      global_acc[0] =
          atomicAdd(d_dump_counter, static_cast<size_t>(block_acc[0]));
    }
    asc_syncthreads();

    if (choose_flag == 1) {
      const size_t j{global_acc[0] + local_index};
      d_key[j] = key;
      dst_val[j] = (&bucket_values_ptr[key_idx * dim_in]);

      if (d_score != nullptr) {
        d_score[j] = bucket_scores_ptr[key_idx];
      }
    }

    if (block_acc[0] == 0) {
      continue;
    }
  }
}

template <class K, class V, class S>
__global__ __vector__ void dump_kernel_hybrid(
    __gm__ Table<K, V, S>* table, __gm__ Bucket<K, V, S>* buckets,
    __gm__ K* keys, __gm__ V* __gm__* src_vals, __gm__ S* scores,
    const size_t offset, const size_t search_length,
    __gm__ size_t* dump_counter, uint32_t dim) {
  const uint64_t total_thread_num = THREAD_NUM_2048 * GetBlockNum();

  AscendC::TPipe pipe;

  AscendC::TBuf<AscendC::TPosition::VECCALC> block_acc;
  pipe.InitBuffer(block_acc, sizeof(uint32_t));
  AscendC::LocalTensor<uint32_t> shared_block_acc_tensor =
      block_acc.Get<uint32_t>();
  __ubuf__ uint32_t* ub_shared_block_acc_mem =
      reinterpret_cast<__ubuf__ uint32_t*>(
          shared_block_acc_tensor.GetPhyAddr());

  AscendC::TBuf<AscendC::TPosition::VECCALC> global_acc;
  pipe.InitBuffer(global_acc, sizeof(uint32_t));
  AscendC::LocalTensor<uint32_t> shared_global_acc_tensor =
      global_acc.Get<uint32_t>();
  __ubuf__ uint32_t* ub_shared_global_acc_mem =
      reinterpret_cast<__ubuf__ uint32_t*>(
          shared_global_acc_tensor.GetPhyAddr());

  asc_vf_call<dump_kernel_hybrid_vf<K, V, S>>(
      dim3{static_cast<uint32_t>(THREAD_NUM_2048)}, table, buckets, keys,
      src_vals, scores, offset, search_length, total_thread_num, dump_counter,
      ub_shared_block_acc_mem, ub_shared_global_acc_mem, dim, GetBlockIdx());
}

template <class V>
__global__ __vector__ void export_batch_value_kernel(
    uint32_t block_dim, uint32_t dim, uint32_t max_tile_size, __gm__ V* values,
    __gm__ size_t* dump_counter, __gm__ V* __gm__* d_src_values) {
  size_t n = *dump_counter;
  uint64_t tail_core_move_num = n / block_dim;
  uint64_t former_core_move_num = tail_core_move_num + 1;
  uint32_t former_num = n - tail_core_move_num * block_dim;
  uint32_t tile_size = (dim <= max_tile_size) ? dim : max_tile_size;
  uint32_t num_tiles = (dim + tile_size - 1) / tile_size;

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

  TPipe pipe;
  TQueBind<TPosition::VECIN, TPosition::VECOUT, 0> move_queue;
  pipe.InitBuffer(move_queue, DOUBLE_BUFFER, tile_size * sizeof(V));

  GlobalTensor<V> src_values_gm;
  GlobalTensor<V> dst_values_gm;
  LocalTensor<V> move_local;
  DataCopyPadExtParams<V> pad_params{true, 0, 0, 0};

  dst_values_gm.SetGlobalBuffer(values);

  for (uint64_t i = core_start_idx; i < core_start_idx + core_move_count; i++) {
    __gm__ V* src_value = d_src_values[i];
    if (src_value == nullptr) {
      continue;
    }

    uint64_t dst_offset = i * dim;

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      uint32_t current_tile_size = (tile_idx == num_tiles - 1)
                                       ? (dim - tile_idx * tile_size)
                                       : tile_size;
      DataCopyExtParams copy_params{
          1, static_cast<uint32_t>(current_tile_size * sizeof(V)), 0, 0, 0};

      move_queue.AllocTensor<V>(move_local);

      src_values_gm.SetGlobalBuffer(src_value + tile_idx * tile_size);
      DataCopyPad(move_local, src_values_gm, copy_params, pad_params);
      move_queue.EnQue<V>(move_local);
      move_queue.DeQue<V>(move_local);

      DataCopyPad(dst_values_gm[dst_offset + tile_idx * tile_size], move_local,
                  copy_params);

      move_queue.FreeTensor(move_local);
    }
  }
}

template <class V>
__global__ __vector__ void export_batch_value_kernel_half_once(
    uint32_t block_dim, uint32_t dim, uint32_t max_tile_size, __gm__ V* values,
    __gm__ size_t* dump_counter, __gm__ V* __gm__* d_src_values) {
  size_t n = *dump_counter;
  uint64_t tail_core_move_num = n / block_dim;
  uint64_t former_core_move_num = tail_core_move_num + 1;
  uint32_t former_num = n - tail_core_move_num * block_dim;
  uint32_t num_tiles = max_tile_size;

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

  TPipe pipe;
  TQueBind<TPosition::VECIN, TPosition::VECOUT, 0> move_queue;
  pipe.InitBuffer(move_queue, DOUBLE_BUFFER, num_tiles * sizeof(V) * dim);
  GlobalTensor<V> src_values_gm;
  GlobalTensor<V> dst_values_gm;
  LocalTensor<V> move_local;
  DataCopyPadExtParams<V> pad_params{true, 0, 0, 0};

  dst_values_gm.SetGlobalBuffer(values);
  DataCopyExtParams copy_in_params{1, static_cast<uint32_t>(sizeof(V) * dim), 0,
                                   0, 0};
  uint64_t core_max_idx = core_start_idx + core_move_count;
  for (uint64_t i = core_start_idx; i < core_max_idx; i += num_tiles) {
    uint32_t move_in_num_once =
        i + num_tiles > core_max_idx ? core_max_idx - i : num_tiles;
    DataCopyExtParams copy_params{
        1, static_cast<uint32_t>(move_in_num_once * sizeof(V) * dim), 0, 0, 0};
    move_queue.AllocTensor<V>(move_local);
    for (uint32_t tile_idx = 0; tile_idx < move_in_num_once; tile_idx++) {
      __gm__ V* src_value = d_src_values[i + tile_idx];
      if (src_value == nullptr) {
        continue;
      }
      src_values_gm.SetGlobalBuffer(src_value);

      DataCopyPad(move_local[tile_idx * dim], src_values_gm, copy_in_params,
                  pad_params);
      move_queue.EnQue<V>(move_local);
      move_queue.DeQue<V>(move_local);
    }
    DataCopyPad(dst_values_gm[i * dim], move_local, copy_params);
    move_queue.FreeTensor(move_local);
  }
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_DUMP_KERNEL_H_
