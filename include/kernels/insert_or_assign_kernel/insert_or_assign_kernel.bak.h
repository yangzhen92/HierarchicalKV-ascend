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

#ifndef ASCENDC_INSERT_OR_ASSIGN_KERNEL_H_
#define ASCENDC_INSERT_OR_ASSIGN_KERNEL_H_

#include <cstdint>
#include "kernel_operator.h"
#include "score_functor.h"
#include "simt_vf_dispatcher.h"
#include "types.h"
#include "utils.h"

// 任务队列长度512
constexpr int RING_BUFFER_SZ = 512;
// 512取余使用，快速计算队列索引
constexpr int RING_BUFFER_MASK = RING_BUFFER_SZ - 1;
// UB数据buffer大小128K
constexpr int DATA_BUFFER_SZ = 128 * 1024;

// 512个任务，每个任务dst value地址
constexpr int RING_BUFFER_OUT_VALUE_BASE_ADDR = 0;
// 512个任务，每个任务src value地址
constexpr int RING_BUFFER_IN_VALUE_BASE_ADDR =
    RING_BUFFER_OUT_VALUE_BASE_ADDR + RING_BUFFER_SZ * sizeof(uint64_t);
// 512个任务，每个任务dst key地址
constexpr int RING_BUFFER_OUT_KEY_ADDR =
    RING_BUFFER_IN_VALUE_BASE_ADDR + RING_BUFFER_SZ * sizeof(uint64_t);
// 512个任务，每个任务src key地址
constexpr int RING_BUFFER_KEY_ADDR =
    RING_BUFFER_OUT_KEY_ADDR + RING_BUFFER_SZ * sizeof(uint64_t);

// 128K数据buffer
constexpr uint64_t DATA_BUFFER_START_ADDR =
    RING_BUFFER_KEY_ADDR + RING_BUFFER_SZ * sizeof(uint64_t) * 4;

// MTE消费进展
constexpr int SCALAR_READ_POS_PTR = DATA_BUFFER_START_ADDR + DATA_BUFFER_SZ;
// ？？？SIMT生产任务进展
constexpr int SIMT_ALLOC_POS_PTR = SCALAR_READ_POS_PTR + 64;
// SIMT生产进展计数，当前生成了多少个value
constexpr int SIMT_WR_POS_PTR = SIMT_ALLOC_POS_PTR + 64;
// ？？？读入key的位置
// ？？？为啥不用寄存器？ 为啥是4倍？
constexpr int KEY_OUTPUT_RD_POS_PTR = SIMT_WR_POS_PTR + 64;
// ？？？写出key的位置
// ？？？为啥不用寄存器？ 为啥是4倍？
constexpr int KEY_OUTPUT_WR_POS_PTR = KEY_OUTPUT_RD_POS_PTR + 64;

constexpr int WORK_DONE_PTR = KEY_OUTPUT_WR_POS_PTR + 64;

namespace npu {
namespace hkv {
using namespace AscendC;

template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          int32_t Strategy = -1, int32_t GROUP_SIZE = 16,
          uint32_t THREAD_NUM = 512, int32_t EVICT_GROUP_SIZE = 16>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void insert_or_assign_kernel_with_digest_vf(
    GM_ADDR buckets_addr_gm, GM_ADDR buckets_size_addr_gm, uint64_t capacity,
    uint32_t bucket_max_size, uint32_t dim, GM_ADDR keys_addr_gm,
    GM_ADDR values_addr_gm, GM_ADDR scores_gm, S cur_score, uint64_t n,
    uint32_t thread_all, uint64_t global_epoch, uint32_t block_index,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift, uint64_t n_align_warp) {
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
  using BUCKET = Bucket<K, V, S>;
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  __gm__ Bucket<K, V, S>* __restrict__ buckets =
      reinterpret_cast<__gm__ Bucket<K, V, S>*>(buckets_addr_gm);
  __gm__ int32_t* __restrict__ buckets_size =
      reinterpret_cast<__gm__ int32_t*>(buckets_size_addr_gm);
  __gm__ const K* __restrict__ keys =
      reinterpret_cast<__gm__ const K*>(keys_addr_gm);
  __gm__ V* __restrict__ values = reinterpret_cast<__gm__ V*>(values_addr_gm);
  __gm__ S* __restrict__ scores = reinterpret_cast<__gm__ S*>(scores_gm);
  S score = static_cast<S>(EMPTY_SCORE);
  constexpr uint32_t STRIDE = sizeof(VecD_Comp) / sizeof(D);

  uint32_t key_pos = 0;
  K key = 0;
  __gm__ K* bucket_keys = nullptr;
  uint64_t bucket_values_uintptr = 0;
  __gm__ S* bucket_scores = nullptr;
  __gm__ int32_t* bucket_size = nullptr;
  for (uint64_t kv_idx = block_index * blockDim.x + threadIdx.x;
       kv_idx < n_align_warp; kv_idx += thread_all) {
    VecD_Comp target_digests{0};
    OccupyResult occupy_result{OccupyResult::INITIAL};
    // 1. 每个线程处理一个key
    if (kv_idx < n) {
      key = keys[kv_idx];
      if (IS_RESERVED_KEY<K>(key)) {
        occupy_result = OccupyResult::ILLEGAL;
      }
      score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch,
                                                cur_score);

      // 2. 计算key的hash值 && 定位key
      const K hashed_key = Murmur3HashDevice(key);
      target_digests = digests_from_hashed<K>(hashed_key);
      uint64_t global_idx = get_global_idx(hashed_key, capacity_divisor_magic,
                                           capacity_divisor_shift, capacity);
      key_pos = global_idx & (bucket_max_size - 1);
      uint64_t bkt_idx = global_idx >> (max_bucket_shift);

      bucket_size = buckets_size + bkt_idx;
      int32_t cur_bucket_size = *bucket_size;
      __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
      bucket_keys = bucket->keys_;
      bucket_values_uintptr = reinterpret_cast<uint64_t>(bucket->vectors);
      bucket_scores = bucket->scores_;

      // 3. 遍历桶，找key/空位
      for (uint32_t offset = 0; offset < bucket_max_size + STRIDE;
           offset += STRIDE) {
        if (occupy_result != OccupyResult::INITIAL) {
          break;
        }
        uint32_t pos_cur = align_to<STRIDE>(key_pos);
        pos_cur = (pos_cur + offset) & (bucket_max_size - 1);

        __gm__ D* digests_ptr =
            BUCKET::digests(bucket_keys, bucket_max_size, pos_cur);
        VecD_Comp probe_digests =
            *reinterpret_cast<__gm__ VecD_Comp*>(digests_ptr);
        // 3.1 遍历digest，4个比较
        uint32_t possible_pos = 0;
        uint32_t cmp_result = vcmpeq4(probe_digests, target_digests);
        cmp_result &= 0x01010101;
        do {
          if (cmp_result == 0) {
            break;
          }
          uint32_t index =
              (Simt::Ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
          cmp_result &= (cmp_result - 1);
          possible_pos = pos_cur + index;

          auto current_key_ptr = BUCKET::keys(bucket_keys, possible_pos);
          auto try_key =
              Simt::AtomicCas(current_key_ptr, key, static_cast<K>(LOCKED_KEY));
          // 3.2 找到key，尝试抢占
          if (try_key == key) {
            occupy_result = OccupyResult::DUPLICATE;
            key_pos = possible_pos;
            ScoreFunctor::update_score_only(bucket_keys, key_pos, scores,
                                            kv_idx, score, bucket_max_size,
                                            false);
            break;
          }
        } while (true);
        // 3.3 找到了，跳出循环
        if (occupy_result == OccupyResult::DUPLICATE) {
          break;
          // 3.4 未找到，且桶已满，进行下一波对比
        } else if (cur_bucket_size == bucket_max_size) {
          continue;
        }
        // 3.5 未找到，桶未满，找空桶
        VecD_Comp empty_digests_ = empty_digests<K>();
        cmp_result = vcmpeq4(probe_digests, empty_digests_);
        cmp_result &= 0x01010101;
        do {
          if (cmp_result == 0) {
            break;
          }
          uint32_t index =
              (Simt::Ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
          cmp_result &= (cmp_result - 1);
          possible_pos = pos_cur + index;
          if (offset == 0 && possible_pos < key_pos) {
            continue;
          }

          auto current_key_ptr = BUCKET::keys(bucket_keys, possible_pos);
          auto try_key =
              Simt::AtomicCas(current_key_ptr, static_cast<K>(EMPTY_KEY),
                              static_cast<K>(LOCKED_KEY));
          // 3.6 找到空位，尝试抢占
          if (try_key == static_cast<K>(EMPTY_KEY)) {
            occupy_result = OccupyResult::OCCUPIED_EMPTY;
            key_pos = possible_pos;
            ScoreFunctor::update_with_digest(bucket_keys, key_pos, scores,
                                             kv_idx, score, bucket_max_size,
                                             get_digest<K>(key), true);
            atomicAdd(bucket_size, 1);
            break;
          }
        } while (true);
        // 3.7 抢占到空位，跳出循环，否则进行下一波对比
        if (occupy_result == OccupyResult::OCCUPIED_EMPTY) {
          break;
        }
      }
    } else {
      occupy_result = OccupyResult::ILLEGAL;
    }

    // 前面查找会有3种结果
    // * OccupyResult::DUPLICATE 抢占key
    // * OccupyResult::OCCUPIED_EMPTY 抢占空位
    // * OccupyResult::INITIAL 均抢占失败
    // 4. 开始准入淘汰
    auto cg_rank_id = threadIdx.x % EVICT_GROUP_SIZE;
    // 遍历组内线程，每个线程都要有可能淘汰
    for (int32_t i = 0; i < EVICT_GROUP_SIZE; i++) {
      auto res_sync = __shfl(occupy_result, i, EVICT_GROUP_SIZE);
      while (res_sync == OccupyResult::INITIAL) {
        S min_score = MAX_SCORE;
        uint32_t min_pos = key_pos;
        // 4.1 遍历桶，找最小值
        uint64_t bucket_scores_sync = __shfl(
            reinterpret_cast<uint64_t>(bucket_scores), i, EVICT_GROUP_SIZE);
        uint64_t bucket_keys_sync = __shfl(
            reinterpret_cast<uint64_t>(bucket_keys), i, EVICT_GROUP_SIZE);
        for (uint32_t current_pos = cg_rank_id; current_pos < bucket_max_size;
             current_pos += EVICT_GROUP_SIZE) {
          S current_score = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                                  L1CacheType::NON_CACHEABLE>(
              reinterpret_cast<__gm__ S*>(bucket_scores_sync) + current_pos);
          K current_key = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                                L1CacheType::NON_CACHEABLE>(
              reinterpret_cast<__gm__ K*>(bucket_keys_sync) + current_pos);
          if (current_score < min_score &&
              current_key != static_cast<K>(LOCKED_KEY) &&
              current_key != static_cast<K>(EMPTY_KEY)) {
            min_score = current_score;
            min_pos = current_pos;
          }
        }
        // 分治法求最小值，最终所有线程获得相同的min_score和min_pos
        for (int32_t offset = EVICT_GROUP_SIZE / 2; offset > 0; offset /= 2) {
          S other_score = __shfl_xor(min_score, offset, EVICT_GROUP_SIZE);
          uint32_t other_pos = __shfl_xor(min_pos, offset, EVICT_GROUP_SIZE);
          if (other_score < min_score) {
            min_score = other_score;
            min_pos = other_pos;
          }
        }
        // 拿到了最小值和位置，后续要进行value搬运，每个线程要维护自己的occupy_result，key_pos
        if (cg_rank_id == i) {
          // 4.2 分数不足，无法准入
          if (score < min_score) {
            occupy_result = OccupyResult::REFUSED;
          } else {
            // 4.3 分数满足，尝试准入
            auto current_key_ptr = bucket_keys + min_pos;
            auto current_key =
                __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                      L1CacheType::NON_CACHEABLE>(current_key_ptr);
            if (current_key != static_cast<K>(LOCKED_KEY) &&
                current_key != static_cast<K>(EMPTY_KEY)) {
              auto try_key = Simt::AtomicCas(current_key_ptr, current_key,
                                             static_cast<K>(LOCKED_KEY));
              // 4.4 抢占成功
              if (try_key == current_key) {
                // 4.4.1 确认分数是不是变更小
                if (__ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                          L1CacheType::NON_CACHEABLE>(bucket_scores +
                                                      min_pos) <= min_score) {
                  key_pos = min_pos;
                  ScoreFunctor::update_with_digest(
                      bucket_keys, key_pos, scores, kv_idx, score,
                      bucket_max_size, get_digest<K>(key), true);
                  if (try_key == static_cast<K>(RECLAIM_KEY)) {
                    occupy_result = OccupyResult::OCCUPIED_RECLAIMED;
                    atomicAdd(bucket_size, 1);
                  } else {
                    occupy_result = OccupyResult::EVICT;
                  }
                } else {
                  // 4.4.2 分数变大，淘汰失败，把key还原回去，重新遍历
                  (void)Simt::AtomicExch(current_key_ptr, current_key);
                }
              }
              // 4.5 抢占失败，重新遍历
            }
          }
        }
        res_sync = __shfl(occupy_result, i, EVICT_GROUP_SIZE);
      }
    }

    uint32_t need_write_value = occupy_result != OccupyResult::REFUSED &&
                                occupy_result != OccupyResult::ILLEGAL;
    uint32_t write_mask = __ballot(need_write_value);
    // 累加本warp内任务数
    uint32_t warp_write_count = __popc(write_mask);
    if (warp_write_count == 0) {
      continue;
    }

    size_t bucket_value_start = key_pos * dim;
    size_t value_start = kv_idx * dim;

    uint64_t bucket_value_base_addr =
        (uint64_t)(bucket_values_uintptr + bucket_value_start * sizeof(V));
    uint64_t value_base_addr = (uint64_t)(values + value_start);

    uint32_t key_output_wr_pos = 0;
    uint32_t key_output_rd_pos = 0;
    uint32_t curr_simt_alloc_pos = 0;
    uint32_t new_simt_alloc_pos = 0;
    // 下发任务，由于多个warp并行，因此使用cas确保任务下发安全
    if (laneid() == 0) {
      while (true) {
        curr_simt_alloc_pos = *(__ubuf__ volatile uint32_t*)SIMT_ALLOC_POS_PTR;
        new_simt_alloc_pos = curr_simt_alloc_pos + warp_write_count;

        if (atomicCAS((__ubuf__ uint32_t*)SIMT_ALLOC_POS_PTR,
                      curr_simt_alloc_pos,
                      new_simt_alloc_pos) == curr_simt_alloc_pos) {
          break;
        }
      }
    }
    curr_simt_alloc_pos = __shfl(curr_simt_alloc_pos, 0);
    new_simt_alloc_pos = curr_simt_alloc_pos + warp_write_count;

    while (true) {
      // 正常来说，512个位置一定是够所有thread的任务数的，这里会超过512，肯定是因为MTE消费速度跟不上生产速度
      // warp来到了第二轮下发任务，发现第一轮的任务还没结束，因此要轮询确认任务是否完成
      key_output_wr_pos = *(__ubuf__ volatile uint32_t*)KEY_OUTPUT_WR_POS_PTR;
      // 获取已完成任务数（KEY_OUTPUT_WR_POS_PTR），完成任务数+队列长度512 >=
      // 当前任务终数，说明当前可以启动任务
      if (new_simt_alloc_pos <= key_output_wr_pos + RING_BUFFER_SZ) {
        break;
      }

      // 剩余队列长度不足，因此等待MTE消费完成，启动写key任务，尝试释放队列；这里释放队列本该由MTE消费者负责，但是任务由两段，一是MTE搬运以后，二是SIMT写出key，因此最终释放由SIMT负责
      uint32_t scalar_read_pos =
          *(__ubuf__ volatile uint32_t*)SCALAR_READ_POS_PTR;
      key_output_rd_pos = *(__ubuf__ volatile uint32_t*)KEY_OUTPUT_RD_POS_PTR;
      // 获取MTE消费进度（SCALAR_READ_POS_PTR），写key任务数量（KEY_OUTPUT_RD_POS_PTR），只有当MTE消费进度大于写key任务数量，才能触发写key任务
      if (key_output_rd_pos < scalar_read_pos) {
        // MTE消费进度已经大于任务完成数，说明可以启动写key任务
        uint32_t output_key_size =
            min(32U, scalar_read_pos - key_output_rd_pos);
        uint32_t my_lane_id = laneid();
        uint32_t prev_key_output_rd_pos = 0;
        // 尝试占领释放队列
        if (my_lane_id == 0) {
          prev_key_output_rd_pos =
              atomicCAS((__ubuf__ uint32_t*)KEY_OUTPUT_RD_POS_PTR,
                        key_output_rd_pos, key_output_rd_pos + output_key_size);
        }
        prev_key_output_rd_pos = __shfl(prev_key_output_rd_pos, 0);
        // 抢占成功，启动写key任务，由当前warp写key并释放队列；抢占失败，由其他warp释放，并重新进入循环
        if (prev_key_output_rd_pos == key_output_rd_pos) {
          if (my_lane_id < output_key_size) {
            uint32_t slot_pos =
                (key_output_rd_pos + my_lane_id) & RING_BUFFER_MASK;
            *(__gm__ volatile uint64_t*)((
                (__ubuf__ uint64_t*)RING_BUFFER_OUT_KEY_ADDR)[slot_pos]) =
                ((__ubuf__ uint64_t*)(RING_BUFFER_KEY_ADDR))[slot_pos];
          }
          __threadfence();
          // 写key任务完成，释放空间，累加任务完成数量
          if (my_lane_id == 0) {
            while (true) {
              // 这里会有多个warp分别释放不同部分队列的场景，后来的warp可能先完成了写key任务，
              // 但是后来的warp不能立刻释放，因为队列buffer是严格顺序的，后来的先释放会导致前面还在释放中的buffer使用错乱，
              // 因此这里需要循环等待，按warp顺序释放队列
              if (atomicCAS((__ubuf__ uint32_t*)KEY_OUTPUT_WR_POS_PTR,
                            prev_key_output_rd_pos,
                            key_output_rd_pos + output_key_size) ==
                  prev_key_output_rd_pos) {
                // 释放完后继续申请空间
                break;
              }
              asm volatile("NOP wait:0b0000000 stall:15" ::);
              asm volatile("NOP wait:0b0000000 stall:15" ::);
              asm volatile("NOP wait:0b0000000 stall:15" ::);
              asm volatile("NOP wait:0b0000000 stall:15" ::);
              asm volatile("NOP wait:0b0000000 stall:15" ::);
              asm volatile("NOP wait:0b0000000 stall:15" ::);
              asm volatile("NOP wait:0b0000000 stall:15" ::);
              asm volatile("NOP wait:0b0000000 stall:15" ::);
            }
          }
        }
      }

      {
        asm volatile("NOP wait:0b0000000 stall:15" ::);
        asm volatile("NOP wait:0b0000000 stall:15" ::);
        asm volatile("NOP wait:0b0000000 stall:15" ::);
        asm volatile("NOP wait:0b0000000 stall:15" ::);
        asm volatile("NOP wait:0b0000000 stall:15" ::);
        asm volatile("NOP wait:0b0000000 stall:15" ::);
        asm volatile("NOP wait:0b0000000 stall:15" ::);
        asm volatile("NOP wait:0b0000000 stall:15" ::);
      }

    }  // while(true)

    // 前面已经明确由足够队列空间存放任务
    // 填充各任务数据
    curr_simt_alloc_pos = __shfl(curr_simt_alloc_pos, 0);
    if (need_write_value) {
      uint32_t offset_in_warp = __popc(write_mask & lanemask_lt());
      uint32_t my_slot_id =
          (curr_simt_alloc_pos + offset_in_warp) & RING_BUFFER_MASK;

      ((__ubuf__ uint64_t*)RING_BUFFER_OUT_VALUE_BASE_ADDR)[my_slot_id] =
          bucket_value_base_addr;
      ((__ubuf__ uint64_t*)RING_BUFFER_IN_VALUE_BASE_ADDR)[my_slot_id] =
          value_base_addr;
      ((__ubuf__ uint64_t*)RING_BUFFER_OUT_KEY_ADDR)[my_slot_id] =
          (uint64_t)(bucket_keys + key_pos);
      ((__ubuf__ uint64_t*)RING_BUFFER_KEY_ADDR)[my_slot_id] = key;
    }

    __threadfence();  // FENCE SIMT -> MTE

    if (laneid() == 0) {
      while (true) {
        // 任务数据填充完成，下发任务
        if (atomicCAS((__ubuf__ uint32_t*)SIMT_WR_POS_PTR, curr_simt_alloc_pos,
                      new_simt_alloc_pos) == curr_simt_alloc_pos) {
          break;
        }

        {
          asm volatile("NOP wait:0b0000000 stall:15" ::);
          asm volatile("NOP wait:0b0000000 stall:15" ::);
          asm volatile("NOP wait:0b0000000 stall:15" ::);
          asm volatile("NOP wait:0b0000000 stall:15" ::);
          asm volatile("NOP wait:0b0000000 stall:15" ::);
          asm volatile("NOP wait:0b0000000 stall:15" ::);
          asm volatile("NOP wait:0b0000000 stall:15" ::);
          asm volatile("NOP wait:0b0000000 stall:15" ::);
        }
      }
    }
  }

  __syncthreads();

  // 所有任务已经处理完成，MTE任务结束
  if (threadIdx.x == 0) {
    *(__ubuf__ volatile uint32_t*)WORK_DONE_PTR = 1;
  }

  // MTE任务下发完成了，处理最后的写key任务！
  uint32_t simt_wr_pos = *(__ubuf__ volatile uint32_t*)SIMT_WR_POS_PTR;
  while (true) {
    uint32_t key_output_rd_pos =
        *(__ubuf__ volatile uint32_t*)KEY_OUTPUT_RD_POS_PTR;
    if (simt_wr_pos == key_output_rd_pos) {
      // output done
      break;
    }

    uint32_t scalar_read_pos =
        *(__ubuf__ volatile uint32_t*)SCALAR_READ_POS_PTR;
    key_output_rd_pos = *(__ubuf__ volatile uint32_t*)KEY_OUTPUT_RD_POS_PTR;
    if (key_output_rd_pos < scalar_read_pos) {
      //{
      // try to output key to release ring buffer space
      uint32_t output_key_size = min(32U, scalar_read_pos - key_output_rd_pos);
      uint32_t my_lane_id = laneid();
      uint32_t prev_key_output_rd_pos = 0;
      if (my_lane_id == 0) {
        prev_key_output_rd_pos =
            atomicCAS((__ubuf__ uint32_t*)KEY_OUTPUT_RD_POS_PTR,
                      key_output_rd_pos, key_output_rd_pos + output_key_size);
      }
      prev_key_output_rd_pos = __shfl(prev_key_output_rd_pos, 0);
      if (prev_key_output_rd_pos == key_output_rd_pos) {
        // output keys
        if (my_lane_id < output_key_size) {
          uint32_t slot_pos =
              (key_output_rd_pos + my_lane_id) & RING_BUFFER_MASK;
          *(__gm__ volatile uint64_t*)((
              (__ubuf__ uint64_t*)RING_BUFFER_OUT_KEY_ADDR)[slot_pos]) =
              ((__ubuf__ uint64_t*)(RING_BUFFER_KEY_ADDR))[slot_pos];
        }
        // release key slot
        if (my_lane_id == 0) {
          while (true) {
            if (atomicCAS((__ubuf__ uint32_t*)KEY_OUTPUT_WR_POS_PTR,
                          prev_key_output_rd_pos,
                          key_output_rd_pos + output_key_size) ==
                prev_key_output_rd_pos) {
              break;
            }
            asm volatile("NOP wait:0b0000000 stall:15" ::);
            asm volatile("NOP wait:0b0000000 stall:15" ::);
            asm volatile("NOP wait:0b0000000 stall:15" ::);
            asm volatile("NOP wait:0b0000000 stall:15" ::);
            asm volatile("NOP wait:0b0000000 stall:15" ::);
            asm volatile("NOP wait:0b0000000 stall:15" ::);
            asm volatile("NOP wait:0b0000000 stall:15" ::);
            asm volatile("NOP wait:0b0000000 stall:15" ::);
          }
        }
      }
    }

    asm volatile("NOP wait:0b0000000 stall:15" ::);
    asm volatile("NOP wait:0b0000000 stall:15" ::);
    asm volatile("NOP wait:0b0000000 stall:15" ::);
    asm volatile("NOP wait:0b0000000 stall:15" ::);
    asm volatile("NOP wait:0b0000000 stall:15" ::);
    asm volatile("NOP wait:0b0000000 stall:15" ::);
    asm volatile("NOP wait:0b0000000 stall:15" ::);
    asm volatile("NOP wait:0b0000000 stall:15" ::);
  }
}

template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          typename VecV = int32_t, int32_t Strategy = -1,
          int32_t EVICT_GROUP_SIZE = 16>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_1024) inline void insert_or_assign_kernel_with_digest_vf_1024(
    __gm__ void* buckets_addr_gm, __gm__ int32_t* buckets_size_addr_gm,
    uint64_t capacity, uint32_t bucket_max_size, uint32_t dim,
    __gm__ void* keys_addr_gm, __gm__ void* values_addr_gm,
    __gm__ void* scores_gm, S cur_score, uint64_t n, uint32_t thread_all,
    uint64_t global_epoch, uint32_t block_index, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift,
    uint64_t n_align_warp, int32_t group_size) {
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
  using BUCKET = Bucket<K, V, S>;
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  __gm__ Bucket<K, V, S>* __restrict__ buckets =
      reinterpret_cast<__gm__ Bucket<K, V, S>*>(buckets_addr_gm);
  __gm__ int32_t* __restrict__ buckets_size =
      reinterpret_cast<__gm__ int32_t*>(buckets_size_addr_gm);
  __gm__ const K* __restrict__ keys =
      reinterpret_cast<__gm__ const K*>(keys_addr_gm);
  __gm__ VecV* __restrict__ values =
      reinterpret_cast<__gm__ VecV*>(values_addr_gm);
  __gm__ S* __restrict__ scores = reinterpret_cast<__gm__ S*>(scores_gm);
  S score = static_cast<S>(EMPTY_SCORE);
  constexpr uint32_t STRIDE = sizeof(VecD_Comp) / sizeof(D);

  uint32_t key_pos = 0;
  K key = 0;
  __gm__ K* bucket_keys = nullptr;
  uint64_t bucket_values_uintptr = 0;
  __gm__ S* bucket_scores = nullptr;
  __gm__ int32_t* bucket_size = nullptr;
  for (uint64_t kv_idx = block_index * blockDim.x + threadIdx.x;
       kv_idx < n_align_warp; kv_idx += thread_all) {
    VecD_Comp target_digests{0};
    OccupyResult occupy_result{OccupyResult::INITIAL};
    // 1. 每个线程处理一个key
    if (kv_idx < n) {
      key = keys[kv_idx];
      if (IS_RESERVED_KEY<K>(key)) {
        occupy_result = OccupyResult::ILLEGAL;
      }
      score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch,
                                                cur_score);

      // 2. 计算key的hash值 && 定位key
      const K hashed_key = Murmur3HashDevice(key);
      target_digests = digests_from_hashed<K>(hashed_key);
      uint64_t global_idx = get_global_idx(hashed_key, capacity_divisor_magic,
                                           capacity_divisor_shift, capacity);
      key_pos = global_idx & (bucket_max_size - 1);
      uint64_t bkt_idx = global_idx >> (max_bucket_shift);

      bucket_size = buckets_size + bkt_idx;
      int32_t cur_bucket_size = *bucket_size;
      __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
      bucket_keys = bucket->keys_;
      bucket_values_uintptr = reinterpret_cast<uint64_t>(bucket->vectors);
      bucket_scores = bucket->scores_;

      // 3. 遍历桶，找key/空位
      for (uint32_t offset = 0; offset < bucket_max_size + STRIDE;
           offset += STRIDE) {
        if (occupy_result != OccupyResult::INITIAL) {
          break;
        }
        uint32_t pos_cur = align_to<STRIDE>(key_pos);
        pos_cur = (pos_cur + offset) & (bucket_max_size - 1);

        __gm__ D* digests_ptr =
            BUCKET::digests(bucket_keys, bucket_max_size, pos_cur);
        VecD_Comp probe_digests =
            *reinterpret_cast<__gm__ VecD_Comp*>(digests_ptr);
        // 3.1 遍历digest，4个比较
        uint32_t possible_pos = 0;
        uint32_t cmp_result = vcmpeq4(probe_digests, target_digests);
        cmp_result &= 0x01010101;
        do {
          if (cmp_result == 0) {
            break;
          }
          uint32_t index = (__ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
          cmp_result &= (cmp_result - 1);
          possible_pos = pos_cur + index;

          auto current_key_ptr = BUCKET::keys(bucket_keys, possible_pos);
          auto try_key =
              asc_atomic_cas(current_key_ptr, key, static_cast<K>(LOCKED_KEY));
          // 3.2 找到key，尝试抢占
          if (try_key == key) {
            occupy_result = OccupyResult::DUPLICATE;
            key_pos = possible_pos;
            ScoreFunctor::update_score_only(bucket_keys, key_pos, scores,
                                            kv_idx, score, bucket_max_size,
                                            false);
            break;
          }
        } while (true);
        // 3.3 找到了，跳出循环
        if (occupy_result == OccupyResult::DUPLICATE) {
          break;
          // 3.4 未找到，且桶已满，进行下一波对比
        } else if (cur_bucket_size == bucket_max_size) {
          continue;
        }
        // 3.5 未找到，桶未满，找空桶
        VecD_Comp empty_digests_ = empty_digests<K>();
        cmp_result = vcmpeq4(probe_digests, empty_digests_);
        cmp_result &= 0x01010101;
        do {
          if (cmp_result == 0) {
            break;
          }
          uint32_t index = (__ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
          cmp_result &= (cmp_result - 1);
          possible_pos = pos_cur + index;
          if (offset == 0 && possible_pos < key_pos) {
            continue;
          }

          auto current_key_ptr = BUCKET::keys(bucket_keys, possible_pos);
          auto try_key =
              asc_atomic_cas(current_key_ptr, static_cast<K>(EMPTY_KEY),
                             static_cast<K>(LOCKED_KEY));
          // 3.6 找到空位，尝试抢占
          if (try_key == static_cast<K>(EMPTY_KEY)) {
            occupy_result = OccupyResult::OCCUPIED_EMPTY;
            key_pos = possible_pos;
            ScoreFunctor::update_with_digest(bucket_keys, key_pos, scores,
                                             kv_idx, score, bucket_max_size,
                                             get_digest<K>(key), true);
            asc_atomic_add(bucket_size, 1);
            break;
          }
        } while (true);
        // 3.7 抢占到空位，跳出循环，否则进行下一波对比
        if (occupy_result == OccupyResult::OCCUPIED_EMPTY) {
          break;
        }
      }
    } else {
      occupy_result = OccupyResult::ILLEGAL;
    }

    // 前面查找会有3种结果
    // * OccupyResult::DUPLICATE 抢占key
    // * OccupyResult::OCCUPIED_EMPTY 抢占空位
    // * OccupyResult::INITIAL 均抢占失败
    // 4. 开始准入淘汰
    auto cg_rank_id = threadIdx.x % EVICT_GROUP_SIZE;
    // 遍历组内线程，每个线程都要有可能淘汰
    for (int32_t i = 0; i < EVICT_GROUP_SIZE; i++) {
      auto res_sync = asc_shfl(occupy_result, i, EVICT_GROUP_SIZE);
      while (res_sync == OccupyResult::INITIAL) {
        S min_score = MAX_SCORE;
        uint32_t min_pos = key_pos;
        // 4.1 遍历桶，找最小值
        uint64_t bucket_scores_sync = asc_shfl(
            reinterpret_cast<uint64_t>(bucket_scores), i, EVICT_GROUP_SIZE);
        uint64_t bucket_keys_sync = asc_shfl(
            reinterpret_cast<uint64_t>(bucket_keys), i, EVICT_GROUP_SIZE);
        for (uint32_t current_pos = cg_rank_id; current_pos < bucket_max_size;
             current_pos += EVICT_GROUP_SIZE) {
          S current_score = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                                  L1CacheType::NON_CACHEABLE>(
              reinterpret_cast<__gm__ S*>(bucket_scores_sync) + current_pos);
          K current_key = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                                L1CacheType::NON_CACHEABLE>(
              reinterpret_cast<__gm__ K*>(bucket_keys_sync) + current_pos);
          if (current_score < min_score &&
              current_key != static_cast<K>(LOCKED_KEY) &&
              current_key != static_cast<K>(EMPTY_KEY)) {
            min_score = current_score;
            min_pos = current_pos;
          }
        }
        // 分治法求最小值，最终所有线程获得相同的min_score和min_pos
        for (int32_t offset = EVICT_GROUP_SIZE / 2; offset > 0; offset /= 2) {
          S other_score = asc_shfl_xor(min_score, offset, EVICT_GROUP_SIZE);
          uint32_t other_pos = asc_shfl_xor(min_pos, offset, EVICT_GROUP_SIZE);
          if (other_score < min_score) {
            min_score = other_score;
            min_pos = other_pos;
          }
        }
        // 拿到了最小值和位置，后续要进行value搬运，每个线程要维护自己的occupy_result，key_pos
        if (cg_rank_id == i) {
          // 4.2 分数不足，无法准入
          if (score < min_score) {
            occupy_result = OccupyResult::REFUSED;
          } else {
            // 4.3 分数满足，尝试准入
            auto current_key_ptr = bucket_keys + min_pos;
            auto current_key =
                __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                      L1CacheType::NON_CACHEABLE>(current_key_ptr);
            if (current_key != static_cast<K>(LOCKED_KEY) &&
                current_key != static_cast<K>(EMPTY_KEY)) {
              auto try_key = asc_atomic_cas(current_key_ptr, current_key,
                                            static_cast<K>(LOCKED_KEY));
              // 4.4 抢占成功
              if (try_key == current_key) {
                // 4.4.1 确认分数是不是变更小
                if (__ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                          L1CacheType::NON_CACHEABLE>(bucket_scores +
                                                      min_pos) <= min_score) {
                  key_pos = min_pos;
                  ScoreFunctor::update_with_digest(
                      bucket_keys, key_pos, scores, kv_idx, score,
                      bucket_max_size, get_digest<K>(key), true);
                  if (try_key == static_cast<K>(RECLAIM_KEY)) {
                    occupy_result = OccupyResult::OCCUPIED_RECLAIMED;
                    asc_atomic_add(bucket_size, 1);
                  } else {
                    occupy_result = OccupyResult::EVICT;
                  }
                } else {
                  // 4.4.2 分数变大，淘汰失败，把key还原回去，重新遍历
                  (void)asc_atomic_exch(current_key_ptr, current_key);
                }
              }
              // 4.5 抢占失败，重新遍历
            }
          }
        }
        res_sync = asc_shfl(occupy_result, i, EVICT_GROUP_SIZE);
      }
    }

    cg_rank_id = threadIdx.x % group_size;
    for (int32_t i = 0; i < group_size; i++) {
      auto res_sync = asc_shfl(occupy_result, i, group_size);
      if ((res_sync != OccupyResult::REFUSED &&
           res_sync != OccupyResult::ILLEGAL)) {
        auto kv_idx_sync = asc_shfl(kv_idx, i, group_size);
        auto value_start = kv_idx_sync * dim;

        auto key_pos_sync = asc_shfl(key_pos, i, group_size);
        uint64_t value_ddr_sync =
            asc_shfl(bucket_values_uintptr, i, group_size);
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

    // 5. 抢占成功，写入value
    if (occupy_result != OccupyResult::REFUSED &&
        occupy_result != OccupyResult::ILLEGAL) {
      // key也是原子标记位，所有key的操作必须原子化
      (void)asc_atomic_exch(bucket_keys + key_pos, key);
    }
  }
}

template <class K, class V, class S, int Strategy = -1>
__global__ __vector__ void insert_or_assign_kernel(
    __gm__ void* buckets, __gm__ int* buckets_size, uint64_t capacity,
    uint32_t bucket_max_size, uint32_t dim, __gm__ void* keys,
    __gm__ void* values, __gm__ void* scores, uint64_t n, uint64_t global_epoch,
    uint32_t value_size, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift,
    uint64_t n_align_warp, int32_t group_size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  if ASCEND_IS_AIV {
    __ubuf__ volatile uint32_t* scalar_read_pos =
        (__ubuf__ volatile uint32_t*)(SCALAR_READ_POS_PTR);
    __ubuf__ volatile uint32_t* simt_alloc_pos =
        (__ubuf__ volatile uint32_t*)(SIMT_ALLOC_POS_PTR);
    __ubuf__ volatile uint32_t* simt_wr_pos =
        (__ubuf__ volatile uint32_t*)(SIMT_WR_POS_PTR);
    __ubuf__ volatile uint32_t* key_output_rd_pos =
        (__ubuf__ volatile uint32_t*)(KEY_OUTPUT_RD_POS_PTR);
    __ubuf__ volatile uint32_t* key_output_wr_pos =
        (__ubuf__ volatile uint32_t*)(KEY_OUTPUT_WR_POS_PTR);
    __ubuf__ volatile uint32_t* work_done =
        (__ubuf__ volatile uint32_t*)(WORK_DONE_PTR);

    __ubuf__ uint64_t* ring_buffer_out_value_ptr =
        (__ubuf__ uint64_t*)RING_BUFFER_OUT_VALUE_BASE_ADDR;
    __ubuf__ uint64_t* ring_buffer_in_value_ptr =
        (__ubuf__ uint64_t*)RING_BUFFER_IN_VALUE_BASE_ADDR;
    __ubuf__ uint64_t* ring_buffer_out_key_ptr =
        (__ubuf__ uint64_t*)RING_BUFFER_OUT_KEY_ADDR;
    __ubuf__ uint64_t* ring_buffer_key =
        (__ubuf__ uint64_t*)RING_BUFFER_KEY_ADDR;

    // 标记位清0
    *scalar_read_pos = 0;
    *simt_alloc_pos = 0;
    *simt_wr_pos = 0;
    *key_output_rd_pos = 0;
    *key_output_wr_pos = 0;
    *work_done = 0;

    uint32_t scalar_process_value = 0;

    int value_dim_size = dim * value_size;
    int value_dim_size_aligned = (value_dim_size + 32 - 1) / 32 * 32;
    constexpr uint32_t issue_queue_limited = 32;  // 16 - 2
    // 1次能消费多少value
    uint32_t value_batch_size = DATA_BUFFER_SZ / (value_dim_size_aligned);
    value_batch_size = min(value_batch_size, issue_queue_limited);

    pipe_barrier(PIPE_ALL);

    constexpr uint32_t thread_num = 512;
    const uint32_t thread_all = thread_num * GetBlockNum();
    uint64_t cur_score =
        (Strategy == npu::hkv::EvictStrategyInternal::kLru ||
         Strategy == npu::hkv::EvictStrategyInternal::kEpochLru)
            ? static_cast<uint64_t>(GetSystemCycle())
            : 0;

    DISPATCH_GROUP_SIZE(
        group_size,
        DISPATCH_VALUE_SIZE(
            value_size,
            DISPATCH_EVICT_STRATEGY(
                Strategy,
                (Simt::VF_CALL<insert_or_assign_kernel_with_digest_vf<
                     uint64_t, DTYPE, uint64_t, STRATEGY, GROUP_SIZE>>(
                    Simt::Dim3{static_cast<uint32_t>(thread_num)},
                    (GM_ADDR)buckets, (GM_ADDR)buckets_size, capacity,
                    bucket_max_size, dim, (GM_ADDR)keys, (GM_ADDR)values,
                    (GM_ADDR)scores, cur_score, n, thread_all, global_epoch,
                    GetBlockIdx(), max_bucket_shift, capacity_divisor_magic,
                    capacity_divisor_shift, n_align_warp)))));

    uint32_t curr_simt_wr_pos = 0;

    while (true) {
      // 获取simt生产计数
      curr_simt_wr_pos = *simt_wr_pos;
      // 是否已经消费完成？
      if (curr_simt_wr_pos == scalar_process_value) {
        // 消费完成 && 任务全完成 => 结束
        if (*work_done == 1) {
          break;
        }
        // 消费完成 && 任务未完成 => 继续等待
        else {
          continue;
        }
      }

      // 计算一次性能消费的数量
      uint32_t curr_batch_size =
          min(value_batch_size, curr_simt_wr_pos - scalar_process_value);
      // 消费
      for (uint32_t i = 0; i < curr_batch_size; ++i) {
        uint32_t slot_pos = (scalar_process_value + i) & RING_BUFFER_MASK;
        copy_gm_to_ubuf_align_v2(
            (__ubuf__ uint8_t*)(DATA_BUFFER_START_ADDR +
                                i * value_dim_size_aligned),
            (__gm__ uint8_t*)(ring_buffer_in_value_ptr[slot_pos]), 0, 1,
            value_dim_size, 0, 0, 0, 0, 0, 0);
      }

      set_flag(PIPE_MTE2, PIPE_MTE3, (event_t)0);
      wait_flag(PIPE_MTE2, PIPE_MTE3, (event_t)0);

      for (uint32_t i = 0; i < curr_batch_size; ++i) {
        uint32_t slot_pos = (scalar_process_value + i) & RING_BUFFER_MASK;
        copy_ubuf_to_gm_align_v2(
            (__gm__ uint8_t*)(ring_buffer_out_value_ptr[slot_pos]),
            (__ubuf__ uint8_t*)(DATA_BUFFER_START_ADDR +
                                i * value_dim_size_aligned),
            0, 1, value_dim_size, 0, 0, 0);
      }

      set_flag(PIPE_MTE3, PIPE_S, (event_t)0);
      wait_flag(PIPE_MTE3, PIPE_S, (event_t)0);

      // 消费完成，累加消费进展
      scalar_process_value += curr_batch_size;
      *scalar_read_pos = scalar_process_value;
    }

  }  // ASCEND_IS_AIV
}

template <class K, class V, class S, int Strategy = -1>
__global__ __vector__ void insert_or_assign_kernel_with_thread_1024(
    __gm__ void* buckets, __gm__ int* buckets_size, uint64_t capacity,
    uint32_t bucket_max_size, uint32_t dim, __gm__ void* keys,
    __gm__ void* values, __gm__ void* scores, uint64_t n, uint64_t global_epoch,
    uint32_t value_size, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift,
    uint64_t n_align_warp, int32_t group_size) {
  const uint32_t thread_all = THREAD_NUM_1024 * GetBlockNum();
  uint64_t cur_score = (Strategy == npu::hkv::EvictStrategyInternal::kLru ||
                        Strategy == npu::hkv::EvictStrategyInternal::kEpochLru)
                           ? static_cast<uint64_t>(GetSystemCycle())
                           : 0;
  DISPATCH_VALUE_SIZE(
      value_size,
      (asc_vf_call<insert_or_assign_kernel_with_digest_vf_1024<K, V, S, DTYPE,
                                                               Strategy>>(
          dim3{THREAD_NUM_1024}, buckets, buckets_size, capacity,
          bucket_max_size, dim, keys, values, scores, cur_score, n, thread_all,
          global_epoch, GetBlockIdx(), max_bucket_shift, capacity_divisor_magic,
          capacity_divisor_shift, n_align_warp, group_size)));
}

template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          int32_t Strategy = -1, int32_t EVICT_GROUP_SIZE = 16>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_512) inline void upsert_kernel_lock_key_hybrid_vf(
    __gm__ void* buckets_addr_gm, __gm__ int32_t* buckets_size_addr_gm,
    uint64_t capacity, uint32_t bucket_max_size, uint32_t dim,
    __gm__ void* keys_addr_gm, __gm__ void* scores_gm, S cur_score, uint64_t n,
    uint64_t global_epoch, uint32_t block_index, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift,
    uint64_t n_align_warp, uint32_t thread_all, __gm__ V* __gm__* d_dst_values,
    __gm__ K* __gm__* d_dst_keys) {
  if (buckets_addr_gm == nullptr) {
    return;
  }
  if (buckets_size_addr_gm == nullptr) {
    return;
  }
  if (keys_addr_gm == nullptr) {
    return;
  }
  using BUCKET = Bucket<K, V, S>;
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  __gm__ Bucket<K, V, S>* __restrict__ buckets =
      reinterpret_cast<__gm__ Bucket<K, V, S>*>(buckets_addr_gm);
  __gm__ int32_t* __restrict__ buckets_size =
      reinterpret_cast<__gm__ int32_t*>(buckets_size_addr_gm);
  __gm__ const K* __restrict__ keys =
      reinterpret_cast<__gm__ const K*>(keys_addr_gm);
  __gm__ S* __restrict__ scores = reinterpret_cast<__gm__ S*>(scores_gm);
  S score = static_cast<S>(EMPTY_SCORE);
  constexpr uint32_t STRIDE = sizeof(VecD_Comp) / sizeof(D);

  uint32_t key_pos = 0;
  K key = 0;
  __gm__ K* bucket_keys = nullptr;
  __gm__ V* bucket_values = nullptr;
  __gm__ S* bucket_scores = nullptr;
  __gm__ int32_t* bucket_size = nullptr;
  for (uint64_t kv_idx = block_index * blockDim.x + threadIdx.x;
       kv_idx < n_align_warp; kv_idx += thread_all) {
    VecD_Comp target_digests{0};
    OccupyResult occupy_result{OccupyResult::INITIAL};
    // 1. 每个线程处理一个key
    if (kv_idx < n) {
      key = keys[kv_idx];
      if (IS_RESERVED_KEY<K>(key)) {
        occupy_result = OccupyResult::ILLEGAL;
      }
      score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch,
                                                cur_score);

      // 2. 计算key的hash值 && 定位key
      const K hashed_key = Murmur3HashDevice(key);
      target_digests = digests_from_hashed<K>(hashed_key);
      uint64_t global_idx = get_global_idx(hashed_key, capacity_divisor_magic,
                                           capacity_divisor_shift, capacity);
      key_pos = global_idx & (bucket_max_size - 1);
      uint64_t bkt_idx = global_idx >> (max_bucket_shift);

      bucket_size = buckets_size + bkt_idx;
      int32_t cur_bucket_size = *bucket_size;
      __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
      bucket_keys = bucket->keys_;
      bucket_values = bucket->vectors;
      bucket_scores = bucket->scores_;

      // 3. 遍历桶，找key/空位
      for (uint32_t offset = 0; offset < bucket_max_size + STRIDE;
           offset += STRIDE) {
        if (occupy_result != OccupyResult::INITIAL) {
          break;
        }
        uint32_t pos_cur = align_to<STRIDE>(key_pos);
        pos_cur = (pos_cur + offset) & (bucket_max_size - 1);

        __gm__ D* digests_ptr =
            BUCKET::digests(bucket_keys, bucket_max_size, pos_cur);
        VecD_Comp probe_digests =
            *reinterpret_cast<__gm__ VecD_Comp*>(digests_ptr);
        // 3.1 遍历digest，4个比较
        uint32_t possible_pos = 0;
        uint32_t cmp_result = vcmpeq4(probe_digests, target_digests);
        cmp_result &= 0x01010101;
        do {
          if (cmp_result == 0) {
            break;
          }
          uint32_t index = (__ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
          cmp_result &= (cmp_result - 1);
          possible_pos = pos_cur + index;

          auto current_key_ptr = BUCKET::keys(bucket_keys, possible_pos);
          auto try_key =
              asc_atomic_cas(current_key_ptr, key, static_cast<K>(LOCKED_KEY));
          // 3.2 找到key，尝试抢占
          if (try_key == key) {
            occupy_result = OccupyResult::DUPLICATE;
            key_pos = possible_pos;
            ScoreFunctor::update_score_only(bucket_keys, key_pos, scores,
                                            kv_idx, score, bucket_max_size,
                                            false);
            break;
          }
        } while (true);
        // 3.3 找到了，跳出循环
        if (occupy_result == OccupyResult::DUPLICATE) {
          break;
          // 3.4 未找到，且桶已满，进行下一波对比
        } else if (cur_bucket_size == bucket_max_size) {
          continue;
        }
        // 3.5 未找到，桶未满，找空桶
        VecD_Comp empty_digests_ = empty_digests<K>();
        cmp_result = vcmpeq4(probe_digests, empty_digests_);
        cmp_result &= 0x01010101;
        do {
          if (cmp_result == 0) {
            break;
          }
          uint32_t index = (__ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
          cmp_result &= (cmp_result - 1);
          possible_pos = pos_cur + index;
          if (offset == 0 && possible_pos < key_pos) {
            continue;
          }

          auto current_key_ptr = BUCKET::keys(bucket_keys, possible_pos);
          auto try_key =
              asc_atomic_cas(current_key_ptr, static_cast<K>(EMPTY_KEY),
                             static_cast<K>(LOCKED_KEY));
          // 3.6 找到空位，尝试抢占
          if (try_key == static_cast<K>(EMPTY_KEY)) {
            occupy_result = OccupyResult::OCCUPIED_EMPTY;
            key_pos = possible_pos;
            ScoreFunctor::update_with_digest(bucket_keys, key_pos, scores,
                                             kv_idx, score, bucket_max_size,
                                             get_digest<K>(key), true);
            asc_atomic_add(bucket_size, 1);
            break;
          }
        } while (true);
        // 3.7 抢占到空位，跳出循环，否则进行下一波对比
        if (occupy_result == OccupyResult::OCCUPIED_EMPTY) {
          break;
        }
      }
    } else {
      occupy_result = OccupyResult::ILLEGAL;
    }

    // 前面查找会有3种结果
    // * OccupyResult::DUPLICATE 抢占key
    // * OccupyResult::OCCUPIED_EMPTY 抢占空位
    // * OccupyResult::INITIAL 均抢占失败
    // 4. 开始准入淘汰
    auto cg_rank_id = threadIdx.x % EVICT_GROUP_SIZE;
    // 遍历组内线程，每个线程都要有可能淘汰
    for (int32_t i = 0; i < EVICT_GROUP_SIZE; i++) {
      auto res_sync = asc_shfl(occupy_result, i, EVICT_GROUP_SIZE);
      while (res_sync == OccupyResult::INITIAL) {
        S min_score = MAX_SCORE;
        uint32_t min_pos = key_pos;
        // 4.1 遍历桶，找最小值
        uint64_t bucket_scores_sync = asc_shfl(
            reinterpret_cast<uint64_t>(bucket_scores), i, EVICT_GROUP_SIZE);
        uint64_t bucket_keys_sync = asc_shfl(
            reinterpret_cast<uint64_t>(bucket_keys), i, EVICT_GROUP_SIZE);
        for (uint32_t current_pos = cg_rank_id; current_pos < bucket_max_size;
             current_pos += EVICT_GROUP_SIZE) {
          S current_score = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                                  L1CacheType::NON_CACHEABLE>(
              reinterpret_cast<__gm__ S*>(bucket_scores_sync) + current_pos);
          K current_key = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                                L1CacheType::NON_CACHEABLE>(
              reinterpret_cast<__gm__ K*>(bucket_keys_sync) + current_pos);
          if (current_score < min_score &&
              current_key != static_cast<K>(LOCKED_KEY) &&
              current_key != static_cast<K>(EMPTY_KEY)) {
            min_score = current_score;
            min_pos = current_pos;
          }
        }
        // 分治法求最小值，最终所有线程获得相同的min_score和min_pos
        for (int32_t offset = EVICT_GROUP_SIZE / 2; offset > 0; offset /= 2) {
          S other_score = asc_shfl_xor(min_score, offset, EVICT_GROUP_SIZE);
          uint32_t other_pos = asc_shfl_xor(min_pos, offset, EVICT_GROUP_SIZE);
          if (other_score < min_score) {
            min_score = other_score;
            min_pos = other_pos;
          }
        }
        // 拿到了最小值和位置，后续要进行value搬运，每个线程要维护自己的occupy_result，key_pos
        if (cg_rank_id == i) {
          // 4.2 分数不足，无法准入
          if (score < min_score) {
            occupy_result = OccupyResult::REFUSED;
          } else {
            // 4.3 分数满足，尝试准入
            auto current_key_ptr = bucket_keys + min_pos;
            auto current_key =
                __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                      L1CacheType::NON_CACHEABLE>(current_key_ptr);
            if (current_key != static_cast<K>(LOCKED_KEY) &&
                current_key != static_cast<K>(EMPTY_KEY)) {
              auto try_key = asc_atomic_cas(current_key_ptr, current_key,
                                            static_cast<K>(LOCKED_KEY));
              // 4.4 抢占成功
              if (try_key == current_key) {
                // 4.4.1 确认分数是不是变更小
                if (__ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                          L1CacheType::NON_CACHEABLE>(bucket_scores +
                                                      min_pos) <= min_score) {
                  key_pos = min_pos;
                  ScoreFunctor::update_with_digest(
                      bucket_keys, key_pos, scores, kv_idx, score,
                      bucket_max_size, get_digest<K>(key), true);
                  if (try_key == static_cast<K>(RECLAIM_KEY)) {
                    occupy_result = OccupyResult::OCCUPIED_RECLAIMED;
                    asc_atomic_add(bucket_size, 1);
                  } else {
                    occupy_result = OccupyResult::EVICT;
                  }
                } else {
                  // 4.4.2 分数变大，淘汰失败，把key还原回去，重新遍历
                  (void)asc_atomic_exch(current_key_ptr, current_key);
                }
              }
              // 4.5 抢占失败，重新遍历
            }
          }
        }
        res_sync = asc_shfl(occupy_result, i, EVICT_GROUP_SIZE);
      }
    }

    // 5. 抢占成功，写入value
    if (occupy_result != OccupyResult::REFUSED &&
        occupy_result != OccupyResult::ILLEGAL) {
      d_dst_values[kv_idx] = bucket_values + key_pos * dim;
      d_dst_keys[kv_idx] = bucket_keys + key_pos;
    } else if (kv_idx < n) {
      d_dst_values[kv_idx] = nullptr;
      d_dst_keys[kv_idx] = nullptr;
    }
  }
}

template <class K, class V, class S, int Strategy = -1>
__global__ __vector__ void upsert_kernel_lock_key_hybrid(
    __gm__ void* buckets, __gm__ int* buckets_size, uint64_t capacity,
    uint32_t bucket_max_size, uint32_t dim, __gm__ void* keys,
    __gm__ void* scores, uint64_t n, uint64_t global_epoch,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift, uint64_t n_align_warp,
    __gm__ V* __gm__* d_dst_values, __gm__ K* __gm__* d_dst_keys) {
  const uint32_t thread_all = THREAD_NUM_512 * GetBlockNum();
  uint64_t cur_score = (Strategy == npu::hkv::EvictStrategyInternal::kLru ||
                        Strategy == npu::hkv::EvictStrategyInternal::kEpochLru)
                           ? static_cast<uint64_t>(GetSystemCycle())
                           : 0;

  asc_vf_call<upsert_kernel_lock_key_hybrid_vf<K, V, S, Strategy>>(
      dim3{THREAD_NUM_512}, buckets, buckets_size, capacity, bucket_max_size,
      dim, keys, scores, cur_score, n, global_epoch, GetBlockIdx(),
      max_bucket_shift, capacity_divisor_magic, capacity_divisor_shift,
      n_align_warp, thread_all, d_dst_values, d_dst_keys);
}

template <typename K>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_2048) inline void write_key_vf(
    uint64_t n, __gm__ K* keys, __gm__ K* __gm__* d_dst_keys,
    uint32_t thread_all, uint32_t block_index) {
  for (uint64_t i = block_index * blockDim.x + threadIdx.x; i < n;
       i += thread_all) {
    if (d_dst_keys[i] != nullptr) {
      *(d_dst_keys[i]) = keys[i];
    }
  }
}

template <class K, class V, bool UNLOCK_KEY>
__global__ __vector__ void write_with_half_ub_kernel(
    uint32_t former_num, uint64_t former_core_move_num,
    uint64_t tail_core_move_num, uint32_t num_tiles, uint32_t dim,
    __gm__ K* keys, __gm__ V* values, uint64_t n,
    __gm__ V* __gm__* d_dst_values, __gm__ K* __gm__* d_dst_keys) {
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
  AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 0>
      move_queue;

  pipe.InitBuffer(move_queue, DOUBLE_BUFFER, num_tiles * sizeof(V) * dim);
  AscendC::GlobalTensor<V> src_values_gm;
  AscendC::GlobalTensor<V> dst_values_gm;
  AscendC::LocalTensor<V> move_local;
  DataCopyPadExtParams<V> pad_params{true, 0, 0, 0};

  src_values_gm.SetGlobalBuffer(values);
  DataCopyExtParams copy_out_params{1, static_cast<uint32_t>(dim * sizeof(V)),
                                    0, 0, 0};
  uint64_t core_max_idx = core_start_idx + core_move_count;
  for (uint64_t i = core_start_idx; i < core_max_idx; i += num_tiles) {
    uint32_t move_in_num_once =
        i + num_tiles > core_max_idx ? core_max_idx - i : num_tiles;
    DataCopyExtParams copy_in_params{
        1, static_cast<uint32_t>(move_in_num_once * dim * sizeof(V)), 0, 0, 0};

    move_queue.AllocTensor<V>(move_local);
    AscendC::DataCopyPad(move_local, src_values_gm[i * dim], copy_in_params,
                         pad_params);
    move_queue.EnQue<V>(move_local);

    move_queue.DeQue<V>(move_local);
    for (uint32_t move_idx = 0; move_idx < move_in_num_once; move_idx++) {
      __gm__ V* dst_value = d_dst_values[i + move_idx];
      if (dst_value == nullptr) {
        continue;
      }

      dst_values_gm.SetGlobalBuffer(dst_value);
      AscendC::DataCopyPad(dst_values_gm, move_local[move_idx * dim],
                           copy_out_params);
    }

    move_queue.FreeTensor(move_local);
  }

  if constexpr (UNLOCK_KEY) {
    const uint32_t thread_all = THREAD_NUM_2048 * GetBlockNum();

    asc_vf_call<write_key_vf<K>>(dim3{THREAD_NUM_2048}, n, keys, d_dst_keys,
                                 thread_all, GetBlockIdx());
  }
}

template <class K, class V, bool UNLOCK_KEY>
__global__ __vector__ void write_kernel(
    uint32_t former_num, uint64_t former_core_move_num,
    uint64_t tail_core_move_num, uint32_t tile_size, uint32_t num_tiles,
    uint32_t dim, __gm__ K* keys, __gm__ V* values, uint64_t n,
    __gm__ V* __gm__* d_dst_values, __gm__ K* __gm__* d_dst_keys) {
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
  AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 0>
      move_queue;

  pipe.InitBuffer(move_queue, DOUBLE_BUFFER, tile_size * sizeof(V));
  AscendC::GlobalTensor<V> src_values_gm;
  AscendC::GlobalTensor<V> dst_values_gm;
  AscendC::LocalTensor<V> move_local;
  DataCopyPadExtParams<V> pad_params{true, 0, 0, 0};
  src_values_gm.SetGlobalBuffer(values);
  for (uint64_t i = core_start_idx; i < core_start_idx + core_move_count; i++) {
    __gm__ V* dst_value = d_dst_values[i];
    if (dst_value == nullptr) {
      continue;
    }

    uint64_t src_offset = i * dim;
    __gm__ V* dst_offset = dst_value;

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      uint32_t current_tile_size = (tile_idx == num_tiles - 1)
                                       ? (dim - tile_idx * tile_size)
                                       : tile_size;
      DataCopyExtParams copy_params{
          1, static_cast<uint32_t>(current_tile_size * sizeof(V)), 0, 0, 0};

      move_queue.AllocTensor<V>(move_local);
      AscendC::DataCopyPad(move_local,
                           src_values_gm[src_offset + tile_idx * tile_size],
                           copy_params, pad_params);
      move_queue.EnQue<V>(move_local);
      move_queue.DeQue<V>(move_local);

      dst_values_gm.SetGlobalBuffer(dst_offset + tile_idx * tile_size);
      AscendC::DataCopyPad(dst_values_gm, move_local, copy_params);

      move_queue.FreeTensor(move_local);
    }
  }

  if constexpr (UNLOCK_KEY) {
    const uint32_t thread_all = THREAD_NUM_2048 * GetBlockNum();

    asc_vf_call<write_key_vf<K>>(dim3{THREAD_NUM_2048}, n, keys, d_dst_keys,
                                 thread_all, GetBlockIdx());
  }
}

template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          bool MoveV, int32_t Strategy = -1, int32_t TILE_SIZE = 32>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_512) inline void upsert_kernel_vf(
    __gm__ Bucket<K, V, S>* buckets, __gm__ int32_t* buckets_size,
    uint64_t capacity, uint32_t bucket_max_size, uint32_t dim,
    __gm__ const K* keys, __gm__ const S* scores, S cur_score, uint64_t n,
    __gm__ V* values, __gm__ V* __gm__* d_dst_values, uint32_t thread_all,
    uint64_t global_epoch, uint32_t block_index, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift) {
  if (buckets == nullptr || buckets_size == nullptr || keys == nullptr) {
    return;
  }
  using BUCKET = Bucket<K, V, S>;
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  auto lane_id = threadIdx.x % TILE_SIZE;
  const uint64_t N = n * TILE_SIZE;

  for (uint64_t t = block_index * blockDim.x + threadIdx.x; t < N;
       t += thread_all) {
    uint64_t key_idx = t / TILE_SIZE;

    K key = keys[key_idx];
    if (IS_RESERVED_KEY<K>(key)) {
      continue;
    }
    S score = ScoreFunctor::desired_when_missed(scores, key_idx, global_epoch,
                                                cur_score);

    // 1. 计算 hash 值，定位 bucket
    const K hashed_key = Murmur3HashDevice(key);
    uint64_t global_idx = get_global_idx(hashed_key, capacity_divisor_magic,
                                         capacity_divisor_shift, capacity);
    uint32_t key_pos = global_idx & (bucket_max_size - 1);
    uint64_t bkt_idx = global_idx >> max_bucket_shift;

    __gm__ int32_t* bucket_size = buckets_size + bkt_idx;
    __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
    __gm__ K* bucket_keys = bucket->keys_;
    __gm__ V* bucket_vectors = bucket->vectors;
    __gm__ S* bucket_scores = bucket->scores_;

    // 2. 32 线程协作查找 key / 空位 / 淘汰
    K evicted_key = static_cast<K>(EMPTY_KEY);
    OccupyResult occupy_result;
    do {
      occupy_result = find_and_lock<K, S, TILE_SIZE>(
          bucket_keys, bucket_scores, bucket_max_size, key, score, key_pos,
          evicted_key, lane_id);
    } while (occupy_result == OccupyResult::CONTINUE);

    if (occupy_result == OccupyResult::REFUSED) {
      continue;
    }

    // 3. 新插入位置需要更新 bucket_size
    if ((occupy_result == OccupyResult::OCCUPIED_EMPTY ||
         occupy_result == OccupyResult::OCCUPIED_RECLAIMED) &&
        lane_id == 0) {
      asc_atomic_add(bucket_size, 1);
    }

    if constexpr (MoveV) {
      copy_vector<V, TILE_SIZE>(values + key_idx * dim,
                                bucket_vectors + key_pos * dim, dim, lane_id);
    }

    // 4. 更新，解锁key
    if (lane_id == 0) {
      if constexpr (!MoveV) {
        d_dst_values[key_idx] = bucket_vectors + key_pos * dim;
      }
      ScoreFunctor::update_with_digest(
          bucket_keys, key_pos, scores, key_idx, score, bucket_max_size,
          get_digest<K>(key), (occupy_result != OccupyResult::DUPLICATE));

      asc_threadfence();

      (void)asc_atomic_exch(bucket_keys + key_pos, key);
    }
  }
}

template <class K, class V, class S, bool MoveV, int Strategy = -1>
__global__ __vector__ void upsert_kernel(
    __gm__ Bucket<K, V, S>* buckets, __gm__ int* buckets_size,
    uint64_t capacity, uint32_t bucket_max_size, uint32_t dim,
    __gm__ const K* keys, __gm__ const S* scores, uint64_t n, __gm__ V* values,
    __gm__ V* __gm__* d_dst_values, uint64_t global_epoch,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift) {
  const uint32_t thread_all = THREAD_NUM_512 * GetBlockNum();
  uint64_t cur_score = (Strategy == npu::hkv::EvictStrategyInternal::kLru ||
                        Strategy == npu::hkv::EvictStrategyInternal::kEpochLru)
                           ? static_cast<uint64_t>(GetSystemCycle())
                           : 0;

  asc_vf_call<upsert_kernel_vf<K, V, S, MoveV, Strategy>>(
      dim3{static_cast<uint32_t>(THREAD_NUM_512)}, buckets, buckets_size,
      capacity, bucket_max_size, dim, keys, scores, cur_score, n, values,
      d_dst_values, thread_all, global_epoch, GetBlockIdx(), max_bucket_shift,
      capacity_divisor_magic, capacity_divisor_shift);
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_INSERT_OR_ASSIGN_KERNEL_H_
