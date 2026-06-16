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

#ifndef ASCENDC_MTE_TASK_QUEUE_H_
#define ASCENDC_MTE_TASK_QUEUE_H_

#include <cstdint>
#include "kernel_operator.h"
#include "utils.h"

namespace npu {
namespace hkv {

using namespace AscendC;

// RING_BUFFER_SZ 必须是2的整数次幂，且必须大于等于8，否则内存不按32对齐
template <uint32_t RING_BUFFER_SZ = 512>
class MTETaskQueue {
public:
  __aicore__ MTETaskQueue() {}

  __aicore__ inline virtual void init() {
    *reinterpret_cast<__ubuf__ volatile uint32_t*>(GLOBAL_ID_ADDR) = 0;
    *reinterpret_cast<__ubuf__ volatile uint32_t*>(MTE_FINISH_ID_ADDR) = 0;
    *reinterpret_cast<__ubuf__ volatile uint32_t*>(SIMT_ASSIGN_ID_ADDR) = 0;
    *reinterpret_cast<__ubuf__ volatile uint32_t*>(TASK_FINISH_FLAG_ADDR) = 0;
  }

  __simt_callee__ inline uint32_t apply_ID(const uint32_t &warp_task_count) {
    return asc_atomic_add(reinterpret_cast<__ubuf__ uint32_t *>(GLOBAL_ID_ADDR), warp_task_count);
  }

  __simt_callee__ inline void fill_task_info(uint32_t task_id, const uint64_t &in_ddr, const uint64_t &out_ddr) {
    reinterpret_cast<__ubuf__ volatile uint64_t*>(RING_BUFFER_IN_ADDR_QUEUE_ADDR)[task_id] = in_ddr;
    reinterpret_cast<__ubuf__ volatile uint64_t*>(RING_BUFFER_OUT_ADDR_QUEUE_ADDR)[task_id] = out_ddr;
  }

  __simt_callee__ inline void assign_task(const uint32_t &new_task_start_id, const uint32_t &new_task_end_id) {
    while (true) {
      // 任务数据填充完成，下发任务
      if (atomicCAS(reinterpret_cast<__ubuf__ uint32_t*>(SIMT_ASSIGN_ID_ADDR), new_task_start_id, new_task_end_id) == new_task_start_id) {
        return;
      }
    }
  }

  __simt_callee__ inline void assignment_finish() {
    *reinterpret_cast<__ubuf__ volatile uint32_t*>(TASK_FINISH_FLAG_ADDR) = FINISH_FLAG;
  }

  __aicore__ inline void run_task(const uint32_t &value_size) {
    int value_size_aligned = (value_size + UB_ALIGN_SIZE - 1) / UB_ALIGN_SIZE * UB_ALIGN_SIZE;
    uint32_t max_proc_batch = DATA_BUFFER_SZ / (value_size_aligned);
    max_proc_batch = min(max_proc_batch, queue_limited);

    __ubuf__ volatile uint32_t* simt_assign_id_addr = reinterpret_cast<__ubuf__ volatile uint32_t*>(SIMT_ASSIGN_ID_ADDR);
    __ubuf__ volatile uint32_t* mte_finish_id_addr = reinterpret_cast<__ubuf__ volatile uint32_t*>(MTE_FINISH_ID_ADDR);
    __ubuf__ volatile uint32_t* task_finish_flag_addr = reinterpret_cast<__ubuf__ volatile uint32_t*>(TASK_FINISH_FLAG_ADDR);
    __ubuf__ volatile uint64_t* ring_buffer_in_addr_queue_addr = reinterpret_cast<__ubuf__ volatile uint64_t*>(RING_BUFFER_IN_ADDR_QUEUE_ADDR);
    __ubuf__ volatile uint64_t* ring_buffer_out_addr_queue_addr = reinterpret_cast<__ubuf__ volatile uint64_t*>(RING_BUFFER_OUT_ADDR_QUEUE_ADDR);
    uint32_t mte_finish_id = 0;
    uint32_t simt_assign_id = 0;
    while (true) {
      simt_assign_id = *simt_assign_id_addr;
      if (simt_assign_id == mte_finish_id) {
        if (*task_finish_flag_addr == FINISH_FLAG) {
          return;
        } else {
          continue;
        }
      }

      uint32_t cur_batch = min(max_proc_batch, simt_assign_id - mte_finish_id);
      for (uint32_t i = 0; i < cur_batch; ++i) {
        uint32_t slot_pos = (mte_finish_id + i) & RING_BUFFER_MASK;
        copy_gm_to_ubuf_align_v2(
          reinterpret_cast<__ubuf__ uint8_t*>(DATA_BUFFER_START_ADDR + i * value_size_aligned),
          reinterpret_cast<__gm__ uint8_t*>(ring_buffer_in_addr_queue_addr[slot_pos]),
          0, 1, value_size, 0, 0, 0, 0, 0, 0);
      }

      set_flag(PIPE_MTE2, PIPE_MTE3, 0);
      wait_flag(PIPE_MTE2, PIPE_MTE3, 0);

      for (uint32_t i = 0; i < cur_batch; ++i) {
        uint32_t slot_pos = (mte_finish_id + i) & RING_BUFFER_MASK;
        copy_ubuf_to_gm_align_v2(
            reinterpret_cast<__gm__ uint8_t*>(ring_buffer_out_addr_queue_addr[slot_pos]),
            reinterpret_cast<__ubuf__ uint8_t*>(DATA_BUFFER_START_ADDR + i * value_size_aligned),
            0, 1, value_size, 0, 0, 0);
      }

      set_flag(PIPE_MTE3, PIPE_S, 0);
      wait_flag(PIPE_MTE3, PIPE_S, 0);

      // 消费完成，累加消费进展
      mte_finish_id += cur_batch;
      *mte_finish_id_addr = mte_finish_id;
    }
  }

protected:
  const uint32_t queue_limited = 32;  // 一次最多处理32个任务
  const uint32_t FINISH_FLAG = 1;  // 任务完成标志1，未完成0
  const uint32_t RING_BUFFER_MASK = RING_BUFFER_SZ - 1;
  const uint32_t DATA_BUFFER_SZ = 128 * 1024;  // 默认缓冲区大小128KB，可根据实际情况调整

  // 队列数据
  const uint32_t RING_BUFFER_IN_ADDR_QUEUE_ADDR = 0;
  const uint32_t RING_BUFFER_OUT_ADDR_QUEUE_ADDR = RING_BUFFER_IN_ADDR_QUEUE_ADDR + RING_BUFFER_SZ * sizeof(uint64_t);

  // 控制数据
  // 全局任务ID
  const uint32_t GLOBAL_ID_ADDR = RING_BUFFER_OUT_ADDR_QUEUE_ADDR + RING_BUFFER_SZ * sizeof(uint64_t);
  // MTE已完成ID
  const uint32_t MTE_FINISH_ID_ADDR = GLOBAL_ID_ADDR + UB_ALIGN_SIZE;
  // SIMT下发ID
  const uint32_t SIMT_ASSIGN_ID_ADDR = MTE_FINISH_ID_ADDR + UB_ALIGN_SIZE;
  const uint32_t TASK_FINISH_FLAG_ADDR = SIMT_ASSIGN_ID_ADDR + UB_ALIGN_SIZE;

  // buffer区
  const uint32_t DATA_BUFFER_START_ADDR = TASK_FINISH_FLAG_ADDR + UB_ALIGN_SIZE;
  const uint32_t DATA_BUFFER_END_ADDR = DATA_BUFFER_START_ADDR + DATA_BUFFER_SZ;
};

}
}

#endif