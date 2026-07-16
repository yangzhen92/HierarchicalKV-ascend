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

/*
 * MteTaskPipelineIntf
 * 定义MTE任务队列的接口，参考cuda::pipeline的接口定义与使用范式
 * 但有以下不同点：
 * 1）cuda粒度为thread，本接口粒度为warp级别；
 * 2）接口新增额外入参并且新增部分额外接口如init、run_task等；
 * 不同原因在于：
 * 1）thread粒度会引入更多的atomic操作，导致性能下降；
 * 2）节省寄存器，NPU寄存器吃紧，尽量避免类中带有私有数据成员；
 * 3）由于本质是驱动scalar下发mte指令，因此需额外设计__aicore__接口；
 */
class MteTaskPipelineIntf {
public:
  // 初始化任务控制数据
  __aicore__ inline virtual void init() = 0;

  // 获取warp级别需要下发任务的mask
  __simt_callee__ inline virtual uint32_t get_acquire_mask(const bool &need_assign) const = 0;

  // 申请任务ID
  __simt_callee__ inline virtual void producer_acquire(const bool &need_assign, const uint32_t &warp_acquire_mask, uint32_t &curr_simt_alloc_pos, uint32_t &new_simt_alloc_pos, uint32_t &assign_task_id) = 0;

  // 下发任务
  __simt_callee__ inline virtual void memcpy_async(const bool &need_assign, const uint32_t & assign_task_id, void *dst, const void *src, size_t len) = 0;

  // 提交任务
  __simt_callee__ inline virtual void producer_commit(const uint32_t & curr_simt_alloc_pos, const uint32_t & new_simt_alloc_pos) = 0;

  // 等待任务完成
  __simt_callee__ inline virtual void consumer_wait(const uint32_t& new_task_end_id) = 0;

  // 释放任务空间
  __simt_callee__ inline virtual void consumer_release(const uint32_t& new_task_end_id) = 0;

  // 下发任务结束
  __simt_callee__ inline virtual void producer_finish() = 0;

  // scalar执行任务
  __aicore__ inline virtual void run_task(const uint32_t len = 0) = 0;
};

// MTE搬运任务队列，默认仅支持GM->GM任务
// RING_BUFFER_SZ 必须是2的整数次幂，且必须大于等于8，否则内存不按32对齐
template <uint32_t RING_BUFFER_SZ = 512>
class GM2GMTaskPipelineQueueBase : public MteTaskPipelineIntf {
 public:
  __aicore__ GM2GMTaskPipelineQueueBase() {}

  // 初始化控制量
  __aicore__ inline void init() override {
    *reinterpret_cast<__ubuf__ volatile uint32_t*>(GLOBAL_ID_ADDR) = 0;
    *reinterpret_cast<__ubuf__ volatile uint32_t*>(MTE_FINISH_ID_ADDR) = 0;
    *reinterpret_cast<__ubuf__ volatile uint32_t*>(SIMT_ASSIGN_ID_ADDR) = 0;
    *reinterpret_cast<__ubuf__ volatile uint32_t*>(TASK_FINISH_FLAG_ADDR) = 0;
  }

  __simt_callee__ inline uint32_t get_acquire_mask(const bool &need_assign) const override {
    return asc_ballot(need_assign);
  }

  // 申请任务ID
  __simt_callee__ inline void producer_acquire(const bool &need_assign, const uint32_t &warp_acquire_mask, uint32_t &curr_simt_alloc_pos, uint32_t &new_simt_alloc_pos, uint32_t &assign_task_id) override {
    // 1. 统计warp内有多少调用本接口的线程
    uint32_t warp_acquire_count = __popc(warp_acquire_mask);

    // 2. 申请任务ID
    if (laneid() == 0) {
      curr_simt_alloc_pos = 
          asc_atomic_add(reinterpret_cast<__ubuf__ uint32_t*>(GM2GMTaskPipelineQueueBase<RING_BUFFER_SZ>::GLOBAL_ID_ADDR), warp_acquire_count);
    }
    curr_simt_alloc_pos = __shfl(curr_simt_alloc_pos, 0);

    assign_task_id = curr_simt_alloc_pos + __popc(warp_acquire_mask & lanemask_lt());
    assign_task_id &= GM2GMTaskPipelineQueueBase<RING_BUFFER_SZ>::RING_BUFFER_MASK;

    // 3. 等待任务队列空间足够
    new_simt_alloc_pos = curr_simt_alloc_pos + warp_acquire_count;
    drain(new_simt_alloc_pos);
  }

  __simt_callee__ inline void memcpy_async(const bool &need_assign, const uint32_t & assign_task_id, void *dst, const void *src, size_t) override {
    if (!need_assign) {
      return;
    }

    reinterpret_cast<__ubuf__ volatile uint64_t*>(
        RING_BUFFER_IN_ADDR_QUEUE_ADDR)[assign_task_id] = reinterpret_cast<uint64_t>(src);
    reinterpret_cast<__ubuf__ volatile uint64_t*>(
        RING_BUFFER_OUT_ADDR_QUEUE_ADDR)[assign_task_id] = reinterpret_cast<uint64_t>(dst);
  }

  __simt_callee__ inline void producer_commit(const uint32_t & curr_simt_alloc_pos, const uint32_t & new_simt_alloc_pos) override {
    // 确保数据写入UB
    __threadfence();

    if (laneid() == 0) {
      while (true) {
        if (atomicCAS(reinterpret_cast<__ubuf__ uint32_t*>(GM2GMTaskPipelineQueueBase<RING_BUFFER_SZ>::SIMT_ASSIGN_ID_ADDR),
                      curr_simt_alloc_pos, new_simt_alloc_pos) == curr_simt_alloc_pos) {
          return;
        }
      }
    }
  }

  __simt_callee__ inline virtual void drain(const uint32_t &new_task_end_id) {
    while (true) {
      uint32_t finish_id = *reinterpret_cast<__ubuf__ volatile uint32_t*>(MTE_FINISH_ID_ADDR);
      if (new_task_end_id - finish_id <= RING_BUFFER_SZ) {
        break;
      }
    }
  }

  __simt_callee__ inline void consumer_wait(const uint32_t& new_task_end_id) override {
  }

  __simt_callee__ inline void consumer_release(const uint32_t& new_task_end_id) override {
  }

  __simt_callee__ inline void producer_finish() override {
    *reinterpret_cast<__ubuf__ volatile uint32_t*>(TASK_FINISH_FLAG_ADDR) =
        FINISH_FLAG;
  }

  __aicore__ inline void run_task(const uint32_t value_size = 0) override {
    uint32_t value_size_aligned =
        (value_size + UB_ALIGN_SIZE - 1) / UB_ALIGN_SIZE * UB_ALIGN_SIZE;
    uint32_t max_proc_batch = DATA_BUFFER_SZ / (value_size_aligned);
    max_proc_batch = min(max_proc_batch, queue_limited);

    __ubuf__ volatile uint32_t* simt_assign_id_addr =
        reinterpret_cast<__ubuf__ volatile uint32_t*>(SIMT_ASSIGN_ID_ADDR);
    __ubuf__ volatile uint32_t* mte_finish_id_addr =
        reinterpret_cast<__ubuf__ volatile uint32_t*>(MTE_FINISH_ID_ADDR);
    __ubuf__ volatile uint32_t* task_finish_flag_addr =
        reinterpret_cast<__ubuf__ volatile uint32_t*>(TASK_FINISH_FLAG_ADDR);
    __ubuf__ volatile uint64_t* ring_buffer_in_addr_queue_addr =
        reinterpret_cast<__ubuf__ volatile uint64_t*>(
            RING_BUFFER_IN_ADDR_QUEUE_ADDR);
    __ubuf__ volatile uint64_t* ring_buffer_out_addr_queue_addr =
        reinterpret_cast<__ubuf__ volatile uint64_t*>(
            RING_BUFFER_OUT_ADDR_QUEUE_ADDR);
    uint32_t mte_finish_id = 0;
    uint32_t simt_assign_id = 0;
    GlobalTensor<uint8_t> gm_tensor;
    DataCopyParams data_copy_params(1, value_size, 0, 0);
    DataCopyPadParams data_copy_pad_params(false, 0, 0, 0);
    while (true) {
      simt_assign_id = *simt_assign_id_addr;
      if (simt_assign_id == mte_finish_id) {
        if (*task_finish_flag_addr == FINISH_FLAG && *simt_assign_id_addr == simt_assign_id) {
          return;
        } else {
          continue;
        }
      }
      
      uint32_t cur_batch = min(max_proc_batch, simt_assign_id - mte_finish_id);
      for (uint32_t i = 0; i < cur_batch; ++i) {
        uint32_t slot_pos = (mte_finish_id + i) & RING_BUFFER_MASK;
        LocalTensor<uint8_t> local_tensor(TPosition::VECCALC, static_cast<uint32_t>(DATA_BUFFER_START_ADDR +
                                          i * value_size_aligned), static_cast<uint32_t>(0));
        gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(
                ring_buffer_in_addr_queue_addr[slot_pos]));
        DataCopyPad(local_tensor, gm_tensor, data_copy_params, data_copy_pad_params);
      }

      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

      for (uint32_t i = 0; i < cur_batch; ++i) {
        uint32_t slot_pos = (mte_finish_id + i) & RING_BUFFER_MASK;
        LocalTensor<uint8_t> local_tensor(TPosition::VECCALC, static_cast<uint32_t>(DATA_BUFFER_START_ADDR +
                                                i * value_size_aligned), static_cast<uint32_t>(0));
        gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(
                ring_buffer_out_addr_queue_addr[slot_pos]));
        DataCopyPad(gm_tensor, local_tensor, data_copy_params);
      }

      set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);

      // 消费完成，累加消费进展
      mte_finish_id += cur_batch;
      *mte_finish_id_addr = mte_finish_id;
    }
  }

 protected:
  const uint32_t queue_limited = 32;  // 一次最多处理32个任务
  const uint32_t FINISH_FLAG = 1;     // 任务完成标志1，未完成0
  const uint32_t RING_BUFFER_MASK = RING_BUFFER_SZ - 1;
  const uint32_t DATA_BUFFER_SZ = MTE_UB_BUFFER_SZ;

  // 队列数据
  // 各任务src/dst地址
  const uint32_t RING_BUFFER_IN_ADDR_QUEUE_ADDR = 0;
  const uint32_t RING_BUFFER_OUT_ADDR_QUEUE_ADDR =
      RING_BUFFER_IN_ADDR_QUEUE_ADDR + RING_BUFFER_SZ * sizeof(uint64_t);

  // 控制数据
  // 全局任务ID
  const uint32_t GLOBAL_ID_ADDR =
      RING_BUFFER_OUT_ADDR_QUEUE_ADDR + RING_BUFFER_SZ * sizeof(uint64_t);
  // MTE已完成ID
  const uint32_t MTE_FINISH_ID_ADDR = GLOBAL_ID_ADDR + UB_ALIGN_SIZE;
  // SIMT下发ID
  const uint32_t SIMT_ASSIGN_ID_ADDR = MTE_FINISH_ID_ADDR + UB_ALIGN_SIZE;
  // 任务完成标记
  const uint32_t TASK_FINISH_FLAG_ADDR = SIMT_ASSIGN_ID_ADDR + UB_ALIGN_SIZE;

  // buffer区
  const uint32_t DATA_BUFFER_START_ADDR = TASK_FINISH_FLAG_ADDR + UB_ALIGN_SIZE;
  const uint32_t DATA_BUFFER_END_ADDR = DATA_BUFFER_START_ADDR + DATA_BUFFER_SZ;
};

}  // namespace hkv
}  // namespace npu

#endif