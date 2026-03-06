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

#include "./v35/find_miss_with_digest_kernel.h"
#include "kernel_operator.h"
#include "../../include/simt_vf_dispatcher.h"

using namespace npu::hkv;

extern "C" __global__ __aicore__ void find_miss_with_digest_kernel(
    GM_ADDR buckets, uint64_t capacity, uint64_t buckets_num,
    uint32_t bucket_capacity, uint32_t dim, GM_ADDR keys, GM_ADDR values,
    GM_ADDR scores, GM_ADDR missed_keys, GM_ADDR missed_indices,
    GM_ADDR missed_size, uint64_t n, uint64_t global_epoch,
    uint32_t value_size, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift,
    uint64_t n_align_warp, int32_t group_size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  const uint64_t thread_all = FIND_MISS_THREAD_NUM * GetBlockNum();

  DISPATCH_GROUP_SIZE(
      group_size,
      DISPATCH_VALUE_SIZE(
          value_size,
          (Simt::VF_CALL<find_miss_with_digest_kernel_vf<
               uint64_t, DTYPE, uint64_t, GROUP_SIZE>>(
              Simt::Dim3{static_cast<uint32_t>(FIND_MISS_THREAD_NUM), 1, 1},
              buckets, capacity, buckets_num, bucket_capacity, dim, keys,
              values, scores, missed_keys, missed_indices, missed_size,
              n, thread_all, global_epoch,
              GetBlockIdx(), max_bucket_shift, capacity_divisor_magic,
              capacity_divisor_shift, n_align_warp))));
}
