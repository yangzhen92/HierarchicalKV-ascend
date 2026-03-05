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

#ifndef ASCENDC_CONTAINS_KERNEL_H_
#define ASCENDC_CONTAINS_KERNEL_H_

#include <cstdint>
#include <kernel_operator.h>
#include "../../../include/types.h"
#include "../../../include/utils.h"
#include "../../utils_kernel/v35/utils_kernel.h"

namespace npu {
namespace hkv {
using namespace AscendC;

constexpr uint32_t CONTAINS_THREAD_NUM = 512;

template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          uint32_t THREAD_NUM = CONTAINS_THREAD_NUM>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void contains_kernel_vf(
    GM_ADDR buckets_gm, uint64_t capacity, uint64_t buckets_num,
    uint32_t bucket_capacity, GM_ADDR keys_gm, GM_ADDR founds_gm,
    uint64_t n, const uint64_t total_thread_num,
    uint32_t block_id, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift) {
  if (buckets_gm == nullptr || keys_gm == nullptr || founds_gm == nullptr) {
    return;
  }

  __gm__ Bucket<K, V, S>* __restrict__ buckets =
      reinterpret_cast<__gm__ Bucket<K, V, S>*>(buckets_gm);
  __gm__ K* __restrict__ keys =
      reinterpret_cast<__gm__ K*>(keys_gm);
  __gm__ bool* __restrict__ founds =
      reinterpret_cast<__gm__ bool*>(founds_gm);

  for (uint64_t kv_idx = block_id * blockDim.x + threadIdx.x;
       kv_idx < n; kv_idx += total_thread_num) {
    bool found = false;

    const K key = ldg_l2nc_l1c(keys + kv_idx);
    if (!IS_RESERVED_KEY<K>(key)) {
      const K hashed_key = Murmur3HashDevice(key);
      VecD_Comp target_digests = digests_from_hashed<K>(hashed_key);
      uint64_t global_idx = get_global_idx(hashed_key, capacity_divisor_magic,
                                           capacity_divisor_shift, capacity);
      uint32_t key_pos = global_idx & (bucket_capacity - 1);
      uint64_t bkt_idx = global_idx >> max_bucket_shift;

      __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;

      uint32_t target_pos = INVALID_KEY_POS;
      OccupyResult result = find_without_lock<K, V, S>(
          bucket, key, key_pos, target_digests, target_pos, bucket_capacity);
      found = (result == OccupyResult::DUPLICATE);
    }

    founds[kv_idx] = found;
  }
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_CONTAINS_KERNEL_H_
