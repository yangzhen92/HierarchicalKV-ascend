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
#include "../../../include/types.h"
#include "../../../include/utils.h"

namespace npu {
namespace hkv {
using namespace AscendC;

constexpr uint32_t FIND_THREAD_NUM = 512;

template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          int32_t GROUP_SIZE = 16, uint32_t THREAD_NUM = FIND_THREAD_NUM>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void find_with_digest_kernel_vf(
    GM_ADDR buckets_gm, uint64_t capacity, uint64_t buckets_num,
    uint32_t bucket_capacity, uint32_t dim, GM_ADDR keys_gm,
    GM_ADDR values_gm, GM_ADDR scores_gm, GM_ADDR founds_gm,
    uint64_t n, const uint64_t total_thread_num,
    uint64_t global_epoch, uint32_t block_id,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift, uint64_t n_align_warp) {
  if (buckets_gm == nullptr || keys_gm == nullptr || values_gm == nullptr) {
    return;
  }
  using BUCKET = Bucket<K, V, S>;
  constexpr uint32_t STRIDE = sizeof(VecD_Comp) / sizeof(D);

  __gm__ Bucket<K, V, S>* __restrict__ buckets =
      reinterpret_cast<__gm__ Bucket<K, V, S>*>(buckets_gm);
  __gm__ K* __restrict__ keys =
      reinterpret_cast<__gm__ K*>(keys_gm);
  __gm__ V* __restrict__ values =
      reinterpret_cast<__gm__ V*>(values_gm);
  __gm__ S* __restrict__ scores =
      reinterpret_cast<__gm__ S*>(scores_gm);
  __gm__ bool* __restrict__ founds =
      reinterpret_cast<__gm__ bool*>(founds_gm);

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
        VecD_Comp empty_digests_val = empty_digests<K>();
        for (uint32_t offset = 0; offset < bucket_capacity + STRIDE;
             offset += STRIDE) {
          uint32_t pos_cur = align_to<STRIDE>(key_pos);
          pos_cur = (pos_cur + offset) & (bucket_capacity - 1);

          __gm__ D* digests_ptr =
              BUCKET::digests(bucket->keys_, bucket_capacity, pos_cur);
          VecD_Comp probe_digests =
              ldg_l2nc_l1c(reinterpret_cast<__gm__ VecD_Comp*>(digests_ptr));
          uint32_t possible_pos = 0;
          uint32_t cmp_result = vcmpeq4(probe_digests, target_digests);
          cmp_result &= 0x01010101;
          do {
            if (cmp_result == 0) {
              break;
            }
            uint32_t index =
                (AscendC::Simt::Ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
            cmp_result &= (cmp_result - 1);
            possible_pos = pos_cur + index;
            const K current_key = ldg_l2nc_l1c(bucket->keys_ + possible_pos);
            if (current_key == key) {
              found = true;
              target_pos = possible_pos;
              goto WRITE_BACK;
            }
          } while (true);
          cmp_result = vcmpeq4(probe_digests, empty_digests_val);
          cmp_result &= 0x01010101;
          do {
            if (cmp_result == 0) {
              break;
            }
            uint32_t index =
                (AscendC::Simt::Ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
            cmp_result &= (cmp_result - 1);
            possible_pos = pos_cur + index;
            if (offset == 0 && possible_pos < key_pos) {
              continue;
            }
            const K current_key = ldg_l2nc_l1c(bucket->keys_ + possible_pos);
            if (current_key == static_cast<K>(EMPTY_KEY)) {
              goto WRITE_BACK;
            }
          } while (true);
        }

WRITE_BACK:
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
    auto cg_rank_id = threadIdx.x % GROUP_SIZE;
    for (int32_t i = 0; i < GROUP_SIZE; i++) {
      bool found_sync = __shfl(found, i, GROUP_SIZE);
      if (found_sync) {
        auto kv_idx_sync = __shfl(kv_idx, i, GROUP_SIZE);
        auto target_pos_sync = __shfl(target_pos, i, GROUP_SIZE);
        uint64_t bucket_values_sync =
            __shfl(bucket_values_uintptr, i, GROUP_SIZE);
        auto value_start = kv_idx_sync * dim;
        auto bucket_value_start = target_pos_sync * dim;
        for (uint32_t j = cg_rank_id; j < dim; j += GROUP_SIZE) {
          values[value_start + j] =
              __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                    L1CacheType::NON_CACHEABLE>(
                  reinterpret_cast<__gm__ V*>(bucket_values_sync) +
                  bucket_value_start + j);
        }
      }
    }
  }
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_FIND_WITH_DIGEST_KERNEL_H_
