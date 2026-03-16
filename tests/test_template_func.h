#pragma once
#include <cstdint>
#include "../include/cuda2npu.h"
#include "../include/types.h"

template <typename K, typename V, typename S>
struct ForEachExecutionFunc {
  __gm__ V* target_value;
  uint32_t dim;
  K target_key;

  __device__ void operator()(const K& key, __gm__ V* value, __gm__ S*, int) {
    if (key == target_key) {
      for (uint32_t i = 0; i < dim; ++i) {
        target_value[i] = value[i];
      }
    }
  }
};

template <typename K, typename V, typename S>
struct ForEachScoresFilterFunc {
  __gm__ uint64_t* count;
  S threshold;

  __device__ void operator()(const K& key, __gm__ V*, __gm__ S* score, int) {
    S score_val = *score;
    bool match = (!npu::hkv::IS_RESERVED_KEY(key) && score_val >= threshold);
    uint32_t vote = asc_ballot(match);
    int32_t group_count = AscendC::Simt::Popc(vote);
    if (threadIdx.x % warpSize == 0) {
      atomicAdd(count, group_count);
    }
  }
};
