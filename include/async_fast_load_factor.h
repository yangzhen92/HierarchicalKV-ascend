/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <vector>
#include "aclnn_helper.h"
#include "aclnnop/aclnn_reduce_sum.h"
#include "debug.h"

namespace npu {
namespace hkv {

/**
 * @brief Maintains asynchronous sampled load-factor calculation state.
 *
 * The component owns one reusable Event, output pair, and ACLNN workspace.
 * Calls are serialized on the Host, while submitted Device work remains
 * asynchronous. A null stream is passed through to CANN as the default stream
 * of the current Context.
 */
class AsyncFastLoadFactor {
 public:
  AsyncFastLoadFactor() = default;

  ~AsyncFastLoadFactor() { release(); }

  AsyncFastLoadFactor(const AsyncFastLoadFactor&) = delete;
  AsyncFastLoadFactor& operator=(const AsyncFastLoadFactor&) = delete;
  AsyncFastLoadFactor(AsyncFastLoadFactor&&) = delete;
  AsyncFastLoadFactor& operator=(AsyncFastLoadFactor&&) = delete;

  /**
   * @brief Returns an approximate load factor without waiting for a newly
   * submitted sample.
   */
  float calculate(size_t delta, void* buckets_size, size_t buckets_num,
                  size_t max_bucket_size, size_t capacity, aclrtStream stream) {
    HKV_CHECK(capacity != 0, "HashTable capacity must not be zero.");
    HKV_CHECK(max_bucket_size != 0,
              "HashTable max bucket size must not be zero.");

    float cached_load_factor = 0.0f;
    size_t cached_capacity = 0;

    {
      std::lock_guard<std::mutex> lock(mutex_);

      HKV_CHECK(!cleanup_required_,
                "Fast load factor resources require failed-task cleanup.");
      initialize();
      collect_completed(max_bucket_size);

      if (!sample_in_flight_) {
        const size_t bucket_num = std::min(buckets_num, size_t{1024});
        if (bucket_num == 0) {
          cached_load_factor_ = 0.0f;
          cached_capacity_ = capacity;
        } else {
          HKV_CHECK(buckets_size != nullptr,
                    "HashTable buckets_size must not be null.");
          launch_sample(buckets_size, bucket_num, capacity, stream);
        }
      }

      cached_load_factor = cached_load_factor_;
      cached_capacity = cached_capacity_;
    }

    const double capacity_value = static_cast<double>(capacity);
    const double previous_load = static_cast<double>(cached_load_factor) *
                                 static_cast<double>(cached_capacity) /
                                 capacity_value;
    const double increment_load =
        static_cast<double>(delta) * 2.0 / capacity_value;
    return static_cast<float>(previous_load + increment_load);
  }

  /**
   * @brief Waits for the pending sample, then releases owned resources.
   *
   * HashTable must call this before destroying buckets_size. The method is
   * idempotent so the component destructor can call it again safely.
   */
  void release() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      return;
    }

    if (cleanup_required_) {
      const aclError status = aclrtSynchronizeStream(cleanup_stream_);
      if (status != ACL_SUCCESS) {
        HKV_LOG_ERROR(
            "Failed to clean up partially submitted fast load factor task, "
            "error={}",
            status);
        std::terminate();
      }
      cleanup_required_ = false;
      cleanup_stream_ = nullptr;
    }

    if (sample_in_flight_) {
      const aclError status = aclrtSynchronizeEvent(event_);
      if (status != ACL_SUCCESS) {
        HKV_LOG_ERROR("Failed to synchronize fast load factor Event, error={}",
                      status);
        std::terminate();
      }
      sample_in_flight_ = false;
    }

    if (workspace_ != nullptr) {
      const aclError status = aclrtFree(workspace_);
      if (status != ACL_SUCCESS) {
        HKV_LOG_ERROR("Failed to free fast load factor workspace, error={}",
                      status);
        std::terminate();
      }
      CHECK_MEM_RELEASE(workspace_);
      workspace_ = nullptr;
      workspace_capacity_ = 0;
    }

    if (device_out_ != nullptr) {
      const aclError status = aclrtFree(device_out_);
      if (status != ACL_SUCCESS) {
        HKV_LOG_ERROR("Failed to free fast load factor Device output, error={}",
                      status);
        std::terminate();
      }
      CHECK_MEM_RELEASE(device_out_);
      device_out_ = nullptr;
    }

    if (host_out_ != nullptr) {
      const aclError status = aclrtFreeHost(host_out_);
      if (status != ACL_SUCCESS) {
        HKV_LOG_ERROR("Failed to free fast load factor Host output, error={}",
                      status);
        std::terminate();
      }
      CHECK_MEM_RELEASE(host_out_);
      host_out_ = nullptr;
    }

    const aclError status = aclrtDestroyEvent(event_);
    if (status != ACL_SUCCESS) {
      HKV_LOG_ERROR("Failed to destroy fast load factor Event, error={}",
                    status);
      std::terminate();
    }
    event_ = nullptr;
    initialized_ = false;
    cached_load_factor_ = 0.0f;
    cached_capacity_ = 0;
    inflight_capacity_ = 0;
    inflight_bucket_num_ = 0;
  }

 private:
  void initialize() {
    if (initialized_) {
      return;
    }

    aclrtEvent event = nullptr;
    int64_t* device_out = nullptr;
    int64_t* host_out = nullptr;

    try {
      NPU_CHECK(aclrtCreateEvent(&event));

      void* device_ptr = nullptr;
      NPU_CHECK(
          aclrtMalloc(&device_ptr, sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST));
      device_out = static_cast<int64_t*>(device_ptr);
      CHECK_MEM_ALLOC(device_out);

      void* host_ptr = nullptr;
      NPU_CHECK(aclrtMallocHost(&host_ptr, sizeof(int64_t)));
      host_out = static_cast<int64_t*>(host_ptr);
      CHECK_MEM_ALLOC(host_out);
    } catch (...) {
      if (host_out != nullptr) {
        (void)aclrtFreeHost(host_out);
        CHECK_MEM_RELEASE(host_out);
      }
      if (device_out != nullptr) {
        (void)aclrtFree(device_out);
        CHECK_MEM_RELEASE(device_out);
      }
      if (event != nullptr) {
        (void)aclrtDestroyEvent(event);
      }
      throw;
    }

    event_ = event;
    device_out_ = device_out;
    host_out_ = host_out;
    initialized_ = true;
  }

  void ensure_workspace(size_t workspace_size) {
    if (workspace_size <= workspace_capacity_) {
      return;
    }

    void* workspace = nullptr;
    NPU_CHECK(
        aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_MEM_ALLOC(workspace);

    if (workspace_ != nullptr) {
      const aclError status = aclrtFree(workspace_);
      if (status != ACL_SUCCESS) {
        const aclError cleanup_status = aclrtFree(workspace);
        if (cleanup_status == ACL_SUCCESS) {
          CHECK_MEM_RELEASE(workspace);
        } else {
          HKV_LOG_ERROR(
              "Failed to free newly allocated fast load factor workspace, "
              "error={}",
              cleanup_status);
        }
        NPU_CHECK(status);
      }
      CHECK_MEM_RELEASE(workspace_);
    }

    workspace_ = workspace;
    workspace_capacity_ = workspace_size;
  }

  void collect_completed(size_t max_bucket_size) {
    if (!sample_in_flight_) {
      return;
    }

    aclrtEventRecordedStatus status;
    NPU_CHECK(aclrtQueryEventStatus(event_, &status));

    if (status == ACL_EVENT_RECORDED_STATUS_NOT_READY) {
      return;
    }

    HKV_CHECK(status == ACL_EVENT_RECORDED_STATUS_COMPLETE,
              "Unexpected fast load factor Event status.");

    if (inflight_bucket_num_ == 0) {
      cached_load_factor_ = 0.0f;
    } else {
      cached_load_factor_ =
          static_cast<float>(*host_out_) /
          static_cast<float>(max_bucket_size * inflight_bucket_num_);
    }
    cached_capacity_ = inflight_capacity_;
    sample_in_flight_ = false;
  }

  void launch_sample(void* buckets_size, size_t bucket_num, size_t capacity,
                     aclrtStream stream) {
    DeviceTensor input;
    input.init(buckets_size, aclDataType::ACL_INT32,
               {static_cast<int64_t>(bucket_num)});

    DeviceTensor output;
    output.init(device_out_, aclDataType::ACL_INT64, {1});

    std::vector<int64_t> dims = {0};
    bool keep_dims = false;
    auto output_type = aclDataType::ACL_INT64;
    uint64_t workspace_size = 0;
    uint64_t* workspace_size_addr = &workspace_size;
    aclOpExecutor* executor = nullptr;
    aclOpExecutor** executor_addr = &executor;

    auto converted_params =
        convert_types(input, dims, keep_dims, output_type, output,
                      workspace_size_addr, executor_addr);
    TupleDeleter tuple_deleter(converted_params);

    auto workspace_status =
        call(aclnnReduceSumGetWorkspaceSize, converted_params);
    NPU_CHECK(workspace_status);
    ensure_workspace(static_cast<size_t>(workspace_size));

    bool task_submitted = false;
    try {
      NPU_CHECK(aclnnReduceSum(workspace_, workspace_size, executor, stream));
      task_submitted = true;

      NPU_CHECK(aclrtMemcpyAsync(host_out_, sizeof(int64_t), device_out_,
                                 sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST,
                                 stream));
      NPU_CHECK(aclrtRecordEvent(event_, stream));
    } catch (...) {
      if (task_submitted) {
        const aclError status = aclrtSynchronizeStream(stream);
        if (status != ACL_SUCCESS) {
          cleanup_required_ = true;
          cleanup_stream_ = stream;
          HKV_LOG_ERROR(
              "Failed to clean up partially submitted fast load factor task, "
              "error={}",
              status);
        }
      }
      throw;
    }

    inflight_capacity_ = capacity;
    inflight_bucket_num_ = bucket_num;
    sample_in_flight_ = true;
  }

  std::mutex mutex_;
  bool initialized_ = false;
  bool sample_in_flight_ = false;
  bool cleanup_required_ = false;
  aclrtStream cleanup_stream_ = nullptr;

  aclrtEvent event_ = nullptr;
  int64_t* device_out_ = nullptr;
  int64_t* host_out_ = nullptr;
  void* workspace_ = nullptr;
  size_t workspace_capacity_ = 0;

  float cached_load_factor_ = 0.0f;
  size_t cached_capacity_ = 0;
  size_t inflight_capacity_ = 0;
  size_t inflight_bucket_num_ = 0;
};

}  // namespace hkv
}  // namespace npu
