#include <vector>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

template <typename K, typename V, typename S>
class DeviceData {
 public:
  DeviceData() = default;

  ~DeviceData() {
    if (device_keys != nullptr) {
      EXPECT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
      EXPECT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
      EXPECT_EQ(aclrtFree(device_values_ptr), ACL_ERROR_NONE);
      EXPECT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
      EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
      EXPECT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
    }
  }

  void malloc(size_t key_num, size_t dim = test_util::DEFAULT_DIM) {
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                          key_num * each_key_size, ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&device_values),
                    key_num * each_value_size * dim, ACL_MEM_MALLOC_HUGE_FIRST),
        ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values_ptr),
                          key_num * sizeof(V*), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                          key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);
  }

  void copy_keys(std::vector<K>& host_keys, size_t key_num) {
    ASSERT_EQ(
        aclrtMemcpy(device_keys, key_num * each_key_size, host_keys.data(),
                    key_num * each_key_size, ACL_MEMCPY_HOST_TO_DEVICE),
        ACL_ERROR_NONE);
  }

  void copy_values(std::vector<V>& host_values, size_t key_num,
                   size_t dim = test_util::DEFAULT_DIM) {
    ASSERT_EQ(aclrtMemcpy(device_values, key_num * each_value_size * dim,
                          host_values.data(), key_num * each_value_size * dim,
                          ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);
  }

  aclrtStream stream = nullptr;
  K* device_keys = nullptr;
  V* device_values = nullptr;
  V** device_values_ptr = nullptr;
  bool* device_found = nullptr;
  size_t each_key_size = sizeof(K);
  size_t each_score_size = sizeof(S);
  size_t each_value_size = sizeof(V);
};