# Extended HierarchicalKV API for Ascend

This document provides supplementary descriptions of the APIs newly added in the Ascend version of HierarchicalKV.

## `is_pure_hbm_mode`

```cpp
bool is_pure_hbm_mode() const noexcept;
```

### Description

`is_pure_hbm_mode` checks whether the value storage of the current hash table is entirely in the device-side HBM.

This API is an extension interface newly added in the Ascend/NPU version. It is mainly used to determine whether the current table is in pure-HBM value storage mode. You can then distinguish the pure-HBM path from the HBM+DDR hybrid path in service or debugging scenarios.

### Return Values

| Return Value | Description |
| --- | --- |
| `true` | All current value storage is in the device-side HBM. |
| `false` | At least part of the current value storage is in the host-side DDR. |

### Notes

- This API reflects the current value storage status of the HashTable instead of a fixed initialization parameter. After you call APIs that may trigger capacity expansion, such as `reserve`, `insert_or_assign`, `find_or_insert`, and `load`, you are advised to call this API again.
- This API only describes whether all value storage is in the HBM. It does not indicate where keys, scores, or other metadata are stored.

### Example

```cpp
auto pure_hbm = table.is_pure_hbm_mode();
if (pure_hbm) {
  // All values are currently in HBM and can be analyzed as a pure-HBM scenario
} else {
  // Some values are currently in DDR. Pay attention to the value transfer overhead in hybrid storage
}
```

## `HashTableOptions::enable_async_fast_load_factor`

```cpp
bool enable_async_fast_load_factor = false;
```

### Description

`enable_async_fast_load_factor` controls whether the HashTable write APIs use an asynchronous approximate load factor when determining whether capacity expansion is required.

| Parameter Type | Default Value | Description |
| --- | --- | --- |
| `bool` | `false` | `false`: the synchronous load factor calculation is used. `true`: the asynchronous sampling result is used for approximate calculation. |

### Example

```cpp
npu::hkv::HashTableOptions options;
options.init_capacity = 4096;
options.max_capacity = 65536;
options.enable_async_fast_load_factor = true;

npu::hkv::HashTable<uint64_t, float, uint64_t> table;
table.init(options);
```

### Constraints

- In asynchronous mode, the approximate load factor is returned based on the cache of the most recently completed sampling task, without waiting for the sampling task dispatched in the current call. The calculation formula is as follows:

  ```text
  cached_load_factor * cached_capacity / current_capacity
      + 2 * delta / current_capacity
  ```

  where `cached_load_factor` is the load factor of the most recently completed sampling, `cached_capacity` is the capacity corresponding to that sampling, `current_capacity` is the current capacity, and `delta` is the estimated number of elements to be added by the current write operation.
- `2 * delta` is a conservative compensation for the fact that asynchronous sampling usually lags behind by one round. It is not an exact accumulated value. When this API is called for the first time and no historical cache exists, `2 * delta / current_capacity` is returned. When this API is called multiple times in a row before sampling is complete, the result may overestimate or underestimate the actual load factor.
- After the asynchronous load factor mode is enabled, concurrent calls to the HashTable write APIs from multiple host threads are not supported, and concurrent execution of write APIs with `reserve`, `clear`, or HashTable destruction is not supported either. The caller must serialize these operations on the host side.
- This parameter must be set before `HashTable::init` and cannot be dynamically modified while the HashTable is running.
