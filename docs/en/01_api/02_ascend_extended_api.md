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
