# Native API Support

*Note: If `Supported` is `Yes` and `Restrictions and Notes` is `-`, the parameter or API is consistent with native support.*

## HierarchicalKV-ascend `HashTableOptions`

| Parameter | Supported | Restrictions and Notes |
| --- | --- | --- |
| `init_capacity` | Yes | - |
| `max_capacity` | Yes | - |
| `max_hbm_for_vectors` | Yes | - |
| `max_bucket_size` | Yes | Must be greater than or equal to 16. |
| `dim` | Yes | - |
| `max_load_factor` | Yes | - |
| `block_size` | No | Ascend chip operators do not use this parameter. |
| `io_block_size` | No | Ascend chip operators do not use this parameter. |
| `device_id` | Yes | - |
| `io_by_cpu` | Yes | - |
| `use_constant_memory` | Yes | - |
| `reserved_key_start_bit` | No | The default value is 0. The feature for setting this parameter is not yet implemented, and setting it has no effect. |
| `num_of_buckets_per_alloc` | Yes | - |
| `api_lock` | Yes | The default value is `false`. |
| `device_memory_pool` | Yes | - |
| `host_memory_pool` | Yes | - |

## HierarchicalKV-ascend `HashTable`

*Notes:*
- *If an API name contains `*`, the API directly returns the address of values.*
- *`find(miss)` is the interface used to find the miss key among the APIs with the same name.*

| API Name | Supported | Restrictions and Notes |
| --- | --- | --- |
| `init` | Yes | - |
| `insert_or_assign` | Yes | - |
| `insert_and_evict` | Yes | - |
| `accum_or_assign` | Yes | Does not support the `double`, `uint16`, or `uint32` data types. |
| `find_or_insert` | Yes | - |
| `find_or_insert*` | Yes | - |
| `lock_keys` | Yes | - |
| `unlock_keys` | Yes | - |
| `assign` | Yes | - |
| `assign_scores` | Yes | - |
| `assign_values` | Yes | - |
| `find` | Yes | - |
| `find(miss)` | Yes | - |
| `find*` | Yes | - |
| `find_and_update` | Yes | - |
| `contains` | Yes | - |
| `erase` | Yes | - |
| `erase_if` | Yes | - |
| `erase_if_v2` | Yes | In tiered storage scenarios, when values are stored in host-side memory, custom functions cannot perform memory operations on the values. |
| `clear` | Yes | - |
| `export_batch` | Yes | - |
| `export_batch_if` | Yes | - |
| `export_batch_if_v2` | Yes | In tiered storage scenarios, when values are stored in host-side memory, custom functions cannot perform memory operations on the values. |
| `for_each` | Yes | In tiered storage scenarios, when values are stored in host-side memory, custom functions cannot perform memory operations on the values. |
| `empty` | Yes | - |
| `size` | Yes | - |
| `size_if` | Yes | - |
| `capacity` | Yes | - |
| `reserve` | Yes | - |
| `load_factor` | Yes | - |
| `set_max_capacity` | Yes | - |
| `dim` | Yes | - |
| `max_bucket_size` | Yes | - |
| `bucket_count` | Yes | - |
| `save` | Yes | - |
| `load` | Yes | - |
| `set_global_epoch` | Yes | - |
