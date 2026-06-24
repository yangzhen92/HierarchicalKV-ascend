# 原生API支持度

*说明：若"是否支持"为是、"限制与说明"为"-"，说明此参数或接口与原生支持度保持一致。*

## HierarchicalKV-ascend HashTableOptions

| 参数名称                   | 是否支持 | 限制与说明                                                         |
| -------------------------- | -------- | ------------------------------------------------------------------ |
| `init_capacity`            | 是       | -                                                                  |
| `max_capacity`             | 是       | -                                                                  |
| `max_hbm_for_vectors`      | 是       | -                                                                  |
| `max_bucket_size`          | 是       | 必须大于等于16                                                     |
| `dim`                      | 是       | -                                                                  |
| `max_load_factor`          | 是       | -                                                                  |
| `block_size`               | 否       | Ascend芯片算子不涉及此参数                                         |
| `io_block_size`            | 否       | Ascend芯片算子不涉及此参数                                         |
| `device_id`                | 是       | -                                                                  |
| `io_by_cpu`                | 是       | -                                                                  |
| `use_constant_memory`      | 是       | -                                                                  |
| `reserved_key_start_bit`   | 否       | 默认值为0。启动该参数会引入额外的计算开销，因此参数未生效，<br>设置该参数不会影响行为和性能。 |
| `num_of_buckets_per_alloc` | 是       | -                                                                  |
| `api_lock`                 | 是       | 默认值为false                                                      |
| `device_memory_pool`       | 是       | -                                                                  |
| `host_memory_pool`         | 是       | -                                                                  |

## HierarchicalKV-ascend HashTable

*说明：*
- *若API名称中含有 `*`，说明此API直接返回values的地址。*
- *`find(miss)` 是同名API中用于查找miss key的接口。*

| API名称               | 是否支持 | 限制与说明 |
| --------------------- | -------- | ---------- |
| `init`                | 是       | -          |
| `insert_or_assign`    | 是       | -          |
| `insert_and_evict`    | 是       | -          |
| `accum_or_assign`     | 是       | 不支持double、uint16、uint32数据类型 |
| `find_or_insert`      | 是       | -          |
| `find_or_insert*`     | 是       | -          |
| `lock_keys`           | 是       | -          |
| `unlock_keys`         | 是       | -          |
| `assign`              | 是       | -          |
| `assign_scores`       | 是       | -          |
| `assign_values`       | 是       | -          |
| `find`                | 是       | -          |
| `find(miss)`          | 是       | -          |
| `find*`               | 是       | -          |
| `find_and_update`     | 是       | -          |
| `contains`            | 是       | -          |
| `erase`               | 是       | -          |
| `erase_if`            | 是       | -          |
| `erase_if_v2`         | 是       | 分级存储场景下，使用HOST侧内存存储value时，不支持自定义函数对value进行内存操作 |
| `clear`               | 是       | -          |
| `export_batch`        | 是       | -          |
| `export_batch_if`     | 是       | -          |
| `export_batch_if_v2`  | 是       | 分级存储场景下，使用HOST侧内存存储value时，不支持自定义函数对value进行内存操作 |
| `for_each`            | 是       | 分级存储场景下，使用HOST侧内存存储value时，不支持自定义函数对value进行内存操作 |
| `empty`               | 是       | -          |
| `size`                | 是       | -          |
| `size_if`             | 是       | -          |
| `capacity`            | 是       | -          |
| `reserve`             | 是       | -          |
| `load_factor`         | 是       | -          |
| `set_max_capacity`    | 是       | -          |
| `dim`                 | 是       | -          |
| `max_bucket_size`     | 是       | -          |
| `bucket_count`        | 是       | -          |
| `save`                | 是       | -          |
| `load`                | 是       | -          |
| `set_global_epoch`    | 是       | -          |
| `is_pure_hbm_mode`    | 是       | Ascend新增接口，返回当前value是否全部存储在DEVICE侧内存中。<br>结果会根据实际存储状态变化，reserve、insert_or_assign、find_or_insert、load等可能触发扩容的接口执行后建议重新调用 |
