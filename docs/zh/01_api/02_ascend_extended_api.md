# Ascend 扩展 HierarchicalKV API

本文档用于补充说明 Ascend 版本 HierarchicalKV 中新增的 API。

## is_pure_hbm_mode

```cpp
bool is_pure_hbm_mode() const noexcept;
```

### 说明

`is_pure_hbm_mode` 用于查询当前哈希表的 value 存储是否全部位于 DEVICE 侧 HBM 中。

该接口是 Ascend/NPU 版本新增的扩展接口，主要用于判断当前表是否处于纯 HBM value 存储模式，便于业务侧或调试场景区分纯 HBM 路径和 HBM+DDR 混合路径。

### 返回值

| 返回值 | 含义 |
| --- | --- |
| `true` | 当前 value 存储全部位于 DEVICE 侧 HBM 中。 |
| `false` | 当前至少存在一部分 value 存储位于 HOST 侧 DDR 中。 |

### 注意事项

- 该接口反映的是当前 HashTable 的 value 存储状态，不是固定的初始化参数。reserve、insert_or_assign、find_or_insert、load等可能触发扩容的接口执行后建议重新调用。
- 该接口只描述 value 存储是否全部位于 HBM，不表示 key、score 或其他元数据的存储位置。

### 示例

```cpp
auto pure_hbm = table.is_pure_hbm_mode();
if (pure_hbm) {
  // 当前 value 全部位于 HBM，可按纯 HBM 场景分析。
} else {
  // 当前存在 value 位于 DDR，需要关注混合存储下的 value 搬运开销。
}
```

## HashTableOptions::enable_async_fast_load_factor

```cpp
bool enable_async_fast_load_factor = false;
```

### 说明

`enable_async_fast_load_factor` 用于控制 HashTable 写接口在扩容判断时是否使用异步近似负载率。

| 参数类型 | 默认值 | 含义 |
| --- | --- | --- |
| `bool` | `false` | `false` 使用同步负载率计算；`true` 使用异步采样结果进行近似计算。 |

### 示例

```cpp
npu::hkv::HashTableOptions options;
options.init_capacity = 4096;
options.max_capacity = 65536;
options.enable_async_fast_load_factor = true;

npu::hkv::HashTable<uint64_t, float, uint64_t> table;
table.init(options);
```

### 使用约束

- 异步模式使用最近一次已完成的采样缓存返回近似负载率，不会等待本次下发的采样任务。计算公式为：

  ```text
  cached_load_factor * cached_capacity / current_capacity
      + 2 * delta / current_capacity
  ```

  其中，`cached_load_factor` 是最近一次已完成采样的负载率，`cached_capacity` 是该次采样对应的容量，`current_capacity` 是当前容量，`delta` 是本次写操作预计新增的元素数量。
- `2 * delta` 用于对异步采样通常滞后一轮进行保守补偿，不是精确累计值。首次调用没有历史缓存时，返回 `2 * delta / current_capacity`；连续多次调用且采样尚未完成时，结果可能高估或低估实际负载率。
- 启用异步负载率模式后，不支持多个 Host 线程并发调用 HashTable 写接口，也不支持写接口与 `reserve`、`clear` 或 HashTable 析构并发执行。调用者必须在 Host 侧串行化上述操作。
- 必须在 `HashTable::init` 之前设置该参数，不支持在 HashTable 运行期间动态修改。
