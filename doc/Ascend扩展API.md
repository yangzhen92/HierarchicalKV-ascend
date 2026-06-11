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

- 该接口反映的是当前 HashTable 的 value 存储状态，不是固定的初始化参数。执行插入、扩容、加载、rehash 或其他可能改变存储布局的操作后，返回值可能发生变化。
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
