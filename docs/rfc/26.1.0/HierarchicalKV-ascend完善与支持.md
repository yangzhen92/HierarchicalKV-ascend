# 需求背景
            
HierarchicalKV为大规模推荐模型提供了高效的Embedding管理解决方案，通过CPU/GPU协同存储架构，在保证高查询性能的同时极大拓展了存储容量。HierarchicalKV-ascend实现了[HierarchicalKV](https://github.com/NVIDIA-Merlin/HierarchicalKV)在昇腾950等AI芯片上的深度移植与优化，既保持了与原生API的兼容性，又充分利用昇腾平台的并行计算特性显著提升了Embedding操作效率。

# 提议方案
            
当前HierarchicalKV-ascend与社区[HierarchicalKV/master/bbe2ee185](https://github.com/NVIDIA-Merlin/HierarchicalKV)版本的API以及支持特性存在差异，需要完善API、特性支持度。

## 方案目标

社区[HierarchicalKV/master/bbe2ee185](https://github.com/NVIDIA-Merlin/HierarchicalKV)API与特性支持度达100%。
- 支持HOST DDR存储values特性。
- 支持全量API以及全量参数功能。
- 全量测试用例迁移。

## 总体设计

### 支持HOST DDR存储values

HierarchicalKV-ascend在26.0.0版本中仅支持在纯HBM内实现增删改查、导入导出以及扩容等能力。本RFC需要全量接口支持使用`HOST DDR`存储values。

#### 实现方案

1. 通过`aclrtHostRegister`接口，实现`HOST DDR`与`DEVICE DDR`的映射，将超出`max_hbm_for_vectors`参数的values内存存储在HOST侧，同时实现DEVICE算子的无感知访问。
2. 将所有涉及values访问的SIMT算子适配修改为通过SIMD算子进行values访问。

涉及SIMT算子适配修改API列表如下
| API名称 |
| :--------- |
| insert_or_assign |
| accum_or_assign |
| find_or_insert |
| assign |
| assign_values |
| find |
| find(miss) |
| reserve |
| export_batch |
| export_batch_if |
| export_batch_if_v2 |

### 支持全量API以及全量参数功能特性

截止26.0.0版本时**未支持**或**API参数未全量支持**的API，本RFC要求API及其参数全量支持

#### 已有API支持全量参数

`HashTableOptions`补齐功能选项参数：
| 参数名称 |
| :--------- |
| io_by_cpu |
| reserved_key_start_bit |
| api_lock |
| device_memory_pool |
| host_memory_pool |

`HashTable`补充支持全量参数（如`unique_key`、`locked_key_ptrs`等）的API：
| API名称 |
| :--------- |
| insert_or_assign |
| insert_and_evict |
| find_or_insert* |
| assign_scores |
| find* |
| find_and_update* |

#### 新增支持API

`HashTable`新增支持API：
| API名称 | 功能说明 |
| :--------- | :--------- |
| accum_or_assign | 对已有key进行value累加，对未命中的key执行插入 |
| find_or_insert | 查找并返回value，对未命中的key执行插入 |
| lock_keys | 对目标key上锁，并返回目标key地址，和unlock_keys配套使用 |
| unlock_keys | 对目标key解锁，和lock_keys配套使用 |
| assign | 更新目标key的value、score |
| assign_values | 更新目标key的value |
| find | 查找并返回value |
| find(miss) | 查找并返回未命中的key和下标 |
| contains | 返回目标key是否存在 |
| erase | 删除目标key |
| erase_if | 自定义过滤func来筛选删除目标key |
| erase_if_v2 | 自定义过滤class来筛选删除目标key |
| for_each | 遍历指定范围key，执行自定义func |
| export_batch_if | 自定义过滤func来筛选并导出目标key |
| export_batch_if_v2 | 自定义过滤class来筛选并导出目标key |
| size_if | 自定义过滤func来筛选并统计数量 |

# 验收标准

本RFC要求以以下结果作为落地验收标准：
1. 完成全量社区测试用例迁移适配
2. 全量测试用例可以在A5环境上运行
