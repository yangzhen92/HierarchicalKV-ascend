**状态 (Status):** Draft
**作者 (Authors):** @lixiong
**创建日期 (Created):** 2026-04-20
**更新日期 (Updated):** 2026-04-20
**相关 Issue/PR:** #1

---

# 1. 概述

## 1.1 简介

本提案旨在将 HierarchicalKV-ascend 从 26.0.0 版本升级至 26.1.0 版本，核心目标是对齐社区 [HierarchicalKV/master/bbe2ee185](https://github.com/NVIDIA-Merlin/HierarchicalKV) 版本的 API 与特性支持度，实现 100% 兼容。主要包括三大方向：支持 HOST DDR 存储 values 的混合存储模式、补齐全量 API 及全量参数功能、完成全量社区测试用例迁移适配。

## 1.2 动机

HierarchicalKV 为大规模推荐模型提供了高效的 Embedding 管理解决方案，通过 CPU/GPU 协同存储架构，在保证高查询性能的同时极大拓展了存储容量。HierarchicalKV-ascend 实现了在昇腾 950 等 AI 芯片上的深度移植与优化。

当前 26.0.0 版本存在以下痛点：

1. **存储容量受限**：仅支持纯 HBM 存储 values，当 Embedding 表规模超过 HBM 容量时无法继续扩容，严重制约了推荐模型中大规模稀疏特征的存储需求。
2. **API 覆盖不全**：社区版本共 37 个 API 中，有 14 个 API 尚未实现（如 `accum_or_assign`、`contains`、`lock_keys` 等），导致用户无法使用完整的哈希表操作能力。
3. **参数支持不完整**：已有 API 中部分参数（如 `unique_key`、`locked_key_ptrs` 等）未全量支持，限制了功能的灵活性和高级用法。
4. **测试覆盖不足**：缺少大量社区测试用例的迁移适配，无法充分验证功能的正确性和稳定性。

不做此提案的影响：用户在昇腾平台上无法获得与 NVIDIA 平台等价的 HKV 能力，大规模推荐模型的训练和推理场景受限，阻碍昇腾平台在推荐系统领域的落地。

## 1.3 目标

**目标：**

1. 社区 [HierarchicalKV/master/bbe2ee185](https://github.com/NVIDIA-Merlin/HierarchicalKV) API 与特性支持度达 100%
2. 支持 HOST DDR 存储 values 特性，突破 HBM 容量限制
3. 全量 API 及全量参数功能支持
4. 全量社区测试用例迁移适配，且可在 A5 环境上运行

**非目标：**

1. 不涉及性能优化专项（如 Tiling 策略优化、流水线优化等），性能对齐后续版本规划
2. 不涉及新硬件平台适配，仅面向昇腾 A5 系列

# 2. 用例分析

## 2.1 大规模推荐模型训练

**场景描述**：推荐模型训练中，Embedding 表的规模通常达到数十亿级别，远超单卡 HBM 容量。需要利用 HOST DDR 扩展存储容量，同时保证训练过程中的读写性能。

**功能点**：
- HOST DDR 存储 values，HBM 存储热点数据
- 动态扩容（reserve）时自动溢出到 HOST DDR
- 查找、插入、更新操作对 HOST DDR 数据透明访问

**关键性能指标**：
- HOST DDR 模式下 benchmark 性能指标达到 1.0xL20，挑战 2.0xL20

**约束**：
- HOST DDR 模式下不支持自定义函数对 value 进行内存操作（erase_if_v2、export_batch_if_v2、for_each）
- HOST DDR 模式下 accum_or_assign 不支持 double、uint16 数据类型

## 2.2 增量训练与梯度累加

**场景描述**：增量训练场景中，需要对已有 key 的 value 进行梯度累加操作，而非覆盖写入。

**功能点**：
- `accum_or_assign` 接口：已有 key 累加 value，新 key 插入
- 支持多种数据类型（float、half、bfloat16）

**约束**：
- 不支持 double 和 uint16 数据类型

## 2.3 并发训练与键锁定

**场景描述**：多线程训练场景中，需要对特定 key 进行锁定，防止并发写入冲突，实现细粒度的并发控制。

**功能点**：
- `lock_keys`：对目标 key 上锁，返回目标 key 地址
- `unlock_keys`：对目标 key 解锁
- 与 `find_or_insert*` 配合使用，实现安全的读-改-写操作

## 2.4 条件筛选与批量操作

**场景描述**：训练过程中需要根据条件筛选 key 进行删除、导出或统计，如淘汰低频 key、导出特定 epoch 的 key。

**功能点**：
- `erase_if` / `erase_if_v2`：自定义过滤条件删除
- `export_batch_if` / `export_batch_if_v2`：自定义过滤条件导出
- `size_if`：自定义过滤条件统计
- `contains`：批量检查 key 是否存在
- `for_each`：遍历执行自定义函数

**约束**：
- HOST DDR 模式下，`erase_if_v2`、`export_batch_if_v2`、`for_each` 不支持自定义函数对 value 进行内存操作

## 2.5 模型检查点保存与恢复

**场景描述**：训练过程中需要定期保存 Embedding 表的检查点，以及从检查点恢复训练状态。

**功能点**：
- `save` / `load`：支持将哈希表内容保存到文件或从文件加载
- `export_batch` / `export_batch_if`：支持批量导出和条件导出

## 2.6 DFX 要求

| 属性 | 要求 |
| ---- | ---- |
| 兼容性 | API 参数与社区版本完全一致，已有代码无需修改即可迁移 |
| 可维护性 | Kernel 代码统一为头文件架构，便于维护和扩展 |
| 可测试性 | 全量社区测试用例迁移，覆盖所有 API 和参数组合 |
| 可靠性 | 内存管理使用 RAII 机制，避免内存泄漏和越界访问 |

# 3. 方案设计

## 3.1 总体方案

### 3.1.1 整体架构

HierarchicalKV-ascend 采用分层架构设计：

```
┌─────────────────────────────────────────────────────────┐
│                    用户应用层                            │
│  (推荐模型训练/推理)                                     │
├─────────────────────────────────────────────────────────┤
│                    API 接口层                            │
│  hkv_hashtable.h - HashTable<K,V,S> 模板类              │
│  hashtable_options.h - HashTableOptions 配置            │
├─────────────────────────────────────────────────────────┤
│                    Kernel 实现层                         │
│  include/kernels/                                        │
│  ├── insert_or_assign_kernel/  插入或赋值                │
│  ├── insert_and_evict_kernel/  插入并淘汰                │
│  ├── accum_or_assign_kernel/   累加或赋值                │
│  ├── find_kernel/              查找/查找未命中           │
│  ├── find_ptr_kernel/          查找指针                  │
│  ├── find_or_insert_kernel/    查找或插入                │
│  ├── find_or_insert_ptr_kernel/ 查找或插入指针           │
│  ├── find_and_update_kernel/   查找并更新                │
│  ├── assign_kernel/            赋值                      │
│  ├── assign_scores_kernel/     分数赋值                  │
│  ├── assign_values_kernel/     值赋值                    │
│  ├── lock_keys_kernel/         键锁定                    │
│  ├── unlock_keys_kernel/       键解锁                    │
│  ├── contains_kernel/          键存在检查                │
│  ├── size_if_kernel/           条件计数                  │
│  ├── remove_kernel/            删除                      │
│  ├── dump_kernel/              导出/条件导出             │
│  ├── traverse_kernel/          遍历                      │
│  ├── clear_kernel/             清空                      │
│  ├── init_table_kernel/        初始化                    │
│  ├── rehash_kernel/            重哈希                    │
│  └── utils_kernel/             工具/组锁                 │
├─────────────────────────────────────────────────────────┤
│                    基础设施层                            │
│  types.h - 数据类型定义 (Bucket, EvictStrategy, Lock)    │
│  score_functor.h - 分数计算策略                          │
│  memory_pool.h - 内存池管理                              │
│  allocator.h - 内存分配器                                │
│  utils.h - 工具函数                                      │
├─────────────────────────────────────────────────────────┤
│                    硬件抽象层                            │
│  SIMT API (asc_atomic_cas, asc_shfl, __ldg, __stg)      │
│  ACL Runtime (aclrtHostRegister, aclrtMalloc, etc.)      │
└─────────────────────────────────────────────────────────┘
```

### 3.1.2 HOST DDR 混合存储方案

核心思路：通过 `aclrtHostRegister` 接口实现 HOST DDR 与 DEVICE DDR 的映射，将超出 `max_hbm_for_vectors` 参数的 values 内存存储在 HOST 侧，同时实现 DEVICE 算子的无感知访问。

**内存申请流程**：

在 `init` 或 `reserve` 接口申请 device 内存时，计算本次申请的 slice 内存大小，如果累计大小超过 `max_hbm_for_vectors` 参数大小，则会申请 host 内存。

```
计算 slice_real_size
        │
        ▼
┌───────────────────────────────┐
│ remaining_hbm_for_vectors     │
│ >= slice_real_size ?          │
└───────────────────────────────┘
        │
   ┌────┴────┐
   │         │
  YES       NO
   │         │
   ▼         ▼
┌──────┐  ┌──────────────────────────┐
│ 申请 │  │ is_pure_hbm = false      │
│Device│  │ 申请 Pinned Host Memory  │
│Memory│  │ aclrtHostRegister 映射   │
└──────┘  └──────────────────────────┘
```

**SIMT 到 SIMD 适配**：将所有涉及 values 访问的 SIMT 算子适配修改为通过 SIMD 算子进行 values 访问，涉及 API 列表：

| API 名称 |
| :------- |
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

### 3.1.3 限制和约束

| 约束项 | 说明 |
| ------ | ---- |
| 硬件平台 | 仅支持昇腾 A5 系列 |
| Key 类型 | 必须为 uint64_t 或 int64_t |
| Score 类型 | 必须为 uint64_t |
| max_bucket_size | 必须大于等于 16，且为 2 的幂次方 |
| HOST DDR 限制 | 使用 HOST DDR 存储 value 时，不支持自定义函数对 value 进行内存操作 |
| accum_or_assign | 不支持 double、uint16 数据类型 |

## 3.2 技术选型

不涉及。

## 3.3 功能与性能设计

### 3.3.1 HOST DDR 混合存储功能

**核心流程**：

1. 用户通过 `HashTableOptions.max_hbm_for_vectors` 设置 HBM 可用容量上限
2. `init` / `reserve` 时，按 slice 粒度申请内存：
   - 若 `remaining_hbm_for_vectors >= slice_real_size`，申请 Device Memory
   - 否则，申请 Pinned Host Memory，并通过 `aclrtHostRegister` 映射到设备地址空间
3. 设置 `is_pure_hbm` 标志，标记是否为纯 HBM 模式
4. Kernel 侧通过 SIMD 算子访问 values，对 HOST DDR 和 HBM 统一处理

**数据模型变更**：

Table 结构体中新增/修改的字段：
- `is_pure_hbm`：bool 类型，标记是否为纯 HBM 模式
- `remaining_hbm_for_vectors`：size_t 类型，剩余可用 HBM 容量

### 3.3.2 新增 API 功能

#### accum_or_assign

**运行核心流程**：
1. 遍历输入 keys，计算 hash 值定位 bucket
2. 在 bucket 中查找 key：
   - 找到已有 key：根据 `accum_or_assigns` 标志决定累加或赋值 value
   - 未找到：插入新 key-value 对
3. 更新 score

#### lock_keys / unlock_keys

**运行核心流程**：
1. 遍历输入 keys，计算 hash 值定位 bucket
2. `lock_keys`：对目标 key 的内存位置设置锁定标记，返回 key 地址和锁定状态
3. `unlock_keys`：对已锁定的 key 解除锁定标记
4. 与 `find_or_insert*` 配合使用，实现安全的读-改-写操作模式

#### contains

**运行核心流程**：
1. 遍历输入 keys，计算 hash 值定位 bucket
2. 在 bucket 中查找 key，使用 digest 快速比较
3. 返回每个 key 的存在状态

#### erase_if / erase_if_v2 / export_batch_if / export_batch_if_v2 / for_each / size_if

**运行核心流程**：
1. 遍历哈希表中所有 bucket
2. 对每个 entry 调用用户提供的过滤函数/仿函数
3. 根据过滤结果执行对应操作（删除/导出/遍历/计数）

## 3.4 安全隐私与DFX设计

### 3.4.1 兼容性

- **API 兼容性**：所有 API 参数与社区版本完全一致，已有代码无需修改即可迁移
- **配置兼容性**：`HashTableOptions` 新增参数均有默认值，不影响已有配置

**Breaking Change**：
- `api_lock` 默认值从 `true` 改为 `false`，多线程场景需手动开启

### 3.4.2 可维护性

- Kernel 代码统一为头文件架构，消除 `.cpp` + `.h` 分离的维护负担
- 提取公共测试工具类（`test_device_data.h`、`test_util.h`），提升测试代码复用性

### 3.4.3 可测试性

- 全量社区测试用例迁移，覆盖所有 API 和参数组合
- 新增 15+ 个测试文件，覆盖新增 API
- 提取公共测试工具类，减少测试代码重复

### 3.4.4 可靠性

- 内存管理使用 RAII 机制（MemoryPool、Allocator），避免内存泄漏
- 原子操作 + 内存屏障保证并发场景下的数据一致性
- 键锁定机制防止并发写入冲突

## 3.5 编程与调用设计

### 3.5.1 编程模型基本设计

**开发环境**：

| 项目 | 要求 |
| ---- | ---- |
| 硬件平台 | 昇腾 A5 系列 |
| 操作系统 | Linux openEuler |
| CANN 版本 | 9.0.0 及以上 |
| 构建工具 | CMake 3.16+ |
| 测试框架 | Google Test |

**开发约束**：

1. Key 类型必须为 `uint64_t` 或 `int64_t`
2. Score 类型必须为 `uint64_t`
3. `max_bucket_size` 必须大于等于 16 且为 2 的幂次方
4. HOST DDR 模式下不支持自定义函数对 value 进行内存操作
5. `accum_or_assign` 不支持 `double` 和 `uint16_t` 数据类型

**可验收设计**：

1. 功能验收：全量社区测试用例在 A5 环境上通过
2. 兼容性验收：API 参数与社区版本完全一致
3. 性能验收：纯 HBM 模式下性能不低于 26.0.0 版本

### 3.5.2 接口定义与设计

不涉及。

### 3.5.3 编程手册设计

不涉及。

# 4. 缺点和风险

| 风险 | 严重程度 | 影响 | 应对措施 |
| ---- | -------- | ---- | -------- |
| `api_lock` 默认值变更 | 低 | 多线程场景可能出现数据竞争 | 文档中明确说明；提供迁移指南 |
| HOST DDR 模式功能限制 | 中 | 部分高级功能不可用 | 文档中明确限制；后续版本逐步补齐 |
| `accum_or_assign` 数据类型限制 | 低 | 部分数据类型不可用 | 文档中明确限制；后续版本扩展支持 |

**兼容性**：
- API 完全向后兼容，已有代码无需修改

# 5. 现有技术

## 5.1 NVIDIA HierarchicalKV

**参考版本**：[HierarchicalKV/master/bbe2ee185](https://github.com/NVIDIA-Merlin/HierarchicalKV)

**借鉴内容**：
- API 接口设计：完全对齐社区版本的 API 参数和功能
- 淘汰策略：5 种淘汰策略的设计和实现
- 内存管理：HBM/Host 混合存储的架构设计
- 测试用例：社区测试用例的迁移适配

**差异**：
- 底层 Kernel 实现完全不同：社区版使用 CUDA，本提案使用 AscendC SIMT API

# 6. 未解决问题

| 问题 | 说明 | 预计解决时间 |
| ---- | ---- | ------------ |
| HOST DDR 模式下自定义函数对 value 内存操作的支持 | 后续支持自定义函数允许读写 values | 后续版本规划 |

---

附录

* **参考资料链接**：
  * [HierarchicalKV GitHub](https://github.com/NVIDIA-Merlin/HierarchicalKV)
  * [AscendC SIMT API 文档](https://www.hiascend.com/document)
  * [CANN 开发套件文档](https://www.hiascend.com/document)

* **术语表**：

| 术语 | 说明 |
| ---- | ---- |
| HBM | High Bandwidth Memory，设备侧高带宽内存 |
| HOST DDR | 主机侧 DDR 内存 |
| SIMT | Single Instruction Multiple Threads，单指令多线程编程模型 |
| SIMD | Single Instruction Multiple Data，单指令多数据编程模型 |
| Digest | 4 字节的键摘要，用于加速查找比较 |
| Bucket | 哈希表的基本存储单元，包含 keys、scores、digests、vectors |
| EvictStrategy | 淘汰策略，决定桶满时淘汰哪个 entry |
| ScoreFunctor | 分数计算仿函数，根据淘汰策略特化实现 |
| RAII | Resource Acquisition Is Initialization，资源获取即初始化 |
| Pinned Memory | 锁页内存，主机侧不会被换出的内存 |
| aclrtHostRegister | ACL 接口，将主机内存映射到设备地址空间 |

* **文档更新计划**：
  * RFC 通过后更新 README.md


