# upsert_kernel 算子设计与实施文档

---

## 0. 概述

### 0.0 需求类型判断

**判断标准**：
- **特定用例**：用户明确指定了具体的 shape 和 dtype
- **通用**：用户未明确指定 shape 和 dtype，或明确要求支持多种配置

**结论**：**通用** - 支持多种数据类型和配置

### 0.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | upsert_kernel |
| 算子类别 | Hash Table / Memory Management |
| 需求类型 | 通用 |
| 支持数据类型 | K: uint64, V: float/half/bfloat16, S: uint64 |
| 支持服务器 | A2 / A3 / A5 |
| 特殊约束 | bucket_max_size 必须是 2 的幂次方 |

### 0.2 算子类别识别

- **类别**：Hash Table Operation (Insert/Update with Eviction)
- **判断依据**：
  - 核心功能是哈希表的插入或更新操作
  - 支持基于分数的淘汰策略
  - 使用 digest 优化查找性能
  - 涉及并发控制和原子操作

### 0.3 成熟方案查阅

- **是否查阅成熟方案**：是
- **参考文档**：
  - HierarchicalKV 原始实现
  - CUDA Warp-Level Primitives
  - AscendC SIMT API 文档

### 0.4 应用关键设计

| 设计项 | 成熟方案 | 应用到当前算子 |
|--------|---------|----------------|
| Digest 优化 | HKV digest 查找 | 使用 4-byte digest 加速 key 比较 |
| Warp 协作 | CUDA Cooperative Groups | 使用 asc_shfl 进行线程间通信 |
| 淘汰策略 | LRU/Epoch-based | ScoreFunctor 抽象淘汰策略 |
| 原子操作 | CAS/Exchange | asc_atomic_cas/exch 实现并发控制 |

---

## 1. 算子设计

### 1.1 数学公式

```
// 输入输出定义
输入:
  - keys: K[n]           // 待插入/更新的键数组
  - values: V[n * dim]   // 待插入/更新的值数组
  - scores: S[n]         // 分数数组（用于淘汰策略）
  - buckets: Bucket[]    // 哈希表桶数组

输出:
  - buckets: Bucket[]    // 更新后的哈希表
  - d_dst_values: V**    // 插入位置的值指针（可选）
  - d_dst_keys: K**      // 插入位置的键指针（可选）

// 核心逻辑
for each key in keys:
    bucket_idx = hash(key) % num_buckets
    pos = find_key_or_empty_slot(bucket[bucket_idx], key)
    
    if key_found:
        update_score(pos)
    else if empty_slot_found:
        insert_key(pos, key, value, score)
    else if bucket_full:
        if score >= min_score_in_bucket:
            evict_min_score_entry()
            insert_key(pos, key, value, score)
        else:
            refuse_insert()
```

### 1.2 API 映射

| 数学操作 | 对应 API | 关键参数 | 数据布局 | 官方文档 |
|---------|---------|---------|---------|---------|
| Hash 计算 | Murmur3HashDevice | key | 标量 | - |
| Digest 比较 | vcmpeq4 | probe_digests, target_digests | 4-byte 向量 | SIMT API |
| 原子 CAS | asc_atomic_cas | ptr, expected, desired | 标量 | SIMT API |
| 原子交换 | asc_atomic_exch | ptr, value | 标量 | SIMT API |
| 线程间通信 | asc_shfl / asc_shfl_xor | value, offset, width | 标量 | SIMT API |
| 内存屏障 | __threadfence | - | - | SIMT API |

#### 1.2.1 API 语义验证（⚠️ 强制）

| API | 数据布局 | 功能需求 | API选择 | 限制条件 | 匹配 | 文档 |
|-----|---------|---------|---------|---------|-----|------|
| vcmpeq4 | 4-byte 向量对齐 | 4 个 digest 并行比较 | `vcmpeq4(a, b)` | 无 | ✅ | SIMT API |
| asc_atomic_cas | 8-byte 对齐 | 原子比较交换 | `asc_atomic_cas(ptr, expected, desired)` | 仅支持整数类型 | ✅ | SIMT API |
| asc_shfl | Warp 内通信 | 线程间数据交换 | `asc_shfl(value, src_lane, width)` | width 必须是 2 的幂 | ✅ | SIMT API |
| __ldg | GM 读取 | 带缓存的全局内存读取 | `__ldg<LD_L2CacheType, L1CacheType>(ptr)` | 无 | ✅ | SIMT API |
| __stg | GM 写入 | 全局内存写入 | `__stg<ST_L2CacheType, L1CacheType>(ptr, value)` | 无 | ✅ | SIMT API |

### 1.3 数据流

```
输入 keys, values, scores (Global Memory)
    ↓
计算 hash 值，定位 bucket
    ↓
遍历 bucket，使用 digest 快速比较
    ↓
┌─────────────────────────────────────────┐
│  查找结果分支：                           │
│  1. DUPLICATE: 找到相同 key，更新 score   │
│  2. OCCUPIED_EMPTY: 找到空位，插入        │
│  3. INITIAL: 桶满，需要淘汰               │
└─────────────────────────────────────────┘
    ↓
淘汰逻辑（如果需要）：
  - 遍历桶找最小 score
  - 分治法求全局最小
  - 尝试淘汰最小 score 的 entry
    ↓
写入 value 到目标位置
    ↓
释放 key 锁（原子交换写入最终 key）
    ↓
更新后的 buckets (Global Memory)
```

### 1.4 核心计算步骤

**核心计算步骤**:
```
1. Hash 计算 - 使用 Murmur3Hash 计算 key 的哈希值
2. Bucket 定位 - 根据哈希值确定目标 bucket 和起始位置
3. Digest 查找 - 使用 4-byte digest 快速比较，每次比较 4 个位置
4. 原子抢占 - 使用 CAS 原子操作抢占 key 或空位
5. 淘汰决策 - 如果桶满，使用分治法找最小 score 并尝试淘汰
6. Value 写入 - 协作写入 value 到目标位置
7. Key 解锁 - 使用原子交换写入最终 key，释放锁
```

**分支差异对比**:

| 操作 | upsert_kernel (MoveV=true) | upsert_kernel (MoveV=false) | upsert_kernel_lock_key_hybrid |
|------|---------------------------|----------------------------|------------------------------|
| Value 处理 | Kernel 内直接写入 | 返回指针给外部写入 | 返回指针给外部写入 |
| 适用场景 | 小 dim，快速写入 | 大 dim，需要外部处理 | Hybrid 模式，支持 DDR |
| 线程数 | 512 | 512 | 512 |
| 协作组大小 | 32 | 32 | 16 (EVICT_GROUP_SIZE) |

**关键设计要点**:
1. **Digest 优化**: 使用 4-byte digest 替代 8-byte key 比较，每次可比较 4 个位置
2. **原子锁机制**: 使用 LOCKED_KEY 作为临时标记，防止并发冲突
3. **分治法求最小**: 使用 asc_shfl_xor 实现 O(log n) 的最小值查找
4. **Warp 协作**: 32 线程协作处理单个 key，提高内存访问效率

### 1.5 内存管理(Buffer 规划)

| Buffer 名称 | 用途 | 大小计算 | TPosition |
|------------|------|---------|-----------|
| keys_gm | 输入键数组 | n * sizeof(K) | GM |
| values_gm | 输入值数组 | n * dim * sizeof(V) | GM |
| scores_gm | 输入分数数组 | n * sizeof(S) | GM |
| buckets_gm | 哈希表存储 | capacity * bucket_max_size * (sizeof(K) + dim * sizeof(V) + sizeof(S)) | GM |
| d_dst_values | 输出值指针 | n * sizeof(V*) | GM |
| d_dst_keys | 输出键指针 | n * sizeof(K*) | GM |

**注**: 本 kernel 不使用 UB (Unified Buffer)，所有操作直接在 Global Memory 上进行

### 1.6 并发控制设计

#### 1.6.1 锁机制

| 锁状态 | Key 值 | 说明 |
|-------|--------|------|
| EMPTY_KEY | 0xFFFFFFFFFFFFFFFC | 空位，可抢占 |
| LOCKED_KEY | 0xFFFFFFFFFFFFFFFD | 已锁定，等待写入 |
| RECLAIM_KEY | 0xFFFFFFFFFFFFFFFE | 待回收，可抢占 |
| 正常值 | 其他 | 已占用 |

#### 1.6.2 原子操作序列

```
1. asc_atomic_cas(key_ptr, expected, LOCKED_KEY)  // 尝试锁定
2. 写入 value 和 score
3. __threadfence()                                // 内存屏障
4. asc_atomic_exch(key_ptr, final_key)            // 解锁并写入最终值
```

---

## 2. 架构设计

### 2.1 多核切分策略

| 项目 | 说明 |
|-----|------|
| 切分维度 | 按 key 索引切分 |
| 单核任务量 | ceil(n / usedCoreNum) 个 key |
| 使用的核数 | 动态计算，最大 512 核 |
| 负载均衡方式 | 静态分配，每个 block 处理等量 key |

**核数计算规范**：

```cpp
// Host 侧
int64_t availableCoreNum = 8;  // 默认值
aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_AICORE_CORE_NUM, &availableCoreNum);
uint32_t usedNumBlocks = min(n, (uint64_t)availableCoreNum);

// Kernel 侧
for (uint64_t kv_idx = block_index * blockDim.x + threadIdx.x;
     kv_idx < n_align_warp; kv_idx += thread_all) {
    // 处理每个 key
}
```

### 2.2 UB 切分策略

| 项目 | 说明 |
|-----|------|
| UB 容量 | 不适用（直接操作 GM） |
| 单次处理数据量 | 每个 thread 处理 1 个 key |
| 是否需要分 chunk | 否 |
| chunk 大小计算公式 | 不适用 |

### 2.3 分支场景覆盖

| 分支条件 | 处理策略 |
|---------|---------|
| 数据类型 | 通过模板参数 K, V, S 支持 |
| 大 dim | 使用 write_kernel 分块写入 |
| 小 dim | 使用 upsert_kernel_vf 直接写入 |
| 桶满 | 触发淘汰逻辑 |
| 桶未满 | 直接插入空位 |
| Key 已存在 | 更新 score |
| 分数不足 | 拒绝插入 |

### 2.4 类别特有设计

**算子类别**: Hash Table Operation

#### 2.4.1 upsert_kernel (MoveV=true) - 直接写入模式

**适用场景**：dim 较小，value 可在 kernel 内直接写入

**核心流程伪代码**：

```cpp
for each key_idx in assigned_range:
    key = keys[key_idx]
    score = ScoreFunctor::desired_when_missed(key_idx)
    
    // 1. 计算 hash，定位 bucket
    hashed_key = Murmur3HashDevice(key)
    bucket_idx = get_bucket_index(hashed_key)
    key_pos = get_start_position(hashed_key)
    
    // 2. 查找 key 或空位
    occupy_result = find_and_lock(bucket, key, score, key_pos)
    
    // 3. 处理查找结果
    if occupy_result == REFUSED:
        continue
    
    if occupy_result in [OCCUPIED_EMPTY, OCCUPIED_RECLAIMED]:
        atomic_add(bucket_size, 1)
    
    // 4. 协作写入 value
    copy_vector(values + key_idx * dim, bucket_vectors + key_pos * dim, dim)
    
    // 5. 更新 score 并解锁
    ScoreFunctor::update_with_digest(key_pos, score)
    atomic_exch(bucket_keys + key_pos, key)
```

**Buffer 需求**：

| Buffer 名称 | 用途 | 大小计算 |
|------------|------|---------|
| keys_gm | 输入键 | n * sizeof(K) |
| values_gm | 输入值 | n * dim * sizeof(V) |
| scores_gm | 输入分数 | n * sizeof(S) |

#### 2.4.2 upsert_kernel (MoveV=false) - 指针返回模式

**适用场景**：dim 较大，value 需要外部写入

**核心流程伪代码**：

```cpp
for each key_idx in assigned_range:
    key = keys[key_idx]
    score = ScoreFunctor::desired_when_missed(key_idx)
    
    // 1-3. 同上
    occupy_result = find_and_lock(bucket, key, score, key_pos)
    
    if occupy_result == REFUSED:
        d_dst_values[key_idx] = nullptr
        d_dst_keys[key_idx] = nullptr
        continue
    
    // 4. 返回指针而非写入
    d_dst_values[key_idx] = bucket_vectors + key_pos * dim
    d_dst_keys[key_idx] = bucket_keys + key_pos
    
    // 5. 更新 score（不解锁，由外部写入后解锁）
    ScoreFunctor::update_with_digest(key_pos, score)
```

**Buffer 需求**：

| Buffer 名称 | 用途 | 大小计算 |
|------------|------|---------|
| d_dst_values | 输出值指针 | n * sizeof(V*) |
| d_dst_keys | 输出键指针 | n * sizeof(K*) |

#### 2.4.3 upsert_kernel_lock_key_hybrid - Hybrid 模式

**适用场景**：支持 DDR 存储的混合模式

**核心流程伪代码**：

```cpp
for each key_idx in assigned_range:
    // 同上查找逻辑
    occupy_result = find_and_lock_with_eviction(bucket, key, score)
    
    // 返回指针供外部 write_kernel 使用
    if occupy_result not in [REFUSED, ILLEGAL]:
        d_dst_values[key_idx] = bucket_values + key_pos * dim
        d_dst_keys[key_idx] = bucket_keys + key_pos
    else:
        d_dst_values[key_idx] = nullptr
        d_dst_keys[key_idx] = nullptr
```

---

## 3. NPU 优化

### 3.1 SIMD

- 使用 API：vcmpeq4 (4 元素并行比较)
- 数据宽度：每次比较 4 个 1-byte digest

### 3.2 Tiling 参数计算

| 参数 | 公式 | 说明 |
|------|------|------|
| n_align_warp | ((n + 31) / 32) * 32 | 对齐到 warp 边界 |
| thread_all | THREAD_NUM_512 * GetBlockNum() | 总线程数 |
| group_size | 32 (MoveV) / 16 (Hybrid) | 协作组大小 |

### 3.3 双缓冲

- 是否使用：否
- 说明：直接操作 GM，不使用 UB 双缓冲

### 3.4 流水线

- 流水线阶段：查找 → 淘汰 → 写入 → 解锁
- 重叠策略：Warp 内线程协作，隐藏内存延迟

### 3.5 内存访问优化

| 优化项 | 方法 |
|-------|------|
| 合并访问 | 同一 warp 访问连续 key |
| 缓存控制 | 使用 __ldg 带缓存提示 |
| 写入优化 | 使用 __stg 带 L2 缓存提示 |
| 内存屏障 | __threadfence 确保 写入顺序 |

---

## 4. 实施计划

### 4.1 文件清单

#### 通用文件

| 序号 | 文件路径 | 说明 |
|------|---------|------|
| 1 | `include/kernels/insert_or_assign_kernel/insert_or_assign_kernel.h` | Kernel 实现 |
| 2 | `include/hkv_hashtable.h` | API 入口 |
| 3 | `include/types.h` | 类型定义 |
| 4 | `include/utils.h` | 工具函数 |
| 5 | `include/score_functor.h` | 分数计算策略 |

### 4.2 测试计划

#### 功能测试矩阵

| 维度 | 测试值 | 覆盖场景 |
|------|-------|---------|
| key 数量 | 1, 100, 10000, 100000 | 不同规模 |
| dim | 1, 8, 64, 128, 1024 | 不同向量维度 |
| 数据类型 | float, half, bfloat16 | 不同精度 |
| 淘汰策略 | kLru, kEpochLru, kCustom | 不同策略 |
| 负载因子 | 0.1, 0.5, 0.9 | 不同负载 |

#### 测试用例

| 序号 | key_num | dim | dtype | 说明 |
|------|---------|-----|-------|------|
| 1 | 1 | 8 | float | 单 key 插入 |
| 2 | 100 | 64 | float | 批量插入 |
| 3 | 10000 | 128 | half | 大规模半精度 |
| 4 | 1000 | 8 | float | 重复 key 更新 |
| 5 | 10000 | 64 | float | 桶满淘汰场景 |

#### 精度验证

- **验证方法**：与 CPU 实现对比
- **精度标准**：float: rtol=1e-5, half: rtol=1e-3
- **验证项**：key 正确性、value 正确性、score 正确性

#### 边界测试

| 测试项 | 输入 | 预期输出 |
|--------|------|---------|
| 空 key 数组 | n=0 | 正常返回 |
| 保留 key | key=LOCKED_KEY | 拒绝插入 |
| 桶满且分数不足 | 高负载 + 低分数 | 拒绝插入 |
| 并发插入相同 key | 多线程相同 key | 只插入一次 |

### 4.3 风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| Hash 冲突 | 高 | 性能下降 | 使用 digest 优化查找 |
| 活锁 | 低 | 插入失败 | 设置最大重试次数 |
| 内存越界 | 低 | 程序崩溃 | 边界检查 |
| 精度损失 | 中 | 结果错误 | 使用高精度中间类型 |

---

## 5. 确认清单

### 5.1 设计确认

- [x] 多核切分策略已确定
- [x] 并发控制机制已确定
- [x] 分支场景已覆盖
- [x] 淘汰策略已设计
- [x] 原子操作序列已确定

### 5.2 实施确认

- [x] 文件清单完整
- [x] 测试计划完整（功能 + 精度 + 边界）
- [x] 风险识别充分

---

## 6. 参考资源

- 官方示例: `asc-devkit/examples/`
- API 文档: `asc-devkit/docs/api/context/`
- SIMT API: `simt_api/device_atomic_functions.h`, `simt_api/device_warp_functions.h`
- 原始实现: HierarchicalKV CUDA 版本
