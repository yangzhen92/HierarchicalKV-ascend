# 内存申请时序图

## init/reserve 接口内存申请流程

```mermaid
sequenceDiagram
    participant User as 用户代码
    participant API as init/reserve 接口
    participant Table as Table 结构体
    participant Allocator as 内存分配器
    participant Device as Device Memory (HBM)
    participant Host as Host Memory (Pinned)

    User->>API: 调用 init() 或 reserve()
    API->>Table: 获取当前状态
    
    Note over Table: 计算 num_of_memory_slices<br/>和 slice_real_size
    
    loop 遍历每个 memory slice
        Table->>Table: 计算 slice_real_size<br/>= num_of_buckets × bucket_max_size × sizeof(V) × dim
        
        alt remaining_hbm_for_vectors >= slice_real_size
            Note over Table: HBM 空间充足
            Table->>Allocator: alloc(Device, slice_real_size)
            Allocator->>Device: 申请设备内存
            Device-->>Allocator: 返回设备内存地址
            Allocator-->>Table: slices[i] = device_ptr
            Table->>Table: remaining_hbm_for_vectors -= slice_real_size
            
            Note over Table: is_pure_hbm 保持 true<br/>（如果是首次申请）
        else remaining_hbm_for_vectors < slice_real_size
            Note over Table: HBM 空间不足
            Table->>Table: is_pure_hbm = false
            Table->>Table: slice_real_size = max(2MB, slice_real_size)
            Table->>Allocator: alloc(Pinned, slice_real_size)
            Allocator->>Host: 申请锁页主机内存
            Host-->>Allocator: 返回主机内存地址
            Allocator-->>Table: slices[i] = host_ptr
            
            Table->>Device: aclrtHostRegister(host_ptr, slice_real_size, ACL_HOST_REGISTER_MAPPED)
            Note over Device: 将 Host 内存映射到设备地址空间
            Device-->>Table: 映射成功
        end
        
        Table->>Table: 更新 bucket vectors 地址
    end
    
    Table->>Table: num_of_memory_slices += num_of_memory_slices
    Table-->>API: 返回申请结果
    API-->>User: 内存申请完成
```

## 关键判断逻辑

```mermaid
flowchart TD
    A[开始申请 slice 内存] --> B[计算 slice_real_size]
    B --> C{remaining_hbm_for_vectors<br/>>= slice_real_size?}
    
    C -->|是| D[申请 Device Memory]
    D --> E[remaining_hbm_for_vectors -= slice_real_size]
    E --> F{is_pure_hbm?}
    F -->|是| G[保持 is_pure_hbm = true]
    F -->|否| H[设置 mixed_hbm = true]
    G --> I[记录 slices[i] = device_ptr]
    H --> I
    
    C -->|否| J[设置 is_pure_hbm = false]
    J --> K[slice_real_size = max(2MB, slice_real_size)]
    K --> L[申请 Pinned Host Memory]
    L --> M[aclrtHostRegister 映射内存]
    M --> N[记录 slices[i] = host_ptr]
    N --> I
    
    I --> O{还有更多 slice?}
    O -->|是| A
    O -->|否| P[更新 num_of_memory_slices]
    P --> Q[结束]
```

## 内存布局示意

```mermaid
graph TB
    subgraph "HBM (Device Memory)"
        D1[Slice 0: Device Memory]
        D2[Slice 1: Device Memory]
        D3[...]
    end
    
    subgraph "Host Memory (Pinned + Mapped)"
        H1[Slice N: Host Memory<br/>映射到设备地址空间]
        H2[Slice N+1: Host Memory<br/>映射到设备地址空间]
        H3[...]
    end
    
    subgraph "Table 结构体"
        T1[slices: V** 数组]
        T2[remaining_hbm_for_vectors]
        T3[is_pure_hbm: bool]
        T4[num_of_memory_slices]
    end
    
    T1 --> D1
    T1 --> D2
    T1 --> H1
    T1 --> H2
```

## 状态变化说明

| 阶段 | remaining_hbm_for_vectors | is_pure_hbm | 说明 |
|------|---------------------------|-------------|------|
| 初始 | max_hbm_for_vectors | true | 全部 HBM 可用 |
| 申请 Device 内存后 | 减少 | true/混合 | HBM 减少 |
| 申请 Host 内存后 | 不变 | false | 开始使用 Host 内存 |

## 关键代码路径

```cpp
// table.h 第 165-186 行
size_t slice_real_size = num_of_buckets_in_one_slice *
                         (*table)->bucket_max_size * sizeof(V) *
                         (*table)->dim;

if ((*table)->remaining_hbm_for_vectors >= slice_real_size) {
    // 路径1: HBM 空间充足，申请 Device Memory
    allocator->alloc(MemoryType::Device, (void**)&((*table)->slices[i]),
                     slice_real_size);
    (*table)->remaining_hbm_for_vectors -= slice_real_size;
} else {
    // 路径2: HBM 空间不足，申请 Host Memory
    (*table)->is_pure_hbm = false;
    const size_t host_min_size = MB(2);
    slice_real_size = std::max(host_min_size, slice_real_size);
    allocator->alloc(MemoryType::Pinned, (void**)&((*table)->slices[i]),
                     slice_real_size);
    // 注册映射，使设备可以访问主机内存
    NPU_CHECK(aclrtHostRegister(static_cast<void*>((*table)->slices[i]),
                                slice_real_size, ACL_HOST_REGISTER_MAPPED,
                                &stub_ptr));
}
```
