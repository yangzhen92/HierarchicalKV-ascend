# CLAUDE.md

This file provides guidance to kscc (claude.ai/code) when working with code in this repository.

## Project Overview

HierarchicalKV-ascend (HKV) is a high-performance key-value storage acceleration library for Huawei Ascend NPU (ascend950), ported from NVIDIA's HierarchicalKV. It provides large-capacity, high-performance dynamic Embedding table CRUD for recommendation systems. The library is header-only under `include/`, with NPU kernels written in Ascend C (ASC) under `include/kernels/`.

- Namespace: `npu::hkv`
- C++ standard: C++17
- Version: 26.1.0

## Build & Run

Primary build entry point is `run.sh`. Requires CANN toolkit installed.

```bash
bash run.sh -d [DEVICE_ID]           # Build + test + benchmark + demo
bash run.sh -d 0 -c                  # Compile only
bash run.sh -d 0 -t 0                # Build without running tests
bash run.sh -d 0 -g                  # Skip DISABLED_ gtest cases
bash run.sh -i /path/to/cann -d 0    # Specify CANN install path
```

Key options: `-r` (run-mode, only `npu`), `-b` (Debug/Release), `-p` (install prefix), `-d` (NPU card ID, sets `HKV_TEST_DEVICE`).

CMake build manually:
```bash
source $ASCEND_INSTALL_PATH/bin/setenv.bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_TEST=1 ..
make -j16
ctest -V
```

## Test Framework

Google Test (gtest/gmock) via `3rdparty/gtest` submodule. Tests in `tests/` directory.

- `test_main` — aggregated test executable from `tests/*.cpp`
- `test_aclnn_helper`, `test_score_functor` — separate executables
- `tests/community/` — 16 files ported from open-source HKV, each builds as its own executable

Test utilities in `tests/common/`: `test_util.h`, `test_device_data.h`, `test_find_base.h`, `test_template_func.h`.

Run all tests: `ctest -V` from build directory. For disabled tests: `GTEST_ALSO_RUN_DISABLED_TESTS=1 ctest -V`.

## Architecture

### Header-only library

All library source is in `include/`. No `.cpp` files for the library itself — only tests, benchmarks, and demo have `.cpp` files.

### Core template: `HashTable<K, V, S, EvictStrategy>`

- `K`: key type (must be `uint64_t` or `int64_t`)
- `V`: value element type (e.g., `float`, `int32_t`)
- `S`: score type (must be `uint64_t`)
- `EvictStrategy`: `kLru`, `kLfu`, `kEpochLru`, `kEpochLfu`, `kCustomized`

### Key source files

- `include/hkv_hashtable.h` — Main `HashTable` class with all API methods (the primary entry point)
- `include/hashtable_options.h` — `HashTableOptions` configuration struct
- `include/types.h` — Core types: `Bucket`, `Table`, `KVM`, `Lock`, `OccupyResult`, eviction strategy enums
- `include/allocator.h` — `BaseAllocator` / `DefaultAllocator` (wraps `aclrtMalloc`/`aclrtFree`)
- `include/memory_pool.h` — Device/host memory pool
- `include/bucket_memory_pool_manager.h` — Pre-allocated bucket pool (implements `IBucketAddressProvider`)
- `include/score_functor.h` — Score functors for each eviction strategy
- `include/group_lock.h` — Triple-group shared mutex (inserter/updater/reader roles)
- `include/utils.h` — Murmur3 hash, memory size helpers, `SAFE_GET_GRID_SIZE`

### Kernel structure (`include/kernels/`)

22 kernel directories, one per operation. Each contains `.h` host launcher + `.asc` device kernel files. Kernels use Ascend C extensions (`__gm__`, `__simt_callee__`, `__simt_vf__`, `__aicore__`, `LAUNCH_BOUND`, `asc_atomic_cas`, `asc_ballot`, `asc_shfl`).

Key kernels: `find_kernel/`, `find_or_insert_kernel/`, `insert_and_evict_kernel/`, `dump_kernel/`, `lock_keys_kernel/`, `rehash_kernel/`.

### Storage model

- Keys stored in HBM only; values can be in HBM or HOST DDR (hybrid mode)
- Bucket-based hash table with configurable `max_bucket_size` (min 16, default 128)
- Digest-based lookup: 1-byte hash fingerprints for fast bucket probing before full key comparison

### Memory allocation strategies

- Pool mode (default, `buckets_mem_pool=enable`): Pre-allocates entire pool with huge pages (2M/1G), reduces TLB miss
- Non-pool mode (`buckets_mem_pool=disable`): Per-bucket allocation

## Environment Variables

- `HKV_NPU_ALLOC_CONF` — Memory allocation config: `buckets_mem_pool=enable|disable`, `page_table=2m|1g`
- `HKV_TEST_DEVICE` — NPU card ID for test/benchmark/demo (default: 0)
- `ASCEND_INSTALL_PATH` / `ASCEND_HOME_PATH` — CANN toolkit installation path

## Dependencies

From `dependencies.yaml`: CANN >= 9.1.0, HDK >= 26.0.rc2, GCC/G++ >= 8.1.0, CMake >= 3.16.0. CMake links: `tiling_api`, `platform`, `opapi`, `nnopbase`.

## Code Style

- Clang-format: Google style base, IndentWidth 4, ColumnLimit 120, Allman braces
- Pre-commit hooks: clang-format v18.1.8, Ruff, pylint (bug-level), typos checker
- Naming: PascalCase classes, snake_case functions, UPPER_SNAKE_CASE constants
- Commit messages use bracketed tags: `[feat]`, `[fix]`, `[Pref]`, `[docs]`, `[style][format]`
