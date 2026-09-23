<h1 align="center">HierarchicalKV-ascend</h1>

<div align="center">

[![Ascend](https://img.shields.io/badge/Community-MindSDK-blue.svg)](https://www.hiascend.com/cn/developer/software/mindsdk)
[![License](https://badgen.net/badge/License/Apache-2.0/blue)](LICENSE)
[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Ascend/HierarchicalKV-ascend)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/Ascend/HierarchicalKV-ascend)

</div>

## 📖 Preface

This project implements a sample port of the HKV HashTable to the Atlas 950 series architecture, referencing [NVIDIA HierarchicalKV(Beta)](https://github.com/NVIDIA-Merlin/HierarchicalKV).

## ℹ️ Introduction

HierarchicalKV-ascend (hereinafter referred to as HKV) is a high-performance key-value storage acceleration library based on the Ascend platform for recommendation systems.
In recommendation systems, HKV provides the capability to add, delete, update, and query entries in large-capacity, high-performance dynamic embedding tables.

## ✨ Key Features

* Supports dynamic expansion
* Supports customizable eviction strategies
* Supports separate storage of keys and values, with keys stored only in device-side memory
* Supports storing values in host-side memory

## 📘 API Description

The core classes and structures are as follows (all in the namespace `npu::hkv::`):
- `struct EvictStrategy`: eviction strategy
- `class HashTableBase`: HashTable interface class
- `class HashTable`: HashTable implementation class

The `HashTable` interfaces invoke operators to perform all core functions.

### API Matrix

For the currently supported interfaces and their restrictions, see [Native API Support](docs/en/01_api/01_native_api_support.md).

For the Ascend extended interfaces, see [Ascend Extended APIs](docs/en/01_api/02_ascend_extended_api.md).

For the parameter descriptions and constraints of native interfaces, see the [HierarchicalKV API](https://nvidia-merlin.github.io/HierarchicalKV/master/api/index.html).

> **Reference baseline:** The API support of this project is aligned with the master branch of [HierarchicalKV](https://github.com/NVIDIA-Merlin/HierarchicalKV) at commit [`bbe2ee1`](https://github.com/NVIDIA-Merlin/HierarchicalKV/commit/bbe2ee1).

### Eviction Strategy

A score is introduced to define the importance of each key. The higher the score, the more important the key is, and the less likely it is to be evicted. Eviction occurs only when a bucket is full. The `score_type` must be the `uint64_t` type. For more details, see `class EvictStrategy`.
| Eviction Strategy | Meaning of `score` |
| :--- | :--- |
| `Lru` | Device clock (nanosecond-level), which differs slightly from the host clock.|
| `Lfu` | When the interpolation APIs are called, the frequency score specified by the `scores` parameter is added to the frequency score of the corresponding key. |
| `EpochLru` | The high 32 bits are the global epoch set by `set_global_epoch`,<br>the low 32 bits are equal to `(device_clock >> 20) & 0xffffffff`, with an accuracy of approximately 1 ms.|
| `EpochLfu` | The high 32 bits are the global epoch set by `set_global_epoch`,<br>the low 32 bits are the frequency score,<br>and the frequency remains constant after reaching the maximum value `0xffffffff`. |
| `Customized` | The score passed by the caller through the `scores` parameter when calling the interpolation APIs. |

* __Note__:
  - The interpolation APIs include `insert_or_assign`, `insert_and_evict`, `accum_or_assign`, `assign_scores`, `find_or_insert`, and so on.
  - `global_epoch`: set by the caller through `set_global_epoch` and used as the implicit input to the interpolation APIs. It is applied to the EpochLru and EpochLfu strategies.

### How to Use

```cpp
#include "debug.h"
#include "hkv_hashtable.h"

using TableOptions = npu::hkv::HashTableOptions;
using EvictStrategy = npu::hkv::EvictStrategy;

int main(int argc, char *argv[])
{
  // 0. Initialize the environment
  aclInit(nullptr);
  NPU_CHECK(aclrtSetDevice(0));

  using KeyType = uint64_t;
  using ValueType = float;
  using ScoreType = uint64_t;

  // 1. Specify the eviction strategy and create a table object
  using HKVTable = npu::hkv::HashTable<KeyType, ValueType, ScoreType, EvictStrategy::kLru>;
  std::unique_ptr<HKVTable> table = std::make_unique<HKVTable>();

  // 2. Define the configuration options
  TableOptions options;
  options.init_capacity = 16 * 1024 * 1024;
  options.max_capacity = options.init_capacity;
  options.dim = 16;
  options.max_hbm_for_vectors = npu::hkv::GB(16);

  // 3. Call the init interface for initialization
  table->init(options);

  // 4. Perform operations on the table
  // 5. Finish
  return 0;
}
```

### Environment Variable Configuration

The `HKV_NPU_ALLOC_CONF` environment variable controls the memory allocation policy of buckets. The configuration format is as follows:

```
HKV_NPU_ALLOC_CONF=buckets_mem_pool=<enable|disable>;page_table=<1g|2m>
```

| Configuration Item | Value | Description |
|--------|--------|------|
| `buckets_mem_pool` | `enable` | Memory pool mode. This mode pre-allocates the entire memory pool and uses large pages (2M/1G) to improve the Translation Lookaside Buffer (TLB) hit ratio, boosting performance. The memory pool is enabled by default, and the page table defaults to 2M. |
| `buckets_mem_pool` | `disable` | Non-memory-pool mode. This mode is consistent with the open-source community in that memory is allocated on demand. In this mode, the TLB hit ratio is lower and the performance is not as good as that of the memory pool mode. In addition, an excessively large `capacity` causes heavy memory fragmentation and memory allocation failures. You are advised to use the memory pool mode. |
| `page_table` | `2m` | Uses the 2 MB large page (default). |
| `page_table` | `1g` | Uses the 1 GB large page, which suits ultra-large-scale scenarios. |

**Configuration example:**
```bash
# Non-memory-pool mode, which uses little memory and suits small-scale scenarios
export HKV_NPU_ALLOC_CONF="buckets_mem_pool=disable"

# Memory pool mode using 2 MB large pages (default configuration)
export HKV_NPU_ALLOC_CONF="buckets_mem_pool=enable;page_table=2m"

# Memory pool mode using 1 GB large pages, which suits ultra-large-scale scenarios
export HKV_NPU_ALLOC_CONF="buckets_mem_pool=enable;page_table=1g"
```

The `HKV_TEST_DEVICE` environment variable specifies the NPU device number used for running. If it is not set, the default value is 0. Sim simulation supports only device number 0.

**Configuration example:**
```bash
# Run on device 1
export HKV_TEST_DEVICE=1

# Run on device 3
export HKV_TEST_DEVICE=3
```

The `HKV_LOG_LEVEL` environment variable controls the log output level of HKV and supports three levels: `0` (INFO), `1` (WARNING), and `2` (ERROR). If it is not set, the default value is `2`, which outputs only ERROR-level logs. You can keep the default setting in a production environment to reduce the performance overhead.

| Environment Variable Value | Log Level | Output |
|-----------|---------|---------|
| `0` | INFO | Outputs logs of all levels (INFO, WARNING, and ERROR), including the call parameters of each public interface, the capacity and expansion data of `init`, `reserve`, and other operations, the HBM and host memory allocation statistics, and alarms for parameters that do not take effect. |
| `1` | WARNING | Outputs WARNING-level and higher logs, including alarms such as parameters that do not take effect. |
| `2` or other values | ERROR | Outputs only ERROR-level logs (default). |

**Configuration example:**
```bash
# Enable INFO-level logs for fault locating and process tracing
export HKV_LOG_LEVEL=0

# Enable WARNING-level logs and focus on alarms only
export HKV_LOG_LEVEL=1
```

### Usage Constraints

- `key_type` must be `uint64_t` or `int64_t`.
- `score_type` must be `uint64_t`.


## 📂 Directory Structure
```
├── 3rdparty           // Third-party dependency libraries
├── benchmark          // Operator performance benchmark project
├── docs               // Documentation directory
├── include            // Header file directory
├── pre-commit         // Code check configuration directory
├── script             // Script directory (dependency checks and so on)
├── tests              // Unit test and integration test projects
├── CMakeLists.txt     // Top-level CMake build file
├── main.cpp           // Entry of the minimal demo application
└── run.sh             // Build and run script
```

## 📦 Compilation and Build

### Build Artifacts

The build artifacts include benchmark, demo, and unit test executables.

### Build Dependencies
  - [CANN development kit](https://hiascend.com/document/redirect/CannCommunityInstSoftware)

For the version requirements of the dependent software, see [dependencies.yaml](dependencies.yaml).

### Quick Build

#### Downloading the Code
```bash
git clone --recurse-submodules https://atomgit.com/Ascend/HierarchicalKV-ascend
cd HierarchicalKV-ascend
```

#### Configuring the Dependencies
  - Configure the CANN environment variables

    Based on the [installation method](https://hiascend.com/document/redirect/CannCommunityInstSoftware) of the CANN development kit in the current environment, select the corresponding command for configuring the environment variables.
    - Default path, where the root user installs the CANN package
      ```bash
      export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
      ```
    - Default path, where a non-root user installs the CANN package
      ```bash
      export ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
      ```
    - Specified path `install_path`, where the CANN package is installed
      ```bash
      export ASCEND_INSTALL_PATH=${install_path}/ascend-toolkit/latest
      ```

#### Compiling and Running

The `run.sh` parameters are as follows:

| Option | Long Option | Description |
| ---- | ------ | ---- |
| `-h` | `--help` | Prints help information and exits. |
| `-r` | `--run-mode MODE` | Specifies the build mode MODE. You can choose SIM simulation or running on an NPU board. The supported values are `sim` and `npu`, and the default value is `npu`.|
| `-i` | `--install-path PATH` | Specifies the CANN installation path, which has the highest priority when the script searches for the CANN path. |
| `-b` | `--build-type TYPE` | Specifies the build type. The options are `Release`/`Debug`, and the default value is `Release`. |
| `-p` | `--install-prefix PATH` | Specifies the installation path. The default value is the `out` directory in the current directory. |
| `-d` | `--device DEVICE_ID` | Specifies the NPU device number. The testcase, benchmark, and demo run on this device. The default value is `0`. |
| `-c` | `--compile-only` | Compiles and installs the code only, without running the testcases. |
| `-t` | `--enable-test SHIFT` | Test switch. The value `0` means that the testcases are not executed, `1` means that the testcases are executed, and the default value is `1`. |
| `-g` | `--skip-disabled-test` | Skips the `DISABLED_` testcases in gtest. |
| `-v` | `--soc-version SOC` | Specifies the SOC version. The default value is `Ascend950PR_9589`. This option takes effect only in `sim` mode. For how to obtain the value of this parameter, see the following document: [msprof Quick Start](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/optool/docs/en/quick_start/msopprof_quick_start.md). |

Examples:

```bash
# -h: View help information
bash run.sh -h

# -r: Specify the build mode (currently only npu is supported)
bash run.sh -r npu

# -d: Specify the NPU device number
bash run.sh -d 3

# -i: Specify the CANN installation path
bash run.sh -i /usr/local/Ascend/ascend-toolkit/latest

# -b: Specify Debug as the build type
bash run.sh -b Debug

# -p: Specify the installation path
bash run.sh -p /home/user/hkv_output

# -c: Compile and install only, without running the testcases
bash run.sh -c

# -t: Test switch. The value 0 means that the testcases are not executed, and 1 means that the testcases are executed
bash run.sh -t 0
bash run.sh -t 1

# -g: Skip the DISABLED_ testcases in gtest
bash run.sh -g
```

Simulation example:
```bash
bash run.sh -r sim -v Ascend950PR_9589

# The executables generate output files, and running them directly in the main directory causes file pollution
mkdir simtest && cd simtest
LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9589/lib:$LD_LIBRARY_PATH ../build/tests/test_main --gtest_filter="TestAllInterfaces.test_export_batch_with_counter"

# Use msprof for performance profiling
msprof op simulator --soc-version=Ascend950PR_9589 --output=./output --kernel-name="*dump_kernel*" ../build/tests/test_main --gtest_filter="TestAllInterfaces.test_export_batch_with_counter"
```

## 🛠️ Contribution Guide

Contributions are welcome. For the contribution process and standards, see the [Contribution Guide](./contributing.md).
Before contributing code, sign the [Open Project Contributor License Agreement (CLA)](https://clasign.osinfra.cn/sign/690ca9ddf91c03dee6082ab1) first.

1. If you encounter a bug, [submit an issue](https://atomgit.com/Ascend/HierarchicalKV-ascend/issues).
2. If you plan to contribute bug fixes, submit a pull request. See the [specific requirements](./contributing.md#pullrequest).
3. If you plan to contribute new features or functions, create an issue and discuss it with us first. Describe the background or purpose of the requirement, how to design it, and its impact on existing APIs and so on. Submitting a pull request without prior discussion may result in rejection, because the evolution direction of the project may differ from your ideas.

## ⚖️ Related Information

🔹 [Release Notes](https://atomgit.com/Ascend/HierarchicalKV-ascend/releases)<br>
🔹 [License Statement](LICENSE)<br>
🔹 [Third-Party Open Source Notice](Third_Party_Open_Source_Notice.md)<br>

## 🤝 Feedback and Discussion

You are welcome to raise questions and have discussions through the following channels.

| Resource | Description |
| --- | --- |
| [Create an Issue](https://atomgit.com/Ascend/HierarchicalKV-ascend/issues/new) | Submit bugs, requirements, or suggestions. |
| [Community Tasks](https://atomgit.com/Ascend/HierarchicalKV-ascend/issues/6) | View and claim community tasks. |
| [Meeting Calendar](https://meeting.ascend.osinfra.cn/?sig=sig-RecSDK) | Schedules of regular community meetings and activities. |

## License

Apache License Version 2.0. For details, see the [LICENSE file](LICENSE).
