# API 文档

本目录包含 HierarchicalKV-ascend（HKV）的 API 相关文档，具体如下：

| 文档 | 说明 |
| ---- | ---- |
| [01_native_api_support.md](./01_native_api_support.md) | **原生 API 支持度**：介绍 HKV 对 [NVIDIA HierarchicalKV](https://github.com/NVIDIA-Merlin/HierarchicalKV) 原生接口的支持情况。以 `HashTableOptions`、`HashTable` 等核心结构为主线，列出当前已支持的接口、全量参数支持情况以及限制说明，支持度对齐 HierarchicalKV master 分支的参考基线提交点。 |
| [02_ascend_extended_api.md](./02_ascend_extended_api.md) | **Ascend 扩展 API**：介绍 HKV 在昇腾平台上扩展的专属接口（如 `is_pure_hbm_mode` 等），每个接口均包含说明、返回值、注意事项与示例，用于在原生接口基础上补充昇腾平台特有的能力。 |
