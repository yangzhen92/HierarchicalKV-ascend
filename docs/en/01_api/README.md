# API Documentation

This directory contains the API documentation for HierarchicalKV-ascend (HKV), as follows:

| Document | Description |
| --- | --- |
| [01_native_api_support.md](./01_native_api_support.md) | **Native API Support**: describes how HKV supports the native interfaces of [NVIDIA HierarchicalKV](https://github.com/NVIDIA-Merlin/HierarchicalKV). Organized around core structures such as `HashTableOptions` and `HashTable`, the document lists the currently supported interfaces, the support status of all parameters, and the restrictions. The support level is aligned with the reference baseline commit of the master branch of HierarchicalKV. |
| [02_ascend_extended_api.md](./02_ascend_extended_api.md) | **Ascend Extended APIs**: describes the dedicated interfaces that HKV extends on the Ascend platform (for example, `is_pure_hbm_mode`). Each interface includes the description, return values, notes, and an example, supplementing the native interfaces with capabilities specific to the Ascend platform. |
