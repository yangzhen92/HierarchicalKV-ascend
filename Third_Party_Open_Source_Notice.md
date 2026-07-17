# OPEN SOURCE SOFTWARE NOTICE

Please note we provide an open source software notice along with this product and/or this product firmware (in the following just "this product"). The open source software licenses are granted by the respective right holders. And the open source licenses prevail all other license information with regard to the respective open source software contained in the product, including but not limited to End User Software Licensing Agreement. This notice is provided on behalf of Huawei Technologies Co. Ltd. and any of its local subsidiaries which may have provided this product to you in your local country.

## Warranty Disclaimer

THE OPEN SOURCE SOFTWARE IN THIS PRODUCT IS DISTRIBUTED IN THE HOPE THAT IT WILL BE USEFUL, BUT WITHOUT ANY WARRANTY, WITHOUT EVEN THE IMPLIED WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. SEE THE APPLICABLE LICENSES FOR MORE DETAILS.

## Copyright Notice and License Texts

Software: HierarchicalKV-ascend

Copyright notice:
Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

================================================================================
Third-Party Open Source Software Components
================================================================================

--------------------------------------------------------------------------------
1. C++ Bundled Third-Party Components
--------------------------------------------------------------------------------

Software: GoogleTest (googletest + googlemock) 1.14.0
Copyright notice:
Copyright 2008, Google Inc.
All rights reserved.
Source: https://github.com/google/googletest
Introduce: Git submodule located at `3rdparty/gtest/` (mirror: https://gitcode.com/gh_mirrors/googl/googletest.git)
Purpose: C++ unit testing framework, used by the test cases under `tests/`
License: BSD 3-Clause License
Please see below

--------------------------------------------------------------------------------
2. Build & Runtime Dependencies (System-level, Not Bundled)
--------------------------------------------------------------------------------

Software: HDK (Huawei Ascend Hardware Development Kit) >=26.0.rc2
Copyright notice:
Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
Source: https://www.hiascend.com/
Purpose: Minimum hardware development kit version required to run HierarchicalKV-ascend
License: Huawei Proprietary License
Please refer to the corresponding HDK license

Software: CANN (Compute Architecture for Neural Networks) >=9.1.0
Copyright notice:
Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
Source: https://www.hiascend.com/software/cann
Purpose: Huawei heterogeneous computing architecture, the minimum CANN version required to build and run HierarchicalKV-ascend
License: Huawei Proprietary License
Please refer to the corresponding CANN license

Software: GCC >=8.1.0
Copyright notice:
Copyright (c) Free Software Foundation, Inc.
Source: https://gcc.gnu.org/
Purpose: Minimum gcc compiler version required to build HierarchicalKV-ascend
License: GPL v3 with GCC Runtime Library Exception
Please see below

Software: G++ >=8.1.0
Copyright notice:
Copyright (c) Free Software Foundation, Inc.
Source: https://gcc.gnu.org/
Purpose: Minimum g++ compiler version required to build HierarchicalKV-ascend
License: GPL v3 with GCC Runtime Library Exception
Please see below

Software: CMake >=3.16.0
Copyright notice:
Copyright (c) Kitware, Inc. and Contributors.
Source: https://cmake.org/
Purpose: Minimum cmake version required to build HierarchicalKV-ascend
License: BSD 3-Clause License
Please see below

================================================================================
License Texts
================================================================================

--------------------------------------------------------------------------------
License: BSD 3-Clause License
--------------------------------------------------------------------------------

Copyright (c) , All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

--------------------------------------------------------------------------------
License: GPL v3 with GCC Runtime Library Exception
--------------------------------------------------------------------------------

Copyright (c) , All rights reserved.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.

GCC Runtime Library Exception v3.1:

In addition to the permissions in the GNU General Public License, the Free Software Foundation gives you unlimited permission to link the compiled version of this file into combinations with other programs, and to distribute those combinations without any restrictions coming from the use of this file. (The General Public License restrictions do apply in other respects; for example, they cover modification of the file, and distribution when not linked into a combined executable.)

The full text of the GNU General Public License v3.0 can be found at:
https://www.gnu.org/licenses/gpl-3.0.txt
